import torch
from torch.nn.utils.rnn import pad_sequence
from os.path import join
import sys
import numpy
from typing import Optional, Union, List, Dict, Tuple

from transformers import AutoTokenizer, AutoModelForMaskedLM, DistilBertForMaskedLM
from p_tuning.prompt_encoder import PromptEncoder


def get_embedding_layer(args, model):
    if 'albert' in args.model_name_or_path:
        embeddings = model.albert.get_input_embeddings()
    # elif 'bert' in args.model_name_or_path:
    #     embeddings = model.bert.get_input_embeddings()
    elif 'distilbert' in args.model_name_or_path:
        embeddings = model.distilbert.get_input_embeddings()
    else:
        raise NotImplementedError()
    return embeddings


def load_promptencoder(template, hidden_size, args):
    ckpt_name = ""
    encoder_state = torch.load(ckpt_name, map_location='cuda:0')
    prompt_encoder = PromptEncoder(template, hidden_size, args)
    prompt_encoder.load_state_dict(encoder_state['embedding'])
    return prompt_encoder


class PTuneForLAMA(torch.nn.Module):

    def __init__(self, args, template):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

        # load pre-trained model
        self.model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
        self.model = self.model.cuda()
        for param in self.model.parameters():
            param.requires_grad = True
        self.embeddings = get_embedding_layer(self.args, self.model)

        self.template = template
        self.no_prompt = args.no_prompt

        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        self.tokenizer.add_special_tokens({'additional_special_tokens': [self.args.pseudo_token]})
        self.pseudo_token_id = self.tokenizer.get_vocab()[self.args.pseudo_token]
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id

        self.spell_length = sum(self.template)
        if not self.args.no_prompt:
            self.prompt_encoder = load_promptencoder(self.template, self.hidden_size, args)  # Load the trained prompt_encoder
            self.prompt_encoder = self.prompt_encoder.cuda()

    def embed_input(self, queries):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding)

        # For using handcraft prompts
        if self.args.use_original_template:
            return raw_embeds

        blocked_indices = (queries == self.pseudo_token_id).nonzero().reshape(bz, self.spell_length, 2)[:, :, 1]  # bz
        replace_embeds = self.prompt_encoder()
        for bidx in range(bz):
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        return raw_embeds

    def get_query(self, text, prompt_tokens):
        # For using handcraft prompts
        if self.args.use_original_template:
            query = self.relation_templates[self.args.relation_id].replace('[X]', text).replace('[Y]',
                                                                                                self.tokenizer.mask_token)
            return self.tokenizer(' ' + query)['input_ids']
        # For Prompt-tuning
        if prompt_tokens:
            if self.args.do_mlm:
                text_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text)))
                text_ids, label_ids = self.mask_tokens(text_ids)
                text_ids = text_ids.squeeze(0).tolist()

                return [[self.tokenizer.cls_token_id]  # [CLS]
                        + prompt_tokens * self.template[0]
                        + prompt_tokens * self.template[1]
                        + prompt_tokens * self.template[2]
                        + text_ids  # sentence ids
                        + [self.tokenizer.sep_token_id]  # [SEP]
                        ], label_ids
            elif not self.args.do_mlm:
                return [[self.tokenizer.cls_token_id]  # [CLS]
                        + prompt_tokens * self.template[0]
                        + prompt_tokens * self.template[1]
                        + prompt_tokens * self.template[2]
                        + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))  # sentence ids
                        + [self.tokenizer.sep_token_id]  # [SEP]
                        ]

        # no_prompt
        elif prompt_tokens is None:
            if self.args.do_mlm:
                text_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text)))
                text_ids, label_ids = self.mask_tokens(text_ids)
                text_ids = text_ids.squeeze(0).tolist()
                return [[self.tokenizer.cls_token_id]  # [CLS]
                        + text_ids  # sentence ids
                        + [self.tokenizer.sep_token_id]  # [SEP]
                        ], label_ids
            elif not self.args.do_mlm:
                return [[self.tokenizer.cls_token_id]  # [CLS]
                        + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))  # sentence
                        + [self.tokenizer.sep_token_id]  # [SEP]
                        ]

        else:
            raise NotImplementedError("The query template for {} has not been defined.".format(self.args.model_name_or_path))

    def mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None,) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        labels = labels.unsqueeze(0)
        inputs = inputs.clone()
        inputs = inputs.unsqueeze(0)

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.args.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(
                special_tokens_mask, dtype=torch.bool
            )
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
                torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
                torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
                & masked_indices
                & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def forward(self, examples):
        bz = len(examples)
        # no prompt
        if self.no_prompt:
            # add masked loss
            if self.args.do_mlm:
                prompt_tokens = None
                queries_a1 = []
                queries_b1 = []
                mlm_labels_a1 = []
                mlm_labels_b1 = []
                # construct query ids
                for i in range(bz):
                    queries_a, mlm_labels_a = self.get_query(examples[i].text_a, prompt_tokens)
                    queries_a1.append(torch.LongTensor(queries_a).squeeze(0))
                    mlm_labels_a1.append(mlm_labels_a.squeeze(0))
                    queries_b, mlm_labels_b = self.get_query(examples[i].text_b, prompt_tokens)
                    queries_b1.append(torch.LongTensor(queries_b).squeeze(0))
                    mlm_labels_b1.append(mlm_labels_b.squeeze(0))

                queries_a = pad_sequence(queries_a1, True, padding_value=self.pad_token_id).long().cuda()
                queries_b = pad_sequence(queries_b1, True, padding_value=self.pad_token_id).long().cuda()
                mlm_labels_a1 = pad_sequence(mlm_labels_a1, True, padding_value=-100).long().cuda()
                mlm_labels_b1 = pad_sequence(mlm_labels_b1, True, padding_value=-100).long().cuda()
                t1 = torch.zeros((bz, 1)).long().cuda()
                t2 = torch.zeros((bz, 1)).long().cuda()
                t1 += -100
                t2 += -100
                mlm_labels_a1 = torch.cat([t1, mlm_labels_a1, t2], dim=-1)
                mlm_labels_b1 = torch.cat([t1, mlm_labels_b1, t2], dim=-1)
                assert queries_a.shape == mlm_labels_a1.shape
                assert queries_b.shape == mlm_labels_b1.shape

                attention_mask_a = queries_a != self.pad_token_id
                attention_mask_b = queries_b != self.pad_token_id

                output_a = self.model(input_ids=queries_a, attention_mask=attention_mask_a.bool(),
                                      labels=mlm_labels_a1, return_dict=True, output_hidden_states=True)
                output_b = self.model(input_ids=queries_b, attention_mask=attention_mask_b.bool(),
                                      labels=mlm_labels_b1, return_dict=True, output_hidden_states=True)
                return output_a, output_b
            # no masked loss
            elif not self.args.do_mlm:
                prompt_tokens = None
                queries_a = [torch.LongTensor(self.get_query(examples[i].text_a, prompt_tokens)).squeeze(0) for i in
                             range(bz)]
                queries_b = [torch.LongTensor(self.get_query(examples[i].text_b, prompt_tokens)).squeeze(0) for i in
                             range(bz)]
                queries_a = pad_sequence(queries_a, True, padding_value=self.pad_token_id).long().cuda()
                queries_b = pad_sequence(queries_b, True, padding_value=self.pad_token_id).long().cuda()

                attention_mask_a = queries_a != self.pad_token_id
                attention_mask_b = queries_b != self.pad_token_id
                output_a = self.model(input_ids=queries_a,
                                      attention_mask=attention_mask_a.bool(), return_dict=True, output_hidden_states=True)
                output_b = self.model(input_ids=queries_b,
                                      attention_mask=attention_mask_b.bool(), return_dict=True, output_hidden_states=True)
                return output_a, output_b

        # prompt tuning
        else:
            # add masked loss
            if self.args.do_mlm:
                prompt_tokens = [self.pseudo_token_id]
                queries_a1 = []
                queries_b1 = []
                mlm_labels_a1 = []
                mlm_labels_b1 = []
                # construct query ids
                for i in range(bz):
                    queries_a, mlm_labels_a = self.get_query(examples[i].text_a, prompt_tokens)
                    queries_a1.append(torch.LongTensor(queries_a).squeeze(0))
                    mlm_labels_a1.append(mlm_labels_a.squeeze(0))
                    queries_b, mlm_labels_b = self.get_query(examples[i].text_b, prompt_tokens)
                    queries_b1.append(torch.LongTensor(queries_b).squeeze(0))
                    mlm_labels_b1.append(mlm_labels_b.squeeze(0))

                queries_a = pad_sequence(queries_a1, True, padding_value=self.pad_token_id).long().cuda()
                queries_b = pad_sequence(queries_b1, True, padding_value=self.pad_token_id).long().cuda()
                mlm_labels_a1 = pad_sequence(mlm_labels_a1, True, padding_value=-100).long().cuda()
                mlm_labels_b1 = pad_sequence(mlm_labels_b1, True, padding_value=-100).long().cuda()
                t1 = torch.zeros((bz, 1 + self.spell_length)).long().cuda()
                t2 = torch.zeros((bz, 1)).long().cuda()
                t1 += -100
                t2 += -100
                mlm_labels_a1 = torch.cat([t1, mlm_labels_a1, t2], dim=-1)
                mlm_labels_b1 = torch.cat([t1, mlm_labels_b1, t2], dim=-1)
                assert queries_a.shape == mlm_labels_a1.shape
                assert queries_b.shape == mlm_labels_b1.shape

                attention_mask_a = queries_a != self.pad_token_id
                attention_mask_b = queries_b != self.pad_token_id

                # get embedded input
                inputs_embeds_a = self.embed_input(queries_a)
                inputs_embeds_b = self.embed_input(queries_b)
                output_a = self.model(inputs_embeds=inputs_embeds_a, attention_mask=attention_mask_a.bool(),
                                      labels=mlm_labels_a1, return_dict=True, output_hidden_states=True)
                output_b = self.model(inputs_embeds=inputs_embeds_b, attention_mask=attention_mask_b.bool(),
                                      labels=mlm_labels_b1, return_dict=True, output_hidden_states=True)
                return output_a, output_b
            # no masked loss
            elif not self.args.do_mlm:
                prompt_tokens = [self.pseudo_token_id]
                # construct query ids
                queries_a = [torch.LongTensor(self.get_query(examples[i].text_a, prompt_tokens)).squeeze(0) for i in
                             range(bz)]
                queries_b = [torch.LongTensor(self.get_query(examples[i].text_b, prompt_tokens)).squeeze(0) for i in
                             range(bz)]
                queries_a = pad_sequence(queries_a, True, padding_value=self.pad_token_id).long().cuda()
                queries_b = pad_sequence(queries_b, True, padding_value=self.pad_token_id).long().cuda()

                attention_mask_a = queries_a != self.pad_token_id
                attention_mask_b = queries_b != self.pad_token_id

                # get embedded input
                inputs_embeds_a = self.embed_input(queries_a)
                inputs_embeds_b = self.embed_input(queries_b)
                output_a = self.model(inputs_embeds=inputs_embeds_a, attention_mask=attention_mask_a.bool(),
                                      return_dict=True, output_hidden_states=True)
                output_b = self.model(inputs_embeds=inputs_embeds_b, attention_mask=attention_mask_b.bool(),
                                      return_dict=True, output_hidden_states=True)
                return output_a, output_b
