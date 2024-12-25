import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel,AlbertTokenizer,AlbertModel
from prompt_encoder import PromptEncoder


class PTuneForLAMA(torch.nn.Module):

    def __init__(self, args, template):
        super().__init__()
        self.args = args
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

        # load pre-trained model
        self.model = AutoModel.from_pretrained(args.model_name_or_path)
        if not args.use_lm_finetune:
            self.model = self.model.half()
        self.model = self.model.cuda()
        for param in self.model.parameters():
            param.requires_grad = self.args.use_lm_finetune
        self.embeddings = self.model.get_input_embeddings()

        self.template = template

        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        self.tokenizer.add_special_tokens({'additional_special_tokens': [self.args.pseudo_token]})
        self.pseudo_token_id = self.tokenizer.get_vocab()[self.args.pseudo_token]
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id
        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, args)
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
            return [[self.tokenizer.cls_token_id]  # [CLS]
                    + prompt_tokens * self.template[0]
                    + prompt_tokens * self.template[1]
                    + prompt_tokens * self.template[2]
                    + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))  # sentence ids
                    + [self.tokenizer.sep_token_id]  # [SEP]
                    ]
        # no_prompt
        elif prompt_tokens is None:
            return [[self.tokenizer.cls_token_id]  # [CLS]
                    + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))  # sentence
                    + [self.tokenizer.sep_token_id]  # [SEP]
                    ]
        else:
            raise NotImplementedError("The query template for {} has not been defined.".format(self.args.model_name_or_path))

    def sequence(self, def_examples):
        output_a = []
        output_b = []
        prompt_tokens = None
        for i in range(len(def_examples) // self.args.BS + 1):
            self.model.train()
            try:
                examples = def_examples[self.args.BS * i:self.args.BS * (i + 1)]
            except IndexError:
                examples = def_examples[self.args.BS * i:]
            queries_a1 = [torch.LongTensor(self.get_query(examples[i].text_a, prompt_tokens)).squeeze(0) for i in range(len(examples))]
            queries_b1 = [torch.LongTensor(self.get_query(examples[i].text_b, prompt_tokens)).squeeze(0) for i in range(len(examples))]
            queries_a1 = pad_sequence(queries_a1, True, padding_value=self.pad_token_id).long().cuda()
            queries_b1 = pad_sequence(queries_b1, True, padding_value=self.pad_token_id).long().cuda()

            attention_mask_a1 = queries_a1 != self.pad_token_id
            attention_mask_b1 = queries_b1 != self.pad_token_id
            output_a1 = torch.mean(self.model(input_ids=queries_a1, attention_mask=attention_mask_a1.bool(),
                                   return_dict=True).last_hidden_state, dim=1)
            output_b1 = torch.mean(self.model(input_ids=queries_b1, attention_mask=attention_mask_b1.bool(),
                                              return_dict=True).last_hidden_state, dim=1)

            output_a.append(output_a1)
            output_b.append(output_b1)
        output_a = torch.cat(output_a, dim=0)
        output_b = torch.cat(output_b, dim=0)
        return output_a, output_b

    def forward(self, examples):
        bz = len(examples)
        # prompt tuning
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
                              return_dict=True).last_hidden_state
        output_b = self.model(inputs_embeds=inputs_embeds_b, attention_mask=attention_mask_b.bool(),
                              return_dict=True).last_hidden_state
        return output_a, output_b
