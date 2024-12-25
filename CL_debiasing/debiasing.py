import os
import torch
import argparse
import numpy as np
import logging
from transformers import AutoTokenizer

from os.path import join, abspath, dirname
from debiasing_modeling import PTuneForLAMA
import sys
from experiments.def_sent_utils import get_def_pairs
from transformers import AdamW
from info_nce import InfoNCE
import traceback

# from info-nce-pytorch import InfoNCE
logger = logging.getLogger(__name__)

SUPPORT_MODELS = ["bert-base-uncased", "albert-base-v2", "distilbert-base-uncased"]
FEMALES = ["woman", "girl", "she", "mother", "daughter", "gal", "female", "her", "herself", "Mary"]
MALES = ["man", "boy", "he", "father", "son", "guy", "male", "his", "himself", "John"]



def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def construct_generation_args():
    parser = argparse.ArgumentParser()

    # pre-parsing args
    parser.add_argument("--relation_id", type=str, default="")
    parser.add_argument("--key", type=int, default=19)
    parser.add_argument("--model_name_or_path", type=str, default="distilbert-base-uncased", choices=SUPPORT_MODELS)
    parser.add_argument("--pseudo_token", type=str, default='[PROMPT]')

    parser.add_argument("--template", type=str, default="(2, 2, 2)")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=34, help="random seed for initialization")
    parser.add_argument("--decay_rate", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--BS", type=int, default=64)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--use_original_template", type=bool, default=False)
    parser.add_argument("--use_lm_finetune", type=bool, default=True)
    parser.add_argument("--no_prompt", type=bool, default=False)
    parser.add_argument("--lstm_dropout", type=float, default=0.0)

    # directories
    parser.add_argument("--out_dir", type=str, default=join(abspath(dirname(__file__)), '../out'))
    parser.add_argument("--bias_type", default='gender', type=str, help="")
    parser.add_argument("--do_mlm", type=bool, default=True)
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for MLM (only effective if --do_mlm)")
    parser.add_argument("--lam", type=float, default=1,
                        help="Weight of mlm loss")

    args = parser.parse_args()

    # post-parsing args

    # args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.template = eval(args.template) if type(args.template) is not tuple else args.template

    assert type(args.template) is tuple

    set_seed(args)
    return args


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, key_a=None, key_b=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.key_a = key_a  # the bias word position
        self.key_b = key_b  # the bias word position


def get_def_examples(args, def_pairs):
    '''Construct definitional examples from definitional pairs.'''
    def_examples = []
    for group_id in def_pairs:
        def_group = def_pairs[group_id]
        f_sents = def_group['f']
        m_sents = def_group['m']

        if args.bias_type == 'gender':
            f_refer = set(FEMALES)
            m_refer = set(MALES)
        for sent_id, (sent_a, sent_b) in enumerate(zip(f_sents, m_sents)):
            word_a = findpos(f_refer, sent_a)
            word_b = findpos(m_refer, sent_b)
            assert word_a is not None
            assert word_b is not None
            def_examples.append(InputExample(guid='{}-{}'.format(group_id, sent_id),
                                             text_a=sent_a, text_b=sent_b, label=None, key_a=word_a, key_b=word_b))
    return def_examples


def findpos(words, sent):
    sent_list = sent.split(' ')
    for i, token in enumerate(sent_list):
        if token in words:
            return token
    return None


class Trainer(object):
    def __init__(self, args):
        self.args = args
        # self.device = 'cuda:2' if torch.cuda.is_available() and not args.no_cuda else "cpu"

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=False)

        # load datasets and dataloaders
        self.def_pairs = get_def_pairs(args.bias_type)  # Obtaining sentence pairs
        self.def_examples = get_def_examples(args, self.def_pairs)
        self.BS = args.BS
        os.makedirs(self.get_save_path(), exist_ok=True)  # Building the output file
        self.device_ids = []
        self.model = PTuneForLAMA(args, self.args.template)
        self.model.model = torch.nn.DataParallel(self.model.model, device_ids=self.device_ids)
        self.infonce = InfoNCE().cuda()

    def get_task_name(self):
        names = [self.args.model_name_or_path,
                 "template_{}".format(self.args.template if not self.args.use_original_template else 'original'),
                 "fixed" if not self.args.use_lm_finetune else "fine-tuned",
                 "seed_{}".format(self.args.seed)]
        return "_".join(names)

    def get_save_path(self):
        return join(self.args.out_dir, 'fine_tune', self.args.model_name_or_path, self.get_task_name(),
                    self.args.relation_id)

    def train(self):
        params = [{'params': self.model.model.parameters()}]
        # optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        optimizer = AdamW(params, lr=self.args.lr)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)

        for epoch_idx in range(20):
            # run training
            total_loss = 0
            for i in range(len(self.def_examples) // self.BS + 1):
                try:
                    self.model.model.train()
                    try:
                        examples = self.def_examples[self.BS * i:self.BS * (i + 1)]
                    except IndexError:
                        examples = self.def_examples[self.BS * i:]

                    output_a, output_b = self.model(examples)
                    loss_cl = self.infonce(torch.mean(output_a.hidden_states[-1], dim=1),
                                           torch.mean(output_b.hidden_states[-1], dim=1))
                    if self.args.do_mlm:
                        loss_mlm = (torch.sum(output_a.loss, dim=0) + torch.sum(output_b.loss, dim=0)) / 2
                        loss = loss_cl + self.args.lam * loss_mlm
                        print("The {}th batch of the epoch_{}".format(i, epoch_idx))
                        print("Loss:", loss_mlm.item())
                        print("Loss:", loss_cl.item())
                        print("Loss:", loss.item())
                    else:
                        loss = loss_cl
                        print("The {}th batch of the epoch_{}".format(i, epoch_idx))
                        print("Loss:", loss_cl.item())
                    total_loss += loss.item()

                    loss.backward()
                    torch.cuda.empty_cache()
                    optimizer.step()
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                except RuntimeError:
                    traceback.print_exc()
                    continue

            print("Mean loss:", total_loss / (len(self.def_examples) // self.BS + 1))
            my_lr_scheduler.step()
            # Save checkpoint
            ckpt_name = "epoch_{}_{}.ckpt".format(epoch_idx, self.args.model_name_or_path)
            path = self.get_save_path()
            os.makedirs(path, exist_ok=True)
            self.model.model.module.save_pretrained(join(path, ckpt_name))
            print("# {} Checkpoint {} saved.".format(self.args.relation_id, ckpt_name))
        return None


def main():
    args = construct_generation_args()
    if args.no_prompt:
        args.relation_id = "long_1-mlm_no-prompt_lr1e-4"
        if type(args.template) is not tuple:
            args.template = eval(args.template)
        assert type(args.template) is tuple
        print(args.relation_id)
        print(args.model_name_or_path)
        trainer = Trainer(args)
        trainer.train()
    else:
        for i in range(0, 15):
            args.relation_id = "long_1-mlm_005-maha_lr1e-4-epoch{}".format(i)
            args.key = i
            if type(args.template) is not tuple:
                args.template = eval(args.template)
            assert type(args.template) is tuple
            print(args.relation_id)
            print(args.model_name_or_path)
            trainer = Trainer(args)
            trainer.train()


if __name__ == '__main__':
    main()
