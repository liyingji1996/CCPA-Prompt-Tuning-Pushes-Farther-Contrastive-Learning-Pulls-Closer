import os
import torch
import argparse
import numpy as np
import logging

from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AlbertModel

from os.path import join, abspath, dirname
from modeling import PTuneForLAMA
from experiments.def_sent_utils import get_def_pairs
from info_nce import InfoNCE
import traceback

logger = logging.getLogger(__name__)

SUPPORT_MODELS = ["bert-base-uncased", "albert-base-v2", "distilbert-base-uncased"]
FEMALES =["woman","girl","she","mother","daughter","gal","female","her","herself","Mary"]
MALES = ["man","boy","he","father","son", "guy","male","his","himself","John"]

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def construct_generation_args():
    parser = argparse.ArgumentParser()

    # pre-parsing args
    parser.add_argument("--relation_id", type=str, default="")
    parser.add_argument("--model_name_or_path", type=str, default="distilbert-base-uncased", choices=SUPPORT_MODELS)
    parser.add_argument("--pseudo_token", type=str, default='[PROMPT]')

    parser.add_argument("--template", type=str, default="(2,2,2)")

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=34, help="random seed for initialization")
    parser.add_argument("--decay_rate", type=float, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--BS", type=int, default=128)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--use_original_template", type=bool, default=False)
    parser.add_argument("--use_lm_finetune", type=bool, default=False)
    parser.add_argument("--lstm_dropout", type=float, default=0.0)
    parser.add_argument("--no_maha", type=bool, default=True)
    parser.add_argument("--beta", type=float, default=0.005, help="Weight of maha loss")

    parser.add_argument("--out_dir", type=str, default=join(abspath(dirname(__file__)), '../out'))
    parser.add_argument("--bias_type", default='gender', type=str, help="")

    args = parser.parse_args()
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.template = eval(args.template) if type(args.template) is not tuple else args.template

    assert type(args.template) is tuple

    set_seed(args)

    return args


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None,key_a=None,key_b=None):
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
        self.key_a = key_a
        self.key_b = key_b


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
                text_a=sent_a, text_b=sent_b, label=None,key_a=word_a,key_b=word_b))
    return def_examples


def findpos(words, sent):
    sent_list = sent.split(' ')
    for i, token in enumerate(sent_list):
        if token in words:
            return token
    return None


def mahalanobis(x, data):
    """Compute the Mahalanobis Distance between each row of x and the data
        x    : vector or matrix of data with, say, p columns.
        data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    """
    x_minus_mu = x - np.mean(data, axis=0)
    cov = np.cov(data.T)  # Calculating the covariance
    inv_covmat = np.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=False)
        # load datasets and dataloaders
        self.def_pairs = get_def_pairs(args.bias_type)  # Obtaining sentence pairs
        self.def_examples = get_def_examples(args, self.def_pairs)
        self.BS = args.BS
        self.device_ids = []
        os.makedirs(self.get_save_path(), exist_ok=True)  # Building the output file

        self.model = PTuneForLAMA(args, self.args.template)
        self.model.model = torch.nn.DataParallel(self.model.model, device_ids=self.device_ids)  # Parallel running PLM
        self.infonce = InfoNCE().cuda()

    def get_task_name(self):
        names = [self.args.model_name_or_path,
                 "template_{}".format(self.args.template if not self.args.use_original_template else 'original'),
                 "fixed" if not self.args.use_lm_finetune else "fine-tuned",
                 "seed_{}".format(self.args.seed)]
        return "_".join(names)

    def get_save_path(self):
        return join(self.args.out_dir, 'prompt_model', self.args.model_name_or_path, 'search', self.get_task_name(),
                    self.args.relation_id)

    def get_checkpoint(self, epoch_idx):
        ckpt_name = "epoch_{}.ckpt".format(epoch_idx)
        return {'embedding': self.model.prompt_encoder.state_dict(),
                'ckpt_name': ckpt_name,
                'time': datetime.now(),
                'args': self.args}

    def save(self, best_ckpt):
        ckpt_name = best_ckpt['ckpt_name']
        path = self.get_save_path()
        os.makedirs(path, exist_ok=True)
        torch.save(best_ckpt, join(path, ckpt_name))
        print("# {} Checkpoint {} saved.".format(self.args.relation_id, ckpt_name))

    def train(self):
        best_ckpt = None
        params = [{'params': self.model.prompt_encoder.parameters()}]
        optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)
        if not self.args.no_maha:
            output_a1, output_b1 = self.model.sequence(self.def_examples)

        for epoch_idx in range(15):
            # run training
            total_loss = 0
            for i in range(len(self.def_examples) // self.args.BS + 1):
                print("The {}th batch of the epoch_{}".format(i, epoch_idx))
                self.model.train()
                try:
                    examples = self.def_examples[self.args.BS * i:self.args.BS * (i + 1)]
                except IndexError:
                    examples = self.def_examples[self.args.BS * i:]
                output_a, output_b = self.model(examples)

                # Calculate the Cosin Similarity of sentences
                cos_matrix = torch.nn.functional.cosine_similarity(torch.mean(output_a, dim=1).float(), torch.mean(output_b, dim=1).float(), dim=1)
                loss_cos = torch.mean(cos_matrix, dim=0)
                print("Loss_Cos:", loss_cos.item())

                # Calculate the Mahalanobis Distance
                if not self.args.no_maha:
                    maha_matrix_a = torch.from_numpy(mahalanobis(torch.mean(output_a, dim=1).detach().cpu().numpy(),
                                                                 output_a1.detach().cpu().numpy()))
                    maha_matrix_b = torch.from_numpy(mahalanobis(torch.mean(output_b, dim=1).detach().cpu().numpy(),
                                                                 output_b1.detach().cpu().numpy()))
                    loss_maha = (torch.mean(maha_matrix_a, dim=0) + torch.mean(maha_matrix_b, dim=0)) / 2
                    loss = loss_cos + self.args.beta * loss_maha
                    print("Loss_Maha:", loss_maha.item())
                else:
                    loss = loss_cos

                print("Loss:", loss.item())
                total_loss += loss.item()
                loss.requires_grad_(True)

                loss.backward()
                torch.cuda.empty_cache()
                optimizer.step()
                torch.cuda.empty_cache()
                optimizer.zero_grad()

            print("Mean loss:", total_loss / (len(self.def_examples) // self.args.BS + 1))
            my_lr_scheduler.step()
            best_ckpt = self.get_checkpoint(epoch_idx)
            self.save(best_ckpt)  # Save Prompt Encoder
        return best_ckpt


def main(relation_id=None):
    args = construct_generation_args()
    if relation_id:
        args.relation_id = relation_id
    if type(args.template) is not tuple:
        args.template = eval(args.template)
    assert type(args.template) is tuple
    print(args.relation_id)
    print(args.model_name_or_path)
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
