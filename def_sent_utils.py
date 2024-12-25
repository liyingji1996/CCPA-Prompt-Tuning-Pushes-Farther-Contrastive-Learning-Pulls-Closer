# standard library
from itertools import combinations
import numpy as np
import os, sys
from collections import defaultdict


sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
np.random.seed(42)
words2 = [["woman", "man"], ["girl", "boy"], ["she", "he"], ["mother", "father"], ["daughter", "son"], ["gal", "guy"], ["female", "male"], ["her", "his"], ["herself", "himself"], ["Mary", "John"]]
DIRECTORY = ''
GENDER = 0



def match(a, L):
    for b in L:
        if a == b:
            return True
    return False


def replace(a, new, L):
    Lnew = []
    for b in L:
        if a == b:
            Lnew.append(new)
        else:
            Lnew.append(b)
    return ' '.join(Lnew)


def template1(words, sent, sent_list, all_pairs):
    for i, (race1, race2) in enumerate(words):
        if match(race1, sent_list):
            sent_r1 = sent
            sent_r2 = replace(race1, race2, sent_list)
            all_pairs[i]['m'].append(sent_r1)
            all_pairs[i]['f'].append(sent_r2)
        if match(race2, sent_list):
            sent_r1 = replace(race2, race1, sent_list)
            sent_r2 = sent
            all_pairs[i]['m'].append(sent_r1)
            all_pairs[i]['f'].append(sent_r2)
    return all_pairs


def template2(words, sent, sent_list, all_pairs):
    for i, (female, male) in enumerate(words):
        if match(female, sent_list):
            sent_f = sent
            sent_m = replace(female,male,sent_list)
            all_pairs[i]['f'].append(sent_f)
            all_pairs[i]['m'].append(sent_m)
        if match(male, sent_list):
            sent_f = replace(male,female,sent_list)
            sent_m = sent
            all_pairs[i]['f'].append(sent_f)
            all_pairs[i]['m'].append(sent_m)
    return all_pairs


def template3(words, sent, sent_list, all_pairs):
    for (b1,b2,b3) in words:
        if match(b1, sent_list):
            sent_b1 = sent
            sent_b2 = replace(b1,b2,sent_list)
            sent_b3 = replace(b1,b3,sent_list)
            pair = (sent_b1,sent_b2,sent_b3)
            all_pairs.append(pair)
        if match(b2, sent_list):
            sent_b1 = replace(b2,b1,sent_list)
            sent_b2 = sent
            sent_b3 = replace(b2,b3,sent_list)
            pair = (sent_b1,sent_b2,sent_b3)
            all_pairs.append(pair)
        if match(b3, sent_list):
            sent_b1 = replace(b3,b1,sent_list)
            sent_b2 = replace(b3,b2,sent_list)
            sent_b3 = sent
            pair = (sent_b1,sent_b2,sent_b3)
            all_pairs.append(pair)
    return all_pairs


def get_pom(bais_type):
    all_pairs2 = defaultdict(lambda: defaultdict(list))
    pom_loc = os.path.join(DIRECTORY, 'POM/')

    for file in os.listdir(pom_loc):
        if file.endswith(".txt"):
            f = open(os.path.join(pom_loc, file), 'r')
            data = f.read()
            for sent in data.lower().split('.'):
                sent = sent.strip()
                sent_list = sent.split(' ')
                if bais_type == 'gender' and len(sent_list) < 110:
                    all_pairs = template2(words2, sent, sent_list, all_pairs2)
    return all_pairs


def get_rest(filename, bais_type):
    all_pairs2 = defaultdict(lambda: defaultdict(list))

    f = open(os.path.join(DIRECTORY, filename), 'r')
    data = f.read()
    for sent in data.lower().split('\n'):
        sent = sent.strip()
        sent_list = sent.split(' ')
        if bais_type == 'gender' and len(sent_list) < 110:
            all_pairs = template2(words2, sent, sent_list, all_pairs2)

    print(filename, len(all_pairs))  # print: reddit.txt 9
    return all_pairs


def get_sst(bais_type):
    all_pairs2 = defaultdict(lambda: defaultdict(list))

    for sent in open(os.path.join(DIRECTORY, 'sst.txt'), 'r'):
        try:
            num = int(sent.split('\t')[0])
            sent = sent.split('\t')[1:]
            sent = ' '.join(sent)
        except:
            pass
        sent = sent.lower().strip()
        sent_list = sent.split(' ')
        if bais_type == 'gender' and len(sent_list) < 110:
            all_pairs = template2(words2, sent, sent_list, all_pairs2)
    return all_pairs


def check_bucket_size(D):
    n = 0
    for i in D:
        for key in D[i]:
            n += len(D[i][key])
            break
    return n


# domain: news, reddit, sst, pom, wikitext
def get_single_domain(domain, bais_type):
    if (domain == "pom"):
        def_pairs = get_pom(bais_type)
    elif (domain == "sst"):
        def_pairs = get_sst(bais_type)
    else:
        def_pairs = get_rest("{}.txt".format(domain), bais_type)
    return def_pairs


def get_def_pairs(bais_type):
    domains = ["reddit", "sst", "wikitext", "pom", "meld"]
    print("Get data from {}".format(domains))
    all_data = defaultdict(lambda: defaultdict(list))
    for domain in domains:
        bucket = get_single_domain(domain, bais_type)
        bucket_size = check_bucket_size(bucket)
        print("{} has {} pairs of templates".format(domain, bucket_size))
        for i in bucket:
            for term in bucket[i]:
                all_data[i][term].extend(bucket[i][term])
    total_size = check_bucket_size(all_data)
    print("{} pairs of templates in total".format(total_size))
    return all_data