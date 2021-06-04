import spacy
import nltk
import os
import json
import math
from pathlib import Path

# create freq dict from ref_corp
nlp = spacy.load("en_core_web_md")
ref_corp_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/ref_corp/text.txt"
ct_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/ling_prepr/corpus_CT"


def freq_dict_ref_corp(ref_corp):
    freq_dict = dict()
    total_count = 0

    with open(ref_corp, 'r', encoding='utf8') as f:
        for line in f.readlines():
            doc = nlp(line)
            for sentence in doc.sents:
                sent_list = [i.text for i in sentence]
                for i in reversed(range(1, 7)):
                    ngrams = list(nltk.ngrams(sent_list, i))
                    for ngram in ngrams:
                        ngram_str = " ".join(ngram).lower()
                        total_count += 1
                        if ngram_str in freq_dict:
                            freq_dict[ngram_str] += 1
                        else:
                            freq_dict[ngram_str] = 1
        f.close()
    return freq_dict


def vintar(ling_prep_dict, ref_dict, ref_total_count, texts):

    # we also need a freq_dict for all unigrams in our domain corpus
    unigram_dict = {}
    domain_count = 0
    for text in texts:
        with open(text, 'r', encoding='utf8') as f:
            domain_doc = nlp(f.read())
            f.close()
        for domain_sentence in domain_doc.sents:
            for token in domain_sentence:
                try:
                    if token.lower() in unigram_dict:
                        unigram_dict[token.lower()] += 1
                    else:
                        unigram_dict[token.lower()] = 1
                except:
                    continue

    # now we calculate vintar score for each CT

    vintar_scores = dict()
    ngram_dict = dict()

    for ct in ling_prep_dict:
        ct_split = ct.split()
        summation = 0
        for word in ct_split:
            if word not in unigram_dict:
                continue
            if word in ref_dict:
                summation += (math.log((unigram_dict[word]/domain_count) + 1) - math.log((ref_dict[word]/ref_total_count) + 1))
            else:
                summation += (math.log((unigram_dict[word]/domain_count) + 1) - math.log((1/ref_total_count)) + 1)

        vintar_scores[ct] = ((ling_prep_dict[ct] ** 2)/len(ct_split)) * summation

    return vintar_scores


# create ref_corp in separate file

ref_corp_dict_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/ref_corp/freq_dict"
with open(ref_corp_dict_path, 'w', encoding='utf8') as ref_target:
    json.dump(freq_dict_ref_corp(ref_corp_path), ref_target)

ref_dict_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/ref_corp/freq_dict"
with open(ref_dict_path, 'r') as f:
    ref_dict = json.load(f)
    f.close()

ref_total_count = 0
for value in ref_dict.values():
    ref_total_count += value


source_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/training.json"
target_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/statistical_scores/vintar_values/vintar.json"
with open(source_path, 'r', encoding='utf8') as source:
    ct_list = json.load(source)

base_path = Path("/home/gillesfloreal/PycharmProjects/ASTRA/data/en_train/")
domains = ["corp", "equi", "htfl", "wind"]
vintar_values = []

for ct_dict, domain in zip(ct_list, domains):
    texts = base_path.joinpath(domain).rglob('*.txt')
    vintar_values.append(vintar(ct_dict, ref_dict, ref_total_count, texts))

with open(target_path, 'w', encoding='utf8') as target:
    json.dump(vintar_values, target)
