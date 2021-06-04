import json
import os
import nltk
import math
""""
Created an n-gram_dictionary beforehand, this time sorted on ngram size, with their frequencies.
"""


def c_value(ct_dict):
    ngram_dict = dict(sorted(ct_dict.items(), key=lambda item: len(item[0].split()), reverse=True))

    c_value_count_dict = dict()
    for candidate in ngram_dict:
        candidate_split = candidate.split()
        length = len(candidate_split)
        for i in reversed(range(1, length - 1)):
            children_ngrams = list(nltk.ngrams(candidate_split, i))
            for child_ngram in children_ngrams:
                if child_ngram in ngram_dict:
                    if child_ngram in c_value_count_dict:
                        c_value_count_dict[child_ngram]['nested_count'] += 1
                        c_value_count_dict[child_ngram]['parent_count'] += ngram_dict[candidate]
                    else:
                        c_value_count_dict[child_ngram] = dict()
                        c_value_count_dict[child_ngram]['nested_count'] = 1
                        c_value_count_dict[child_ngram]['parent_count'] = ngram_dict[candidate]
                        c_value_count_dict[child_ngram]['total_count'] = ngram_dict[child_ngram]

    c_value_dict = {}

    for candidate in ngram_dict:
        # if nested
        if candidate in c_value_count_dict:
            c_value_dict[candidate] = ((math.log(0.1 + len(candidate), 2)) * c_value_count_dict[candidate]['total_count'] - ((1/c_value_count_dict[candidate]['nested_count']) * c_value_count_dict[candidate]['parent_count']))

        else:
        # if not nested
            c_value_dict[candidate] = ((math.log(0.1 + len(candidate), 2)) * ngram_dict[candidate])

    return c_value_dict


ct_file_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/training.json"
with open(ct_file_path, 'r', encoding='utf8') as source:
    ct_list = json.load(source)
    source.close()

c_value_list = []
for ct_dict in ct_list:
    c_value_list.append(c_value(ct_dict))

target_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/statistical_scores/c_values/c_values_list.json"
with open(target_path, 'w') as target:
    json.dump(c_value_list, target)

