import json
import os
import nltk
import math

ct_dir_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/ling_prepr/corpus_CT"
stop_words = ['a', 'the', 'on', 'an', 'of', 'from', 'for', 'to', 'into', 'in', 'with', 'by'
              'over', '-', '.', '<', '>', '"', "'", "-", '-']


def c_value(ct_file):

    ct_file_path = ct_dir_path + '/' + ct_file
    with open(ct_file_path, 'r', encoding='utf8') as f:
        ct_dict = json.load(f)
        f.close()
    ngram_dict = dict()
    for i in ct_dict:
        ngram_dict.update(dict(sorted(ct_dict[i].items(), key=lambda item: item[1], reverse=True)))

    c_value_count_dict = dict()
    for candidate in ngram_dict:
        candidate_split = candidate.split()
        length = len(candidate_split)
        for i in reversed(range(1, length - 1)):
            children_ngrams = list(nltk.ngrams(candidate_split, i))
            unwanted = False
            for child_ngram in children_ngrams:
                for child in child_ngram:
                    if child in stop_words or child.isnumeric() is True:
                        unwanted = True
                if unwanted is True:
                    continue
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
        candidate_unwanted = False
        if candidate in c_value_count_dict:
            c_value_dict[candidate] = ((math.log(0.1 + len(candidate), 2)) * c_value_count_dict[candidate]['total_count'] - ((1/c_value_count_dict[candidate]['nested_count']) * c_value_count_dict[candidate]['parent_count']))

        else:
            for word in candidate.split():
                if word in stop_words or word.isnumeric() is True:
                    candidate_unwanted = True

            if candidate_unwanted is True:
                continue
            c_value_dict[candidate] = ((math.log(0.1 + len(candidate), 2)) * ngram_dict[candidate])

    return c_value_dict


for ct_file in os.listdir(ct_dir_path):
    file_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/statistical_scores/c_values/" + "c_value_" + ct_file
    with open(file_path, 'w') as target:
        json.dump(c_value(ct_file), target)

