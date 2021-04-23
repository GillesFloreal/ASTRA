import json
import os
import nltk
import math

ct_dir_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/ling_prepr/corpus_CT"


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

    # highest c-value is 9442, in order to have values between 0 and 1, we divide by 9442

    for candidate in ngram_dict:
        if candidate in c_value_count_dict:
            c_value_dict[candidate] = (math.log(len(candidate), 2) * c_value_count_dict[candidate]['total_count'] - ((1/c_value_count_dict[candidate]['nested_count']) * c_value_count_dict[candidate]['parent_count'])) / 9442

        else:
            c_value_dict[candidate] = (math.log(len(candidate), 2) * ngram_dict[candidate]) / 9442

    return c_value_dict


for ct_file in os.listdir(ct_dir_path):
    file_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/ling_prepr/" + "c_value_" + ct_file
    with open(file_path, 'w') as target:
        json.dump(c_value(ct_file), target)

