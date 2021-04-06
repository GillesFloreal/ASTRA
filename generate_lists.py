import spacy
import os
import json


def generate_term_list(path):
    with open(path, 'r', encoding='utf') as f:
        lines = f.readlines()
        f.close()

    term_label_pairs = {}

    for line in lines:
        line_clean = line.strip('\n')
        pair = tuple(line_clean.split('\t'))
        term_label_pairs[pair[0]] = pair[1]

    return term_label_pairs


def generate_idf_list(collection_path):
    nlp = spacy.load("en_core_web_md")
    idf_list = []

    for filename in os.listdir(collection_path):
        file_path = collection_path + '/' + filename
        with open(file_path, 'r', encoding='utf-8') as f:
            file = f.read()
            f.close()

        doc = nlp(file)
        file_list = [token.lemma_ for token in doc]

        idf_list.append(tuple(file_list))

    return idf_list


for directory in os.listdir("/home/gillesfloreal/PycharmProjects/ASTRA/en_train"):
    dir_path = "/home/gillesfloreal/PycharmProjects/ASTRA/en_train/" + directory + "/texts/annotated"
    idf_path_name = "/home/gillesfloreal/PycharmProjects/ASTRA/idf_lists/" + directory + ".json"
    with open("/home/gillesfloreal/PycharmProjects/ASTRA/idf_lists/htfl.json", 'w') as outfile:
        json.dump(generate_idf_list(dir_path)
                  , outfile)

