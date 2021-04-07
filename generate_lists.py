import spacy
import os
import json
import nltk


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
        file_path = collection_path + '/' + filename        # get filename from directory
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            text_clean = text.strip('\n')
            f.close()

        doc = nlp(text_clean)
        idf_items_file = []
        for sentence in doc.sents:        # loop over sentences to get all n-grams within the sentence
            sent_list = [i.lemma_ for i in sentence] # try to get rid of punctuation and stuff
            for i in range(1, 4):       # n-grams chosen between 1 and 4
                ngrams = list(nltk.ngrams(sent_list, i))
                ngram_string = [" ".join(ngram) for ngram in ngrams]
                idf_items_file.extend(ngram_string)  # add them all together in 1 doc list

        idf_list.append(idf_items_file)     # now all n-grams are in a list per doc
    return idf_list


# create json files for the n-grams to save memory when calculating idf

# for directory in os.listdir("/home/gillesfloreal/PycharmProjects/ASTRA/en_train"):
#    dir_path = "/home/gillesfloreal/PycharmProjects/ASTRA/en_train/" + directory + "/texts/annotated"
#    idf_path_name = "/home/gillesfloreal/PycharmProjects/ASTRA/idf_lists/" + directory + ".json"
#    with open(idf_path_name, 'w') as outfile:
#        json.dump(generate_idf_list(dir_path)
#                 , outfile)

