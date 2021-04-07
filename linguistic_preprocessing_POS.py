from generate_lists import generate_term_list
import os
import spacy
import nltk
import json


def len_ngrams(term_list):
    length = 0
    for term in term_list:
        if len(term.split()) > length:
            length = len(term.split())
    return length


def pos_ngram_generator(nlp_doc, term_list):
    pos_doc = []
    for sentence in nlp_doc.sents:  # loop over sentences to get all n-grams within the sentence
        sent_list = [i.text for i in sentence]
        pos_list = [i.pos_ for i in sentence]
        length = len_ngrams(term_list)
        for i in range(1, length):  # n-grams chosen between 1 and max length of terms
            ngrams = list(nltk.ngrams(sent_list, i))
            pos_grams = list(nltk.ngrams(pos_list, i))
            for ngram, pos in zip(ngrams, pos_grams):
                ngram_string = " ".join(ngram)
                if ngram_string in term_list:
                    pos_string = " ".join(pos)
                    pos_doc.append(pos_string)

    return pos_doc


nlp = spacy.load("en_core_web_md")

en_train_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/en_train"
annotations_name = "_en_terms.ann"
pos_labels_global = []

for subdir in os.listdir(en_train_path):
    annotations_name = subdir + "_en_terms.ann"
    annotations_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/en_train/" \
                       + subdir + "/annotations/" + annotations_name

    term_label_pairs = generate_term_list(annotations_path)

    texts_dir = "/home/gillesfloreal/PycharmProjects/ASTRA/data/en_train/" + subdir + "/texts"
    for text_dir in os.listdir(texts_dir):
        next_path = "/home/gillesfloreal/PycharmProjects/data/ASTRA/en_train/" + subdir + "/texts/" + text_dir
        for text in os.listdir(next_path):
            text_path = "/home/gillesfloreal/PycharmProjects/data/ASTRA/en_train/" + subdir + "/texts/" \
                        + text_dir + "/" + text

            with open(text_path, 'r', encoding='utf8') as f:
                doc = nlp(f.read())
                f.close()

            pos_labels_global.extend(pos_ngram_generator(doc, term_label_pairs))


# for the financial (test) corpus, we need a slightly different approach

fin_annotations_dir = "/home/gillesfloreal/PycharmProjects/ASTRA/data/finc_en_test/annotations"
fin_texts_dir = "/home/gillesfloreal/PycharmProjects/ASTRA/data/finc_en_test/texts"

# first get all unique annotations in a list (data is slightly different so first method does not apply)
fin_annotation_labels = []
for fin_annotation in os.listdir(fin_annotations_dir):
    fin_annotation_path = fin_annotations_dir + "/" + fin_annotation
    with open(fin_annotation_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line_split = line.split('\t')
            fin_annotation_labels.append(line_split[0])

fin_annotation_labels = list(set(fin_annotation_labels))

for fin_text in os.listdir(fin_texts_dir):
    fin_text_path = fin_texts_dir + "/" + fin_text
    with open(fin_text_path, 'r', encoding='utf8') as f:
        fin_doc = nlp(f.read())
        f.close()

        pos_labels_global.extend(pos_ngram_generator(fin_doc, fin_annotation_labels))

target_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/ling_prepr/pos_tags.json"

with open(target_path, 'w', encoding='utf8') as target:
    json.dump(list(set(pos_labels_global)), target)

# results in 134 unique POS-tags
