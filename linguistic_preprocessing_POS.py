from generate_lists import generate_term_list
import os
import spacy
import nltk
import json
from pathlib import Path
import pickle


"""" 
In this document, we first extract the viable POS-tags of terms from the training corpora. These are used to select
CTs in the linguistic preprocessing step. To select CTs, we generate ngrams and filter them so they contain no 
stopwords. We create 1 list with four embedded lists, one for each corpus.
"""

stop_words = ['a', 'the', 'on', 'an', 'of', 'from', 'for', 'to', 'into', 'in', 'with', 'by'
              'over', '-', '.', '<', '>', '"', "'", "and", "against", "this", "also", "is", "was", "%",
              "were", "are", "be", "or", "as", "more", "will", "by", "that", "has", "these", "been", "which",
              "but", "then", "such", "shall", "may", "must", "at", "have"]


def ngram_generator_pos_check(doc, pos):
    ngrams_doc = []
    for sentence in doc.sents:  # loop over sentences to get all n-grams within the sentence
        sent_list = [i.text for i in sentence]
        pos_list = [i.pos_ for i in sentence]
        for i in range(1, 7):  # n-grams chosen between 1 and 7
            ngrams = list(nltk.ngrams(sent_list, i))
            pos_grams = list(nltk.ngrams(pos_list, i))
            for ngram, pos_gram in zip(ngrams, pos_grams):
                if " ".join(pos_gram) in pos:
                    unwanted = False
                    for ngram_word in ngram:
                        if ngram_word.lower() in stop_words or ngram_word.isnumeric() is True:
                            unwanted = True
                    if unwanted is False:
                        ngram_string = " ".join(ngram).lower()
                        ngrams_doc.append(ngram_string)

    return ngrams_doc


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


#def pos_extraction():
#    # extract all viable POS-tags from training data, checking if they are in the annotations list
#    nlp = spacy.load("en_core_web_md")

#    en_train_path = Path("/home/gillesfloreal/PycharmProjects/ASTRA/data/en_train")
#    pos_labels_global = []

#    for subdir in os.listdir(en_train_path):

#        annotations_name = subdir + "_en_terms.ann"
#        annotations_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/en_train/" \
#                           + subdir + "/annotations/" + annotations_name

#        term_label_pairs = generate_term_list(annotations_path)

#        # look for all .txt files in subdirectory
#        texts_dir = en_train_path.joinpath(subdir).rglob('*.txt')
#        for text_path in texts_dir:

#            with open(text_path, 'r', encoding='utf8') as f:
#                doc = nlp(f.read())
#                f.close()

#            pos_labels_global.extend(pos_ngram_generator(doc, term_label_pairs))

#    return list(set(pos_labels_global))


#with open('pos_labels.pickle', 'wb') as handle:
#    pickle.dump(pos_extraction(), handle)


def ling_preprocessing(path):
    nlp = spacy.load("en_core_web_md")
    en_train_path = Path(path)
    with open('pos_labels.pickle', 'rb') as source:
        pos_labels_list = pickle.load(source)
    ct_train_terms = []
    for subdir in os.listdir(en_train_path):
        ct_domain_terms = {}
        texts_dir = en_train_path.joinpath(subdir).rglob('*.txt')
        for text_path in texts_dir:
            with open(text_path, 'r', encoding='utf8') as f:
                doc = nlp(f.read())
                f.close()

            for ngram in (ngram_generator_pos_check(doc, pos_labels_list)):
                if ngram in ct_domain_terms:
                    ct_domain_terms[ngram] += 1
                else:
                    ct_domain_terms[ngram] = 1

        ct_train_terms.append(ct_domain_terms)
    return ct_train_terms


corpora_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/en_train"
target_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/training.json"
with open(target_path, 'w', encoding='utf8') as f:
    json.dump(ling_preprocessing(corpora_path), f)

# results in 134 unique POS-tags
# max length POS = 7
