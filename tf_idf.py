import spacy
import nltk
import json
import math


def ngram_generator(doc):
    ngrams_doc = []
    for sentence in doc.sents:  # loop over sentences to get all n-grams within the sentence
        sent_list = [i.text for i in sentence]
        for i in range(1, 6):  # n-grams chosen between 1 and 4
            ngrams = list(nltk.ngrams(sent_list, i))
            ngram_string = [" ".join(ngram) for ngram in ngrams]
            ngrams_doc.extend(ngram_string)
    return ngrams_doc


def tf_idf(file_path, domain):
    nlp = spacy.load('en_core_web_md')
    with open(file_path, 'r', encoding='utf') as f:
        text = f.read()
        text_clean = text.strip('\n')
        doc = nlp(text_clean)
        f.close()

    domain_idf_list_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/idf_lists/"\
                           + domain + ".json"

    # calculate tf for the terms
    ngrams = ngram_generator(doc)
    tf_dict = {}
    for ngram in ngrams:
        if ngram in tf_dict:
            tf_dict[ngram] += 1
        else:
            tf_dict[ngram] = 1

    # calculate idf and then tf-idf all at once
    with open(domain_idf_list_path, 'r', encoding='utf8') as f:
        idf_list = json.load(f)
        f.close()

    tfidf_dict = {}
    for ngram in tf_dict:
        idf_ngram = 0
        for document in idf_list:
            if ngram in document:
                idf_ngram += 1
        if idf_ngram == 0:
            print(ngram)
            print('OH NO')
        tfidf_dict[ngram] = math.log(tf_dict[ngram] + 1, 10) * math.log(len(idf_list)/idf_ngram, 10)
        # this calculates TF-IDF of all ngrams in a given document

    return tfidf_dict



