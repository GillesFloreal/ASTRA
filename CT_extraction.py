import json
import os
import nltk
import spacy
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_md")


def extract_ct(pos_path, corpus_dir):
    corpus_terms = dict()
    corpus_terms['7'] = {}
    corpus_terms['6'] = {}
    corpus_terms['5'] = {}
    corpus_terms['4'] = {}
    corpus_terms['3'] = {}
    corpus_terms['2'] = {}
    corpus_terms['1'] = {}
    with open(pos_path, 'r', encoding='utf8') as f:
        pos_list = json.load(f)
        f.close()

    ann_dir = corpus_dir + '/texts'
    for ann_unann in os.listdir(ann_dir): #loop both through annotated and unannotated
        texts_dir = ann_dir + '/' + ann_unann
        for text in os.listdir(texts_dir):
            text_path = texts_dir + '/' + text
            with open(text_path, 'r', encoding='utf8') as f:
                doc = nlp(f.read())
                f.close()

            for sentence in doc.sents:  # loop over sentences to get all n-grams within the sentence
                sent_list = [i.text for i in sentence]
                sent_pos_list = [i.pos_ for i in sentence]
                for i in reversed(range(1, 7)):  # n-grams chosen between 1 and max length of terms
                    ngrams = list(nltk.ngrams(sent_list, i))
                    pos_grams = list(nltk.ngrams(sent_pos_list, i))
                    for ngram, pos in zip(ngrams, pos_grams):
                        for word in ngram:
                            if word in stop_words:
                                break
                            else:
                                pos_string = " ".join(pos)
                                if pos_string in pos_list:
                                    ngram_string = " ".join(ngram)
                                    if ngram_string.lower() in corpus_terms[str(i)]:
                                        corpus_terms[str(i)][ngram_string.lower()] += 1
                                    else:
                                        corpus_terms[str(i)][ngram_string.lower()] = 1
    return corpus_terms


en_train_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/en_train"
pos_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/ling_prepr/pos_tags.json"

for corpus in os.listdir(en_train_path):
    corpus_path = en_train_path + '/' + corpus
    target_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/ling_prepr/" + corpus + "_CT.json"
    with open(target_path, 'w') as target:
        json.dump(extract_ct(pos_path, corpus_path), target)
