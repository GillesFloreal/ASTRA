import spacy
import nltk
import os
import json
import math

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
                        ngram_str = " ".join(ngram)
                        total_count += 1
                        if ngram_str in freq_dict:
                            freq_dict[ngram_str] += 1
                        else:
                            freq_dict[ngram_str] = 1
        f.close()

    return freq_dict


def vintar(domain, ref_dict, ref_total_count):

    # we also need a freq_dict for all unigrams in our domain corpus

    domain_count = 0
    domain_freq_dict = {}
    ann_dir = "/home/gillesfloreal/PycharmProjects/ASTRA/data/en_train/" + domain + '/texts'
    for ann_unann in os.listdir(ann_dir):  # loop both through annotated and unannotated
        texts_dir = ann_dir + '/' + ann_unann
        for text in os.listdir(texts_dir):
            text_path = texts_dir + '/' + text
            with open(text_path, 'r', encoding='utf8') as f:
                domain_doc = nlp(f.read())
                f.close()
            for domain_sentence in domain_doc.sents:
                for unigram in domain_sentence:
                    domain_count += 1
                    if unigram.text in domain_freq_dict:
                        domain_freq_dict[unigram.text] += 1
                    else:
                        domain_freq_dict[unigram.text] = 1

    # now we calculate vintar score for each CT
    ct_domain_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/ling_prepr/corpus_CT/" + domain + "_CT.json"
    with open(ct_domain_path, 'r') as f:
        ct_dict = json.load(f)
        f.close()

    vintar_scores = dict()
    ngram_dict = dict()
    for i in ct_dict:
        ngram_dict.update(dict(sorted(ct_dict[i].items(), key=lambda item: item[1], reverse=True)))
    for ct in ngram_dict:
        ct_split = ct.split()
        summation = 0
        for word in ct_split:
            if word not in domain_freq_dict:
                continue
            if word in ref_dict:
                summation += (math.log((domain_freq_dict[word]/domain_count) + 1) - math.log((ref_dict[word]/ref_total_count) + 1))
            else:
                summation += (math.log((domain_freq_dict[word]/domain_count) + 1) - math.log((1/ref_total_count)) + 1)

        vintar_scores[ct] = ((ngram_dict[ct] ** 2)/len(ct_split)) * summation

    return vintar_scores


# create ref_corp in separate file

#ref_corp_dict_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/ref_corp/freq_dict"
#with open(ref_corp_dict_path, 'w') as ref_target:
#    json.dump(freq_dict_ref_corp(ref_corp_path), ref_target)



ref_dict_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/ref_corp/freq_dict"
with open(ref_dict_path, 'r') as f:
    ref_dict = json.load(f)
    f.close()

ref_total_count = 0
for value in ref_dict.values():
    ref_total_count += value

domains = ["corp", "equi", "htfl", "wind"]

for name in domains:
    target_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/statistical_scores/vintar_values/vintar_" + name + ".json"
    with open(target_path, 'w') as target:
        json.dump(vintar(name, ref_dict, ref_total_count), target)
