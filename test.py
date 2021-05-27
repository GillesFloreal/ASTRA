import json

with open('/home/gillesfloreal/PycharmProjects/ASTRA/data/mapr/false_neg.json', 'r', encoding='utf-8') as f:
    false_neg_list = json.load(f)
    f.close()

ngram_dict = dict()

for ann in false_neg_list:
    ann_split = ann.split()
    if len(ann_split) in ngram_dict:
        ngram_dict[len(ann_split)] += 1
    else:
        ngram_dict[len(ann_split)] = 1

print(ngram_dict)