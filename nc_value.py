import json
import os

c_value_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/statistical_scores/c_values"

for c_value_domain in os.listdir(c_value_path):
    c_value_domain_path = c_value_path + '/' + c_value_domain
    with open(c_value_domain_path, 'r', encoding='utf8') as f:
        c_value_dict = json.load(f)
        f.close()

    c_value_sorted = sorted(c_value_dict.items(), key=lambda item: item[1], reverse=True)
    percentage = int(len(c_value_sorted)/100)

    c_value_list = [c_value_item[0] for c_value_item in c_value_sorted[:percentage]]
    print(c_value_list)

