import json

paths = ["/home/gillesfloreal/PycharmProjects/ASTRA/data/ling_prepr/c_value_corp_CT.json",
         "/home/gillesfloreal/PycharmProjects/ASTRA/data/ling_prepr/c_value_equi_CT.json",
         "/home/gillesfloreal/PycharmProjects/ASTRA/data/ling_prepr/c_value_htfl_CT.json",
         "/home/gillesfloreal/PycharmProjects/ASTRA/data/ling_prepr/c_value_wind_CT.json"]

max = 0

for path in paths:
    with open(path, 'r') as f:
        c_value = json.load(f)
        f.close()
    ct_sorted = sorted(c_value.items(), key=lambda item: item[1], reverse=True)

    if ct_sorted[0][1] > max:
        max = ct_sorted[0][1]

print(max)