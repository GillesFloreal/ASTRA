import os
import json
import spacy
from tf_idf import ngram_generator
nlp = spacy.load("en_core_web_md")


def freq_dict_corp(corpus_directory):
    ngram_dict = {}
    for text in os.listdir(corpus_directory):
        text_path = corpus_directory + '/' + text
        with open(text_path, 'r', encoding='utf8') as f:
            doc = nlp(f.read())
            f.close()
        ngrams = ngram_generator(doc)
        for ngram in ngrams:
            if ngram in ngram_dict:
                ngram_dict[ngram] += 1
            else:
                ngram_dict[ngram] = 1
    return ngram_dict


def evaluation(output, annotations, dom_corp):
    prob = list()
    ngram_output_dict = dict()
    total_annotations = 0
    ngram_dict = freq_dict_corp(dom_corp)
    # create a list of all output
    with open(output, 'r', encoding='utf8') as f:
        output_list = list()
        for line in f.readlines():
            line_split = line.split('\t')
            prob.append(line_split[1])
            output_list.append(line_split[0])
            term_split = line_split[0].split()
            len_term_split = len(term_split)
            if len_term_split in ngram_output_dict:
                ngram_output_dict[len_term_split] += 1
            else:
                ngram_output_dict[len_term_split] = 1
        f.close()
    print("len_output:", len(output_list))
    pick_list = ['Term_Common', 'Term', 'Term_Out_of_Domain']
    label_dict = dict()
    label_dict['Term_Common'] = 0
    label_dict['Term'] = 0
    label_dict['Term_Out_of_Domain'] = 0
    label_dict['Named_Entity'] = 0
    label_unique_dict = dict()
    label_unique_dict['Term_Common'] = 0
    label_unique_dict['Term'] = 0
    label_unique_dict['Term_Out_of_Domain'] = 0
    label_unique_dict['Named_Entity'] = 0
    # create a dictionary of all annotations; key is the annotation, value its label
    annotations_dict = dict()
    annotations_full_list = list()
    ne_ann_list = list()
    ann_term_label_dict = dict()
    for annotation_file in os.listdir(annotations):
        annotation_path = annotations + '/' + annotation_file
        with open(annotation_path, 'r', encoding='utf8') as f:

            for line in f.readlines():
                line_split = line.split('\t')
                label_split = line_split[1].split()
                label = label_split[0]
                total_annotations += 1
                if label in pick_list:
                    label_dict[label] += 1
                    annotated_term = line_split[-1].strip('\n')
                    annotated_term_clean = annotated_term.replace(u'\xa0', u' ')
                    annotations_full_list.append(annotated_term_clean.lower())
                    if annotated_term_clean.lower() in annotations_dict:
                        annotations_dict[annotated_term_clean.lower()] += 1
                    else:
                        label_unique_dict[label] += 1
                        annotations_dict[annotated_term_clean.lower()] = 1

                    if annotated_term_clean.lower() not in ann_term_label_dict:
                        ann_term_label_dict[annotated_term_clean.lower()] = label

                elif label == 'Named_Entity':
                    annotated_term = line_split[-1].strip('\n')
                    annotated_term_clean = annotated_term.replace(u'\xa0', u' ')
                    ne_ann_list.append(annotated_term_clean.lower())

    annotations_full_list = list(set(annotations_full_list))
    print(len(annotations_full_list), len(ne_ann_list))
    print("len dict:", len(annotations_dict), "len list:", len(annotations_full_list))

    annotations_one_list = list()
    true_positives = 0
    pos_list = list()
    neg_list = list()
    for ann in annotations_dict:
        if annotations_dict[ann] != 1:
            annotations_one_list.append(ann)

    output_one_list = list()
    for ngram in output_list:
        if ngram in ngram_dict:
            if ngram_dict[ngram] != 1:
                output_one_list.append(ngram)

    ann_label_one = dict()
    one_true_pos = 0
    for ct in output_one_list:
        if ct in annotations_one_list:
            one_true_pos += 1
            if ann_term_label_dict[ct] in ann_label_one:
                ann_label_one[ann_term_label_dict[ct]] += 1
            else:
                ann_label_one[ann_term_label_dict[ct]] = 1

    annotations_list_one_labels = dict()
    for ann in annotations_one_list:
        print(ann_term_label_dict[ann])
        if ann_term_label_dict[ann] in annotations_list_one_labels:
            annotations_list_one_labels[ann_term_label_dict[ann]] += 1
        else:
            annotations_list_one_labels[ann_term_label_dict[ann]] = 1

    #pr_one = one_true_pos/len(output_one_list)
    #re_one = one_true_pos/len(annotations_one_list)
    #f_one = 2 * ((pr_one * re_one)/ (pr_one + re_one))
    print("ann_list_one:", len(annotations_one_list))
    print("output_list_one:", len(output_one_list))
    #print("pr_one:", pr_one)
    #print("re_one", re_one)
    #print("f_score_one:", f_one)
    counter = 0
    ne_count = 0
    negatives = 0
    ann_label_full = dict()

    precision_list = list()
    recall_list = list()
    fscore_list = list()
    for ct in output_list:
        #print("counter:", counter)
        #print(ct)
        counter += 1
        if ct in ne_ann_list:
            ne_count += 1
        if ct in annotations_full_list:
            pos_list.append(ct)
            true_positives += 1
            if ann_term_label_dict[ct] in ann_label_full:
                ann_label_full[ann_term_label_dict[ct]] += 1
            else:
                ann_label_full[ann_term_label_dict[ct]] = 1
            pr = true_positives/counter
            precision_list.append(pr)
            re = true_positives/len(annotations_full_list)
            recall_list.append(re)
            f_score_temp = 2 * ((pr * re)/(pr + re))
            fscore_list.append(f_score_temp)
            #print("#positives:", true_positives)
            #print("pr_true:", true_positives/counter)
            #print("re_true:", true_positives / len(annotations_full_list))

        else:
            neg_list.append(ct)
            negatives += 1
            pr = true_positives / counter
            precision_list.append(pr)
            re = true_positives / len(annotations_full_list)
            recall_list.append(re)
            f_score_temp = 2 * ((pr * re) / (pr + re))
            fscore_list.append(f_score_temp)
            #print("#negatives:", negatives)
            #print("pr_false:", true_positives / counter)
            #print("re_false:", true_positives / len(annotations_full_list))

    precision = true_positives / len(output_list)
    recall = true_positives / (len(annotations_full_list))

    #print(len(annotations_dict))
    #print(len(output_list))
    n_gram_dict = dict()
    false_neg = list()
    for ct in annotations_full_list:
        ct_len = len(ct.split())
        if ct in output_list:
            if ct_len in n_gram_dict:
                n_gram_dict[ct_len]["pos"] += 1
            else:
                n_gram_dict[ct_len] = dict()
                n_gram_dict[ct_len]["pos"] = 1
                n_gram_dict[ct_len]["neg"] = 0
        else:
            false_neg.append(ct)
            if ct_len in n_gram_dict:
                n_gram_dict[ct_len]["neg"] += 1
            else:
                n_gram_dict[ct_len] = dict()
                n_gram_dict[ct_len]["neg"] = 1
                n_gram_dict[ct_len]["pos"] = 0

    print("n_gram_dict:", n_gram_dict)
    print("ne_count:", ne_count)
    print("ngram_output_dict:", ngram_output_dict)
    print("total labels all annotations:", label_dict)
    print("all unique hapax annotations:", annotations_list_one_labels)
    print("all unique annotations labels:", label_unique_dict)
    print("unique term extraction labels:", ann_label_one)
    print("all extracted labels:", ann_label_full)
    return false_neg


output_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/mapr/HAMLET_predictions/D_en_POSallDomains_TrainAllDomains_POS3_CT1_11100_statRobust_genRobust_210511a.txt"
annotations_dir = "/home/gillesfloreal/PycharmProjects/ASTRA/data/finc_en_test/annotations"
domain_corp = "/home/gillesfloreal/PycharmProjects/ASTRA/data/finc_en_test/texts"


hamlet_precision = evaluation(output_path, annotations_dir, domain_corp)
# f_score = 2 * ((hamlet_precision * hamlet_recall)/ (hamlet_precision + hamlet_recall))
# print("pr:", hamlet_precision)
# print("re:", hamlet_recall)
# print("f_score:", f_score)


with open("/home/gillesfloreal/PycharmProjects/ASTRA/data/mapr/false_neg.json", 'w', encoding='utf8') as f:
    json.dump(hamlet_precision, f)

#with open("/home/gillesfloreal/PycharmProjects/ASTRA/data/mapr/false_pos.json", 'w', encoding='utf8') as f:
#    json.dump(hamlet_recall, f)

#with open("/home/gillesfloreal/PycharmProjects/ASTRA/data/mapr/fscore.json", 'w', encoding='utf8') as f:
#    json.dump(hamlet_fscore, f)