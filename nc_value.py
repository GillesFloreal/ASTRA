import json
from pathlib import Path
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_md")

c_value_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/statistical_scores/c_values/c_values_list.json"
domains_path = Path("/home/gillesfloreal/PycharmProjects/ASTRA/data/en_train")

domains = ["corp", "equi", "htfl", "wind"]
with open(c_value_path, 'r', encoding='utf8') as source:
    c_value_list = json.load(source)
    source.close()

nc_value = []

for c_value_domain, domain in zip(c_value_list, domains):
    domain_path = Path("/home/gillesfloreal/PycharmProjects/ASTRA/data/en_train/" + domain)
    c_value_sorted = sorted(c_value_domain.items(), key=lambda item: item[1], reverse=True)
    percentage = int(len(c_value_sorted)/10)

    c_value_list = c_value_sorted[:percentage]
    c_value_list_bottom = c_value_sorted[percentage:]
    # make dict to put all top c_values in a dict to later add context words
    term_context_word = {}
    term_context_word_bottom = {}
    for item in c_value_list:
        term_context_word[item[0]] = {}
        
    for item in c_value_list_bottom:
        term_context_word_bottom[item[0]] = {}
    domain_paths = domains_path.joinpath(domain).rglob('*.txt')

    p = list(domain_paths)
    for text in tqdm(p):
        with open(text, 'r', encoding='utf8') as f:
            doc = nlp(f.read())
            f.close()

        for sent in doc.sents:
            sent_list = [i.text for i in sent]

            term_indices = []
            #generate ngrams max length 7
            for n in range(1, 7):
                ngrams_list = []
                # to follow indices closely, we generate ngrams manually
                for i in range(len(sent_list) - (n-1)):
                    start = i
                    end = i + n
                    ngram_list = sent_list[start:end]
                    # if an ngram is in our c_value_list, we gather the context words (window 3)
                    ngram = " ".join(ngram_list).lower()
                    if ngram in c_value_list:
                        context = []
                        # first start of ngram to three below (or until beginning)
                        if start - 3 < 0:
                            context.extend(sent_list[0:start])
                        else:
                            context.extend(sent_list[start-3:start])
                        # then end, same idea

                        if end + 3 > len(sent_list):
                            context.extend(sent_list[end:len(sent_list)])
                        else:
                            context.extend(sent_list[end:end+3])

                        for word in context:
                            if word.lower() in term_context_word[ngram]:
                                term_context_word[ngram][word.lower()] += 1
                            else:
                                term_context_word[ngram][word.lower()] = 1
                    if ngram in c_value_list_bottom:
                        context = []
                        if start - 3 < 0:
                            context.extend(sent_list[0:start])
                        else:
                            context.extend(sent_list[start-3:start])
                        # then end, same idea

                        if end + 3 > len(sent_list):
                            context.extend(sent_list[end:len(sent_list)])
                        else:
                            context.extend(sent_list[end:end+3])

                        for word in context:
                            if word.lower() in term_context_word[ngram]:
                                term_context_word_bottom[ngram][word.lower()] += 1
                            else:
                                term_context_word_bottom[ngram][word.lower()] = 1

    #context word are collected, now calculate weights for all context words
    context_weights = {}
    # first we need to find out how many times a context word occurs in a term
    context_df = {}
    for term in term_context_word:
        for context_word in term_context_word[term]:
            if context_word in context_df:
                context_df[context_word] += 1
            else:
                context_df[context_word] = 1


    # now calculate weights

    for context_word in context_df:
        context_weights[context_word] = context_df[context_word]/len(c_value_list)

    # now onto the calculation of the nc-value

    nc_value_domain = {}
    for term in c_value_list:
        context_sum = 0
        for context_word in term_context_word[term[0]]:
            # summation: number of occurrences times weight of the context word
            context_sum += (term_context_word[term][context_word]) * context_weights[context_word]
        nc_value_domain[term[0]] = (0.8 * term[1]) + 0.2 * context_sum

    # do the same for the bottom 90%
    for term in c_value_list_bottom:
        context_sum = 0
        for context_word in term_context_word_bottom[term[0]]:
            if context_word in context_weights:
                context_sum += (term_context_word_bottom[term][context_word]) * context_weights[context_word]

        nc_value_domain[term[0]] = (0.8 * term[1]) + 0.2 * context_sum

    nc_value.append(nc_value_domain)

target_path = "/home/gillesfloreal/PycharmProjects/ASTRA/data/statistical_scores/c_values/nc_value.json"
with open(target_path, 'w', encoding='utf8') as target:
    json.dump(nc_value, target)

