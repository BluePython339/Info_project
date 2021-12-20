import wikipedia as wiki
import re
from collections import Counter
from tqdm import tqdm



def preproccess_text(text, n):
    text = text.lower()
    stripped = re.sub(r'([^A-Za-z ])+', "", text)
    prepped = re.sub(r'([^A-Za-z])+', "_", stripped)
    res = [prepped[i:i + n] for i in range(len(prepped) - n + 1)]
    counts = list(zip(Counter(res).keys(),Counter(res).values()))
    counts.sort(key=lambda tup: tup[1], reverse=True)
    return counts

if __name__ == "__main__":

    wiki.set_lang('nl')

    items = wiki.random(200)

    #To create a suitable dataset we first want to know the most frequently appearing n-grams in the dutch language.
    #
    # code for tri-grams
    trigrams = {}
    for it in tqdm(items, "trigram making"):
        try:
            counts = preproccess_text(wiki.page(it).content, 3)
            for i in counts:
                if i[0] in trigrams.keys():
                    trigrams[i[0]] += i[1]
                else:
                    trigrams[i[0]] = i[1]
        except KeyboardInterrupt:
            print('Interrupted')
            exit(0)
        except:
            pass

    tri_profile = [(k, v) for k, v in trigrams.items()]
    tri_profile.sort(key=lambda tup: tup[1], reverse=True)

    # code for quad-grams
    quadgrams = {}
    for it in tqdm(items, "quadgram making"):
        try:
            counts = preproccess_text(wiki.page(it).content, 4)
            for i in counts:
                if i[0] in quadgrams.keys():
                    quadgrams[i[0]] += i[1]
                else:
                    quadgrams[i[0]] = i[1]
        except KeyboardInterrupt:
            print('Interrupted')
            exit(0)
        except :
            pass

    quad_profile = [(k, v) for k, v in quadgrams.items()]
    quad_profile.sort(key=lambda tup: tup[1], reverse=True)

    with open("tri_profile.bin", "w+") as tri:
        for i in tri_profile:
            tri.write(f"{i[0]} {i[1]}\n")

    with open("quad_profile.bin", "w+") as quad:
        for i in quad_profile:
            quad.write(f"{i[0]} {i[1]}\n")
