import wikipedia as wiki
import re
from collections import Counter
from tqdm import tqdm
import random


def write_to_log(msg):
    with open("dataset.log", "a+") as log:
        log.write(f"{msg}\n")

def preproccess_text(text, n):
    text = text.lower()
    stripped = re.sub(r'([^A-Za-z ])+', "", text)
    prepped = re.sub(r'([^A-Za-z])+', "_", stripped)
    res = [prepped[i:i + n] for i in range(len(prepped) - n + 1)]
    length = len(prepped)
    counts = list(zip(Counter(res).keys(),Counter(res).values()))
    counts.sort(key=lambda tup: tup[1], reverse=True)
    return counts, length

class DatasetGenerator(object):


    def __init__(self, profile, langs):
        self.langs = langs
        self.dataset = []
        self.base = {}

        self.check_list = {}

        with open(profile, 'r')as prof:
            a = prof.readlines()
            for i in range(300):
                self.base[a[i].split(" ")[0]] = 0
            self.base["length"] = 0


    def gen_quad_train_dataset(self):
        wiki.set_lang('nl')
        picks = []
        write_to_log("______start of dataset_______")
        while len(picks) < 2000:
            for _ in range(3):
                lang = 'nl'
                wiki.set_lang(lang)
                items = wiki.random(500)
                confirmed = 0
                for it in tqdm(items, "quadgram making NL"):
                    try:
                        count, length = preproccess_text(wiki.page(it).content, 4)
                        it_count = self.base.copy()
                        for i in count:
                            if i[0] in it_count.keys():
                                it_count[i[0]] = i[1]

                        it_count["length"] = length
                        it_count["lang"] = 1
                        confirmed +=1
                        picks.append([x[1] for x in it_count.items()])
                    except KeyboardInterrupt:
                        exit()
                    except wiki.exceptions.DisambiguationError:
                        pass
                    except:
                        pass
                write_to_log(f"{confirmed} samples of {lang}")
            for lang in self.langs:
                wiki.set_lang(lang)
                confirmed = 0
                items = wiki.random(250)
                for it in tqdm(items, "quadgram making"):
                    try:
                        count, length = preproccess_text(wiki.page(it).content, 4)
                        it_count = self.base.copy()
                        for i in count:
                            if i[0] in it_count.keys():
                                it_count[i[0]] = i[1]
                        it_count["length"] = length
                        it_count["lang"] = 0
                        confirmed += 1
                        picks.append([x[1] for x in it_count.items()])
                    except KeyboardInterrupt:
                        exit()
                    except wiki.exceptions.DisambiguationError:
                        pass
                    except:
                        pass
                write_to_log(f"{confirmed} samples of {lang}")
        write_to_log("_______end of dataset_______")
        return picks



a = DatasetGenerator("tri_profile.bin", ["en","fr","de","es","it","af","srn","vls"])
#
data = a.gen_quad_train_dataset()
with open("dataset_train_tri_2.csv", "w+") as csv:
    for i in data:
        csv.write(",".join(map(str, i))+"\n")

data = a.gen_quad_train_dataset()
with open("dataset_val_tri_2.csv", "w+") as csv:
    for i in data:
        csv.write(",".join(map(str, i))+"\n")
