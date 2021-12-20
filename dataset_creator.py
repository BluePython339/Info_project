



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


    def gen_train_dataset(self):
        wiki.set_lang('nl')
        for i in range(500):


    def gen_n_samples(self):



