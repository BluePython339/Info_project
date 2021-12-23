from model import Net, preprocess_text
import torch
import multiprocessing as mp
from scraper_v3 import scrape
import time


manager = mp.Manager()
archive_list = manager.list()
archive_counter = mp.Value('i', 0)
target_list = manager.list()
check_set = manager.list()


def write_to_archive():
    with open("archive.bin", "a+") as arc:
        while True:
            if len(archive_list) > 0:
                tar = archive_list.pop(0)
                if tar not in check_set:
                    check_set.append(tar)
                    archive_counter.value += 1
                    arc.write(f"{tar}\n")


def spider_ling():
    model = Net(input_shape=301)
    model.load_state_dict(torch.load("quad_model.pth"))
    while True:

        if len(target_list) > 0:
            try:
                target = target_list.pop(0)
                if target not in check_set:
                    text, child_urls = scrape(target)
                    if len(text) > 5000:
                        text = text[:5000]
                    pred = model(preprocess_text(text)).item()

                    if pred > 0.70:
                        archive_list.append(target)
                        for i in child_urls:
                            if i not in archive_list:
                                target_list.append(i)
            except KeyboardInterrupt:
                exit()
            except:
                pass


def logger():
    while True:
        print(f"{len(target_list)} Targets, {archive_counter.value} targets archived, {len(archive_list)} backlogged")
        time.sleep(1)


def main():

    with open("urls.txt", "r") as urls:
        for i in urls.readlines():
            target_list.append(i[:-1])
    spiders = []
    archiver = mp.Process(target=write_to_archive)
    log = mp.Process(target=logger)
    for i in range(20):
        spiders.append(mp.Process(target=spider_ling))

    for i in spiders:
       i.start()
    archiver.start()
    log.start()

    for i in spiders:
        i.join()
    archiver.join()
    log.join()


if __name__ == "__main__":
    main()



