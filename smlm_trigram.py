import json
import os.path
from collections import defaultdict

import numpy as np


class SMLM1:
    # tokens[n]指n代表的词语
    tk_to_word = []
    word_to_tk = {}

    # base_possibility[t1][t2]代表t2在t1后出现的频次
    base_frequency: defaultdict[defaultdict[defaultdict[int]]]
    base_possibility: np.array

    def __init__(self, path: str = None):
        if path is None:
            self.base_frequency = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
            self.tokenize(["<start>", "<end>"])
        else:
            with open(path, "r+") as f:
                cfg = json.load(f)
                self.tokenize(cfg["tokens"])
                self.base_frequency = np.array(cfg["base_possibility"])
                f.close()


    def tokenize(self, text: list[str]):
        tokens = list()
        for i in text:
            if i not in self.word_to_tk:
                self.tk_to_word.append(i)
                self.word_to_tk[i] = len(self.tk_to_word) - 1
            tokens.append(self.word_to_tk[i])
        return tokens

    def train(self, text: str):
        words = ["<start>"]
        words += ((text.lower().
                replace(",", " , ").
                replace(".", " . ").
                replace("!", " ! ").
                replace("?", " ? ").
                replace("\"", " \" ").replace("(", " ( ").replace(")", " ) ")
                ).split()
            )
        words.append("<end>")

        tokens = self.tokenize(words)

        for x, y, z in zip(tokens, tokens[1:], tokens[2:]):
            self.base_frequency[x][y][z] += 1


    def save(self, path: str):
        cfg = {
            "tokens": self.tk_to_word,
            "base_possibility": self.base_frequency.tolist()
        }
        with open(path, "w+") as f:
            f.write(json.dumps(cfg))
            f.close()

    def predict(self, t1, t2):
        probs = np.array(list(self.base_frequency[t1][t2].values()))
        return list(self.base_frequency[t1][t2].keys())[np.random.choice(len(probs), p = probs / probs.sum())]

    def generate(self, w1, w2):
        t1 = self.word_to_tk[w1]
        t2 = self.word_to_tk[w2]
        while True:
            print(self.tk_to_word[t2], end=" ")
            next = self.predict(t1, t2)
            if next == self.word_to_tk["<end>"]:
                break
            t1, t2 = t2, next


model = SMLM1()

for i in os.listdir("./stories"):
    with open("./stories/{}".format(i), "r+", encoding="utf-8") as f:
        model.train(f.read())

model.generate(
    "of",
    "the"
)
