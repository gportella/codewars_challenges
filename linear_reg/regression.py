from typing import List, Tuple


class Datamining:
    def __init__(self, train_set: List[Tuple[float, float]]):
        self.a = 0
        self.b = 0
        self.fit(train_set)

    def fit(self, train_set):
        av_x = sum([x[0] for x in train_set]) / len(train_set)
        av_y = sum([y[1] for y in train_set]) / len(train_set)
        nom = 0
        denom = 0
        for xx, yy in train_set:
            diff_x = xx - av_x
            diff_y = yy - av_y
            nom += diff_x * diff_y
            denom += diff_x * diff_x
        self.a = nom / denom
        self.b = av_y - self.a * av_x

    def predict(self, x):
        for val_x in x:
            return self.a * val_x + self.b
