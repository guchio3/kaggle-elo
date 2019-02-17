import numpy as np
import pandas as pd


class UniformKFold():
    def __init__(self, fold_num, random_state=71):
        np.random.seed(random_state)
        self.fold_num = fold_num

    def split(self, X, y):
        indices = np.argsort(y)
        indice_templates = np.arange(len(y)) % self.fold_num
        for i in range(self.fold_num):
            train = indices[indice_templates != i]
            valid = indices[indice_templates == i]
            yield np.random.permutation(train), valid


if __name__ == '__main__':
    SRC_PATH = '/home/naoya.taguchi/workspace/kaggle-elo/inputs/train/'
    sample_df = pd.concat([
        pd.read_pickle(SRC_PATH + 'target.pkl.gz'),
        pd.read_pickle(SRC_PATH + 'feature_1.pkl.gz'),
        pd.read_pickle(SRC_PATH + 'feature_2.pkl.gz')
        ], axis=1)

    ukf = UniformKFold(5)
    folds = ukf.split(sample_df.drop('target', axis=1), sample_df['target'])
    for trn, val in folds:
        print(len(trn), len(val))
