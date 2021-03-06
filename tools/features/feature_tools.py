import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils.general_utils import dec_timer, sel_log


@dec_timer
def split_df(base_df, target_df, split_name,
             target_name, n_sections, logger=None):
    '''
    policy
    ------------
    * split df based on split_id, and set split_id as index
        because of efficiency.

    '''
    sel_log(
        f'now splitting a df to {n_sections} dfs using {split_name} ...',
        logger)
    split_ids = base_df[split_name].unique()
    splitted_ids = np.array_split(split_ids, n_sections)
    if split_name == target_name:
        target_ids = splitted_ids
    else:
        target_ids = [base_df.set_index(split_name)
                      .loc[splitted_id][target_name]
                      for splitted_id in splitted_ids]
    # Pay attention that this is col-wise splitting bacause of the
    #   data structure of this competition.
    target_df = target_df.set_index(target_name)
    dfs = [target_df.loc[target_id.astype(str)].reset_index()
           for target_id in target_ids]
    return dfs


def get_all_features(path):
    files = os.listdir(path)
    features = [_file.split('.')[0] for _file in files]
    return features


@dec_timer
def load_features(features, base_dir, logger=None):
    loaded_features = []
    sel_log(f'now loading features ... ', None)
    for feature in tqdm(features):
        load_filename = base_dir + feature + '.pkl.gz'
        loaded_feature = pd.read_pickle(load_filename, compression='gzip')
        loaded_features.append(loaded_feature)

    features_df = pd.concat(loaded_features, axis=1)
    return features_df


def _save_feature(feature_pair, base_dir, logger=None):
    feature, feature_df = feature_pair
    save_filename = base_dir + feature + '.pkl.gz'
    if os.path.exists(save_filename):
        sel_log(f'already exists at {save_filename} !', None)
    else:
        sel_log(f'saving to {save_filename} ...', logger)
        feature_df.to_pickle(save_filename, compression='gzip')


@dec_timer
def save_features(features_df, base_dir, nthread, logger=None):
    feature_pairs = [[feature, features_df[feature]] for feature in
                     features_df.columns]
    with Pool(nthread) as p:
        iter_func = partial(_save_feature, base_dir=base_dir, logger=logger)
        _ = p.map(iter_func, feature_pairs)
        p.close()
        p.join()
        del _


def select_features(df, importance_csv_path, metric='gain_mean', topk=10):
#    if metric == :
    importance_df = pd.read_csv(importance_csv_path)
    importance_df.sort_values(metric, ascending=False, inplace=True)
    selected_df = df[importance_df.head(topk).features]
    return selected_df


@dec_timer
def _mk_features(load_func, feature_func, nthread, exp_ids,
                 series_df=None, trn_meta_df=None, tst_meta_df=None,
                 logger=None):
    # Load dfs
    # Does not load if the exp_ids are not the targets.
    series_df, trn_meta_df, tst_meta_df = load_func(
        exp_ids, series_df, trn_meta_df, tst_meta_df, logger)
    # Finish before feature engineering if the exp_ids are not the targets.
    if series_df is None:
        return None, None, None

    # Test meta is only 20338, so i use splitting only for series.
    # The n_sections is devided by 3 because it's 3 phase data.
    series_dfs = split_df(
        series_df,
        series_df,
        'card_id',
        'card_id',
        nthread,
        #        series_df['card_id'].nunique(),
        logger=logger)

    with Pool(nthread) as p:
        sel_log(f'feature engineering ...', None)
        # Using partial enable to use constant argument for the iteration.
        iter_func = partial(feature_func, exp_ids=exp_ids)
        features_list = p.map(iter_func, series_dfs)
        p.close()
        p.join()
        features_df = pd.concat(features_list, axis=0)

    # Merge w/ meta.
    # This time, i don't remove the original features because
    #   this is the base feature function.
    sel_log(f'merging features ...', None)
    trn_features_df = trn_meta_df.merge(features_df, on='card_id', how='left')
    tst_features_df = tst_meta_df.merge(features_df, on='card_id', how='left')

    # Save the features
    trn_dir = './inputs/train/'
    tst_dir = './inputs/test/'
    sel_log(f'saving features ...', logger)
    save_features(trn_features_df, trn_dir, nthread, logger)
    save_features(tst_features_df, tst_dir, nthread, logger)

    return series_df, trn_meta_df, tst_meta_df
