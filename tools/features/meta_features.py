import datetime

import numpy as np
import pandas as pd

from ..utils.general_utils import dec_timer, sel_log
from .feature_tools import get_all_features, load_features


def e005_meta_basic_features(df):
    # make base df
    features = pd.DataFrame()
    features['card_id'] = df.card_id
    # start feature engineering
    features['dayofweek'] = df['first_active_month'].dt.dayofweek
    features['weekofyear'] = df['first_active_month'].dt.weekofyear
    features['month'] = df['first_active_month'].dt.month
    features['elapsed_time'] = (
        datetime.datetime(2019, 1, 1) -
        df['first_active_month']).dt.days
    features['hist_first_buy_date_diff'] = (
        df['e001_hist_trans_basic_numerical_purchase_date_first'] -
        df['first_active_month']).dt.days
    features['new_merc_trans_first_buy_date_diff'] = (
        df['e003_new_merc_trans_basic_numerical_purchase_date_first'] -
        df['first_active_month']).dt.days
    for f in ['e001_hist_trans_basic_numerical_purchase_date_first',
              'e001_hist_trans_basic_numerical_purchase_date_last',
              'e003_new_merc_trans_basic_numerical_purchase_date_first',
              'e003_new_merc_trans_basic_numerical_purchase_date_last']:
        features[f] = df[f].astype(np.int64) * 1e-9
    features['card_id_total'] = \
        df['e001_hist_trans_basic_numerical_card_id_size'] + \
        df['e003_new_merc_trans_basic_numerical_card_id_size']
    features['purchase_amount_total'] = \
        df['e001_hist_trans_basic_numerical_purchase_amount_sum'] + \
        df['e003_new_merc_trans_basic_numerical_purchase_amount_sum']

    for f in ['feature_1', 'feature_2', 'feature_3']:
        order_label = df.groupby([f])['outliers'].mean()
        features[f + '_outlier_enc'] = df[f].map(order_label)

    features.set_index('card_id', inplace=True)
    features = features.add_prefix('e005_meta_basic_')
    return features


def _meta_features(df, exp_ids):
    _features = []
    if 'e005' in exp_ids:
        _features.append(e005_meta_basic_features(df))
    features = pd.concat(_features, axis=1)
    return features


@dec_timer
def _load_meta_features_src(
        exp_ids, trn_tst_meta_df, trn_meta_df, tst_meta_df, logger):
    target_ids = [
        'e005',
    ]
    if len(set(target_ids) & set(exp_ids)) < 1:
        sel_log(f'''
                ======== {__name__} ========
                Stop feature making because even 1 element in exp_ids
                    {exp_ids}
                does not in target_ids
                    {target_ids}''', logger)
        return None, None, None

    # Load dfs if not input.
    features = get_all_features('./inputs/test/')

    sel_log(f'loading train features ...', None)
    trn_meta_df = load_features(features, './inputs/train/')
    trn_meta_df['target'] = pd.read_pickle(
        './inputs/train/target.pkl.gz', compression='gzip')
    trn_meta_df['outliers'] = 0
    trn_meta_df.loc[trn_meta_df['target'] < -30, 'outliers'] = 1

    sel_log(f'loading test features ...', None)
    tst_meta_df = load_features(features, './inputs/test/')

    sel_log(f'merging train and test ...', None)
    tst_meta_df['target'] = np.nan
    tst_meta_df['outliers'] = 0
    trn_tst_meta_df = pd.concat([
        trn_meta_df.sort_index(axis=1),
        tst_meta_df.sort_index(axis=1),
    ], axis=0)
    trn_tst_meta_df['first_active_month'] = \
        pd.to_datetime(trn_tst_meta_df['first_active_month'])
    trn_meta_df = trn_meta_df[['card_id', 'outliers']]
    tst_meta_df = tst_meta_df[['card_id']]

    return trn_tst_meta_df, trn_meta_df, tst_meta_df
