import datetime

import pandas as pd

from ..utils.general_utils import dec_timer, sel_log
from .feature_funcs import (get_0th_count_rat, get_0th_value,
                            get_1st_count_rat, get_1st_value,
                            get_2nd_count_rat, get_2nd_value,
                            get_last_count_rat, get_last_value, percentile10,
                            percentile25, percentile50, percentile75,
                            percentile90)


def e003_basic_numerical_features(df):
    # define aggregations
    e003_aggs = {
        'card_id': ['size'],
        'installments': ['sum', 'max', 'min', 'mean', 'var',
                         'nunique', 'first', 'last', 'sem', 'mad'],
        'month_diff': ['sum', 'max', 'min', 'mean', 'var',
                       'nunique', 'first', 'last', 'sem', 'mad'],
        'month_lag': ['sum', 'max', 'min', 'mean', 'var',
                      'nunique', 'first', 'last', 'sem', 'mad'],
        'purchase_amount': ['sum', 'max', 'min', 'mean', 'var',
                            'nunique', 'first', 'last', 'sem', 'mad',
                            percentile10, percentile25, percentile50,
                            percentile75, percentile90],
        'purchase_date': ['first', 'last'],
    }
    features = df.groupby('card_id').agg(e003_aggs)
    features.columns = [col[0] + '_' + col[1] for col in features.columns]
    # feature engineering after aggregations
    features['purchase_date_first_last_diff'] = (
        features['purchase_date_first'] -
        features['purchase_date_last']).dt.days
    features['purchase_date_average'] = \
        features['purchase_date_first_last_diff'] / \
        features['card_id_size']
    features['purchase_date_uptonow'] = (
        datetime.datetime(2019, 1, 1) -
        features['purchase_date_last']).dt.days
    # add prefix
    features = features.add_prefix('e003_new_merc_trans_basic_numerical_')
    return features


def e004_basic_categorical_features(df):
    # change categorical values to numerical
    df['authorized_flag'] = df['authorized_flag'].map({'Y': 1, 'N': 0})
    df['category_1'] = df['category_1'].map({'Y': 1, 'N': 0})

    e004_aggs = {
        'month': ['nunique', get_0th_count_rat, get_0th_value,
                  get_1st_count_rat, get_1st_value],
        'hour': ['nunique', get_0th_count_rat, get_0th_value,
                 get_1st_count_rat, get_1st_value,
                 get_2nd_count_rat, get_2nd_value,
                 get_last_count_rat, get_last_value],
        'year': ['nunique', get_0th_count_rat, get_0th_value,
                 get_1st_count_rat, get_1st_value],
        'weekofyear': ['nunique', get_0th_count_rat, get_0th_value,
                       get_1st_count_rat, get_1st_value,
                       get_2nd_count_rat, get_2nd_value,
                       get_last_count_rat, get_last_value],
        'dayofweek': ['nunique', get_0th_count_rat, get_0th_value,
                      get_1st_count_rat, get_1st_value,
                      get_2nd_count_rat, get_2nd_value,
                      get_last_count_rat, get_last_value],
        'weekend': ['sum', 'mean'],
        'city_id': ['nunique', get_0th_count_rat, get_0th_value,
                    get_1st_count_rat, get_1st_value,
                    get_2nd_count_rat, get_2nd_value,
                    get_last_count_rat, get_last_value],
        'state_id': ['nunique', get_0th_count_rat, get_0th_value,
                     get_1st_count_rat, get_1st_value,
                     get_2nd_count_rat, get_2nd_value,
                     get_last_count_rat, get_last_value],
        'subsector_id': ['nunique', get_0th_count_rat, get_0th_value,
                         get_1st_count_rat, get_1st_value,
                         get_2nd_count_rat, get_2nd_value,
                         get_last_count_rat, get_last_value],
        'merchant_id': ['nunique', get_0th_count_rat, get_0th_value,
                        get_1st_count_rat, get_1st_value,
                        get_2nd_count_rat, get_2nd_value,
                        get_last_count_rat, get_last_value],
        'merchant_category_id': ['nunique', get_0th_count_rat, get_0th_value,
                                 get_1st_count_rat, get_1st_value,
                                 get_2nd_count_rat, get_2nd_value,
                                 get_last_count_rat, get_last_value],
        'authorized_flag': ['sum', ],
        'category_1': ['sum', 'mean'],
        'category_2': ['nunique', get_0th_count_rat, get_0th_value,
                       get_1st_count_rat, get_1st_value,
                       get_2nd_count_rat, get_2nd_value,
                       get_last_count_rat, get_last_value],
        'category_3': ['nunique', get_0th_count_rat, get_0th_value,
                       get_1st_count_rat, get_1st_value,
                       get_2nd_count_rat, get_2nd_value,
                       get_last_count_rat, get_last_value],
    }
    features = df.groupby('card_id').agg(e004_aggs)
    features.columns = [col[0] + '_' + col[1] for col in features.columns]
    # feature engineering after aggregations
    features = features.add_prefix(
        'e004_new_merc_trans_basic_categorical_')#.reset_index(drop=False)
    return features


def _new_merc_trans_features(df, exp_ids):
    _features = []
    if 'e003' in exp_ids:
        _features.append(e003_basic_numerical_features(df))
    if 'e004' in exp_ids:
        _features.append(e004_basic_categorical_features(df))
    features = pd.concat(_features, axis=1)#.reset_index(drop=False)
    return features


@dec_timer
def _load_new_merc_trans_features_src(
        exp_ids, series_df, trn_meta_df, tst_meta_df, logger):
    target_ids = [
        'e003',
        'e004',
    ]
    if len(set(target_ids) & set(exp_ids)) < 1:
        sel_log(f'''
                ======== {__name__} ========
                Stop feature making because even 1 element in exp_ids
                    {exp_ids}
                does not in target_ids
                    {target_ids}''', logger)
        return None, None, None

    trn_meta_path = './inputs/prep/train.pkl.gz'
    tst_meta_path = './inputs/prep/test.pkl.gz'
    series_path = './inputs/prep/new_merchant_transactions_v1.pkl.gz'

    # Load dfs if not input.
    if series_df is None:
        sel_log(f'loading {series_path} ...', None)
        series_df = pd.read_pickle(series_path, compression='gzip')
    if trn_meta_df is None:
        sel_log(f'loading {trn_meta_path} ...', None)
        trn_meta_df = pd.read_pickle(trn_meta_path, compression='gzip')
    if tst_meta_df is None:
        sel_log(f'loading {tst_meta_path} ...', None)
        tst_meta_df = pd.read_pickle(tst_meta_path, compression='gzip')

    return series_df, trn_meta_df, tst_meta_df
