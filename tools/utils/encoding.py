from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def label_encoding(df, le_dict={}, fit_columns=[], inplace=False):
    '''
    label encoding object type object

    '''
    if not inplace:
        df = df.copy()

    # fit
    for col in fit_columns:
        if df[col].dtype == 'object':
            filled = df[col].fillna('NAN!!!!!!!!')
        elif df[col].dtype in ['int64', 'float64']:
            filled = df[col].fillna(-11111)
        le = LabelEncoder()
        le.fit(filled)
        le_dict[col] = le

    # transform
    transformed_cols = []
    for col in df.columns:
        if col not in le_dict:
            continue
        transformed_cols.append(col)
        le = le_dict[col]

        # $B7gB;$,$"$k>l9g(B
        is_deficit = df[col].isnull().sum() > 0
        if is_deficit:
            if df[col].dtype == 'object':
                df[col].fillna('NAN!!!!!!!!', inplace=True)
                fill_symbol = le.transform(['NAN!!!!!!!!'])[0]
            elif df[col].dtype in ['int64', 'float64']:
                df[col].fillna(-11111, inplace=True)
                fill_symbol = le.transform([-11111])[0]

        df[col] = le.transform(df[col])
        df[col] = df[col].astype(int)
        # $B7gB;$,$"$k>l9g(B, nan $B$r%;%C%H(B
        if is_deficit:
            df[col] = df[col].replace(fill_symbol, np.nan)
    print(f'transformed_cols: {transformed_cols}')
    return le_dict if inplace else df, le_dict


def _fill_unseen(serieses, fill_value):
    base_series, target_series = serieses
    unseen_vals = set(target_series.unique()) - set(base_series.unique())
    replace_dict = {val: fill_value for val in unseen_vals}
    return target_series.replace(replace_dict)


def fill_unseens(base_df, target_df, target_cols, nthread,
                 fill_value=np.nan, inplace=False):
    if not inplace:
        target_df = target_df.copy()

    with Pool(nthread) as p:
        series_pairs = [[base_df[col], target_df[col]] for col in target_cols]
        iter_func = partial(_fill_unseen, fill_value=fill_value)
        filled = p.map(iter_func, series_pairs)
        p.close()
        p.join()
    filled_df = pd.concat(filled, axis=1)

    for col in target_cols:
        target_df[col] = filled_df[col]
    return None if inplace else target_df
