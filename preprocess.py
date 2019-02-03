import gc
import os
import sys
import time
import datetime
from logging import getLogger
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from tools.utils.general_utils import (dec_timer, logInit,
                                       send_line_notification)


@dec_timer
def preprocess(logger):
    '''
    policy
    ------------
    * Directly edit the src for each preprocessing because preprocess
        should be ran only once.

    '''
    # First preprocessing, change the formats
    if False:
        logger.info('start converting origin to better format one.')
        df = pd.read_csv(
            './inputs/origin/historical_transactions.csv.zip',
            compression='zip')
        df.to_pickle(
            './inputs/prep/historical_transactions.pkl.gz',
            compression='gzip')
        df = pd.read_csv(
            './inputs/origin/merchants.csv.zip',
            compression='zip')
        df.to_pickle('./inputs/prep/merchants.pkl.gz',
                     compression='gzip')
        df = pd.read_csv(
            './inputs/origin/new_merchant_transactions.csv.zip',
            compression='zip')
        df.to_pickle('./inputs/prep/new_merchant_transactions.pkl.gz',
                     compression='gzip')
        df = pd.read_csv(
            './inputs/origin/train.csv.zip',
            compression='zip')
        df.to_pickle('./inputs/prep/train.pkl.gz',
                     compression='gzip')
        df = pd.read_csv(
            './inputs/origin/test.csv.zip',
            compression='zip')
        df.to_pickle('./inputs/prep/test.pkl.gz',
                     compression='gzip')
    if True:
        logger.info('adding datetime info to transaction data')
        df = pd.read_pickle(
            './inputs/prep/historical_transactions.pkl.gz',
            compression='gzip')
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        df['year'] = df['purchase_date'].dt.year
        df['weekofyear'] = df['purchase_date'].dt.weekofyear
        df['month'] = df['purchase_date'].dt.month
        df['dayofweek'] = df['purchase_date'].dt.dayofweek
        df['weekend'] = (df.purchase_date.dt.weekday >= 5).astype(int)
        df['hour'] = df['purchase_date'].dt.hour
        # https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73244
        df['month_diff'] = (
            (datetime.datetime(2019, 1, 1) - df['purchase_date']).dt.days) // 30
        df['month_diff'] += df['month_lag']
        df.sort_values(['card_id', 'purchase_date'], inplace=True)
        df.to_pickle(
            './inputs/prep/historical_transactions_v1.pkl.gz',
            compression='gzip')
        df = pd.read_pickle(
            './inputs/prep/new_merchant_transactions.pkl.gz',
            compression='gzip')
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        df['year'] = df['purchase_date'].dt.year
        df['weekofyear'] = df['purchase_date'].dt.weekofyear
        df['month'] = df['purchase_date'].dt.month
        df['dayofweek'] = df['purchase_date'].dt.dayofweek
        df['weekend'] = (df.purchase_date.dt.weekday >= 5).astype(int)
        df['hour'] = df['purchase_date'].dt.hour
        # https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73244
        df['month_diff'] = (
            (datetime.datetime(2019, 1, 1) - df['purchase_date']).dt.days) // 30
        df['month_diff'] += df['month_lag']
        df.sort_values(['card_id', 'purchase_date'], inplace=True)
        df.to_pickle('./inputs/prep/new_merchant_transactions_v1.pkl.gz',
                     compression='gzip')


if __name__ == '__main__':
    t0 = time.time()

    logger = getLogger(__name__)
    logger = logInit(logger, './logs/', 'preprocess.log')
    preprocess(logger)

    prec_time = time.time() - t0
    send_line_notification(f'Finished pre-processing in {prec_time:.1f} s !')
