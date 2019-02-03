import datetime
import pickle
import sys
import time
import warnings
from itertools import tee
from logging import getLogger

import lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                     StratifiedKFold)
from tqdm import tqdm

import tools.models.my_lightgbm as mlgb
from tools.features.feature_tools import (get_all_features, load_features,
                                          select_features)
from tools.utils.encoding import fill_unseens, label_encoding
from tools.utils.general_utils import (dec_timer, get_locs, load_configs,
                                       log_evaluation, logInit, parse_args,
                                       sel_log, send_line_notification,
                                       test_commit)
from tools.utils.metrics import calc_best_MCC, calc_MCC, lgb_MCC
from tools.utils.samplings import resampling
from tools.utils.visualizations import save_importance

warnings.simplefilter(action='ignore', category=FutureWarning)


@dec_timer
def train(args, logger):
    '''
    policy
    ------------
    * use original functions only if there's no pre-coded functions
        in useful libraries such as sklearn.

    todos
    ------------
    * load features
    * train the model
    * save the followings
        * logs
        * oofs
        * importances
        * trained models
        * submissions (if test mode)

    '''
    # -- Prepare for training
    exp_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    train_base_dir = './inputs/train/'
    configs = load_configs('./config.yml', logger)

    # -- Load train data
    sel_log('loading training data ...', None)
    target = pd.read_pickle(
        train_base_dir + 'target.pkl.gz', compression='gzip')
    outliers = pd.read_pickle(
        train_base_dir + 'outliers.pkl.gz', compression='gzip')
    # Cache can be used only in train
    if args.use_cached_features:
        features_df = pd.read_pickle(
            './inputs/train/cached_featurse.pkl.gz', compression='gzip')
    else:
        if configs['train']['all_features']:
            _features = get_all_features('./inputs/train/')
        else:
            _features = configs['features']
        features_df = load_features(
            _features, train_base_dir, logger)
        # gen cache file if specified for the next time
        if args.gen_cached_features:
            features_df.to_pickle(
                './inputs/train/cached_featurse.pkl.gz', compression='gzip')
    # remove invalid features
    features_df.drop(configs['invalid_features'], axis=1, inplace=True)
    # label encoding categorical features
    sel_log('loading test data ...', None)
    test_base_dir = './inputs/test/'
    test_features_df = load_features(
        features_df.columns, test_base_dir, logger)
    trn_tst_df = pd.concat([features_df, test_features_df], axis=0)
    if configs['categorical_features']:
        sel_log('label encoding ...', None)
        trn_tst_df, le_dict = label_encoding(trn_tst_df,
                                             fit_columns=configs['categorical_features'])
    features_df = trn_tst_df.iloc[:features_df.shape[0]]
    test_features_df = trn_tst_df.iloc[features_df.shape[0]:]
    # feature selection if needed
    if configs['train']['feature_selection']:
        features_df = select_features(features_df,
                                      configs['train']['feature_select_path'],
                                      'gain_mean',
                                      configs['train']['feature_topk'])
    features = features_df.columns
    if configs['categorical_features']:
        categorical_features = sorted(
            list(set(features) &
                 set(configs['categorical_features'])))
    else:
        categorical_features = None

    # categorical_features = get_locs(
    #     features_df, configs['categorical_features'])

    # -- Data resampling
    # Stock original data for validation
#    if configs['preprocess']['resampling']:
#        target, id_measurement, features_df = resampling(
#            target, id_measurement, features_df,
#            configs['preprocess']['resampling_type'],
#            configs['preprocess']['resampling_seed'], logger)
    sel_log(f'the shape features_df is {features_df.shape}', logger)

    # -- Split using group k-fold w/ shuffling
    # NOTE: this is not stratified, I wanna implement it in the future
    if configs['train']['fold_type'] == 'skf':
        skf = StratifiedKFold(configs['train']['fold_num'], random_state=71)
        folds = skf.split(features_df, outliers)
    else:
        print(f"ERROR: wrong fold_type, {configs['train']['fold_type']}")
    folds, pred_folds = tee(folds)

    # -- Make training dataset
    train_set = mlgb.Dataset(features_df, target,
                             categorical_feature=categorical_features)
#    train_set = mlgb.Dataset(features_df.values, target.values,)
#                             feature_name=features,
#                             categorical_feature=configs['categorical_features'])

    # -- CV
    # Set params
    PARAMS = configs['lgbm_params']
    PARAMS['nthread'] = args.nthread
    # PARAMS['categorical_feature'] = categorical_features

    sel_log('start training ...', None)
    hist, cv_model = mlgb.cv(
        params=PARAMS,
        num_boost_round=10000,
        folds=folds,
        train_set=train_set,
        verbose_eval=100,
        early_stopping_rounds=200,
        metrics='rmse',
        callbacks=[log_evaluation(logger, period=100)],
    )

    # -- Prediction
    if configs['train']['single_model']:
        best_iter = cv_model.best_iteration
        single_train_set = lightgbm.Dataset(features_df.values, target.values)
        single_booster = lightgbm.train(
            params=PARAMS,
            num_boost_round=int(best_iter * 1.3),
            train_set=single_train_set,
            valid_sets=[single_train_set],
            verbose_eval=100,
            early_stopping_rounds=200,
            callbacks=[log_evaluation(logger, period=100)],
        )
        oofs = [single_booster.predict(features_df.values)]
        y_trues = [target]
        val_idxes = [features_df.index]
        scores = []
        y_true, y_pred = target, oofs[0]
        fold_importance_df = pd.DataFrame()
        fold_importance_df['split'] = single_booster.\
            feature_importance('split')
        fold_importance_df['gain'] = single_booster.\
            feature_importance('gain')
        fold_importance_dict = {0: fold_importance_df}
    else:
        sel_log('predicting using cv models ...', logger)
        oofs = []
        y_trues = []
        val_idxes = []
        scores = []
        fold_importance_dict = {}
        for i, idxes in tqdm(list(enumerate(pred_folds))):
            trn_idx, val_idx = idxes
            booster = cv_model.boosters[i]

            # Get and store oof and y_true
            y_pred = booster.predict(features_df.values[val_idx])
            y_true = target.values[val_idx]
            oofs.append(y_pred)
            y_trues.append(y_true)
            val_idxes.append(val_idx)

            # Calc RMSE
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            scores.append(rmse)

            # Save importance info
            fold_importance_df = pd.DataFrame()
            fold_importance_df['split'] = booster.feature_importance('split')
            fold_importance_df['gain'] = booster.feature_importance('gain')
            fold_importance_dict[i] = fold_importance_df

        rmse_mean, rmse_std = np.mean(scores), np.std(scores)
        sel_log(
            f'RMSE_mean: {rmse_mean}, RMSE_std: {rmse_std}',
            logger)

    # -- Post processings
    filename_base = f'{args.exp_ids[0]}_{exp_time}_{rmse_mean:.4}'

    # Save oofs
    with open('./oofs/' + filename_base + '_oofs.pkl', 'wb') as fout:
        pickle.dump([val_idxes, oofs], fout)

    # Save importances
    # save_importance(configs['features'], fold_importance_dict,
    save_importance(features, fold_importance_dict,
                    './importances/' + filename_base + '_importances',
                    topk=100)

    # Save trained models
    with open(
            './trained_models/' + filename_base + '_models.pkl', 'wb') as fout:
        pickle.dump(
            single_booster if configs['train']['single_model'] else cv_model,
            fout)

    # --- Make submission file
    if args.test:
        #        # -- Prepare for test
        #        test_base_dir = './inputs/test/'
        #
        #        sel_log('loading test data ...', None)
        #        test_features_df = load_features(
        #            features, test_base_dir, logger)
        #        # label encoding
        #        sel_log('encoding categorical features ...', None)
        #        test_features_df = fill_unseens(features_df, test_features_df,
        #                                        configs['categorical_features'],
        #                                        args.nthread)
        #        test_features_df, le_dict = label_encoding(test_features_df, le_dict)

        # -- Prediction
        sel_log('predicting for test ...', None)
        preds = []
        for booster in tqdm(cv_model.boosters):
            pred = booster.predict(test_features_df.values)
            preds.append(pred)
        target_values = np.mean(preds, axis=0)

        # -- Make submission file
        sel_log(f'loading sample submission file ...', None)
        sub_df = pd.read_csv(
            './inputs/origin/sample_submission.csv.zip',
            compression='zip')
        sub_df.target = target_values

        # print stats
        submission_filename = f'./submissions/{filename_base}_sub.csv.gz'
        sel_log(f'saving submission file to {submission_filename}', logger)
        sub_df.to_csv(submission_filename, compression='gzip', index=False)


if __name__ == '__main__':
    t0 = time.time()
    logger = getLogger(__name__)
    logger = logInit(logger, './logs/', 'lgb_train.log')
    args = parse_args(logger)

    logger.info('')
    logger.info('')
    logger.info(
        f'============ EXP {args.exp_ids[0]}, START TRAINING =============')
    train(args, logger)
    test_commit(args, './logs/test_commit.log')
    prec_time = time.time() - t0
    send_line_notification(
        f'Finished: {" ".join(sys.argv)} in {prec_time:.1f} s !')
