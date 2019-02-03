import pandas as pd

from ..utils.general_utils import sel_log


def get_neg_us_index(target, random_state=None):
    '''
    Get negative undersamling index.

    Positive and negative is not important here, so it can be more generalized.

    '''
    res_index = target[target == 1].index
    res_index = res_index.append(target[target == 0].sample(
        len(res_index), random_state=random_state).index)
    return res_index


def get_pos_os_index(target, random_state=None):
    '''
    Get positive oversamling index.

    Positive and negative is not important here, so it can be more generalized.

    '''
    pos_size = target[target == 1].count()
    neg_size = target[target == 0].count()
    assert neg_size >= pos_size, \
        f'The sample of pos is more than neg ! ({pos_size} : {neg_size})'

    res_index = target.index
    res_index = res_index.append(target[target == 1].sample(
        neg_size - pos_size, replace=True, random_state=random_state).index)
    return res_index


def resampling(target, id_measurement, features_df,
               resampling_type, random_state, logger):
    if resampling_type == 'under':
        sel_log('now under sampling ...', None)
        resampled_index = get_neg_us_index(target, random_state)
        target = target.loc[resampled_index]
        id_measurement = id_measurement.loc[resampled_index]
        features_df = features_df.loc[resampled_index]
    elif resampling_type == 'over':
        sel_log('now down sampling ...', None)
        resampled_index = get_pos_os_index(target, random_state)
        target = target.loc[resampled_index]
        id_measurement = id_measurement.loc[resampled_index]
        features_df = features_df.loc[resampled_index]
    else:
        sel_log(f'ERROR: wrong resampling type ({resampling_type})', logger)
        sel_log('plz specify "under" or "over".', logger)
    return target, id_measurement, features_df
