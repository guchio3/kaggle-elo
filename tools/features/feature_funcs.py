import numpy as np


def percentile10(series):
    return np.percentile(series, 10)


def percentile25(series):
    return np.percentile(series, 10)


def percentile50(series):
    return np.percentile(series, 10)


def percentile75(series):
    return np.percentile(series, 10)


def percentile90(series):
    return np.percentile(series, 10)


def _get_nth_count_rat(series, n):
    count_list = series.value_counts()
    return count_list.iloc[n] / series.value_counts().sum() \
        if n < count_list.size and 0 <= n else np.nan


def get_0th_count_rat(series):
    return _get_nth_count_rat(series, 0)


def get_1st_count_rat(series):
    return _get_nth_count_rat(series, 1)


def get_2nd_count_rat(series):
    return _get_nth_count_rat(series, 2)


def get_last_count_rat(series):
    return _get_nth_count_rat(series, series.nunique() - 1)


def _get_nth_value(series, n):
    count_list = series.value_counts()
    return count_list.index[n] if n < count_list.size and 0 <= n else np.nan


def get_0th_value(series):
    return _get_nth_value(series, 0)


def get_1st_value(series):
    return _get_nth_value(series, 1)


def get_2nd_value(series):
    return _get_nth_value(series, 2)


def get_last_value(series):
    return _get_nth_value(series, series.nunique() - 1)
