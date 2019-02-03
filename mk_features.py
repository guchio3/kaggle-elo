import gc
import sys
import time
from logging import getLogger

from tools.features.feature_tools import _mk_features
from tools.features.hist_trans_features import (_hist_trans_features,
                                                _load_hist_trans_features_src)
from tools.features.meta_features import (_load_meta_features_src,
                                          _meta_features)
from tools.features.new_merc_trans_features import (_load_new_merc_trans_features_src,
                                                    _new_merc_trans_features)
from tools.utils.general_utils import (dec_timer, logInit, parse_args,
                                       send_line_notification)


@dec_timer
def mk_features(args, logger):
    trn_meta_df = None
    tst_meta_df = None
    series_df = None
    trn_tst_meta_df = None
    # base features
    series_df, trn_meta_df, tst_meta_df = _mk_features(
        _load_hist_trans_features_src, _hist_trans_features,
        args.nthread, args.exp_ids, series_df,
        trn_meta_df, tst_meta_df, logger=logger)
    series_df, trn_meta_df, tst_meta_df = _mk_features(
        _load_new_merc_trans_features_src, _new_merc_trans_features,
        args.nthread, args.exp_ids, series_df,
        trn_meta_df, tst_meta_df, logger=logger)
    trn_tst_meta_df, trn_meta_df, tst_meta_df = _mk_features(
        _load_meta_features_src, _meta_features,
        args.nthread, args.exp_ids, trn_tst_meta_df,
        trn_meta_df, tst_meta_df, logger=logger)
    gc.collect()


if __name__ == '__main__':
    t0 = time.time()
    logger = getLogger(__name__)
    logger = logInit(logger, './logs/', 'mk_features.log')
    args = parse_args(logger)

    logger.info('')
    logger.info('')
    logger.info(
        f'============ EXP {args.exp_ids[0]}, START MAKING FEATURES =============')
    mk_features(args, logger)
    prec_time = time.time() - t0
    send_line_notification(
        f'Finished: {" ".join(sys.argv)} in {prec_time:.1f} s !')
