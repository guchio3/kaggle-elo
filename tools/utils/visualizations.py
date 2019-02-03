from itertools import chain

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# ==========================================
#  tools for train info
# ==========================================
def save_importance(features, fold_importance_dict,
                    filename_base, topk=30, main_metric='gain'):
    assert main_metric in ['gain', 'split'], \
        f'please specify gain or split as main_metric'
    dfs = []
    for fold in fold_importance_dict:
        df = fold_importance_dict[fold]
        df = df.add_suffix(f'_{fold}')
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    df['features'] = features
    splits = df.loc[:, df.columns.str.contains('split')]
    gains = df.loc[:, df.columns.str.contains('gain')]

    # stats about splits
    df['split_mean'] = splits.mean(axis=1)
    df['split_std'] = splits.std(axis=1)
    df['split_cov'] = df.split_std / df.split_mean

    # stats about gains
    df['gain_mean'] = gains.mean(axis=1)
    df['gain_std'] = gains.std(axis=1)
    df['gain_cov'] = df.gain_std / df.gain_mean

    # sort and save to csv
    df.sort_values(by=main_metric + '_mean', ascending=False, inplace=True)
    df.to_csv(filename_base + '.csv', index=False)

    # plot and save fig
    plt_dfs = []
    for fold in fold_importance_dict:
        plt_df = pd.DataFrame(fold_importance_dict[fold][main_metric])
        plt_df['features'] = features
        plt_dfs.append(plt_df)
    plt_df = pd.concat(plt_dfs, axis=0)

    # Plot! note that use only top-k
    plt.figure(figsize=(20, 10))
    sns.barplot(x=main_metric, y='features', data=plt_df,
                order=df.features.head(topk))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(filename_base + '.png')


# ==========================================
#  tools for EDA
# ==========================================
def _plt_3phase_current(df, meta_df, id_measurement,
                        ax, ylim, alpha, fontsize):
    # get info from meta_df
    target_df = meta_df.query(f'id_measurement == {id_measurement}')
    signal_ids = target_df.signal_id.astype(str)
    phase = target_df.phase.tolist()
    targets = target_df.target.tolist()
    y_preds = target_df['y_pred'].round(4).tolist() \
        if 'y_pred' in target_df.columns else None
    plt_df = df.loc[:, signal_ids]

    # plot
    for i in range(3):
        ax.plot(plt_df.iloc[:, i], label=f'phase {phase[i]}', alpha=alpha)

    # decoration
    if y_preds is None:
        ax.set_title(
            f'id_measurement: {id_measurement}, targets: {targets}',
            fontsize=fontsize)
    else:
        ax.set_title(
            f'id_measurement: {id_measurement}, targets: {targets}, y_preds: {y_preds}',
            fontsize=fontsize)
    ax.set_xlabel('time', fontsize=fontsize)
    ax.set_ylabel('signal', fontsize=fontsize)
    ax.legend()
    ax.set_ylim(ylim)


def plt_3phase_currents(df, meta_df, id_measurements, fig_title=None,
                        height_base=3, width_base=8, col_num=4,
                        ylim=None, alpha=1.):
    fontsize = int(height_base * width_base / 2)
    if hasattr(id_measurements, "__iter__"):
        height, width = len(id_measurements) // col_num, col_num
        if len(id_measurements) % col_num != 0:
            height += 1
        fig, axs = plt.subplots(height, width, figsize=(
            width * width_base, height * height_base, ))
        axs = list(chain.from_iterable(axs))
    else:
        id_measurements = [id_measurements]
        fig = plt.figure(figsize=(width_base, height_base))
        ax = fig.add_subplot(111)
        axs = [ax]

    for i, id_measurement in enumerate(id_measurements):
        ax = axs[i]
        _plt_3phase_current(
            df,
            meta_df,
            id_measurement,
            ax,
            ylim,
            alpha,
            fontsize)

    # decoration
    if fig_title:
        if len(id_measurements) == 1:
            fig.suptitle(fig_title, fontsize=fontsize, va='bottom')
        else:
            fig.suptitle(fig_title, fontsize=fontsize * 1.5, va='bottom')
    # apply tight_layout
    fig.tight_layout(rect=[0, 0.08, 1, 0.99])
