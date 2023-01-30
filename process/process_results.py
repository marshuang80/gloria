import csv
from functools import reduce
import json
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

MODELS = ['imagenet_supervised_pretrain', 'chexpert_supervised_pretrain', 'chexpert_moco_pretrain', 'chexpert_gloria_pretrain']
DATASETS = ['chexpert', 'intermountain', 'candid_ptx']
FT_METHODS = ['lh']
METRICS = ['macro_auroc', 'macro_auprc', 'macro_f1', 'micro_auroc', 'micro_auprc', 'micro_f1']
TRAIN_PCT = ['0.01', '0.05', '0.1', '0.2']

def generate_stats(df):
    group_cols = ['model','method','train_pct']
    stats = []

    for metric in METRICS:
        stats_for_metric = df.groupby(group_cols)[metric].agg(['mean', 'count', 'std'])
        ci95_hi = []
        ci95_lo = []

        for i in stats_for_metric.index:
            m, c, s = stats_for_metric.loc[i]
            ci95_hi.append(m + 1.96*s/math.sqrt(c))
            ci95_lo.append(m - 1.96*s/math.sqrt(c))

        stats_for_metric[f'{metric}_mean'] = stats_for_metric['mean']
        stats_for_metric[f'{metric}_ci95_hi'] = ci95_hi
        stats_for_metric[f'{metric}_ci95_lo'] = ci95_hi

        stats_for_metric = stats_for_metric.reset_index()
        stats_for_metric = stats_for_metric[group_cols + [f'{metric}_mean', f'{metric}_ci95_hi', f'{metric}_ci95_lo']]

        stats.append(stats_for_metric)

    stats = reduce(lambda x, y: pd.merge(x, y, on=group_cols), stats)
    return stats


for dataset in DATASETS:
    for method in FT_METHODS:
        data = []
        models = MODELS
        # if dataset == 'chexpert':
        #     models = MODELS
        # else:
        #     models = MODELS + [f'{dataset}_spt_gloria_pretrain', f'{dataset}_dapt_gloria_pretrain']

        for model in models:
            for train_pct in TRAIN_PCT:
                for split in range(1, 6):
                    filename = f'/home/cvanuden/git-repos/gloria/data/output/{model}_{method}_{dataset}_classifier_{train_pct}/{split}/results.csv'
                    try:
                        with open(filename) as f:
                            curr_dict = json.load(f)
                            curr_dict['model'] = model
                            curr_dict['method'] = method
                            curr_dict['train_pct'] = float(train_pct)
                            data.append(curr_dict)
                    except:
                        print(filename)

            all_data_dir = f'/home/cvanuden/git-repos/gloria/data/output/{model}_{method}_{dataset}_classifier_1.0'
            date = [f for f in os.listdir(all_data_dir)][0]
            filename = f'{all_data_dir}/{date}/results.csv'
            with open(filename) as f:
                curr_dict = json.load(f)
                curr_dict['model'] = model
                curr_dict['method'] = method
                curr_dict['train_pct'] = 1
                data.append(curr_dict)

        df = pd.DataFrame.from_dict(data)
        stats = generate_stats(df)
        print(stats)

        df.to_csv(f'/home/cvanuden/git-repos/gloria/data/results/{dataset}_{method}_results.csv', index=False)
        stats.to_csv(f'/home/cvanuden/git-repos/gloria/data/results/{dataset}_{method}_stats.csv', index=False)

        for metric in METRICS:
            plt.figure()
            sns.lineplot(data=df, x="train_pct", y=metric, hue="model")
            plt.xscale('log')
            plt.xlabel(f'Fraction of {dataset} finetuning data used')
            plt.ylabel(metric)
            plt.title(f"{dataset} ({metric})")
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.savefig(f"/home/cvanuden/git-repos/gloria/data/images/{dataset}_{metric}.png", bbox_inches='tight', dpi=300)

data = []
for dataset in DATASETS:
    for method in FT_METHODS:
        data.append(pd.read_csv(f'/home/cvanuden/git-repos/gloria/data/results/{dataset}_{method}_results.csv'))

df = pd.concat(data).reset_index()
df = df.replace(
    {
        'model': {
            'intermountain_spt_gloria_pretrain': 'spt_gloria_pretrain',
            'intermountain_dapt_gloria_pretrain': 'dapt_gloria_pretrain',
            'candid_ptx_spt_gloria_pretrain': 'spt_gloria_pretrain',
            'candid_ptx_dapt_gloria_pretrain': 'dapt_gloria_pretrain',
        }
    }
)

for metric in METRICS:
    plt.figure()
    sns.lineplot(data=df, x="train_pct", y=metric, hue="model")
    plt.xscale('log')
    plt.title(f"{metric}")
    plt.xlabel(f'Fraction of finetuning data used')
    plt.ylabel(metric)
    plt.savefig(f"/home/cvanuden/git-repos/gloria/data/images/all_tasks_{metric}.png", dpi=300)