import re
from statsmodels.stats.proportion import proportion_confint
import yaml
import pandas as pd
from scipy.stats import t
import numpy as np



def pres(word):    
    if word == 'optimal':
        return 'Shortest-Path'
    elif word == 'path':
        return 'Path'
    elif word == 'dfs':
        return 'DFS-BT'
    elif word == 'dfs-pruned':
        return 'DFS'
    elif word == "avg_thoughts":
        return "Avgerage Length of Thoughts"
    elif word == "avg_valid_thoughts":
        return "Avgerage Length of CoTs"
    elif word == "avg_verified_thoughts":
        return "Avgerage Length of Verified CoTs"
    elif 'play' in word:
        teach, _, num_play = re.split('[-_]', word)
        return f"{pres(teach)} RL-{num_play}"
    elif 'walk' in word:
        _, L = re.split('[-_]', word)
        return f"Walk-{L}"
    else:
        return ' '.join([word.capitalize() for word in re.split('[-_]', word)])    
    
def confidence_interval(succes, trials, confidence=0.95, method='wilson'):
    return proportion_confint(succes, trials, alpha=1-confidence, method='wilson')


def normal_confidence_interval(mean, std, n):
    z = 1.96
    se = std / np.sqrt(n)
    margin = z * se
    lower = mean - margin
    upper = mean + margin
    return lower, upper


def load_eval(depth, filter=None):
    with open(f'eval/config/base.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    metric_name = base_config['metric_for_best_model']['name']
    with open(f'eval/config/d{depth}.yaml', 'r') as f:
        config = yaml.safe_load(f)
    sweep_name = config['sweep_name']
    eval_file = f"csv/evals/{sweep_name}_best_{metric_name}_gd_n5000.csv"
    df = pd.read_csv(eval_file)
    # print(depth)
    # print(df)
    if filter:
        df = df[df['teach'].isin(filter)]
        df['order'] = df['teach'].apply(lambda x: filter.index(x))    
        df = df.sort_values(by='order', ascending=True)
        df.reset_index(drop=True, inplace=True)
    return df, sweep_name