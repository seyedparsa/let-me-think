import wandb
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import yaml

# sweep_name = "delta_flower_d2-s3-l5-b3_t8.5e6"
# sweep_ids = ["tqp3opz9", "44rx44p1", "xc3wofvr", "tq4eaubd"]

# sweep_name = "delta_flower_d1-s3-l5-b3_t4e6"
# sweep_ids = ["bdlg4lzm"]


useful_metrics = ['eval/evidence_accuracy', 'eval/decision_accuracy', 'eval/loss',]                    
useful_columns = ['teach', 'lr', 'seed']


def retrieve_runs(sweep_ids, metric, filename=None):
    data = []
    for sweep_id in sweep_ids:
        sweep = wandb.Api().sweep(f"{config['entity']}/{config['project']}/{sweep_id}")
        for run in tqdm(sweep.runs):
            history = run.history(pandas=True)
            if metric['name'] not in history.columns:
                continue    
            metric_values = history[metric['name']]        
            best_step_idx = metric_values.idxmax() if metric['mode'] == 'max' else metric_values.idxmin()
            best_step = history.loc[best_step_idx]                       
            row = run.summary._json_dict.copy()
            row.update(run.config)                    
            row = {key: row[key] for key in useful_columns}
            row["name"] = run.name
            row["run_id"] = run.id
            row["sweep_id"] = sweep_id
            metrics = {key.replace("eval/", ""): best_step[key] for key in useful_metrics}
            row.update(metrics)
            data.append(row)

    df = pd.DataFrame(data)
    if filename:
        df.to_csv(filename, index=False)
        print(f"CSV saved to {filename}.csv")
    return df
    

def select_best_run(df, metric, filename=None):
    metric_name = metric['name']
    def select_best_model(df):
        df = df.assign(teach=df.name)
        if df['teach'].iloc[0] == "dfs-pruned":
            return df.loc[df['loss'].idxmin()]    
        if metric['mode'] == 'max':
            return df.loc[df[metric_name].idxmax()]
        else:
            return df.loc[df[metric_name].idxmin()]

    df = df.groupby("teach").apply(select_best_model, include_groups=False)
    df.reset_index(drop=True, inplace=True)
    df.set_index("teach", inplace=True)
    if filename:
        df.to_csv(filename)
        print(f"CSV saved to {filename}")
    return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--retrieve', action='store_true')
    args = parser.parse_args()
    with open('eval/config/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open(f'eval/config/d{args.depth}.yaml', 'r') as f:
        config.update(yaml.safe_load(f))

    sweep_name = config['sweep_name']
    sweep_file = f"csv/sweeps/{sweep_name}.csv"
    if args.retrieve:
        runs_df = retrieve_runs(sweep_ids=config['sweep_ids'], metric=config['metric_for_best_step'], filename=sweep_file)
    else:
        runs_df = pd.read_csv(sweep_file)
    
    sweep_best_file = f"csv/sweeps/{sweep_name}_best_{config['metric_for_best_model']['name']}.csv"    
    df = select_best_run(runs_df, metric=config['metric_for_best_model'], filename=sweep_best_file)
    print(df)

    


