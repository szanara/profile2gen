"""Code to run the hyperparameter optimization of synthetic data generation models.
Code from  original reposittory https://github.com/vanderschaarlab/data-centric-synthetic-data/tree/main """
import argparse
from functools import partial

import optuna
import pandas as pd
from synthcity.benchmark import Benchmarks
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.utils.optuna_sample import suggest_all
import sys 
sys.path.append('/path to utils/utils')
from data2noise import *
from pickle_functions import *
from functions import *

def objective(trial: optuna.Trial, model_name: str) -> float:
    """Optuna objective function for hyperparameter optimization of synthetic
    data generation models."""
    hp_space = Plugins().get(model_name).hyperparameter_space()
    params = suggest_all(trial=trial, distributions=hp_space)
    if "batch_size" in params:
        params["batch_size"] = 15
    if model_name == "ddpm":
        params["is_classification"] = False
    trial_id = f"trial_{trial.number}"
    try:
        report = Benchmarks.evaluate(
            [(trial_id, model_name, params)],
            loader,
            repeats=1,
            metrics={"detection": ["detection_xgb"]},
        )
    except Exception as e:  # invalid set of params
        print(f"{type(e).__name__}: {e}")
        print(params)
        raise optuna.TrialPruned
    return report[trial_id]["mean"].mean()


def uint_cols_to_int(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "uint8":
            df[col] = df[col].astype(int)
    return df


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_name", type=str, required=True)
    argparser.add_argument("--path2save", type= str, required=True)
    args = argparser.parse_args()
    model_name = args.model_name
    RESULTS_DIR  = args.path2save
    
    df = read_data(path= ' YOUR PATH TO DATASETS/ datasets/prepared/fat' 
                    file_name='urinary_train.csv'
    )
    df = uint_cols_to_int(df)
    p = PrepareDataToGen(df)
    df = p.withoutnan()
    save_dir = RESULTS_DIR+"/hparams"
    os.makedirs(save_dir,exist_ok=True)
    #save_dir.mkdir(exist_ok=True, parents=True)

    loader = GenericDataLoader(
        df,
        target_column='target',
    )

    print(f"Optimizing hyperparameters for {model_name}")
    study = optuna.create_study(direction="minimize")
    obj_fun = partial(objective, model_name=model_name)
    study.optimize(obj_fun, n_trials=10)
    save_to_pickle(study, path=save_dir+f"/{model_name}.pkl")