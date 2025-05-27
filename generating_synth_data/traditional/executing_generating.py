import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("--model_name", type=str, required=True)
argparser.add_argument("--dataset_name", type= str, required=True)
args = argparser.parse_args()


import os
import pandas as pd
from pathlib import Path
import sys
sys.path.append('/path/utils')

from generating import *
from pickle_functions import *

data_path = '/path/datasets/prepared'
path2data_complete= os.path.join(data_path,args.dataset_name)
data = pd.read_csv(path2data_complete+f'/{args.dataset_name}_train.csv')


params_path = '/path/models/hparams'
params = load_from_pickle(path=Path(params_path+f'/{args.model_name}.pkl'))


synthetic_data = fit_and_generate_synth_data(
    data=data,
    model_name=args.model_name,
    model_params=  params.best_params,
    target_column= 'target')


path =  '/path/generated'
path_complete = os.path.join(path,args.dataset_name)
os.makedirs(path_complete, exist_ok=True)
synthetic_data.to_csv(path_complete+f'/{args.model_name}.csv')