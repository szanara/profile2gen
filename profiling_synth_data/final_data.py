import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset_name", type= str, required=True)
argparser.add_argument('--model_name',type=str, required=True)
args = argparser.parse_args()
import pandas as pd
import numpy as np
import sys 
sys.path.append('you root path/utils')
from functions import * 
from sculping_data import *

#path_data = 'you root path/generated/data-centric'
path_data = 'you root path/generated/data-centric'

path_labels = f'you root path/indices/{args.dataset_name}/{args.model_name}'
hard_path = os.path.join(path_labels, 'hard.npy')

path = os.path.join(f'you root path/FinalData/{args.dataset_name}/{args.model_name}')
os.makedirs(path, exist_ok=True)
# Verifica a existência do arquivo hard.npy
if os.path.exists(hard_path):
    hard = np.load(hard_path, allow_pickle=True)
else:
    hard = None

dataset_path = os.path.join(path_data, f'{args.dataset_name}/{args.model_name}/dataset.csv')

# Verifica a existência do arquivo dataset.csv
try:
    dataset_train = pd.read_csv(dataset_path, sep=';')
except FileNotFoundError:
    print(f'Arquivo {dataset_path} não encontrado. Pulando para o próximo.')
else:
    # Manipulação condicional do dataset
    if hard is not None:
        new_dataset_train = dataset_train.drop(hard)
        new_dataset_train.to_csv(os.path.join(path, 'dataset.csv'), sep=',', index=False)
    else:
        print('None hard')
        dataset_train.to_csv(os.path.join(path, 'dataset.csv'), sep=',', index=False) 