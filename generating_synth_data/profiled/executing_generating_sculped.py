import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("--model_name", type= str, required=True)
argparser.add_argument("--dataset_name")
args = argparser.parse_args()

import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys
sys.path.append('/path/utils')
from generating_sculped import *
from generating import *
from pickle_functions import *
sys.path.append('path/utils/')
from functions import * 

data_path = '/path to it /datasets/prepared'
path2data_complete= os.path.join(data_path,args.dataset_name)
data = pd.read_csv(path2data_complete+f'/{args.dataset_name}_train.csv')
p= PrepareDataToGen(data)
data = p.withoutnan()
#print(data)
params_path = '/path to it /models/hparams' 

params = load_from_pickle(path=Path(params_path+f'/{args.model_name}.pkl'))

bests_path = '/path to it /models/frameworks'


bests = np.load(bests_path+f'/{args.dataset_name}/best_framework.npy', allow_pickle=True)
print(params.best_params)
path_saving = f'{args.dataset_name}/{args.model_name}'

gen = GeneratingSculped(best =bests,
                         synthetic_model= args.model_name, 
                         params = params.best_params,
                           data_df = data, 
                           path2save = path_saving)

gen.generating()