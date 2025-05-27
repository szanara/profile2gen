import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset_name", type= str, required=True)
argparser.add_argument('--model_name',type=str, required=True)
args = argparser.parse_args()
import pandas as pd
import numpy as np
import sys 
sys.path.append('your path to utils/utils')
from functions import * 
from sculping_data import *
path_data = 'your path to synthetic profiled_data/generated/data-centric'
path_framework = f'your path to framework for the best frameworks /models/frameworks/post/{args.dataset_name}/{args.model_name}'
fr = np.load(path_framework+'/best_framework.npy',allow_pickle=True)
framework = fr[1]
thresh = float(fr[2])

dataset_train = pd.read_csv(path_data+f'/{args.dataset_name}/{args.model_name}/dataset.csv',sep=';')
dataset = PrepareDataToGen(dataset_train)
data_full  = dataset.filtering_transformer()
X = data_full.drop(columns='target')
y = data_full['target']

if framework == 'cleanlab':      
    
    X_easy, X_ambiguous,X_hard, y_easy, y_ambiguous,  y_hard,  indices,  percentile_threshold, data_centric_threshold = sculpt_with_cleanlab(  
                                                                       X = X,
                                                                       y = y,
                                                                       data_centric_threshold =thresh ,
                                                                        ) 

else: 
    X_easy, X_ambiguous,X_hard, y_easy, y_ambiguous,  y_hard,  indices,  percentile_threshold, data_centric_threshold = sculpt_with_dataiq(
    X=X,
    y = y,
    percentile_threshold= None,
    data_centric_threshold=thresh,
    method = "dataiq"
) 


indices_dict = dict(indices[1:][0])
easy = indices_dict['easy']                    
hard = indices_dict['hard'] 
ambi = indices_dict['ambiguous']                   
path = os.path.join(f'your path to save the indices /indices/{args.dataset_name}/{args.model_name}')
os.makedirs(path, exist_ok=True)
if ambi is not None:
    np.save(path+f'/ambi.npy',ambi)
else:
    np.save(path+f'/easy.npy',easy)
    np.save(path+f'/hard.npy',hard)
print(f'SAVED')









