import argparse
parser = argparse.ArgumentParser(description="AutoGluon Training Script")

# Adicione argumentos
parser.add_argument('--data_path_real', type=str, required=True, help="Path to the training data")
parser.add_argument('--data_path_synt', type=str, required=True)
parser.add_argument('--dataset_name', type=str, required=True, help="Name of the label column in the dataset")
parser.add_argument('--output_path', type=str, required=True, help="Path to save the trained model")
parser.add_argument('--performs_path', type=str, help="Time limit for training in seconds")
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--stage', type= str, required=True)

args = parser.parse_args()
from models_to_evaluate import *
import sys
sys.path.append('/root to utils/utils')
from functions import *
import pandas as pd 

print('PATH', args.data_path_synt)
os.makedirs(args.output_path, exist_ok=True)
if args.stage=='pt1':
    dataset_train = pd.read_csv(args.data_path_synt+f'/{args.dataset_name}/{args.model_name}/dataset.csv',sep=';')  
elif args.stage=='final':
    dataset_train = pd.read_csv(args.data_path_synt+f'/{args.dataset_name}/{args.model_name}/dataset.csv',sep=',')
else:
    dataset_train = pd.read_csv(args.data_path_synt+f'/{args.dataset_name}/{args.model_name}.csv')             
#dataset_test = pd.read_csv(args.data_path_real+f'/{args.dataset_name}/{args.dataset_name}_test.csv')

### for generated after sculpting
print(dataset_train)
dataset_test = pd.read_csv(args.data_path_real+f'/{args.dataset_name}/{args.dataset_name}_test.csv')

prepare_train= PrepareDataToGen(dataset_train)
prepare_test = PrepareDataToGen(dataset_test)

X_train = prepare_train.filtering_transformer()
X_test = prepare_test.filtering_transformer()


eval = EvaluateData(data_df_train = X_train,
                    data_df_test =X_test,
                    path2saveweights =args.output_path
                    ,path2saveperfs = args.performs_path
                    )

eval.evaluate_and_save()