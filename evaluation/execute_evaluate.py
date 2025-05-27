import argparse
parser = argparse.ArgumentParser(description="AutoGluon Training Script")

# Adicione argumentos
parser.add_argument('--data_path', type=str, required=True, help="Path to the training data")
parser.add_argument('--dataset_name', type=str, required=True, help="Name of the label column in the dataset")
parser.add_argument('--output_path', type=str, required=True, help="Path to save the trained model")
parser.add_argument('--performs_path', type=str, help="Time limit for training in seconds")

args = parser.parse_args()
import sys
sys.path.append('/root /utils')
from models_to_evaluate import *
from functions import *
import pandas as pd 

os.makedirs(args.output_path, exist_ok=True)
dataset_train = pd.read_csv(args.data_path+f'/{args.dataset_name}/{args.dataset_name}_train.csv')
dataset_test = pd.read_csv(args.data_path+f'/{args.dataset_name}/{args.dataset_name}_test.csv')

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