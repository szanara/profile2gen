#!/bin/bash

datasets_name=('fat')
#model_names=('ctgan' 'tvae' 'nflow' 'bayesian_network')
#datasets_name=('urinary')
model_names=('ctgan')

for name in  "${datasets_name[@]}"; do
	for model in "${model_names[@]}"; do
        echo "Iniciando execução para dataset: $name e modelo: $model"
		nohup python3 final_data.py  --dataset_name "$name" --model_name  "$model"  > your root outs/final_data.out &
	done		
done

