#!/bin/bash

#model_names=('ctgan' 'tvae' 'ddpm' 'nflow' 'bayesian_network')
model_names=('ctgan')

#datasets_names=('urinary' 'parkinson' 'cholesterol' 'diabetes' 'fat' 'plasma')
datasets_names=('fat')
path2save='/path/models/'

for model in "${model_names[@]}"; do
    for dataset in "${datasets_names[@]}"; do
        python /executing_generating.py --model_name "$model" --dataset_name "$dataset"
    done
done

