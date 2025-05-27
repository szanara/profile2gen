#!/bin/bash

#model_names=('ctgan' 'tvae' 'ddpm' 'nflow' 'bayesian_network')
model_names=('ctgan')
#datasets_names=('urinary' 'parkinson' 'cholesterol' 'diabetes' 'fat' 'plasma')
datasets_names=('fat')

for model in "${model_names[@]}"; do
    for dataset in "${datasets_names[@]}"; do
        nohup python3 executing_generating_sculped.py --model_name "$model" --dataset_name "$dataset" > /path /model_status/generating_sculped_"$dataset"_"$model".out
    done
done

