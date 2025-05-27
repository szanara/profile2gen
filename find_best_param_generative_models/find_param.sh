#!/bin/bash

model_names=('ctgan' 'tvae' 'ddpm' 'nflow' 'bayesian_network' 'tabformer' 'great')
path2save='YOR PATH TO APPEND HERE/models/urinary'

for model in "${model_names[@]}"; do
    python YOR PATH /get_parameters.py --path2save "$path2save" --model_name "$model"
done

