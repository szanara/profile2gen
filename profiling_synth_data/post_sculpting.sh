#!/bin/bash

#datasets_name=('urinary' 'parkinson' 'cholesterol' 'diabetes' 'fat' 'plasma')
#models_name=('ctgan' 'tvae' 'nflow' 'bayesian_network')
datasets_name=('fat')
models_name=('ctgan')
output_path='you path to check it up/model_status/outs'

# Cria o diretório de saída se ele não existir
mkdir -p "$output_path"

for name in "${datasets_name[@]}"; do
    echo "$name"
    for model in "${models_name[@]}"; do
        echo "$model"
        nohup python3 after_profile_synth_data.py --dataset_name "$name" --model_name "$model" > "${output_path}/${name}_${model}.out" 2>&1 &
    done
done
