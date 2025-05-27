#!/bin/bash

datasets_name=('fat' 'cholesterol' 'diabetes' 'fat' 'urinary' 'parkinson' 'plasma')

for name in "${datasets_name[@]}"; do
     echo "DATASETTTT is $name"
    nohup python3 executing_chosing_framework.py \
    --data_path_real /root to datasets prepared /datasets/prepared \
    --path2save "$name" \
    --noise_level "[0, 0.02, 0.08, 0.1, 0.25, 0.20]" \
    --dataset_name "$name" \
    --datacentric_treshold "[0.1, 0.125, 0.15, 0.175, 0.2, 0.25]"\
    --stage pre | tee your path to check the evolution/model_status/choosing_framework_"$name".out &
done


