#!/bin/bash

datasets_name=('urinary' 'parkinson' 'urinary' 'parkinson' 'cholesterol' 'diabetes' 'fat' 'plasma')

for name in  "${datasets_name[@]}"; do
	nohup python3 execute_evaluate.py --data_path /root/datasets/prepared --dataset_name "$name" --output_path /root/model_status/originais/"$name"  --performs_path /root/results_evaluation/"$name" | tee /root/model_status/evaluate_originals_"$name".out &
done
