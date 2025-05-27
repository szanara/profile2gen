#!/bin/bash

#datasets_name=('urinary' 'parkinson' 'cholesterol' 'diabetes' 'fat' 'plasma')
model_names=('ctgan')
datasets_name=('fat' )
#model_names=('ctgan' 'tvae' 'nflow' 'bayesian_network')

for name in  "${datasets_name[@]}"; do
	for model in "${model_names[@]}"; do
		nohup python3 execute_evaluate_synthetic.py --data_path_synt /root to this fold/generated --dataset_name "$name" --output_path root to this fold /model_status/synthetic/"$name"  --performs_path root to this fold results_evaluation/synthetic/"$name"/"$model" --model_name  "$model"  --data_path_real /root to this fold/model_status/evaluate_synthetic_"$name"_"$model".out &
	done		
done
