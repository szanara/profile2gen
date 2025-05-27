#!/bin/bash

datasets_name=('diabetes')
#model_names=('ctgan' 'tvae' 'nflow' 'bayesian_network')
#datasets_name=('urinary')
model_names=('great')

for name in  "${datasets_name[@]}"; do
	for model in "${model_names[@]}"; do
        echo "Iniciando execução para dataset: $name e modelo: $model"
		nohup python3 execute_evaluate_synthetic.py --data_path_synt /root to the fold /FinalData --dataset_name "$name" --output_path /root to the fold/model_status/"$model"/data_final/"$name"  --performs_path /root to it /results_evaluation/data-centric/data_final/"$name"/"$model" --model_name  "$model"  --data_path_real /root/datasets/prepared --stage final  > /root/model_status/great/evaluate_final_sculpted_"$name"_"$model".out &
	done		
done

