#!/bin/bash

datasets_name=( 'fat' )
#model_names=('ctgan' 'tvae' 'nflow' 'bayesian_network')
#datasets_name=('urinary')
model_names=('ctgan')

for name in  "${datasets_name[@]}"; do
	for model in "${model_names[@]}"; do
        echo "Iniciando execução para dataset: $name e modelo: $model"
		nohup python3 execute_evaluate_synthetic.py --data_path_synt root to the fold/generated/data-centric --dataset_name "$name" --output_path  root ot the fold/model_status/data-centric/parte1/"$name"/"$model"  --performs_path root to the fold/ results_evaluation/data-centric/parte1/"$name"/"$model" --model_name  "$model"  --data_path_real root to the fold/datasets/prepared --stage pt1  > root to the fold/model_status/evaluate_pt1_sculpted_"$name"_"$model".out &
	done		
done

