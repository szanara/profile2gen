#!/bin/bash
"""Attention in path2save. You shoul keep the 'post' as it is. 
Its is separing real framework from snthetic framework"""

#datasets_name=('urinary' 'parkinson' 'cholesterol' 'diabetes' 'fat' 'plasma')
#model_name=('ctgan' 'tvae' 'nflow' 'bayesian_network')
datasets_name=('fat')
model_name=('ctgan')
for name in "${datasets_name[@]}"; do
	for model in "${model_name[@]}"; do
	    nohup python3 executing_chosing_framework.py \
	    --data_path_real /path to the synthetic data generaed after the profiling /generated/data-centric \
	    --path2save post/"$name"/"$model" \
	    --noise_level "[0, 0.02, 0.08, 0.1, 0.25, 0.20]" \
	    --dataset_name "$name" \
	    --datacentric_treshold "[0.1, 0.125, 0.15, 0.175, 0.2, 0.25]" \
	    --stage post \
	    --model "$model" | tee /path to you check the status /model_status/choosing_framework_post_"$model"_"$name".out &
	done
done



