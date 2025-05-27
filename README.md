# profile2gen
This repository corresponds to the implementation of the research work titled:

"Breaking the Barrier of Hard Samples: A Data-Centric Approach to Synthetic Data for Medical Tasks"
Accepted at ICML 2025.
 
 The repositoy is structured as:
```

├── datasets/
│   ├── originals/
│   └── prepared/
│   └── notebooks_to_prepare_data
├── choosing_framework/
│   └── utils
│   └── choosing_data_framework.py
│   └── executing_choosing_framework.py
│   └── choosing_framework_2realdatasets.sh
│   └── choosing_framework_p2syntheticdata.sh
│
├── generating_synth_data/
│   └── utils
│   └── traditional
│   └── profiled
├── profiling_synth_data/
│   └── utils
│   └── after_profile_synth_data.py
│   └── post_sculpting.sh
│   └── finl_data.py
│   └──final_data.sh
├── models/
├── generated
├──outs
├── evaluation
│   └── execute_evaluate_synthetic.py
│   └── evaluate_synthetic.sh
│   └── evaluate_step1_synthetic.sh
│   └── evalaute_originals.sh
│   └──execure_evaluate.py
├──indices
├──FinalData
├── requirements.txt
└── README.md

```











## **Project Structure and Execution Guide** 
The folder names in this project are self-explanatory and follow the logical flow of the Profile2Gen process.

Each of the following folders:
* choosing_framework/

* generating_synth_data/

* profiling_synth_data/

* evaluation/

contains a subfolder named utils/ that includes base utility code required by each corresponding main script.
Each main script has a corresponding .sh file that executes it.

**Important:** Before running any script, you must manually update the paths inside the code files. Folder names should not be changed. Paths were written using "root" or "root for this folder" as placeholders — these should be replaced with the actual root path on your system.

**Some folders may appear empty** — this is intentional. Certain scripts depend on these folders being present and may crash if they are missing.



## **Execution Order**


--> **To Generate Synthetic Data**


```
# 1. Find best parameters for all generative models
bash /root/find_best_param_generative_models/find_param.sh

# 2. Select the best framework to initiate preprocessing based on flipping rate
bash /root/choosing_framework/choosing_framework_2realdatasets.sh

# 3. Generate synthetic data in the traditional way
bash /root/generating_synth_data/traditional/generating.sh

# 4. Generate synthetic data using best profiler
bash /root/generating_synth_data/profiled/generating_Sculped.sh

# 5. Select the best profiling framework for the synthetic data
bash /root/choosing_framework/choosing_framework_p2syntheticdata.sh

# 6. Profile synthetic data using the selected framework
bash /root/profiling_synth_data/post_sculpting.sh

# 7. Remove hard samples to produce the final synthetic dataset
bash /root/profiling_synth_data/final_data.sh
```


-->** For evaluation process**


```# Evaluate the real dataset
bash /root/evaluation/evaluate_originals.sh

# Evaluate traditionally generated synthetic data
bash /root/evaluation/evaluate_synthetic.sh

# Evaluate preprocessed synthetic data (after first profiling step)
bash /root/evaluation/evaluate_step1_synthetic.sh

# Evaluate final Profile2Gen data (after full processing)
bash /root/evaluation/evaluate_final_synthetic.sh
```



