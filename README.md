# profile2gen
 This repository corresponds to the implementation for the research work titled "Breaking the Barrier of Hard Samples: A Data-Centric Approach to Synthetic Data for Medical Tasks" published at ICML 2025.

 
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











**Project Structure and Execution Guide** 
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



----> **Execution Order**
--> **To Generate Synthetic Data**



