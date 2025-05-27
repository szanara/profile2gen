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
