# Contents

- [Contents](#contents)
- [SRC Description](#SRC-description)
- [Dataset](#dataset)
- [Simulator](#simulator)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Preparing ASSISTments Problem Logs](#preparing-assistments-problem-logs)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [SRC Description](#contents)

we propose a novel framework named Set-to-Sequence Ranking-based Concept-aware Learning Path Recommendation (SRC), which formulates the recommendation task under a set-to-sequence paradigm. Specifically, we first design a concept-aware encoder module which can capture the correlations among the input learning concepts. 
The outputs are then fed into a decoder module that sequentially generates a path through an attention mechanism that handles correlations between the learning and target concepts. Our recommendation policy is optimized by policy gradient. 
In addition, we also introduce an auxiliary module based on knowledge tracing to enhance the model’s stability by evaluating students’ learning effects on learning concepts

Set-to-Sequence Ranking-based Concept-aware Learning Path Recommendation

AAAI2023

# [Dataset](#contents)

- [assist09](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data)
- [junyi](https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy)

# [Simulator](#contents)

- [DKT](https://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf)
- [CoKT](https://dl.acm.org/doi/10.1145/3488560.3498374)

# [Environment Requirements](#contents)

- Hardware（CPU and GPU）
    - Prepare hardware environment with CPU processor and GPU of Nvidia.
- Framework
    - [PyTorch](https://pytorch.org/get-started/locally/)
- Requirements
  - numpy
  - tqdm
  - longling
  - torch
  - scikit-learn
  - elasticsearch
  
- For more information, please check the resources below：
  - [PyTorch Tutorials](https://pytorch.org/tutorials/)
  - [PyTorch Python API](https://pytorch.org/docs/stable/index.html)

# [Quick Start](#contents)

After installing the dependencies, you can start training and evaluation as follows:

- running on GPU

  ```shell
  # Build Simulator first
  python trainDKT.py -d dataset -m model
  # Train the SRC agent with the simulator
  python trainSRC.py -d dataset -p path_type
  ```

The checkpoints written by `trainDKT.py` land in the `SavedModels/` folder
using the pattern `<model>_<dataset>.pt`, because the trainer builds
`exp_name` from the selected model and dataset before appending the `.pt`
suffix.【F:SRC-mamba/KTScripts/options.py†L48-L93】 Therefore, switching to
CoKT produces `CoKT_<dataset>.pt` (for example, `CoKT_assist17.pt`). If a
run still leaves a file such as `DKT_assist17.pt`, it means the simulator was
started with the default `-m DKT`, so double-check the CLI flags or any code
overrides that set `args.model`.
# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text 
.
└─SRC
  ├─README.md             # descriptions of SRC
  ├─KTScripts             # KT module for simulator, SRC, MPC, etc.
    ├─__init__.py
    ├─BackModels.py       # Some basic model
    ├─DataLoader.py       # Dataset file
    ├─options.py          # Parameter option
    ├─PredictModel.py     # The framework of the prediction model
    ├─utils.py            # Some utility functions
  ├─Scripts               # Various simulators, baseline models, SRC.
    ├─__init__.py
    ├─Agent               
      ├─__init__.py
      ├─MPC.py            # MPC model
      ├─SRC.py            # SRC model
      ├─utils.py          # Some utility functions
    ├─Envs               
      ├─__init__.py
      ├─KES
        ├─Env.py          # Simulator based on DKT
        ├─EnvCo.py        # Simulator based on CoKT
        ├─utils.py        # Some utility functions
    ├─Optimizer.py        # Definition of ModelWithLoss and ModelWithOptimizer
    ├─options.py          # Parameter option
    ├─utils.py            # Some utility functions
  ├─generate_data.py      # generate the datasets required for CoKT
  ├─testRule.py           # The performance of three baselines -- random, rule-based, and GRU4Rec -- was tested.
  ├─trainDKT.py           # Train simulator, GRU4Rec 
  ├─trainMPC.py           # Train and validate MPC
  ├─trainSRC.py           # Train and validate SRC
```
## [Script Parameters](#contents)

- Parameters of trainDKT.py

See [KTScripts/options.py](./KTScripts/options.py) for detailed parameters and explanations

- Parameters of trainSRC.py and trainMPC.py

See [Scripts/options.py](./Scripts/options.py) for detailed parameters and explanations

## [Preparing ASSISTments Problem Logs](#contents)

To convert the ASSISTments 2012 problem-log export into the `.npz` + concept
mapping pair used by SRC, place the raw CSV anywhere on disk and run:

```bash
python Scripts/prepare_assist2012.py \
  --input /path/to/assistments_problem_log.csv \
  --output-dir data/assist2012
```

The `--input` flag must point to the CSV that contains the columns such as
`problem_log_id`, `skill`, `assistment_id`, `answer_text`, and the confidence
scores shown in the example. The script writes `assist2012.npz` and
`knowledge_concept_mapping_assist2012.csv` into `--output-dir` (defaulting to
the directory where the CSV lives).

To process the tab-separated ASSISTments 2017 (ITEST) export, run:

```bash
python Scripts/prepare_assist2017.py \
  --input /path/to/assistments_itest.tsv \
  --output-dir data/assist2017
```

The script automatically fabricates the columns that are missing from the
compact 2017 schema (e.g. numeric skill identifiers, attempt counters, hint
counts) and produces `assist2017.npz` alongside
`knowledge_concept_mapping_assist2017.csv` in the chosen output directory.

After preparing one or more datasets you can generate a Markdown table with the
headline statistics (learners, exercises, concepts, interaction counts, and
average sequence lengths) using:

```bash
python Scripts/dataset_statistics.py \
  --dataset ASSIST09=data/assist09 \
  --dataset ASSIST12=/path/to/assist2012 \
  --dataset "SLP-Math=/data/slp"
```

When given a directory the tool searches for `<name>.npz` inside it. Specify
`--format csv` or `--format tsv` to export delimited text instead of Markdown.


# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters          | Value                                                                |
|---------------------|----------------------------------------------------------------------|
| Resource            | AMD Ryzen 2990WX 32-Core Processor; 256G Memory; NVIDIA GeForce 2080Ti |
| Uploaded Date       | 01/15/2023 (month/day/year)                                          |
| Framework           | PyTorch                                                              |
| Dataset             | assist09, junyi                                                      |
| Simulator           | DKT, CoKT                                                            |
| Training Parameters | epoch=10, batch_size=256, lr=1e-3                                    |
| Optimizer           | Adam                                                                 |
| Loss Function       | Policy Gradient                                                      |
| Outputs             | Reward                                                               |
| Results             | Based on simulator training, it has strong randomness. The comparison of different baselines can be referred to the paper. |
| Per Step Time       | 54.97 ms                                                             |

### Inference Performance

| Parameters      | Value                                                                  |
|-----------------|------------------------------------------------------------------------|
| Resource        | AMD Ryzen 2990WX 32-Core Processor; 256G Memory; NVIDIA GeForce 2080Ti |
| Uploaded Date   | 01/15/2023 (month/day/year)                                            |
| Framework       | PyTorch                                                                |
| Dataset         | assist09, junyi                                                        |
| Simulator       | DKT, CoKT                                                              |
| Outputs         | Reward                                                                 |
| Results         | Based on simulator training, it has strong randomness. The comparison of different baselines can be referred to the paper. |
| Per Step Time   | 40.61 ms                                                               |

# [Description of Random Situation](#contents)

- Simulator Training.
- Random initialization of model weights.

# [ModelZoo Homepage](#contents)

 Please check the official project homepage for more information.