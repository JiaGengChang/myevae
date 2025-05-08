# MyeVAE

[![PyPI version](https://badge.fury.io/py/myevae.svg)](https://badge.fury.io/py/myevae)
[![GitHub latest commit](https://badgen.net/github/last-commit/JiaGengChang/myevae)](https://GitHub.com/JiaGengChang/myevae/commit/)
[![GitHub license](https://img.shields.io/github/license/JiaGengChang/myevae.svg)](https://github.com/JiaGengChang/myevae/blob/master/LICENSE)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

MyeVAE is a variational autoencoder leveraging multi-omics for risk prediction in newly diagnosed multiple myeloma patients.

This repository contains the Python source code to preprocess multimoodal features, train MyeVAE, score its performance, and perform SHAP analysis. 

All computation is done on the CPU, using the PyTorch library.

<p align="left"><img src="https://raw.githubusercontent.com/JiaGengChang/myevae/refs/heads/main/assets/myeVAE.png" alt="Illustration of MyeVAE architecture, hosted on github" width="600"></p>

# Setup

## Install MyeVAE
Requires **Python 3.9** or later

Install through PyPI 
```bash
pip install myevae
```

Install through github
```bash
pip install git+https://github.com/JiaGengChang/myevae.git
```

## Download data

Download the raw multi-omics dataset and example outputs from the following link:

1. Input datasets are stored at https://myevae.s3.us-east-1.amazonaws.com/example_inputs.tar.gz

2. Example output files are stored at https://myevae.s3.us-east-1.amazonaws.com/example_outputs.tar.gz

Please use the download links sparingly.

If you use the AWS S3 CLI, the bucket ARN is `arn:aws:s3:::myevae` - you can list its contents and download specific files.

# Step-by-step guide
## 1. (Optional) Feature preprocessing

As there are over 50,000 raw features and contains missing values, this step performs supervised-feature selection, scaling, and imputation.

```bash
myevae preprocess \
    -e [endpoint: os | pfs] \
    -i [/inputs/folder]
```

The following output files will be created in the same directory as the inputs.
```bash
inputs
├──train_features_os_processed.csv
├──train_features_pfs_processed.csv
├──valid_features_os_processed.csv
└──valid_features_pfs_processed.csv
```

Preprocessing of features is resource-intensive. Alternatively, proceed to step 3 using the downloaded processed features.


## 2. Fit model

Place the hyperparameter grid (`param_grid.py`) in `/params/folder`.

```bash
myevae train \
    -e [endpoint: os | pfs] \
    -n [model_name] \
    -i [/inputs/folder] \
    -p [/params/folder] \
    -t [nthreads] \
    -o [/output/folder]
```

## 3. Score model

```bash
myevae score \
    -e [endpoint: os | pfs] \
    -n [model_name] \
    -i [/inputs/folder] \
    -o [/output/folder]
```

## 4. Generate SHAP plots
```bash
myevae shap \
    -e [endpoint: os | pfs] \
    -n [model_name] \
    -i [/inputs/folder] \
    -o [/output/folder]
```

# Requirements

## Files

<p align="left"><img src="https://raw.githubusercontent.com/JiaGengChang/myevae/refs/heads/main/assets/directory.png" alt="Illustration of folder structure, hosted on github" width="300"></p>


### Input csv files
Place the following `.csv` files inside your inputs folder (`/inputs/folder`), and the `param_grid.py` inside your preferred folder. Also create an empty folder for the outputs.
```bash
├──inputs
│     ├──train_features.csv [ feature matrix of training samples ]
│     ├──valid_features.csv [ feature matrix of validation samples ]
│     ├──train_labels.csv [ survival files of training samples ]
│     └──valid_labels.csv [ survival files of validation samples ]
├──params
│     └──param_grid.py
└──outputs
      └──[output files created here]
```

For `features*.csv` and `labels*.csv`, column 0 is read in as the index, which should be the patient IDs.

Example input csv files can be downloaded from AWS S3 link above.

### Hyperparameter grid python file
Place a python file named `param_grid.py` containing the hyperparameter grid dictionary in the params folder (`/params`). 

This contains the set of hyperparameters that grid search will be performed on. 

Otherwise, use the default provided in `src/param_grid.py`.

The default hyperparameter grid is only meant for testing purposes:
```python
from torch.nn import LeakyReLU, Tanh

# the default hyperparameter grid for debugging uses
# this is not meant to be used for real training, as the search space is only on z_dim
param_grid = {
    'z_dim': [8,16,32],
    'lr': [5e-4], 
    'batch_size': [1024],
    'input_types': [['exp','cna','gistic','fish','sbs','ig']],
    'input_types_subtask': [['clin']],
    'layer_dims': [[[32, 4],[16,4],[4,1],[4,1],[4,1],[4,1]]],
    'layer_dims_subtask' : [[4,1]],
    'kl_weight': [1],
    'activation': [LeakyReLU()],
    'subtask_activation': [Tanh()],
    'epochs': [100],
    'burn_in': [20],
    'patience': [5],
    'dropout': [0.3],
    'dropout_subtask': [0.3]
}
```

An actual paramater grid is available at https://myevae.s3.us-east-1.amazonaws.com/param_grid.py

## Software
These dependencies will be automatically installed.
```
python >= 3.9
torch >= 1.9.0
scikit-learn >= 0.24.1
scikit-survival >= 0.23.1
importlib_resources
matplotlib
shap-0.47.3.dev8-offline-fork-for-myevae==0.0.1
```

The last dependency is a offline fork of shap (https://pypi.org/project/shap/) which has been modified to work MyeVAE.


## Recommended hardware

1. Minimum 4 CPU cores (8 cores is recommended)

2. Minimum 16 GB RAM (64GB is required for feature preprocessing)

No GPU is required.


# Citation

If you use MyeVAE in your research, please consider citing:

Jia Geng Chang, Jianbin Chen, Guo-Liang Chew, Wee Joo Chng. *MyeVAE: a multi-modal variational autoencoder for risk profiling of newly diagnosed multiple myeloma*. 2 May 2025. Manuscript under review.
