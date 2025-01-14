# ISSMP_DIS

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7%2B-green.svg)](https://www.python.org/downloads/)

The official implementation for ISSMP_DIS: A project focused on disentangling and analyzing sleep stage data from multimodal inputs. 

## Overview
ISSMP_DIS provides tools for preprocessing, training, and evaluating models on multimodal datasets including RRI, actigraphy, BMI, and Apple Watch data. 

## Features
- Preprocessing tools for aligning RRI and actigraphy data
- Ready-to-use processed datasets
- Configurable experimental setup
- Scripts for training and testing various models

![teaser](assets/example_teaser.gif)

## Getting Started

### 1. Data Downloading
Before running experiments, preprocess the data by following these steps:

1. **RRI and Actigraphy Data**
   - Download RRI and actigraphy data from the [Sleepdata.org](https://sleepdata.org/) website after obtaining access.
   - Save the data in the `RRI` and `Actigraph` folders.

2. **BMI Data**
   - Obtain BMI data from the [BioNCC website](https://bioncc.nih.gov/).

3. **Apple Watch Dataset**
   - Download the processed Apple Watch dataset from [AppleWatchDisentangle](https://drive.google.com/file/d/1xVeS-8ngJ1av5GN_s6eHpr-SQfeGL4ZV/view?usp=drive_link).

### 2. Data Processing
Align and clean the HRV and activity count features using the provided script:

```shell
python align_actigraphy_rri.py
```
Refer to [align_actigraphy_rri.py](https://github.com/bzhai/multimodal_sleep_stage_benchmark/blob/master/dataset_builder_loader/align_actigraphy_rri.py) for more details.

### 3. Experiment Setup
Configure experimental settings in the `sleep_stage_config.py` file. Detailed descriptions are provided as comments in the code.

## Training

### Example Training Commands

#### DSleepNet (Univariate & Multivariate PA)
```shell
CUDA_VISIBLE_DEVICES="0" python -m train_dis_normal_filter --seq_len 100 --feature_type all --nn_type GSNMSE3DRES --dis_type bmi5c --num_train 2002 --epochs 20 --batch_size 512 --debug 0 --train_test_group group51 --beta_d 10 --beta_y 10 --aux_loss_multiplier_y 100 --aux_loss_multiplier_d 100

CUDA_VISIBLE_DEVICES="0" python -m train_dis_normal_filter --seq_len 100 --feature_type all --nn_type GSNMSE3DRES --num_train 2002 --epochs 20 --batch_size 1024 --debug 0 --beta_y 10 --beta_d 10 --dis_type bmi5c ahi4pa5 sleepage5c --train_test_group group2 --aux_loss_multiplier_y 10 --aux_loss_multiplier_d 10
```

#### VAE with Ancillary Tasks
```shell
CUDA_VISIBLE_DEVICES="0" python -m train_dis_no_ie --seq_len 100 --feature_type full --nn_type GSN_NO_IE --dis_type ahi4pa5 --train_test_group group53 --num_train 2002 --epochs 20 --batch_size 1024 --debug 0 --beta_y 1e-3 --beta_d 1e-5 --aux_loss_multiplier_y 1 --aux_loss_multiplier_d 1e-5

CUDA_VISIBLE_DEVICES="0" python -m train_dis_no_ie --seq_len 100 --feature_type all --nn_type GSN_NO_IE --num_train 2002 --epochs 20 --batch_size 1024 --debug 0 --beta_y 1e-3 --beta_d 1e-5 --dis_type bmi5c ahi4pa5 sleepage5c --train_test_group group2 --aux_loss_multiplier_y 10 --aux_loss_multiplier_d 10
```

#### Baseline Model
```shell
CUDA_VISIBLE_DEVICES="0" python -m train_non_dis_baseline --seq_len 100 --nn_type qzd --feature_type all --dis_type ahi4pa5 --train_test_group group53 --num_train 2002 --epochs 20 --batch_size 1024 --debug 0

CUDA_VISIBLE_DEVICES="0" python -m train_non_dis_baseline --seq_len 100 --nn_type qzd --feature_type all --dis_type bmi5c ahi4pa5 sleepage5c --train_test_group group2 --num_train 2002 --epochs 20 --batch_size 1024 --debug 0
```

## Testing

### Apple Watch Dataset
Run the following command to test trained models on the Apple Watch dataset:

```shell
python test_dis_on_apple.py
```

## Results
The results will be saved as tables and plots. The `test_dis_on_apple.py` script generates outputs used in Table 2 of the paper.

## Acknowledgments
We extend our gratitude to the contributors of the following projects, which provided foundational support:

- [Sleepdata.org](https://sleepdata.org/)
- [BioNCC](https://bioncc.nih.gov/)
- [AppleWatchDisentangle](https://drive.google.com/file/d/1xVeS-8ngJ1av5GN_s6eHpr-SQfeGL4ZV/view?usp=drive_link)

## License
This project is distributed under the [MIT License](LICENSE).

For more details, please refer to the individual licenses of dependent libraries and datasets.
