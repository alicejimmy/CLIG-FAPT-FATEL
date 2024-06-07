# Unreliable Partial Label Learning: Novel Dataset Generation Method and Solution Frameworks
Unreliable Partial Label (UPLL) can be used to address the problem of multiple annotators providing different answers, thus saving annotation costs.

The main contributions of this research are:
* [Two UPLL datasets](https://github.com/alicejimmy/CLIG-FAPT-FATEL/tree/main/dataset_collection) have been collected and made publicly available, which as far as we know, is the first study to collect data and make it publicly available.
* A new method for generating UPLL datasets, named Candidate Label Inference Generation (CLIG), is proposed, and experiments demonstrate that it aligns more closely with real-world labeling tendencies.
* Two new UPLL frameworks are proposed: 
    * Feature Alignment Pseudo-Target Learning (FAPT), which optimizes feature extraction through contrastive learning and uses supervised learning to make classification results close to pseudo-targets.
    * Feature Alignment Temporarily Expanded Label Set (FATEL), which optimizes feature extraction through contrastive learning, and trains models by temporarily adding items to the candidate label set using supervised learning.
* Extensive experiments on different image datasets demonstrate that FAPT and FATEL outperform state-of-the-art methods for UPLL in multiple experiments.

The detailed content is written in the master's thesis, and the complete source code is published in [this project](https://github.com/alicejimmy/CLIG-FAPT-FATEL).

## Environment
We run and develop on Ubuntu, you can execute the following command to install the required packages:
```python=
conda create --name FAPT_FATEL python=3.11.4 ipykernel=6.25.0
conda activate FAPT_FATEL
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install wheel==0.41.2
pip install pandas==2.2.1
pip install Pillow==9.3.0
pip install numpy==1.24.1
```
Alternatively, you can directly use the `CLIG-FAPT-FATEL/Dockerfile` to build your environment.

## Datasets
We use the following five image datasets:

| Abbreviation | Full Name | Path of Images | Path of Human-annotated Labels | Candidate Labels Set Generation Method |
| --------------- | -------------------------------------- | ---------- | --------- |-------- |
| CIFAR10 | CIFAR10 | In `CLIG-FAPT-FATEL\data` | `CLIG-FAPT-FATEL\dataset_collection\CIFAR-10_human.pt` | Generated using CLIG |
| CIFAR100_SM | Complete dataset of small mammals in CIFAR100 (2500 images) | In `CLIG-FAPT-FATEL\data` | None | Generated using CLIG |
| CIFAR100_T | Complete dataset of trees in CIFAR100 (2500 images) | In `CLIG-FAPT-FATEL\data` | None | Generated using CLIG |
| CIFAR100_SM_500 | Partial dataset of small mammals in CIFAR100 (500 images) | In `CLIG-FAPT-FATEL\data` | `CLIG-FAPT-FATEL\dataset_collection\small_mammals_result.csv` | Directly use the organized human-annotated labels |
| CIFAR100_T_500  | Partial dataset of trees in CIFAR100 (500 images) | In `CLIG-FAPT-FATEL\data` | `CLIG-FAPT-FATEL\dataset_collection\trees_result.csv` | Directly use the organized human-annotated labels |

* During training, `data` folder will be automatically created, and the CIFAR10 and CIFAR100 datasets will be downloaded to `CLIG-FAPT-FATEL/data`.

## File Description
* `CLIG-FAPT-FATEL/`
    * `Dockerfile`: Text file used to build the image
    * `main.py`: Main code
    * `made_labelset_model.py`: Code for training model $M$ in CLIG
    * `made_labelset_model`: Directory to store trained model $M$
    * `dataset_collection`: Human-annotated labels

## Usage of `CLIG-FAPT-FATEL/made_labelset_model.py`
* We provide trained models $M$ in `CLIG-FAPT-FATEL/made_labelset_model`.
* If needed, you can use `made_labelset_model.py` to train your own model $M$.
* Usage:
    ```python=
    # Training model M on CIFAR10 dataset
    python made_labelset_model.py --epochs 200 --dataset cifar10 --lr 0.01 --momentum 0.9 --wd 1e-3 --target_acc 97.86 --deviation 0.1
    ```
    ```python=
    # Training model M on CIFAR100_SM
    python made_labelset_model.py --epochs 200 --dataset cifar100_SM --lr 0.1 --momentum 0.95 --wd 1e-4 --target_acc 96.60 --deviation 2
    ```
    ```python=
    # Training model M on CIFAR100_T
    python made_labelset_model.py --epochs 200 --dataset cifar100_T --lr 0.1 --momentum 0.95 --wd 1e-4 --target_acc 94.19 --deviation 2
    ```
* Options:

| Name | Default | Description |
| -------- | -------- | -------- |
| `--epochs` | `200` | Number of total epochs |
| `--batch_size` | `64` | Batch size |
| `--dataset` | `cifar10` | Dataset name <br> Options: `cifar10`, `cifar100_SM`, `cifar100_T` |
| `--model` | `resnet18` | Training model name <br> Options: `resnet18` |
| `--lr` | `0.01` | Learning rate |
| `--momentum` | `0.9` | Momentum |
| `--wd` | `1e-3` | Weight decay |
| `--target_acc` | `97.86` | $=\delta_2 \times 100$ <br> Accuracy of early stopping  |
| `--deviation` | `0.1` | Acceptable deviation value for early stopping |

## Quick Start
### Start Running FAPT
* Usage:
    ```python=
    # Run FAPT on CIFAR10 dataset
    python main.py --epochs 800 --model resnet18 --dataset cifar10 --creation_method CLIG --framework FAPT --lr 0.01 --momentum 0.9 --wd 1e-3 --warm_up 10 --phi 0.9
    ```
    ```python=
    # Run FAPT on CIFAR100_SM dataset
    python main.py --epochs 500 --model resnet18 --dataset cifar100_SM --creation_method CLIG --framework FAPT --lr 0.005 --momentum 0.9 --wd 1e-3 --warm_up 10 --phi 0.9
    ```
    ```python=
    # Run FAPT on CIFAR100_T dataset
    python main.py --epochs 500 --model resnet18 --dataset cifar100_T --creation_method CLIG --framework FAPT --lr 0.005 --momentum 0.9 --wd 1e-3 --warm_up 10 --phi 0.9
    ```
    ```python=
    # Run FAPT on CIFAR100_SM_500 dataset
    python main.py --epochs 200 --model resnet18 --dataset cifar100_SM_500 --creation_method CLIG --framework FAPT --lr 0.005 --momentum 0.9 --wd 1e-3 --warm_up 10 --phi 0.9
    ```
    ```python=
    # Run FAPT on CIFAR100_T_500 dataset
    python main.py --epochs 200 --model resnet18 --dataset cifar100_T_500 --creation_method CLIG --framework FAPT --lr 0.005 --momentum 0.9 --wd 1e-3 --warm_up 10 --phi 0.9
    ```
### Start Running FATEL
* Usage:
    ```python=
    # Run FATEL on CIFAR10 dataset
    python main.py --epochs 800 --model resnet18 --dataset cifar10 --creation_method CLIG --framework FATEL --lr 0.01 --momentum 0.9 --wd 1e-3 --warm_up 10 --phi 0.9
    ```
    ```python=
    # Run FATEL on CIFAR100_SM dataset
    python main.py --epochs 500 --model resnet18 --dataset cifar100_SM --creation_method CLIG --framework FATEL --lr 0.005 --momentum 0.9 --wd 1e-3 --warm_up 10 --phi 0.9
    ```
    ```python=
    # Run FATEL on CIFAR100_T dataset
    python main.py --epochs 500 --model resnet18 --dataset cifar100_T --creation_method CLIG --framework FATEL --lr 0.005 --momentum 0.9 --wd 1e-3 --warm_up 10 --phi 0.9
    ```
    ```python=
    # Run FATEL on CIFAR100_SM_500 dataset
    python main.py --epochs 200 --model resnet18 --dataset cifar100_SM_500 --creation_method CLIG --framework FATEL --lr 0.005 --momentum 0.9 --wd 1e-3 --warm_up 10 --phi 0.9
    ```
    ```python=
    # Run FATEL on CIFAR100_T_500 dataset
    python main.py --epochs 200 --model resnet18 --dataset cifar100_T_500 --creation_method CLIG --framework FATEL --lr 0.005 --momentum 0.9 --wd 1e-3 --warm_up 10 --phi 0.9
    ```
### Options

| Name | Default | Description |
| -------- | -------- | -------- |
| `--epochs` | `200` | Number of total epochs |
| `--batch_size` | `64` | Batch size |
| `--dataset` | `cifar10` | Dataset name <br> Options: `cifar10`, `cifar100_SM`, `cifar100_T`, `cifar100_SM_500`, `cifar100_T_500` |
| `--model` | `resnet18` | Training model name <br> Options: `resnet18`, `wideresnet` |
| `--lr` | `0.01` | Learning rate |
| `--momentum` | `0.9` | Momentum |
| `--wd` | `1e-3` | Weight decay |
| `--framework` | `FAPT` | UPLL framework name <br> Options: `FAPT`, `FATEL` |
| `--creation_method` | `CLIG` | How to generate a dataset <br> Options: `CLIG`, `APLG` |
| `--partial_rate` | `0.1` | For APLG. The probability that each class (except the true label) will be added to the candidate labelset. |
| `--noisy_rate` | `0.3` | For APLG. The probability that the true label is not in the candidate labelset. |
| `--warm_up` | `10` | $=R$ <br> Number of warm-up epochs |
| `--phi` | `0.9` | $=\phi$ <br> For FAPT. Pseudo target update ratio |
| `--data_ratio` | `1.0` | For the experiment of reduce the amount of data. Control the amount of training data. |
