# MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model

## 🚩 News

- [2024/04/08] upload paper and release code.

## 👨‍🏫 Quick Start

<details>
  <summary><b>Setup and download</b></summary>
  
### 1. Conda environment

```
conda create python=3.10 --name motionlcm
conda activate motionlcm
```

Install the packages in `requirements.txt` and install [PyTorch 1.13.1](https://pytorch.org/).

```
pip install -r requirements.txt
```

We test our code on Python 3.10.12 and PyTorch 1.13.1.

### 2. Dependencies

Run the script to download dependencies materials:

```
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
bash prepare/perpare_t5.sh
```

### 3. Pretrained models

Run the script to download the pretrained models:

```
bash prepare/download_pretrained_models.sh
```

The folders `experiments_t2m` and `experiments_control` store pretrained models for text-to-motion and motion control respectively.

### 4. (Optional) Download manually

Visit the [Google Driver](https://drive.google.com/drive/folders/1SIhb6srXWS0PNvZ2fs40QiE3Rk764u6z?usp=sharing) to download the previous dependencies and models.

</details>

## ▶️ Demo

<details>
  <summary><b>Text-to-motion & Motion Control</b></summary>

Text-to-Motion (using provided prompts and lengths in `demo/example.txt`): 
```
python demo.py --cfg configs/motionlcm_t2m.yaml --example demo/example.txt --plot
```
Text-to-Motion (using prompts from HumanML3D test set): 
```
python demo.py --cfg configs/motionlcm_t2m.yaml --plot
```

Motion Control (using prompts and trajectory from HumanML3D test set): 
```
python demo.py --cfg configs/motionlcm_control --plot
```

The outputs will be stored in `${cfg.TEST_FOLDER} / ${cfg.NAME} / demo_${timestamp}` (`experiments_t2m_test/motionlcm_humanml/demo_2024-04-06T23-05-07`).

</details>

## 🚀 Train your own models

<details>
  <summary><b>Training guidance</b></summary>

### 1. Prepare the datasets

Please refer to [HumanML3D](https://github.com/EricGuo5513/HumanML3D) for text-to-motion dataset setup. Copy the result dataset to our repository:
```
cp -r ../HumanML3D/HumanML3D ./datasets/humanml3d
```

### 2. Important args in the config yaml

The parameters required for model training and testing are recorded in the corresponding YAML file (e.g., `configs/motionlcm_t2m.yaml`). Below are some of the important parameters in the file:

- `${FOLDER}`: The folder for the specific training task (i.e., `experiments_t2m` and `experiments_control`).
- `${TEST_FOLDER}`: The folder for the specific testing task (i.e., `experiments_t2m_test` and `experiments_control_test`).
- `${NAME}`: The name of the model (e.g., `motionlcm_humanml`). `${FOLDER}`, `${NAME}`, and the current timestamp constitute the training output folder (for example, `experiments_t2m/motionlcm_humanml/2024-04-06T23-05-07`). The same applies to `${TEST_FOLDER}` for testing.
- `${TRAIN.PRETRAINED}`: The path of the pretrained model.
- `${TEST.CHECKPOINTS}`: The path of the testing model.

### 3.1. Ready to train MotionLCM model

Please first check the parameters in `configs/motionlcm_t2m.yaml`. Then, run the following command:

```
python -m train_motionlcm --cfg configs/motionlcm_t2m.yaml
```

### 3.2. Ready to train motion ControlNet

Please update the parameters in `configs/motionlcm_control.yaml`. Then, run the following command:

```
python -m train_motion_control --cfg configs/motionlcm_control.yaml
```

### 4. Evaluate the model

Text-to-Motion: 

```
python -m test --cfg configs/motionlcm_t2m.yaml
```

Motion Control:

```
python -m test --cfg configs/motionlcm_control.yaml
```

</details>
