# MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model


Author list 

links

## 🤩 Abstract

> xxx

## 📢 News

- **[2024/04/08]** upload paper and release code.

## 👨‍🏫 Quick Start

This section provides a quick start guide to set up the environment and run the demo. The following steps will guide you through the installation of the required dependencies, downloading the pretrained models, and preparing the datasets. 

<details>
  <summary><b> 1. Conda environment </b></summary>

```
conda create python=3.10 --name motionlcm
conda activate motionlcm
```

Install the packages in `requirements.txt` and install [PyTorch 1.13.1](https://pytorch.org/).

```
pip install -r requirements.txt
```

We test our code on Python 3.10.12 and PyTorch 1.13.1.

</details>

<details>
  <summary><b> 2. Dependencies </b></summary>

Run the script to download dependencies materials:

```
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
bash prepare/perpare_t5.sh
```

</details>

<details>
  <summary><b> 3. Pretrained models </b></summary>

Run the script to download the pretrained models:

```
bash prepare/download_pretrained_models.sh
```

The folders `experiments_t2m` and `experiments_control` store pretrained models for text-to-motion and motion control respectively.

</details>


<details>
  <summary><b> 4. (Optional) Download manually </b></summary>

Visit the [Google Driver](https://drive.google.com/drive/folders/1SIhb6srXWS0PNvZ2fs40QiE3Rk764u6z?usp=sharing) to download the previous dependencies and models.

</details>

<details>
  <summary><b> 5. Prepare the datasets </b></summary>

Please refer to [HumanML3D](https://github.com/EricGuo5513/HumanML3D) for text-to-motion dataset setup. Copy the result dataset to our repository:
```
cp -r ../HumanML3D/HumanML3D ./datasets/humanml3d
```

</details>

<details>
  <summary><b> 6. Folder Structure </b></summary>

After the whole setup pipeline, the folder structure will look like:

```
MotionLCM
├── configs
├── datasets
│   ├── humanml3d
│   │   ├── new_joint_vecs
│   │   ├── new_joints
│   │   ├── texts
│   │   ├── Mean.npy
│   │   ├── Std.npy
│   │   ├── ...
│   └── humanml_spatial_norm
│       ├── Mean_raw.npy
│       └── Std_raw.npy
├── deps
│   ├── glove
│   ├── sentence-t5-large
│   └── t2m
├── experiments_control
│   └── motionlcm_humanml
│       └── motionlcm_humanml.ckpt
├── experiments_t2m
│   ├── mld_humanml
│   │   └── mld_humanml.ckpt
│   └── motionlcm_humanml
│       └── motionlcm_humanml.ckpt
├── ...
```

</details>

## 🎬 Demo

MotionLCM provides two main functionalities: text-to-motion and motion control. The following commands demonstrate how to use the pretrained models to generate motions. The outputs will be stored in `${cfg.TEST_FOLDER} / ${cfg.NAME} / demo_${timestamp}` (`experiments_t2m_test/motionlcm_humanml/demo_2024-04-06T23-05-07`).

<details>
  <summary><b> 1. Text-to-Motion (using provided prompts and lengths in `demo/example.txt`) </b></summary>

```
python demo.py --cfg configs/motionlcm_t2m.yaml --example assets/example.txt
```
</details>

<details>
  <summary><b> 2. Text-to-Motion (using prompts from HumanML3D test set) </b></summary>

```
python demo.py --cfg configs/motionlcm_t2m.yaml
```
</details>

<details>
  <summary><b> 3. Motion Control (using prompts and trajectory from HumanML3D test set) </b></summary>

```
python demo.py --cfg configs/motionlcm_control.yaml
```

</details>

<details>
  <summary><b> 4. Render SMPL </b></summary>

After running the demo, the output folder will store the stick figure animation for each generated motion (e.g., `assets/example.gif`).

![example](assets/example.gif)

To record the necessary information about the generated motion, a pickle file with the following keys will be saved simultaneously (e.g., `assets/example.pkl`):

- `joints`: The XYZ positions of the generated motion (numpy array) with the shape of `(num_frames, njoints, 3)`.
- `text`: The text prompt.
- `length`: The length of the generated motion.
- `hint`: The trajectory for motion control (optional).

#### 4.1 Create SMPL meshes

To create SMPL meshes for a specific pickle file, let's use `assets/example.pkl` as an example:

```
python fit.py --pkl assets/example.pkl
```

The SMPL meshes (numpy array) will be stored in `assets/example_mesh.pkl` with the shape `(num_frames, 6890, 3)`.

You can also fit all pickle files within a folder. The code will traverse all `.pkl` files in the directory and filter out files that have already been fitted.

```
python fit.py --dir assets/
```

#### 4.2 Render SMPL meshes

Refer to [TEMOS-Rendering motions](https://github.com/Mathux/TEMOS) for blender setup (only **Installation** section). 

We support three rendering modes for SMPL mesh, namely `sequence` (default), `video` and `frame`.


<details>
  <summary><b> 4.2.1 sequence </b></summary>

```
YOUR_BLENDER_PATH/blender --background --python render.py -- --pkl assets/example_mesh.pkl --mode sequence --num 8
```

You will get a rendered image of `num=8` keyframes as shown in `assets/example_mesh.png`. The darker the color, the later the time.

<img src="assets/example_mesh_show.png" alt="example" width="30%">

</details>

<details>
  <summary><b> 4.2.2 video </b></summary>

```
YOUR_BLENDER_PATH/blender --background --python render.py -- --pkl assets/example_mesh.pkl --mode video --fps 20
```

You will get a rendered video with `fps=20` as shown in `assets/example_mesh.mp4`.

![example](assets/example_mesh_show.gif)

</details>

<details>
  <summary><b> 4.2.3 frame </b></summary>

```
YOUR_BLENDER_PATH/blender --background --python render.py -- --pkl assets/example_mesh.pkl --mode frame --exact_frame 0.5
```

You will get a rendered image of the keyframe at `exact_frame=0.5` (i.e., the middle frame) as shown in `assets/example_mesh_0.5.png`.

<img src="assets/example_mesh_0.5_show.png" alt="example" width="25%">

</details>

</details>

## 🚀 Train your own models

We provide the training guidance for both text-to-motion and motion control tasks. The following steps will guide you through the training process.

<details>
  <summary><b> 1. Important args in the config yaml </b></summary>

The parameters required for model training and testing are recorded in the corresponding YAML file (e.g., `configs/motionlcm_t2m.yaml`). Below are some of the important parameters in the file:

- `${FOLDER}`: The folder for the specific training task (i.e., `experiments_t2m` and `experiments_control`).
- `${TEST_FOLDER}`: The folder for the specific testing task (i.e., `experiments_t2m_test` and `experiments_control_test`).
- `${NAME}`: The name of the model (e.g., `motionlcm_humanml`). `${FOLDER}`, `${NAME}`, and the current timestamp constitute the training output folder (for example, `experiments_t2m/motionlcm_humanml/2024-04-06T23-05-07`). The same applies to `${TEST_FOLDER}` for testing.
- `${TRAIN.PRETRAINED}`: The path of the pretrained model.
- `${TEST.CHECKPOINTS}`: The path of the testing model.

</details>

<details>
  <summary><b> 2. Train MotionLCM and ControlNet </b></summary>

#### 2.1. Ready to train MotionLCM model

Please first check the parameters in `configs/motionlcm_t2m.yaml`. Then, run the following command:

```
python -m train_motionlcm --cfg configs/motionlcm_t2m.yaml
```

#### 2.2. Ready to train motion ControlNet

Please update the parameters in `configs/motionlcm_control.yaml`. Then, run the following command:

```
python -m train_motion_control --cfg configs/motionlcm_control.yaml
```

</details>

<details>
  <summary><b> 3. Evaluate the model </b></summary>

Text-to-Motion: 

```
python -m test --cfg configs/motionlcm_t2m.yaml
```

Motion Control:

```
python -m test --cfg configs/motionlcm_control.yaml
```

</details>

## 📊 Results

We provide the quantitative and qualitative results in the paper. 

## 🌹 Acknowledgement

We would like to thank the authors of the following repositories for their excellent work: 


## 📜 Citation

If you find this work useful, please consider citing our paper:

```bash
@inproceedings{motionlcm,
  title={MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model},
  author={xxx},
  booktitle={xxx},
  year={2024}
}
```