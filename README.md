## TextMEF: Text-guided Prompt Learning for Multi-exposure Image Fusion [IJCAI 2025]

[[paper]](https://www.ijcai.org/proceedings/2025/0175.pdf)

## :memo: TODO
- :white_check_mark: Release training and testing code.
- :white_check_mark: Release our training sets and test sets.
- :white_check_mark: Release pretrained checkpoints.

## :airplane: Environment

```shell
git clone https://github.com/hqj9994ever/SPRFusion.git
conda create -n SPRFusion python=3.8.18
conda activate SPRFusion
conda install cudatoolkit==11.8 -c nvidia
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
conda install packaging
pip install causal_conv1d==1.1.1 # or download causal_conv1d-1.1.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl from https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.1.1 to install manually.
pip install mamba_ssm==1.1.1 # or download mamba_ssm-1.1.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl from https://github.com/state-spaces/mamba/releases/tag/v1.1.1 to install manually.
pip install -r requirements.txt
```
Note: After installing the mamba library, replace the file content of `mamba_ssm/ops/selective_scan_interface.py` with that of `selective_scan_interface.py` from [Vim](https://github.com/hustvl/Vim).


#### 📂 Data Preparation

You can download our training set and test set on [Google Drive](https://drive.google.com/drive/folders/1VsckjV152tF5As6gq7G1u7LOmzuIKhE3?usp=drive_link).

You should put the data in the correct place in the following form.

```
TextMEF ROOT
└── Dataset
    ├── train_data
    │   └── SICE
    │       ├── trainA
    │       ├── trainB
    │       └── trainC
    └── test_data
        ├── 31nogt
        │   ├── over
        │   └── under
        ├── Mobile
        │   ├── over
        │   ├── under
        │   └── gt
        └── SICE
            ├── over
            ├── under
            └── gt
```


#### :cookie: Pretrained Checkpoints

Download the model weights on [Google Drive](https://drive.google.com/drive/folders/1VsckjV152tF5As6gq7G1u7LOmzuIKhE3?usp=drive_link) and put it in `model_zoo/`.

## :tennis: Train

### Prompt learning
```shell
python train_prompt.py \
--task_name train0 \
--lowlight_images_path 'path to ue image dir' \
--overlight_images_path 'path to oe image dir' \
--normallight_images_path 'path to gt image dir' \
--num_epochs 180 \
--prompt_lr 5e-5 \
--patch_size 224 \
--length_prompt 16 \
--load_pretrain False
```

### Train fusion network
```shell
python train.py \
--task_name train1 \
--lowlight_images_path 'path to ue image dir' \
--overlight_images_path 'path to oe image dir' \
--normallight_images_path 'path to gt image dir' \
--num_epochs 400 \
--train_lr 2e-4 \
--patch_size 224 \
--length_prompt 16 \
--load_pretrain True
```

## :gun: Evaluation

```shell
python test.py \
--input_u 'path to ue image dir' \
--input_o 'path to oe image dir' \
--gt 'path to GT dir (if GT exists)' \
--need_H (if GT exists) 
```

## :pushpin: Citation

If this work has been helpful to you, we would appreciate it if you could cite our paper! 

```BibTex
@inproceedings{liu2025textmef,
  title={TextMEF: text-guided prompt learning for multi-exposure image fusion},
  author={Liu, Jinyuan and Huang, Qianjun and Wu, Guanyao and Wang, Di and Jiang, Zhiying and Ma, Long and Liu, Risheng and Fan, Xin},
  booktitle={Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence},
  pages={1567--1575},
  year={2025}
}
```

## :email: Contact
  If you have any other questions about the code, please open an issue in this repository or email us at  `hqj9994ever@gmail.com`.
