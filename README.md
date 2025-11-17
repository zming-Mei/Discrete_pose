## Requirements
- Ubuntu 22.04
- Python 3.9
- CUDA 11.8
- NVIDIA RTX 3090

## Installation

- ### Install pytorch
Create a new conda environment and activate the environment.
```bash
conda create -n DFMArt python=3.9
conda activate DFMArt
```

``` bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```
- ### Install from requirements.txt
``` bash
pip install -r requirements.txt 
```
- ### Install flow_matching
```
pip install flow_matching
```
- ### Install pytorch3d from a local clone
``` bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
pip install --no-build-isolation -e . --config-settings editable_mode=compat
```

- ### Compile pointnet2
``` bash
cd networks/pts_encoder/pointnet2_utils/pointnet2
pip install -e .
```

## Training
Set the parameter '--data_path' in scripts/train.sh 

- ### Training network

``` bash
bash scripts/train.sh
```
- ### Eval network
``` bash
bash scripts/eval.sh
```
