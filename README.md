# Brainnetome4Depression
## 1. Environment and Data
#### Environment
``` shell
conda create --name bn4depression python=3.9.1
source activate bn4depression

conda install -c conda-forge nibabel
conda install -c conda-forge nilearn
conda install matplotlib
conda install yaml
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install tqdm
```

#### Data
adjust your own path in `load_path.py`
Dataset in OpenNeuro: [depression_ds002748](https://openneuro.org/datasets/ds002748/versions/1.0.5)
```shell
.
├─Brainnetome4Depression
│  └─BN_Atlas
├─connection_matrix
└─depression_ds002748
    ├─sub-01
    │  ├─anat
    │  └─func
    ├─ ... ...
```

## 2. Run on Windows
Get functional connection: `python functional_connection.py`
Change your hyperparameter and save_model_weights/save_result_txt in `config.yaml`
Change your aggregation type in `run.py` and then `python run.py`

## 3. Run on BSCC Platform
```shell
module load anaconda/2021.11 
module load cudnn/8.8.1_cuda11.x 
# create bn4depression via Conda
chmod -x run.sh
source activate bn4depression
dsub  -s run.sh #提交作业
djob # 查看作业id
djob  -T 作业ID #取消作业
```

## 4. Original Data
To save public store, they were zipped as `ori_data.zip`.
`result_lobe1/2/3.txt`: aggregation type is `lobe`
`result_gyrus1/2/3.txt`: aggregation type is `gyrus`
some of the results in: `draw.ipynb`
