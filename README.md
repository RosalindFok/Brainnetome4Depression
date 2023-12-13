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
Get functional connection: `python functional_connection.py`<br>
Change your hyperparameter and save_model_weights/save_result_txt in `config.yaml`<br>
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
`result_lobe1/2/3.txt`: aggregation type is `lobe`.<br>
`result_gyrus1/2/3.txt`: aggregation type is `gyrus`.<br>
To save public store, all txt files were zipped as `ori_data.zip`.<br>
`draw.ipynb`: AUC and LogLoss with lobe/gyrus or different epochs/learning_rate.

## 5. Methods
### 5.1. Brainnetome Atlas
![viewer](.\figs\Atlas_1.svg)
![correlation matrix](.\figs\Atlas_2.svg)
## 5.2. preprocessing_pipeline
![pipeline of preprocessing](.\figs\preprocessing_pipeline.svg)

## 6. Results
### 6.1. Aggregation via lobe or gyrus
![lobe](.\figs\lobe_auc_logloss.svg)
![gyrus](.\figs\gyrus_auc_logloss.svg)
### 6.2. Different epochs or learning rate
![learning rate](.\figs\diff_learningrate.svg)
![epochs](.\figs\diff_epochs.svg)
