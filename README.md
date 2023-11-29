# Brainnetome4Depression
## 1. Environment
``` shell
conda create --name bn4depression python=3.9.1
source activate bn4depression

conda install -c conda-forge nibabel
conda install -c conda-forge nilearn
conda install matplotlib
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install tqdm
```

## 2. Run on Windows
Get functional connection: `python functional_connection.py`
Change your aggregation type in `run.py` and then `python run.py`

## 3. Run on BSCC Platform
```shell
chmod -x run.sh
dsub -s run.sh #提交作业
djob # 查看
djob -T 作业ID #取消作业
```