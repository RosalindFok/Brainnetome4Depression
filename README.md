# Brainnetome4Depression

``` shell
conda create --name bn4depression python=3.9.1
source activate bn4depression

conda install -c conda-forge nibabel
conda install -c conda-forge nilearn
conda install matplotlib
```

提交作业
```shell
chmod -x run.sh
dsub -s run.sh # 提交
djob # 查看
djob -T 作业ID #取消作业
```