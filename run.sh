#!/bin/bash
#DSUB -A root.bingxing2.gpuuser486
#DSUB -q root.default
#DSUB -l wuhanG5500
#DSUB --job_type cosched
#此参数用于指定资源。如申请 6核CPU，1卡GPU，48GB内存。
#DSUB -R 'cpu=6;gpu=1;mem=45000'
#DSUB -N 1
#DSUB -e %J.out
#DSUB -o %J.out
export export PYTHONUNBUFFERED=1
python run.py