# -*- coding: UTF-8 -*-
import os
from load_path import aggregation_lobe, aggregation_gyrus, aggregation_not

# Change your aggregation type here
aggregation_type = aggregation_gyrus

max_sector = 7 if aggregation_type==aggregation_lobe else 24 if aggregation_type==aggregation_gyrus else 246 if aggregation_type==aggregation_not else -1
for counterfactual_sector in range(-1, max_sector): # 24个脑回
    parameters_string = f' --counterfactual_sector {counterfactual_sector} --aggregation_type {aggregation_type}'
    exit_code = os.system(f'python GraphNeuralNetworks.py {parameters_string}')
    if not exit_code == 0: # exit_code is not 0 <--> sth wrong with the training
        print(f'Error: running {parameters_string}, please check')
        break