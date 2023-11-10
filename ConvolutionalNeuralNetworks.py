""" CNN """
import os, torch, time
import numpy as np
from load_path  import *
from tqdm import tqdm
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from torch import FloatTensor, Tensor, optim, nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, auc, roc_curve, log_loss
from torch.nn import (
    Module,
    Linear,
    Tanh,
    Sigmoid,
    Sequential
)

sub_id_arr, state_arr, matrix_arr = [], [], [] 
for file in select_path_list(CONNECTION_MATRIX, 'npy'):
    sub_id = file[file.find('sub'):file.find('sub')+len('sub-00')]
    state = file[-(len('.npy')+1)]
    matrix = np.load(file)
    sub_id_arr.append(sub_id)
    state_arr.append(state)
    matrix_arr.append(matrix)
assert len(sub_id_arr) == len(state_arr) == len(matrix_arr)

# 划分正负样本 正样本-健康人群-21 负样本-抑郁症-51(0~50)
# 训练集+验证集 : 测试集 = 63 ： 9
# 训练集+验证集63 = 45正样本 + 18负样本
# 测试集9 = 6正样本 + 3负样本
train_valid_dict, test_dict = {}, {} # key(sub-id_state) : matrix
for index, _ in zip(range(len(sub_id_arr)), tqdm(range(len(sub_id_arr)))):
    if index <= 44 or (index >= 51 and index <= 68):
        train_valid_dict[sub_id_arr[index]+'_'+state_arr[index]] = matrix[index] 
    elif (index >= 45 and index <= 50) or index >= 69:
        test_dict[sub_id_arr[index]+'_'+state_arr[index]] = matrix[index] 
    else:
        print(f'{index} must be covered!')
        exit(1)

""" 超参数 """
batch_size = 1

""" 算力设备 """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
print(f'Device = {device}')
np.random.seed(0)

""" 制作datasets """
class GetData(Dataset):
    def __init__(self, user : list, features : np.array, targets : list) -> None:
        # features为特征矩阵 每行对应到一个短视频 每列对应到一种特征
        # targets为点击率目标01值
        self.user = user
        self.features = FloatTensor(features)
        self.targets = FloatTensor(targets)
    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        # 每个loader返回特征矩阵+点击目标
        return self.user[index], self.features[index], self.targets[index]
    def __len__(self) -> int:
        assert len(self.features) == len(self.targets) == len(self.user)
        return len(self.features)

""" 划分训练集 验证集 测试集 """
def get_train_value_dataloader(root_dir:str, label_tag:str):
    def make_xy(data : np.array): 
        x, y, user = [],[],[] # y向量为矩阵的最后一列 即点击的01值; x为特征矩阵; user用用户id
        for i, _ in zip(data, tqdm(range(len(data)))):
            y.append(int(i[-1]))
            tmp = [] 
            for v in i[1:-1]:
                tmp.append(float(v))
            x.append(tmp)
            user.append(int(i[0]))
        # 特征矩阵部分进行归一化
        start = time.time()
        x = MinMaxScaler().fit_transform(np.array(x))
        end = time.time()
        print(f'It took {end-start} seconds to normalize feature matrix.')
        return user, x, y
    
    def make_dataloader(user : list, x : np.array, y : list) -> DataLoader:
        dataset = GetData(user, x, y)
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    # 加载训练矩阵
    file_path = os.path.join(root_dir, label_tag+'_matrix.npy')
    matrix = []
    start = time.time()
    matrix = (np.load(file_path, allow_pickle=True))
    end = time.time()
    print(f'Reading {file_path} took {round((end-start), 3)} seconds...')
    
    # 训练集 : 测试集 : 验证集 = 8 : 1 : 1
    train_matrix = matrix[ : int(len(matrix)*0.8)]
    val_matrix = matrix[int(len(matrix)*0.8) : int(len(matrix)*0.9)]
    test_matrix = matrix[int(len(matrix)*0.9) : ]

    val_user, val_x, val_y = make_xy(val_matrix)
    test_user, test_x, test_y = make_xy(test_matrix)
    train_user, train_x, train_y = make_xy(train_matrix)
    
    # 返回 train_loader, val_loader, test_loader
    return make_dataloader(train_user,train_x,train_y),make_dataloader(val_user,val_x,val_y),make_dataloader(test_user,test_x,test_y)
# Dataloader

# CNN 卷积核的问题，我觉得功能连接更强的矩阵应该再空间上更靠近一些，做个消融
# CNN就不用side info了 GNN再用吧
# 可解释性：找到最有价值的哪几个脑区

# 性能评估