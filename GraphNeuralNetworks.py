# -*- coding: UTF-8 -*-
""" GNN """
import os, torch, time, json, random, copy, argparse
import numpy as np
from load_path  import *
from tqdm import tqdm
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from torch import FloatTensor, Tensor, optim, nn
#在分类、聚类算法中，需要使用距离来度量相似性的时候、或者使用PCA技术进行降维的时候，StandardScaler表现更好（避免不同量纲对方差、协方差计算的影响）；
#在不涉及距离度量、协方差、数据不符合正态分布、异常值较少的时候，可使用MinMaxScaler。（eg：图像处理中，将RGB图像转换为灰度图像后将其值限定在 [0, 255] 的范围）；
#在带有的离群值较多的数据时，推荐使用RobustScaler。
from sklearn.preprocessing import MinMaxScaler,RobustScaler, StandardScaler
from sklearn.metrics import roc_auc_score, auc, roc_curve, log_loss
from torch.nn import (
    Module,
    Linear,
    Tanh,
    Sigmoid,
    Sequential
)
from load_path import *
np.random.seed(0)

""" 参数解析 """
parser = argparse.ArgumentParser(description='parameter')
parser.add_argument('--counterfactual_sector', type=int)
parser.add_argument('--aggregation_type', type=str)
args = parser.parse_args()
counterfactual_sector = args.counterfactual_sector
aggregation_type = args.aggregation_type

""" 加载脑图谱分区信息 """
if not os.path.exists(BNA_SUBREGION_PATH):
    print(f'Pleas check {BNA_SUBREGION_PATH}, make sure it is there.')
    exit(1)
with open(BNA_SUBREGION_PATH, 'r') as file:
    # {lobe : {gyrus : "name labelID_start labelID_end", ...}, ...}
    subregion_info = json.load(file)
lobe_index =  {} # {name : "startIdx endIdx"}. Idx = labelID - 1
gyrus_index = {} # {name : "startIdx endIdx"}. Idx = labelID - 1
lobe_full_name, gyrus_full_name = [],[]
for lobe in subregion_info:
    lobe_full_name.append(lobe)
    l_startIdx_list, l_endIdx_list = [], []
    for gyrus in subregion_info[lobe]:
        gyrus_full_name.append(subregion_info[lobe][gyrus].split(',')[0])
        startIdx = int(subregion_info[lobe][gyrus].split(',')[-2]) - 1
        endIdx   = int(subregion_info[lobe][gyrus].split(',')[-1]) - 1
        gyrus_index[gyrus] = [startIdx, endIdx]
        l_startIdx_list.append(startIdx)
        l_endIdx_list.append(endIdx)      
    l_startIdx, l_endIdx = min(l_startIdx_list), max(l_endIdx_list)
    lobe_index[lobe] = [l_startIdx, l_endIdx]


""" 构建每个脑区的embedding """
# 聚合
def aggregation_embeddings(embeddings : list[np.array], start : int, end : int)->np.array:
    array = embeddings[start : end+1]
    Euclidean_distance = np.zeros([end-start+1, end-start+1], dtype=float)
    for i in range(len(array)):
        for j in range(i+1, len(array)):
            Euclidean_distance[i][j] = Euclidean_distance[j][i] = np.linalg.norm(array[i]-array[j])
    assert np.diag(Euclidean_distance).all() == 0
    weights = [sum(row) for row in Euclidean_distance]
    sum_weights = sum(weights)

    new_weights = [x if sum_weights == 0 else x/sum_weights for x in weights] 
    assert len(new_weights) == len(array)
    result = np.zeros(array[0].shape, dtype=float)
    for weight, embedd in zip(new_weights, array):
        result += weight*embedd
    return result
# 72 participants
all_data_pair = {} # key(sub-id) : value(matirx 246×490, label∈{0,1})
for file in select_path_list(CONNECTION_MATRIX, '.npy'):
    # 1-抑郁症 0-健康人群
    name = file.split(os.sep)[2].split('_')[0]
    label = int(file[-len('.npy')-1])
    # 246×246 matrix for each participants
    connected_matrix = np.load(file) # 246×246 取值(-1,1]

    # /*** 反事实研究 ***/
    if aggregation_type == aggregation_lobe:
        # 对7个脑叶
        mask_list = [lobe_index[lobe] for lobe in lobe_index]
        assert counterfactual_sector < len(mask_list)
        if not counterfactual_sector < 0:
            sector_name = lobe_full_name[counterfactual_sector]
            for i in range(mask_list[counterfactual_sector][0], mask_list[counterfactual_sector][1]+1):
                connected_matrix[i, :] = -1.1#np.random.normal(0, 1, size=connected_matrix.shape[1])
                connected_matrix[:, i] = -1.1#np.random.normal(0, 1, size=connected_matrix.shape[0])
        else:
            sector_name = 'No Counterfactual'
    elif aggregation_type == aggregation_gyrus:
        # 对24个脑回
        mask_list = [gyrus_index[gyrus] for gyrus in gyrus_index]
        assert counterfactual_sector < len(mask_list)
        if not counterfactual_sector < 0:
            sector_name = gyrus_full_name[counterfactual_sector]
            for i in range(mask_list[counterfactual_sector][0], mask_list[counterfactual_sector][1]+1):
                connected_matrix[i, :] = -1.1#np.random.normal(0, 1, size=connected_matrix.shape[1])
                connected_matrix[:, i] = -1.1#np.random.normal(0, 1, size=connected_matrix.shape[0])
        else:
            sector_name = 'No Counterfactual'
    elif aggregation_type == aggregation_not:
        # 对246个亚区
        assert counterfactual_sector < min(connected_matrix.shape[0], connected_matrix.shape[1])
        sector_name = str(counterfactual_sector)
        if not counterfactual_sector < 0:
            connected_matrix[counterfactual_sector, :] = -1.1#np.random.normal(0, 1, size=connected_matrix.shape[1])
            connected_matrix[:, counterfactual_sector] = -1.1#np.random.normal(0, 1, size=connected_matrix.shape[0])
    else:
        print(f'Please check you aggregation type = {aggregation_type}')
        exit(1)
    # /*** 反事实研究 ***/

    embedding_from_edges = {} # key(subregion id) : value(embedding 245 from its edges)
    for i in range(len(connected_matrix)):
        this_subregion_embedding_from_edges = [] 
        for j in range(len(connected_matrix[i])):
            if not i == j: 
                this_subregion_embedding_from_edges.append(connected_matrix[i][j]) 
        
        embedding_from_edges[i] = this_subregion_embedding_from_edges
    
    # 将embedding_from_edge 和 embedding_from_nbrs 进行拼接
    # 需要考虑图层面的任务 可解释性则研究边层面、节点层面
    embeddings = [] # 246×490
    for subregion_id in embedding_from_edges.keys():
        embeddings.append(np.array(embedding_from_edges[subregion_id]))

    results = []
    if aggregation_type == aggregation_lobe:
        # 按照7个lobe(脑叶)进行节点聚合
        for lobe in lobe_index:
            startIdx, endIdx = lobe_index[lobe][0], lobe_index[lobe][1]
            aggregation_result = aggregation_embeddings(embeddings, startIdx, endIdx)
            results.append(aggregation_result) # 7×245
    elif aggregation_type == aggregation_gyrus:
        # 按照24个gyrus(脑回)进行节点聚合
        for gyrus in gyrus_index:
            startIdx, endIdx = gyrus_index[gyrus][0], gyrus_index[gyrus][1]
            aggregation_result = aggregation_embeddings(embeddings, startIdx, endIdx)
            results.append(aggregation_result) # 24×245
    elif aggregation_type == aggregation_not:
        # 不进行节点聚合
        results = embeddings # 246*245
    else:
        print(f'Please check you aggregation type = {aggregation_type}')
        exit(1)
    all_data_pair[name] = (results, label) # sub-01到sub-51 label=1; sub-52~sub72 label=0


""" 数据增强 """
# 从一个m×n的二维矩阵中随机挑选一半的位置
def randomly_choose_half_point(m : int, n : int)->list:
    position = []
    for x in range(m):
        for y in range(n):
            position.append((x,y))
    return random.sample(position, len(position)//2)
# 72 -> 465. 51 * 5 = 255; 21 * 10 = 210
noise = 1e-4
original_keys = copy.deepcopy(list(all_data_pair.keys()))
for name in original_keys:
    (results, label) = all_data_pair[name]
    # 患者
    if label == 1:
        # 扩充5倍样本量, 新增4份样本
        for i in range(1,5):
            new_name = name + str(i)
            position = randomly_choose_half_point(len(results), len(results[0]))
            new_results = copy.deepcopy(results)
            for pos in position:
                add_or_reduce = random.randint(0,1) # 该位置处的值 加上或者减去 噪声值
                new_results[pos[0]][pos[1]] += noise if add_or_reduce == 0 else -noise
            all_data_pair[new_name] = (new_results, label)
    # 正常人群
    elif label == 0:
        # 扩充10倍样本量, 新增9份样本
        for i in range(1,10):
            new_name = name + str(i)
            position = randomly_choose_half_point(len(results), len(results[0]))
            new_results = copy.deepcopy(results)
            for pos in position:
                add_or_reduce = random.randint(0,1) # 该位置处的值 加上或者减去 噪声值
                new_results[pos[0]][pos[1]] += noise if add_or_reduce == 0 else -noise
            all_data_pair[new_name] = (new_results, label)


""" 超参数 """
batch_size = len(all_data_pair)
learning_rate = 0.0001 # 0.0001 great MLP
epochs = 100

""" 算力设备 """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device = {device}')

""" 制作datasets """
class GetData(Dataset):
    def __init__(self, participant : list, embeddings : list, targets : list) -> None:
        # embeddings为每个脑区的embedding
        # targets为01值
        self.participant = participant
        self.embeddings = FloatTensor(embeddings)
        self.targets = FloatTensor(targets)
    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        return self.participant[index], self.embeddings[index], self.targets[index]
    def __len__(self) -> int:
        assert len(self.participant) == len(self.embeddings) == len(self.targets)
        return len(self.embeddings)

""" 划分训练集 验证集 测试集 """
def get_train_value_dataloader():
    def make_dataloader(participants : list, x : np.array, y : list) -> DataLoader:
        dataset = GetData(participants, x, y)
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    # 组织数据
    participants, embeddings, labels = [], [], []
    for name in all_data_pair:
        participants.append(name)
        embeddings.append(MinMaxScaler().fit_transform(all_data_pair[name][0]).flatten().tolist())
        labels.append(all_data_pair[name][1])
    
    # 人工核验点
    assert len(participants) == len(embeddings) == len(labels) == 51*5 + 21*10

    # 全样本 465 = 255 + 210. 255 = [0~50]+[72~275]; 210 = [51~71]+[276~464]
    # 训练集:验证集:测试集 = 403:31:31 = (221:182):(17+14):(17+14) 
    train_loader      = make_dataloader(participants[  :51]+participants[72:242] + participants[51:72]+participants[276:437], 
                                        embeddings[    :51]+embeddings[  72:242] + embeddings[  51:72]+embeddings[  276:437], 
                                        labels[        :51]+labels[      72:242] + labels[      51:72]+labels[      276:437])
    validation_loader = make_dataloader(participants[242:259] + participants[437:451], 
                                        embeddings[  242:259] + embeddings[  437:451], 
                                        labels[      242:259] + labels[      437:451])
    test_loader       = make_dataloader(participants[259:276] + participants[451:465], 
                                        embeddings[  259:276] + embeddings[  451:465], 
                                        labels[      259:276] + labels[      451:465])
    return train_loader, validation_loader, test_loader


""" MLP """
class MLP(Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        # 激活函数
        self.activation_function = Tanh() 
        # lobe: in_features= 1716 -> 2**10+
        # gyrus:in_features= 5880 -> 2**12+
        self._features = Sequential(
            # MLP learning rate = 0.0001 is great
            Linear(in_features, 2**12), self.activation_function,
            Linear(2**12      , 2**8 ), self.activation_function,
            Linear(2**8       , 2**4 ), self.activation_function
        )
        # 分类层
        self._classifier = Linear(2**4, 1)
        # 预测层
        self.predict = Sigmoid() 
    
    def forward(self, x: Tensor) -> Tensor:
        x = self._features(x)
        features_x = x
        x = self._classifier(x)
        x = x.squeeze(-1)
        return features_x, self.predict(x)

# 两种方式的API计算AUC值
def calculate_AUC(pred_list : list, true_list : list):
    pred_np = np.array(pred_list)
    true_np = np.array(true_list)
    
    fpr, tpr, thresholds = roc_curve(true_np, pred_np, pos_label=1)
    roc_auc = auc(fpr, tpr)

    assert roc_auc == roc_auc_score(true_np, pred_np)
    return roc_auc, pred_list
    
if __name__ == '__main__':
    train_loader, validation_loader, test_loader = get_train_value_dataloader()

    start_time = time.time()
    in_features = next(iter(train_loader))[1].shape[-1] # lobe:1716=7*245, gyrus:5880=24*245

    model = MLP(in_features)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The number of trainable parametes is {trainable_parameters}')

    if torch.cuda.is_available: # 将模型迁移到GPU上
        model = model.cuda()

    # 损失函数
    loss = nn.CrossEntropyLoss() # 交叉熵损失
    
    # 优化函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

    # 利用训练集和验证集更新模型参数
    y_train_loss, y_valid_loss = [], [] # 用于损失函数值绘图 
    for ep in range(epochs):
        ep_start = time.time()
        train_loss_list = []
            
        # 训练
        model.train()
        pred_list = []
        true_list = []
        for user, xt, yt in train_loader:
            if torch.cuda.is_available: # 将数据迁移到GPU上
                xt, yt = xt.cuda(), yt.cuda()
            _, y_pred = model(xt)
            pred_list += y_pred.cpu()
            true_list += yt.cpu()
            l = loss(y_pred, yt)
            train_loss_list.append(l.item())
            # 反向传播的三步
            optimizer.zero_grad() # 清除梯度
            l.backward() # 反向传播
            optimizer.step() # 优化更新
        
        # 验证
        model.eval()
        val_loss_list = []
        pred_list = []
        true_list = []
        with torch.no_grad():
            for user, xv, yv in validation_loader:
                if torch.cuda.is_available: # 将数据迁移到GPU上
                    xv, yv = xv.cuda(), yv.cuda()
                _, y_pred = model(xv)
                pred_list += y_pred.cpu()
                true_list += yv.cpu()
                l = loss(y_pred, yv)
                val_loss_list.append(l.item())
                
        roc_auc, _ = calculate_AUC(pred_list, true_list)
        
        ep_end = time.time()
        mean_train_loss, mean_val_loss = round(np.mean(train_loss_list),12), round(np.mean(val_loss_list),12)
        y_train_loss.append(mean_train_loss)
        y_valid_loss.append(mean_val_loss)
        # print(f'epoch: {ep}; train loss: {mean_train_loss}; val loss: {mean_val_loss}; val AUC: {roc_auc}; {round((ep_end-ep_start)/60,3)} minutes.')
    
    assert len(y_train_loss) == len(y_valid_loss)
    # 保存训练和验证损失
    x = np.array(list(range(1, epochs+1)))
    y_train_loss = np.array(y_train_loss)
    y_valid_loss = np.array(y_valid_loss)
    
    # 在测试集上计算AUC, LogLoss
    model.eval()
    pred_list = []
    true_list = []
    with torch.no_grad():
        for user, xv, yv in test_loader:
            if torch.cuda.is_available: # 将模型和数据迁移到GPU上
                xv, yv = xv.cuda(), yv.cuda()
            features_x, y_pred = model(xv)
            pred_list += y_pred.cpu()
            true_list += yv.cpu()
        
    roc_auc, pred_list = calculate_AUC(pred_list, true_list)
    y_true, y_pred = np.array(true_list), np.array(pred_list)
    logLoss = log_loss(y_true, y_pred)

    # 保存模型参数
    torch.save(model.state_dict(), 'model.pth')
    
    end_time = time.time()
    print(f'It took {round((end_time-start_time)/60, 2) } minutes to train. counterfactual_sector is {sector_name}. Test AUC = {roc_auc}. Test LogLoss = {logLoss}\n')

    # 可解释性分析结果记录
    result_file = 'result.txt'
    with open(result_file, 'a' if os.path.exists(result_file) else 'w') as file:
        file.write(f'{sector_name}\tAUC={roc_auc}\tLogLoss={logLoss}\n')
        for x, y in zip(pred_list, true_list):
            file.write(f'{str(x)}\t{str(y)}\n')
        file.write('\n\n')