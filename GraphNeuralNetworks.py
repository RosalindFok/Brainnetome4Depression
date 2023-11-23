""" GNN """
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
from load_path import *

""" 构建每个脑区的embedding """
# 72 participants
all_data_pair = {} # key(sub-id) : value(matirx 246×490, label∈{0,1})
for file in select_path_list(CONNECTION_MATRIX, '.npy'):
    # 1-抑郁症 0-健康人群
    name = file.split(os.sep)[2].split('_')[0]
    label = int(file[-len('.npy')-1])
    # 246×246 matrix for each participants
    connected_matrix = np.load(file) # 246×246 取值(-1,1]
    embedding_from_edges = {} # key(subregion id) : value(embedding 245 from its edges)
    for i in range(len(connected_matrix)):
        this_subregion_embedding_from_edges = [] 
        for j in range(len(connected_matrix[i])):
            if not i == j: 
                this_subregion_embedding_from_edges.append(connected_matrix[i][j]) 
        embedding_from_edges[i] = this_subregion_embedding_from_edges
    
    embedding_from_nbrs = {} # key(subregion id) : value(embedding 245 from its neighbours)
    for i in range(len(connected_matrix)):
        embedding_from_nbrs[i] = np.array([.0] * (len(connected_matrix[i])-1))
    
    for (subregion_id, weights), _ in zip(embedding_from_edges.items(), tqdm(range(len(embedding_from_edges)))):
        this_subregion_embedding_from_nbrs = []
        # 每个节点有245条边、对应245个邻居。邻居的embedding每位乘以边上权重。所有邻居的embedding相加
        weights = np.array(weights).reshape(len(weights),1) # 245×1的列向量
        weights = np.tile(weights, weights.shape[0]) # 扩充为245×245 每行中的元素相同. 
        for neighbours_id in embedding_from_nbrs.keys():
            if not neighbours_id == subregion_id:
                this_subregion_embedding_from_nbrs.append(embedding_from_edges[neighbours_id])  
        this_subregion_embedding_from_nbrs = np.array(this_subregion_embedding_from_nbrs)
        assert weights.shape == this_subregion_embedding_from_nbrs.shape
        # 逐行进行哈达玛积
        for weights_row, embedding_row in zip(weights, this_subregion_embedding_from_nbrs):
            row = weights_row * embedding_row
            embedding_from_nbrs[subregion_id] += row
    
    assert len(embedding_from_edges) == len(embedding_from_nbrs)

    # 将embedding_from_edge 和 embedding_from_nbrs 进行拼接
    embeddings = [] # 246×490
    for subregion_id in embedding_from_edges.keys():
        embeddings.append(np.concatenate((embedding_from_edges[subregion_id], embedding_from_nbrs[subregion_id]),axis=0))

    all_data_pair[name] = (np.array(embeddings), label)


""" 超参数 """
batch_size = len(all_data_pair)/2
learning_rate = 0.001
epochs = 100

""" 算力设备 """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device = {device}')
np.random.seed(0)

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
    
    # TODO 目前全部作为训练集 验证集 测试集
    train_loader = validation_loader = test_loader = make_dataloader(participants, embeddings, labels)
    return train_loader, validation_loader, test_loader

""" MLP """
class MLP(Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        # 激活函数
        self.activation_function = Tanh()
        # in_features = 120540 接近2**17
        self._features = Sequential(
            Linear(in_features, 2**17), self.activation_function,
            Linear(2**17      , 2**12), self.activation_function,
            Linear(2**12      , 2**8 ), self.activation_function,
            Linear(2**8       , 2**4 ), self.activation_function)
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
    in_features = next(iter(train_loader))[1].shape[-1] # 120540 = 246*490
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
        print(f'Training...')
        pred_list = []
        true_list = []
        for (user, xt, yt), _ in zip(train_loader, tqdm(range(len(train_loader)))):
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
        print(f'Validating...')
        val_loss_list = []
        pred_list = []
        true_list = []
        with torch.no_grad():
            for (user, xv, yv), _ in zip(validation_loader, tqdm(range(len(validation_loader)))):
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
        print(f'epoch: {ep}; train loss: {mean_train_loss}; val loss: {mean_val_loss}; val AUC: {roc_auc}; {round((ep_end-ep_start)/60,3)} minutes.')
    
    assert len(y_train_loss) == len(y_valid_loss)
    # 保存训练和验证损失
    x = np.array(list(range(1, epochs+1)))
    y_train_loss = np.array(y_train_loss)
    y_valid_loss = np.array(y_valid_loss)
    
    # 在测试集上计算AUC, LogLoss
    print(f'Testing...')
    model.eval()
    pred_list = []
    true_list = []
    temp_path = os.path.join('..','temp')
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    with open(os.path.join('..','temp', str(dataset)+'_mlp_hidden_layer.txt'), 'w') as f:
        with torch.no_grad():
            for (user, xv, yv), _ in zip(test_loader, tqdm(range(len(test_loader)))):
                if torch.cuda.is_available: # 将模型和数据迁移到GPU上
                    xv, yv = xv.cuda(), yv.cuda()
                features_x, y_pred = model(xv)
                write_x = features_x.cpu().detach().numpy().tolist()
                write_y = yv.cpu().detach().numpy().tolist()
                assert len(write_x) == len(write_y)
                for w_x, w_y in zip(write_x, write_y):
                    for x in w_x:
                        f.write(str(x)+' ')
                    f.write(str(w_y)+'\n')
                pred_list += y_pred.cpu()
                true_list += yv.cpu()
        
    roc_auc, pred_list = calculate_AUC(pred_list, true_list)
    y_true, y_pred = np.array(true_list), np.array(pred_list)
    logLoss = log_loss(y_true, y_pred)

    # 保存模型参数
    # torch.save(model.state_dict(), os.path.join(torch_model_path, label_tag+'.pth'))
    
    end_time = time.time()
    print(f'It took {round((end_time-start_time)/60, 2) } minutes to train model. Test AUC = {roc_auc}. Test LogLoss = {logLoss}\n')
# 可解释性：找到最有价值的哪几个脑区

# 性能评估