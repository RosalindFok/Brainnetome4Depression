""" UMAP降维分析 """
import umap, os, time
import numpy as np
import matplotlib.pyplot as plt
from load_path  import *
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler 
from sklearn.decomposition import PCA

sub_id_arr, state_arr, matrix_arr = [], [], [] 
for file in select_path_list(CONNECTION_MATRIX, 'npy'):
    sub_id = file[file.find('sub'):file.find('sub')+len('sub-00')]
    state = file[-(len('.npy')+1)]
    matrix = np.load(file)
    sub_id_arr.append(sub_id)
    state_arr.append(state)
    matrix_arr.append(matrix)
assert len(sub_id_arr) == len(state_arr) == len(matrix_arr)

color = ['#75bbfd','#ff7f0e']
marker = ['o', 's']

encoding, label = [], []
for sub_id, state, matrix, _ in zip(sub_id_arr, state_arr, matrix_arr, tqdm(range(len(sub_id_arr)))):
    # matrix处理
    # 提取不包含对角线的上三角
    arr = []
    for x in range(len(matrix)):
         for y in range(len(matrix[x])):
              if x < y:
                arr.append(matrix[x][y])
    matrix = np.array(arr)
    # matrix = MinMaxScaler().fit_transform(matrix.reshape(1, -1))
    encoding.append(matrix)
    label.append(int(state))
assert len(sub_id_arr) == len(encoding) == len(label)
encoding = np.array(encoding) # (72, 245+244+243+...+1)

# 二维UMAP
start_time = time.time()
mapper = umap.UMAP(n_neighbors=len(label)-1, n_components=2, metric='euclidean', n_epochs=100, learning_rate=0.01, min_dist=0.001).fit(encoding)
X_umap_2d = mapper.embedding_
end_time = time.time()
print(f'It took {round((end_time-start_time)/60, 2)} minutes to UMAP to 2D')
X_umap_2d = MinMaxScaler().fit_transform(X_umap_2d)
for (index, value), _ in zip(enumerate(X_umap_2d), tqdm(range(len(X_umap_2d)))):
    plt.scatter(value[0], value[1], color=color[label[index]], marker=marker[label[index]])
plt.title('Feature Matrix UMAP 2D', fontdict={'family':'Times New Roman','size':20})
plt.show()

# 三维UMAP
start_time = time.time()
mapper = umap.UMAP(n_neighbors=len(label)-1, n_components=3, metric='euclidean', n_epochs=100, learning_rate=0.01, min_dist=0.001).fit(encoding)
X_umap_3d = mapper.embedding_
end_time = time.time()
print(f'It took {round((end_time-start_time)/60, 2)} minutes to UMAP to 3D')
X_umap_3d = MinMaxScaler().fit_transform(X_umap_3d)
ax = plt.subplot(projection = '3d') 
for (index, value), _ in zip(enumerate(X_umap_3d), tqdm(range(len(X_umap_3d)))):
            ax.scatter3D(value[0], value[1], value[2], color=color[label[index]], marker=marker[label[index]])
plt.title('Feature Matrix UMAP 3D', fontdict={'family':'Times New Roman','size':20})
plt.show()

def pca_process(arr : np.array, dim : int)->list[float]:
    print(f'PCA Start!')
    start = time.time()
    pca_dim = PCA(n_components=dim)
    arr = pca_dim.fit_transform(arr).tolist()
    result = []
    # 只保留4位小数
    for x, _ in zip(arr, tqdm(range(len(arr)))):
        result.append([float('{:.4f}'.format(i)) for i in x])
    end = time.time()
    print(f'PCA took {round((end-start)/60, 3)} minutes. The dim = {len(result[0])}')
    assert len(result) == len(arr)
    return result

# 二维PCA
pca_2d = pca_process(arr=encoding, dim=2)
pca_2d = MinMaxScaler().fit_transform(pca_2d)
for (index, value), _ in zip(enumerate(pca_2d), tqdm(range(len(pca_2d)))):
    plt.scatter(value[0], value[1], color=color[label[index]], marker=marker[label[index]])
plt.title('Feature Matrix PCA 2D', fontdict={'family':'Times New Roman','size':20})
plt.show()

# 三维PCA
pca_3d = pca_process(arr=encoding, dim=3)
pca_3d = MinMaxScaler().fit_transform(pca_3d)
ax = plt.subplot(projection = '3d') 
for (index, value), _ in zip(enumerate(pca_3d), tqdm(range(len(pca_3d)))):
    ax.scatter(value[0], value[1], value[2], color=color[label[index]], marker=marker[label[index]])
plt.title('Feature Matrix PCA 3D', fontdict={'family':'Times New Roman','size':20})
plt.show()