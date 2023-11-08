import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from nibabel.viewers import OrthoSlicer3D
from nilearn import plotting, maskers, connectome, image, datasets
from sklearn.cluster import KMeans

""" 文件路径 """
# 原始数据 REST-meta-MDD-Phase1-Sharing
REST_meta_MDD = os.path.join('..', 'REST-meta-MDD')
Phase1_Sharing = os.path.join(REST_meta_MDD, 'REST-meta-MDD-Phase1-Sharing')
Results = os.path.join(Phase1_Sharing, 'Results')
ReHo_labels = ['ReHo_FunImgARCWF', 'ReHo_FunImgARglobalCWF']
ReHo_paths = [ReHo_FunImgARCWF, ReHo_FunImgARglobalCWF] = [os.path.join(Results, label) for label in ReHo_labels]
ReHo_dict = {label: path for label, path in zip(ReHo_labels, ReHo_paths)} # {key=label : value=path}
# 原始数据 REST-meta-MDD-VBM-Phase1-Sharing
VBM_Phase1_Sharing = os.path.join(REST_meta_MDD, 'REST-meta-MDD-VBM-Phase1-Sharing')
VBM = os.path.join(VBM_Phase1_Sharing, 'VBM')
model_labels = ['c1', 'c2', 'c3', 'mwc1', 'mwc2', 'mwc3', 'wc1', 'wc2', 'wc3']
model_paths = [c1, c2, c3, mwc1, mwc2, mwc3, wc1, wc2, wc3] = [os.path.join(VBM, label) for label in model_labels]
model_dict = {label: path for label, path in zip(model_labels, model_paths)} # {key=label : value=path}
# 伪时序图像
concatenated_images_path = os.path.join('..', 'concatenated_images')

""" 绘制相关矩阵 """
def draw_correlation_matrix(correlation_matrix, labels=None)->None:
    plt.figure(figsize=(10, 10))
    # Mask the main diagonal for visualization:
    # np.fill_diagonal(correlation_matrix, 0)

    plt.imshow(correlation_matrix, interpolation="nearest", cmap="RdBu_r", vmax=0.8, vmin=-0.8)

    if not labels == None:
        # Add labels and adjust margins
        x_ticks = plt.xticks(range(len(labels) - 1), labels[1:], rotation=90)
        y_ticks = plt.yticks(range(len(labels) - 1), labels[1:])
        plt.gca().yaxis.tick_right()
    plt.subplots_adjust(left=.01, bottom=.3, top=.99, right=.62)
    plt.show()

""" 将该路径下的各个患者合并成一个文件 将该路径下的各个正常合并成一个文件 """
def gather_subjects_to_time_series(model_path, save_path):
    for site_name in ['S' + str(i) for i in range(1, 26)]:
        for flag in ['-1-', '-2-']: # 1-MDD 2-NCs
            concatenated_path = os.path.join(save_path, site_name+flag+'concatenated.nii.gz')
            if os.path.exists(concatenated_path):
                return 
            
            files = [os.path.join(model_path, file) for file in os.listdir(model_path) if site_name+flag in file]
            first_img = nib.load(files[0])
            first_data = first_img.get_fdata()
            shape = first_data.shape
            # 创建一个空的拼接后的图像数据数组
            concatenated_data = np.zeros((shape[0], shape[1], shape[2], len(files)))
            # 遍历每个nii文件
            for i, nii_file in enumerate(files):
                # 加载nii文件
                img = nib.load(nii_file)
                # 提取数据数组
                data = img.get_fdata()
                # 将数据数组添加到拼接后的图像数据数组中
                concatenated_data[:, :, :, i] = data

            # 创建一个新的nii图像对象，使用第一个nii文件的头信息
            concatenated_img = nib.Nifti1Image(concatenated_data, affine=first_img.affine, header=first_img.header)
            nib.save(concatenated_img, concatenated_path)
            print(f'{concatenated_path} is saved!')
            

""" Atlas """
### Brainnetome Atlas - Brainnetome Center and National Laboratory of Pattern Recognition(NLPR)
atlas = nib.load('BN_Atlas_246_1mm.nii.gz') # dim[1~3] = [182 218 182]
### Harvard_Oxford Atlas
# dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
# atlas = dataset.maps # dim[1~3] = [91 109  91]
# 映射到MNI空间
# load_mni152_template
# load_mni152_gm_template: grey-matter template.
# load_mni152_wm_template: white-matter template.
# atlas = image.resample_to_img(atlas, datasets.load_mni152_gm_template()) # dim[1~3] = [197 233 189]
# labels = dataset.labels

""" nibabel库读取nii.gz文件
nibabel.load返回一个Nifti1Image类型变量
通过方法get_fdata()来获得其数组, 为numpy类型
通过header对象来获得其头信息,包括维度、像素间距、数据类型等.其维度dim是一个长度为8的数组,通过header['dim']访问
    - dim[0] 表示数据数组的维度数量.例如,如果数据是一个 3D 图像,dim[0] 就是 3.
    - dim[1] 到 dim[3] 分别表示数据在三个空间维度(X、Y、Z)上的大小.例如,如果数据是一个 100x100x100 的 3D 图像,dim[1]、dim[2] 和 dim[3] 就分别是 100.
    - 如果数据有时间维度,dim[4] 就表示时间点的数量.例如,如果数据是一个 4D 时间序列图像,每个时间点是一个 100x100x100 的 3D 图像,且有 10 个时间点,dim[4] 就是 10.
    - dim[5] 到 dim[7] 用于更高维度的数据,但在大多数情况下,这些维度的大小都是 1.
    - MNI空间的数据 dim[1~3] = [121 145 121]

本rest-fMRI数据集中每个像素点存放一个(0,1)之间的灰度值, 周围有很多空值, 有部分值大于1.
本rest-fMRI数据集中  不存在时间序列,即dim[4]=1

静息态(rs)fMRI的重要特征: 功能连接-描述不同脑区的协同性、ReHo(局部一致性)-描述相邻体素区域的活动步调的一致性、低频波动振幅(ALFF)-描述单个体素区域的活动强度
    ReHo(Regional Homogeneity):在于描述给定体素的时间序列(BOLD信号)与其最近邻体素的时间序列(BOLD)的相似性,ReHo认为,当大脑功能区域涉及特定条件时,该区域内的体素在时间上更均匀,是一个经过验证的较为可靠rs-fMRI特征.
    ALFF(Amplitude of Low Frequency Fluctuation):揭示了区域自发活动的 BOLD 信号强度.

使用该数据集计算功能连接出现负相关: http://rfmri.org/node/469
"""

# ReHo_FunImgARglobalCWF_files = os.listdir(ReHo_FunImgARglobalCWF)
# for file , _ in zip(ReHo_FunImgARglobalCWF_files, tqdm(range(len(ReHo_FunImgARglobalCWF_files)))):
#     img = nib.load(os.path.join(ReHo_FunImgARglobalCWF, file))
#     print(img.header['dim'])
# exit(0)

""" 构建伪rs-fMRI时间序列 """
# 所有受试者采集自25个站点(Site, S1~S25). 每个站点都有1-患者、2-正常人(Not Clinically Significant, NCs)
# 每个站点的所有患者合并一个、每个站点的正常人合并一个
# 使用wc1/wc2/wc3: MNI空间的灰质/白质/脑脊液的密度(density)
# 使用mwc1/mwc2/mwc3: MNI空间的灰质/白质/脑脊液的体素(volume)
for model_label, model_path in model_dict.items():
    if 'w' in model_label:
        save_path = os.path.join(concatenated_images_path, model_label)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        gather_subjects_to_time_series(model_path, save_path)

""" """
labels = os.listdir(concatenated_images_path)
for label in os.listdir(concatenated_images_path):
    print(label)
exit(0)


# 定义一个标签掩码器
masker = maskers.NiftiLabelsMasker(labels_img=atlas, standardize='zscore_sample')#True) 

# 提取时间序列
time_series = masker.fit_transform(concatenated_img) 

# 计算连接矩阵
correlation_matrix = connectome.ConnectivityMeasure(kind='correlation', standardize='zscore_sample').fit_transform([time_series])[0] 
for x in correlation_matrix:
    print(x)
exit(0)
# 绘制热力图
draw_correlation_matrix(correlation_matrix)
# exit(0)

# 获取节点坐标
coordinates = plotting.find_parcellation_cut_coords(labels_img=atlas) 

# 绘制连接图
plotting.plot_connectome(correlation_matrix, coordinates)   

# 显示图像
plotting.show()
exit(0)
for file, _ in zip(wc1_files, tqdm(range(len(wc1_files)))):
    # 把S1-1-0001.nii.gz到S1-1-0074.nii.gz拼起来，装作一个时间序列
    
    
    
    
    img = nib.load(os.path.join(wc1, file))
    print(img.header['dim'])
    continue
    # 展示三个剖面
    # OrthoSlicer3D(new_img.dataobj).show()
    # 绘制玻璃脑图像
    # plotting.plot_glass_brain(img, colorbar=True, plot_abs=False)
    # plotting.show()

    # 定义一个标签掩码器
    masker = maskers.NiftiLabelsMasker(labels_img=atlas, standardize=True) 

    # 提取时间序列
    time_series = masker.fit_transform(img) 

    # 计算连接矩阵
    correlation_matrix = connectome.ConnectivityMeasure(kind='correlation').fit_transform([time_series])[0] 
    print(correlation_matrix)
    assert np.count_nonzero(correlation_matrix == 1) == 246
    # exit(0)
    # 获取节点坐标
    coordinates = plotting.find_parcellation_cut_coords(labels_img=atlas) 

    # 绘制连接图
    plotting.plot_connectome(correlation_matrix, coordinates)   

    # 显示图像
    plotting.show()
    # # numpy类型的数组
    # data = img.get_fdata() 
    # # 将NaN值替换为0
    # data = np.nan_to_num(data)