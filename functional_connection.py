# -*- coding: UTF-8 -*-
""" 功能连接网络 """
import os, csv, time, json
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nibabel.viewers import OrthoSlicer3D
from nilearn import plotting, maskers, connectome, image, datasets
from load_path import *

""" 绘制相关矩阵 """
def draw_correlation_matrix(correlation_matrix : np.array, labels=None)->None:
    plt.figure(figsize=(10, 10))
    plt.imshow(correlation_matrix, interpolation="nearest", cmap="RdBu_r", vmax=0.8, vmin=-0.8)

    if not labels == None:
        # Add labels and adjust margins
        x_ticks = plt.xticks(range(len(labels) - 1), labels[1:], rotation=90)
        y_ticks = plt.yticks(range(len(labels) - 1), labels[1:])
        plt.gca().yaxis.tick_right()
    plt.subplots_adjust(left=.01, bottom=.3, top=.99, right=.62)
    plt.show()

""" 保存相关矩阵 """
def save_connection_matrix(correlation_matrix : np.array, save_path : str):
    plt.figure(figsize=(10, 10))
    plt.imshow(correlation_matrix, interpolation="nearest", cmap="RdBu_r", vmax=0.8, vmin=-0.8)
    plt.savefig(save_path)
    plt.close()

""" 对脑图谱的可视化 """
def draw_atlas(atlas : nib.nifti1.Nifti1Image):
    # 展示三个剖面
    OrthoSlicer3D(atlas.dataobj).show()
    # 绘制玻璃脑图像
    plotting.plot_glass_brain(atlas, colorbar=True, plot_abs=False)
    plotting.show()
    # 绘制结构连接矩阵
    with open(BNA_MATRIX_PATH, 'r') as file:
        reader = csv.reader(file)
        matrix = np.array([row for row in reader])
        matrix = np.where(matrix == '0', 0, 1)
        draw_correlation_matrix(matrix)   
        # 获取节点坐标
        coordinates = plotting.find_parcellation_cut_coords(labels_img=atlas)  
        # 绘制脑图谱连接网络
        plotting.plot_connectome(matrix, coordinates, node_size=10)
        # 显示图谱
        plotting.show()

""" Atlas """
# Brainnetome Atlas - Brainnetome Center and National Laboratory of Pattern Recognition(NLPR)
atlas = nib.load(path_join(BNA_PATH, 'BN_Atlas_246_1mm.nii.gz')) # dim[1~3] = [182 218 182]
# Harvard-Oxford
# atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm').maps
# labels = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm').labels
# 对图谱的解析
# draw_atlas(atlas)

# 定义一个标签掩码器
masker = maskers.NiftiLabelsMasker(labels_img=atlas, standardize='zscore_sample')

""" 读取PARICIPANTS_INFO的内容 """
participants_side_info = {} # key(sub-id) : value(information_dictionary);
with open(PARICIPANTS_INFO, 'r') as file:
    reader = csv.reader(file, delimiter='\t')
    side_info = [row for row in reader]
    # head解析'participant_id', 'age', 'gender', 'group', 'IQ_Raven', 'ICD-10', 'MADRS', 'Zung_SDS', 'BDI', 'HADS-anx', 'HADS-depr', 'MC-SDS', 'TAS-26', 'ECR-avoid', 'ECR-anx', 'RRS-sum', 'RRS-reflection', 'RRS-brooding', 'RRS-depr', 'Edinburgh'
    # 其中group下有 depr和control. depr为抑郁症患者, control为健康人群
    head = side_info[0][1:]
    side_info = side_info[1:]
    for each_participants_side_info in side_info:
        assert len(head) == len(each_participants_side_info[1:])
        information_dictionary = {field:value for field, value in zip(head, each_participants_side_info[1:])} # key(field in head) : value(the specific value)
        participants_side_info[each_participants_side_info[0]] = information_dictionary
with open(PARICIPANTS_INFO_JSON, 'w') as file:
    json.dump(participants_side_info, file, indent=4)

""" nibabel库读取nii.gz文件
nibabel.load返回一个Nifti1Image类型变量
通过方法get_fdata()来获得其数组, 为numpy类型
通过header对象来获得其头信息,包括维度、像素间距、数据类型等.其维度dim是一个长度为8的数组,通过header['dim']访问
    - dim[0] 表示数据数组的维度数量.例如,如果数据是一个 3D 图像,dim[0] 就是 3.
    - dim[1] 到 dim[3] 分别表示数据在三个空间维度(X、Y、Z)上的大小.例如,如果数据是一个 100x100x100 的 3D 图像,dim[1]、dim[2] 和 dim[3] 就分别是 100.
    - 如果数据有时间维度,dim[4] 就表示时间点的数量.例如,如果数据是一个 4D 时间序列图像,每个时间点是一个 100x100x100 的 3D 图像,且有 10 个时间点,dim[4] 就是 10.
    - dim[5] 到 dim[7] 用于更高维度的数据,但在大多数情况下,这些维度的大小都是 1.

静息态(rs)fMRI的重要特征: 功能连接-描述不同脑区的协同性、ReHo(局部一致性)-描述相邻体素区域的活动步调的一致性、低频波动振幅(ALFF)-描述单个体素区域的活动强度
    ReHo(Regional Homogeneity):在于描述给定体素的时间序列(BOLD信号)与其最近邻体素的时间序列(BOLD)的相似性,ReHo认为,当大脑功能区域涉及特定条件时,该区域内的体素在时间上更均匀,是一个经过验证的较为可靠rs-fMRI特征.
    ALFF(Amplitude of Low Frequency Fluctuation):揭示了区域自发活动的 BOLD 信号强度.
"""

for sub_func_path,sub_anat_path in zip(SUBJECTS_FUNC_PATH, SUBJECTS_ANAT_PATH):
    start_time = time.time()

    files = select_path_list(sub_func_path[0], '.nii')
    full_name = files[0].split(os.sep)[-1].split('.')[0]
    sub_name = full_name.split('_')[0]
    state = participants_side_info[sub_name]['group']
    state = '0' if state == 'control' else '1' if state == 'depr' else exit(1) # 0-健康人群 1代表患者
    save_matrix_path = path_join(CONNECTION_MATRIX, full_name+'_'+state+'.npy')
    save_pic_path = path_join(CONNECTION_MATRIX, full_name+'_'+state+'.svg') 
    save_ntw_path = path_join(CONNECTION_MATRIX, full_name+'_network_'+state+'.svg') 
    
    # 连接矩阵和热力图均存在
    if os.path.exists(save_matrix_path) and os.path.exists(save_pic_path):
        ### 绘制相互作用图 ###
        correlation_matrix = np.load(save_matrix_path)
        np.fill_diagonal(correlation_matrix, 0)
        correlation_matrix = np.where(correlation_matrix>=0.7, correlation_matrix, 0)
        # 获取节点坐标
        coordinates = plotting.find_parcellation_cut_coords(labels_img=atlas)  
        # 绘制脑图谱相互作用网络
        plotting.plot_connectome(correlation_matrix, coordinates, node_size=10, colorbar=True, output_file=save_ntw_path)
    # 连接矩阵信息不存在
    elif not os.path.exists(save_matrix_path):
        img = nib.load(files[0]) # 本数据集每个img的时间序列为100
        # 删除前5个和后5个时间维度的图像
        img = nib.Nifti1Image(img.get_fdata()[...,5:-5], img.affine, img.header)
        # 对原始图像进行上采样与Altas对齐
        img = image.resample_img(img, target_affine=atlas.affine, target_shape=atlas.shape[:3])
        # 提取时间序列
        time_series = masker.fit_transform(img)
        # 计算连接矩阵 BN Atlas=246×246 其值分布在 (-1.0, 1.0] 之间
        correlation_matrix = connectome.ConnectivityMeasure(kind='correlation', standardize='zscore_sample').fit_transform([time_series])[0] 
        # 保存连接矩阵和其热力图
        np.save(save_matrix_path, correlation_matrix)
        save_connection_matrix(correlation_matrix=correlation_matrix, save_path=save_pic_path)
    # 连接矩阵存在 但是热力图不存在
    elif os.path.exists(save_matrix_path) and not os.path.exists(save_pic_path):
        correlation_matrix = np.load(save_matrix_path)
        # 保存热力图
        save_connection_matrix(correlation_matrix=correlation_matrix, save_path=save_pic_path)
    
    end_time = time.time()
    print(f'It took {round((end_time - start_time)/60, 2)} minutes to process {sub_name}.')
   
    