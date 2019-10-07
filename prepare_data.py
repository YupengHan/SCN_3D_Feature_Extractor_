# Copyright 2019-present, Yupeng HAN
# All rights reserved.

# TODO: labels need to be stored in numpy
# classification need to store in ndarray!
# If anything need to modified: http://localhost:8008/notebooks/Fusion/data/dataandcroptest.ipynb



import glob, numpy as np, multiprocessing as mp, torch
import kitti_utils

KITTI_LABEL_NAMES = (
    'Car',
    'Pedestrian',
    'Cyclist',
    'Truck',
    'Misc',
    'Van',
    'Tram',
    'Person_sitting',)

origin_data_dir = '/disk1/home/yupeng/Data/K3OD/training'
proc_data_dir = '/disk1/home/yupeng/Data/Proc_FusionData'


vel_files=sorted(glob.glob(origin_data_dir+'/velodyne/*.bin'))
img_files=sorted(glob.glob(origin_data_dir+'/image_2/*.png'))
lab_files=sorted(glob.glob(origin_data_dir+'/label_2/*.txt'))
cal_files=sorted(glob.glob(origin_data_dir+'/calib/*.txt'))

assert len(cal_files) == len(lab_files) and len(lab_files) == len(img_files) and len(img_files) == len(vel_files)

def f(img_n):
    idx = fn[-10:-4] #string
    vel_n = origin_data_dir+'/velodyne/'+int(idx)+'.bin'
    cal_n = origin_data_dir+'/calib/'+idx+'.txt'
    lab_n = origin_data_dir+'/label_2/'+idx+'.txt'
    calib_data = kitti_utils.Calibration(cal_n)
    
    vel = kitti_utils.load_velo_scan(vel_n)
    vel = calib_data.project_velo_to_rect(vel)
    # 0 < z < 70
    vel = vel[np.where(vel[:,2] > 0)]
    vel = vel[np.where(vel[:,2] < 70)]
    # -3 < y < 1
    vel = vel[np.where(vel[:,1] < 1)]
    vel = vel[np.where(vel[:,1] > -3)]
    # -40 < y < 40
    vel = vel[np.where(vel[:,0] < 40)]
    vel = vel[np.where(vel[:,0] > -40)]
    # 这里没有中心化！注意变化！！！！
    # coords=np.ascontiguousarray(vel[:,:3]-v[:,:3].mean(0))
    coords = np.ascontiguousarray(vel)

    

    # Get color
    vel_2d = calib_data.project_rect_to_image(vel)
    vel_rgb=np.zeros((len(vel_2d),3))
    colors = kitti_utils.img_xy2rgb(img_n, vel_2d)
    
    # read label
    f = open(lab_n)
    str_label = f.readlines()
    label_3d_boxes = []
    # label.append(KITTI_LABEL_NAMES.index(name))

    for label in str_label:
        if label[0:8] == "DontCare":
            str_label.remove(label)
            continue
        obj = Object3d(label)
        box3d = compute_box_3d(obj, calib_data.P)
        label_3d_boxes.append(box3d[1])

    
    label_3d_boxes = np.asarray(label_3d_boxes)
    label_3d_boxes = label_3d_boxes.astype(np.float)
    
    str_label = [x for x in str_label if x[0:8] != 'DontCare']
    str_label = [x.split(' ') for x in str_label]
    nplabel = np.asarray(str_label)
    for i in range(len(nplabel)): 
        nplabel[i][0] = KITTI_LABEL_NAMES.index(nplabel[i][0])
    nplabel = nplabel.astype(np.float)
    #第1列truck 表示图中出现了卡车（一共有'Car', 'Van', 'Truck','Pedestrian', 'Person_sitting', 'Cyclist', 'Tram','Misc' or 'DontCare'这些类别，Don’t care 是没有3D标注的，原因是雷达扫不了那么远，即使可以视觉检测出来）
    #第2列0.0表示其是否被截断的程度为0。（如果车在图片边缘，那么就有可能发生部分被截断的情况。用0-1 表示被截断的程度。）
    #第3列0表示没有被遮挡。（0表示完全可见，1表示部分遮挡，2表示大部分被遮挡，3表示未知。）
    #第4列 -1.57 表示卡车中心与相机中心构成的矢量与在bird view下的夹角为-1.57，实际上就是说明改开叉在-90，即正前方。这个信息反映目标物体中心在bird view相对方向信息。
    #第5-8列的599.41 156.40 629.75 189.25是目标的2D bounding box 像素位置，形式为xyxy，前两个值为bounding box左上点的x，y位置，后两个点为右下角的x,y位置。
    # 第9-11列 2.85 2.63 12.34 表示该车的高度，宽度，和长度，单位为米。
    # 第12-14列 0.47 1.49 69.44 表示该车的3D中心在相机坐标下的xyz坐标。
    # 第15列 -1.56 表示车体朝向，绕相机坐标系y轴的弧度值。注意和第4列区别开来，第四列不在乎车体朝向，而是车体中心与相机中心所构成矢量在与相机坐标系z轴的夹角（其实这里笔者有点疑虑，如果车体中心位置已知，车体朝向是不知道的，但是第4列的alpha是可以算出来的，那么其实第4列的数据是冗余的？）。



    torch.save((coords,colors,nplabel,label_3d_boxes),proc_data_dir+idx+'.pth')
    print(idx)

p = mp.Pool(processes=mp.cpu_count())
p.map(f,img_files)
p.close()
p.join()


