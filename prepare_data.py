# Copyright 2019-present, Yupeng HAN
# All rights reserved.

import glob, numpy as np, multiprocessing as mp, torch
import kitti_utils


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
    labels = f.readlines()
    label_3d_boxes = []

    for label in labels:
        if label[0:8] == "DontCare":
            continue
        obj = Object3d(label)
        box3d = compute_box_3d(obj, calib_data.P)
        label_3d_boxes.append(box3d[1])


    



    torch.save((coords,colors,labels, label_3d_boxes),proc_data_dir+idx+'.pth')
    print(idx)

p = mp.Pool(processes=mp.cpu_count())
p.map(f,img_files)
p.close()
p.join()


