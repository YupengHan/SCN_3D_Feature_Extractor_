/disk1/home/yupeng/Data/K3OD/ImageSets/train.txt
/disk1/home/yupeng/Data/K3OD/ImageSets/test.txt

import os
import shutil

f = open('/disk1/home/yupeng/Data/K3OD/training/split_train_val.txt','r')
items = f.readlines()
items = [x.strip('\n') for x in items]

oimg = '/home/yupeng/database/K3OD/training/image_2/' #.png
olabel = '/home/yupeng/database/K3OD/training/label_2/' #.txt
ovelodyne = '/home/yupeng/database/K3OD/training/velodyne/' #.bin

nimg = '/home/yupeng/database/voxeldata/training/image_2/' #.png
nlabel = '/home/yupeng/database/voxeldata/training/label_2/' #.txt
nvelodyne = '/home/yupeng/database/voxeldata/training/velodyne/' #.bin

for i in range(len(items)):
  shutil.copyfile(oimg+items[i]+'.png', nimg+items[i]+'.png')
  shutil.copyfile(olabel+items[i]+'.txt', nlabel+items[i]+'.txt')
  shutil.copyfile(ovelodyne+items[i]+'.bin', nvelodyne+items[i]+'.bin')

# f = open('/home/yupeng/database/K3OD/ImageSets/val.txt','r')
# items = f.readlines()
# items = [x.strip('\n') for x in items]

oimg = '/home/yupeng/database/K3OD/training/image_2/' #.png
olabel = '/home/yupeng/database/K3OD/training/label_2/' #.txt
ovelodyne = '/home/yupeng/database/K3OD/training/velodyne/' #.bin

nimg = '/home/yupeng/database/voxeldata/validation/image_2/' #.png
nlabel = '/home/yupeng/database/voxeldata/validation/label_2/' #.txt
nvelodyne = '/home/yupeng/database/voxeldata/validation/velodyne/' #.bin

for i in range(len(items)):
  shutil.copyfile(oimg+items[i]+'.png', nimg+items[i]+'.png')
  shutil.copyfile(olabel+items[i]+'.txt', nlabel+items[i]+'.txt')
  shutil.copyfile(ovelodyne+items[i]+'.bin', nvelodyne+items[i]+'.bin')