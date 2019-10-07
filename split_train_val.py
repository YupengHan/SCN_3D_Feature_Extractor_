import os
import shutil

train_list = '/disk1/home/yupeng/Data/K3OD/ImageSets/train.txt'
test_list = '/disk1/home/yupeng/Data/K3OD/ImageSets/test.txt'

f = open(train_list,'r')
items = f.readlines()
items = [x.strip('\n') for x in items]

opth = '/disk1/home/yupeng/Data/Proc_FusionData/' #.pth
ntra   = '/disk1/home/yupeng/Data/Proc_FusionData/train/'
for i in range(len(items)):
  shutil.copyfile(opth+items[i]+'.pth', ntra+items[i]+'.pth')


f = open(test_list,'r')
items = f.readlines()
items = [x.strip('\n') for x in items]
nval = '/disk1/home/yupeng/Data/Proc_FusionData/valid/'
for i in range(len(items)):
  shutil.copyfile(opth+items[i]+'.pth', nval+items[i]+'.pth')
