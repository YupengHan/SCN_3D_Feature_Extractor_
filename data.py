# Copyright 2019-present, Yupeng HAN
# All rights reserved.

# Options
scale=10  #Voxel size = 1/scale
val_reps=1 # Number of test views, 1 or more
batch_size=8 #For testing


import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp, time

dimension=3
full_scale=4096 #Input field size

train,val=[],[]
for x in torch.utils.data.DataLoader(
        glob.glob('train/*.pth'),
        collate_fn=lambda x: torch.load(x[0]), num_workers=mp.cpu_count()):
    train.append(x)
for x in torch.utils.data.DataLoader(
        glob.glob('val/*.pth'),
        collate_fn=lambda x: torch.load(x[0]), num_workers=mp.cpu_count()):
    val.append(x)
print('Training examples:', len(train))
print('Validation examples:', len(val))



def trainMerge(tbl):
    locs=[]
    feats=[]
    labels=[]
    for idx,i in enumerate(tbl):
        a,b,c=train[i]
        m=np.eye(3)+np.random.randn(3,3)*0.1
        m[0][0]*=np.random.randint(0,2)*2-1
        m*=scale
        theta=np.random.rand()*2*math.pi
        m=np.matmul(m,[[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
        a=np.matmul(a,m)
        m=a.min(0)
        M=a.max(0)
        q=M-m
        offset=-m+np.clip(full_scale-M+m-0.001,0,None)*np.random.rand(3)+np.clip(full_scale-M+m+0.001,None,0)*np.random.rand(3)
        a+=offset
        idxs=(a.min(1)>=0)*(a.max(1)<full_scale)
        a=a[idxs]
        b=b[idxs]
        c=c[idxs]
        a=torch.from_numpy(a).long()
        locs.append(torch.cat([a,torch.LongTensor(a.shape[0],1).fill_(idx)],1))
        feats.append(torch.from_numpy(b)+torch.randn(3)*0.1)
        labels.append(torch.from_numpy(c))
    locs=torch.cat(locs,0)
    feats=torch.cat(feats,0)
    labels=torch.cat(labels,0)
    return {'x': [locs,feats], 'y': labels.long(), 'id': tbl}
train_data_loader = torch.utils.data.DataLoader(
    list(range(len(train))),
    batch_size=batch_size,
    collate_fn=trainMerge,
    num_workers=20, 
    shuffle=True,
    drop_last=True,
    worker_init_fn=lambda x: np.random.seed(x+int(time.time()))
)

valOffsets=[0]
valLabels=[]
for idx,x in enumerate(val):
    valOffsets.append(valOffsets[-1]+x[2].size)
    valLabels.append(x[2].astype(np.int32))
valLabels=np.hstack(valLabels)

def valMerge(tbl):
    locs=[]
    feats=[]
    labels=[]
    point_ids=[]
    for idx,i in enumerate(tbl):
        a,b,c=val[i]
        m=np.eye(3)
        m[0][0]*=np.random.randint(0,2)*2-1
        m*=scale
        theta=np.random.rand()*2*math.pi
        m=np.matmul(m,[[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
        a=np.matmul(a,m)+full_scale/2+np.random.uniform(-2,2,3)
        m=a.min(0)
        M=a.max(0)
        q=M-m
        offset=-m+np.clip(full_scale-M+m-0.001,0,None)*np.random.rand(3)+np.clip(full_scale-M+m+0.001,None,0)*np.random.rand(3)
        a+=offset
        idxs=(a.min(1)>=0)*(a.max(1)<full_scale)
        a=a[idxs]
        b=b[idxs]
        c=c[idxs]
        a=torch.from_numpy(a).long()
        locs.append(torch.cat([a,torch.LongTensor(a.shape[0],1).fill_(idx)],1))
        feats.append(torch.from_numpy(b))
        labels.append(torch.from_numpy(c))
        point_ids.append(torch.from_numpy(np.nonzero(idxs)[0]+valOffsets[i]))
    locs=torch.cat(locs,0)
    feats=torch.cat(feats,0)
    labels=torch.cat(labels,0)
    point_ids=torch.cat(point_ids,0)
    return {'x': [locs,feats], 'y': labels.long(), 'id': tbl, 'point_ids': point_ids}
val_data_loader = torch.utils.data.DataLoader(
    list(range(len(val))),
    batch_size=batch_size,
    collate_fn=valMerge,
    num_workers=20,
    shuffle=True,
    worker_init_fn=lambda x: np.random.seed(x+int(time.time()))
)
