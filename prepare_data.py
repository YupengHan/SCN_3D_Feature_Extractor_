# Copyright 2019-present, Yupeng HAN
# All rights reserved.

import glob, plyfile, numpy as np, multiprocessing as mp, torch

files=sorted(glob.glob('*/*_vh_clean_2.ply'))
files2=sorted(glob.glob('*/*_vh_clean_2.labels.ply'))
assert len(files) == len(files2)

# 把数据存储为 coords, colors, labels的形式
# 1 depth -2 < y < 2; 0 < z < 80
# 2 找到coord对应rgb代码
# 3 存起来，检查一下labels怎么用





def f(fn):
    fn2 = fn[:-3]+'labels.ply'
    a=plyfile.PlyData().read(fn)
    v=np.array([list(x) for x in a.elements[0]])
    coords=np.ascontiguousarray(v[:,:3]-v[:,:3].mean(0))
    colors=np.ascontiguousarray(v[:,3:6])/127.5-1
    a=plyfile.PlyData().read(fn2)
    w=remapper[np.array(a.elements[0]['label'])]
    torch.save((coords,colors,w),fn[:-4]+'.pth')
    print(fn, fn2)

p = mp.Pool(processes=mp.cpu_count())
p.map(f,files)
p.close()
p.join()


