import torch
import grouping_cuda as grouping
import numpy as np

n = 10
m = 3
nsamples = 5
bs = 1

# bs, n, 1
xyz = torch.tensor(np.array([
    0, 2, 0, 2, 0, 2, 1, 2, 2, 1
])).view( bs, 10, 1).to('cuda:0').int()

# bs, m, nsample
idx = torch.ones((bs, m, nsamples)).to('cuda:0').int()

grouping.grouping_wrapper(bs, n, m, nsamples, xyz, idx)

print( idx )
print('test over')

