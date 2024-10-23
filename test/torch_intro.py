# John Eargle (mailto: jeargle at gmail.com)
# torch_intro

import torch


def ch1():
    x = torch.arange(12, dtype=torch.float32)
    print(f'x: {x}')
    print(f'x.numel(): {x.numel()}')
    print(f'x.shape: {x.shape}')
    X = x.reshape(3, 4)
    print(f'X:')
    print(X)


if __name__=='__main__':

    print('***************************')
    print('*** ONTHEDL TORCH INTRO ***')
    print('***************************')

    # ====================
    # Chapter 1
    # ====================

    ch1()
