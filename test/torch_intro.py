# John Eargle (mailto: jeargle at gmail.com)
# torch_intro

import torch


def ch2():
    x = torch.arange(12, dtype=torch.float32)
    print(f'x: {x}')
    print(f'x.numel(): {x.numel()}')
    print(f'x.shape: {x.shape}')
    X = x.reshape(3, 4)
    print(f'X:')
    print(X)
    # Automatically infer a dimension with -1.
    Y = x.reshape(2, -1)
    print(f'Y:')
    print(Y)
    Z = x.reshape(-1, 3)
    print(f'Z:')
    print(Z)
    X = torch.zeros((2, 3, 4))
    print(f'X:')
    print(X)
    X = torch.ones((2, 3, 4))
    print(f'X:')
    print(X)
    X = torch.randn(3, 4)
    print(f'X:')
    print(X)
    X = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    print(f'X:')
    print(X)
    X = x.reshape(3, 4)
    print(f'X:')
    print(X)
    print(f'X[-1]:')
    print(X[-1])
    print(f'X[1:3]:')
    print(X[1:3])
    X[1, 2] = 17
    print(f'X:')
    print(X)
    # Assign multiple rows.
    X[:2, :] = 12
    print(f'X:')
    print(X)


if __name__=='__main__':

    print('***************************')
    print('*** ONTHEDL TORCH INTRO ***')
    print('***************************')

    # ====================
    # Chapter 2
    # ====================

    ch2()
