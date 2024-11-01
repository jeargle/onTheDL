# John Eargle (mailto: jeargle at gmail.com)
# torch_intro

import numpy
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
    print(f'exp(X):')
    print(torch.exp(x))

    # Operations
    x = torch.tensor([1.0, 2, 4, 8])
    y = torch.tensor([2, 2, 2, 2])
    # x + y, x - y, x * y, x / y, x ** y
    print(f'x:')
    print(x)
    print(f'y:')
    print(y)
    print(f'x + y:')
    print(x + y)
    print(f'x - y:')
    print(x - y)
    print(f'x * y:')
    print(x * y)
    print(f'x / y:')
    print(x / y)
    print(f'x ** y:')
    print(x ** y)

    X = torch.arange(12, dtype=torch.float32).reshape((3,4))
    Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    print('X:')
    print(X)
    print('Y:')
    print(Y)
    print('torch.cat((X, Y), dim=0):')
    print(torch.cat((X, Y), dim=0))
    print('torch.cat((X, Y), dim=1):')
    print(torch.cat((X, Y), dim=1))
    print('X == Y:')
    print(X == Y)
    print('X.sum():')
    print(X.sum())

    # Broadcasting
    a = torch.arange(3).reshape((3, 1))
    b = torch.arange(2).reshape((1, 2))
    print('a:')
    print(a)
    print('b:')
    print(b)
    print('a + b:')
    print(a + b)

    # Saving memory
    before = id(Y)
    Y = Y + X
    print('before:')
    print(before)
    print('Y:')
    print(Y)
    print(f'id(Y) == before: {id(Y) == before}')

    Z = torch.zeros_like(Y)
    print('id(Z):', id(Z))
    Z[:] = X + Y
    print('id(Z):', id(Z))

    before = id(X)
    X += Y
    print('before:')
    print(before)
    print('X:')
    print(X)
    print(f'id(X) == before: {id(X) == before}')

    # Conversion to other python objects
    # A = X.numpy()
    A = X.numpy()
    B = torch.from_numpy(A)
    print('A:')
    print(A)
    print('B:')
    print(B)
    print('type(A), type(B):')
    print(type(A), type(B))
    a = torch.tensor([3.5])
    a, a.item(), float(a), int(a)
    print('a:')
    print(a)
    print('a.item():')
    print(a.item())
    print('float(a):')
    print(float(a))
    print('int(a):')
    print(int(a))


if __name__=='__main__':

    print('***************************')
    print('*** ONTHEDL TORCH INTRO ***')
    print('***************************')

    # ====================
    # Chapter 2
    # ====================

    ch2()
