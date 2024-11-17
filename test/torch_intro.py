# John Eargle (mailto: jeargle at gmail.com)
# torch_intro

import os

import numpy
import pandas as pd
import torch


def ch_2_2():
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

    # CSV handling
    data_file = os.path.join('data', 'house_tiny.csv')
    data = pd.read_csv(data_file)
    print(data)

    # Data prep
    inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
    #   Split categories into separate dimensions (one-hot).
    inputs = pd.get_dummies(inputs, dummy_na=True)
    print('inputs:')
    print(inputs)

    #   Fill NA cells with mean values.
    inputs = inputs.fillna(inputs.mean())
    print('inputs:')
    print(inputs)

    #   Conversion to tensors.
    X = torch.tensor(inputs.to_numpy(dtype=float))
    y = torch.tensor(targets.to_numpy(dtype=float))
    print('X:')
    print(X)
    print('y:')
    print(y)


def ex_2_2():
    pass


def ch_2_3():
    # Linear algebra
    # Scalar
    x = torch.tensor(3.0)
    y = torch.tensor(2.0)
    # x + y, x - y, x * y, x / y, x ** y
    print('x:')
    print(x)
    print('y:')
    print(y)
    print('x + y:')
    print(x + y)
    print('x - y:')
    print(x - y)
    print('x * y:')
    print(x * y)
    print('x / y:')
    print(x / y)
    print('x ** y:')
    print(x ** y)

    # Vector
    x = torch.arange(3)
    print('x:')
    print(x)
    print('x[2]:')
    print(x[2])
    print('len(x):')
    print(len(x))
    print('x.shape:')
    print(x.shape)

    # Matrix
    A = torch.arange(6).reshape(3, 2)
    print('A:')
    print(A)
    print('A.T:')
    print(A.T)
    A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
    print('A:')
    print(A)
    print('A == A.T:')
    print(A == A.T)

    # Tensor
    B = torch.arange(24).reshape(2, 3, 4)
    print('B:')
    print(B)

    #   Arithmetic
    A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    B = A.clone()  # Assign a copy of A to B by allocating new memory
    print('A:')
    print(A)
    print('B:')
    print(B)
    print('A + B:')
    print(A + B)
    print('A * B:')
    print(A * B)
    a = 2
    X = torch.arange(24).reshape(2, 3, 4)
    print('a:')
    print(a)
    print('X:')
    print(X)
    print('a + X:')
    print(a + X)
    print('a * X:')
    print(a * X)
    print('(a * X).shape:')
    print((a * X).shape)

    #   Reduction
    x = torch.arange(3, dtype=torch.float32)
    print('x:')
    print(x)
    print('x.sum():')
    print(x.sum())

    print('A:')
    print(A)
    print('A.shape:')
    print(A.shape)
    print('A.sum():')
    print(A.sum())

    print('A.sum(axis=0).shape:')
    print(A.sum(axis=0).shape)
    print('A.sum(axis=1).shape:')
    print(A.sum(axis=1).shape)
    print('A.sum(axis=[0, 1]) == A.sum():')
    print(A.sum(axis=[0, 1]) == A.sum())

    print('A.mean():')
    print(A.mean())
    print('A.sum() / A.numel():')
    print(A.sum() / A.numel())

    print('A.mean(axis=0):')
    print(A.mean(axis=0))
    print('A.sum(axis=0) / A.shape[0]:')
    print(A.sum(axis=0) / A.shape[0])

    #   Non-reduction sum
    sum_A = A.sum(axis=1, keepdims=True)
    print('sum_A:')
    print(sum_A)
    print('sum_A.shape')
    print(sum_A.shape)

    print('A / sum_A:')
    print(A / sum_A)

    print('A.cumsum(axis=0):')
    print(A.cumsum(axis=0))

    #   Dot products

    #   Matrix*vector products

    #   Matrix*matrix products

    #   Norms


def ex_2_3():
    pass


def ch_2():
    ch_2_2()
    ex_2_2()

    ch_2_3()
    ex_2_3()


if __name__=='__main__':

    print('***************************')
    print('*** ONTHEDL TORCH INTRO ***')
    print('***************************')

    # ====================
    # Chapter 2
    # ====================

    ch_2()
