import numpy as np
import pandas as pd
import torch 

'''
In backward propagation, we usually find out the derivatives or slopes. PyTorch has a wonderful way to compute derivatives (have in-build function).
Suppose, my equation is y = x^2 and I try to find out the derivative of y, that is dy/dx = 2x. 
'''

def back_prop():
    x = torch.tensor(4.0, requires_grad=True)
    print(x) 
    y = x**2
    print(y)


    # Back Propagation, y = 2*x
    y.backward()
    gradient = x.grad
    print(f"Gradient: {gradient}")

    # More complex
    lst = [[2.0, 3.0, 1.0], [4.0, 5.0, 3.0], [7.0, 6.0, 4.0]]
    torch_input = torch.tensor(lst, requires_grad=True)
    print(torch_input)

    ### y = x**3 + x**2
    y = torch_input**3 + torch_input**2
    print(y)

    z = y.sum()
    print(z)

    z.backward()
    grads = torch_input.grad
    print(grads)

    '''
    How the grad (backpropagation) works?
    - x**3 + x**2
    - 3*x**2 + 2x
    - 3*2**2 + 2*2 = 16 (tensor_input[0][0])
    - 3*2*3 + 2*3 = 33 (tensor_input[0][1])
    '''

if __name__ == '__main__':
    back_prop()