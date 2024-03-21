import numpy as np
import pandas as pd
import torch 

def basic_config_check():
    print(f"Numpy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Torch version: {torch.__version__}")
    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")
    
    print(f"Index of CUDA devices: {torch.cuda.current_device()}")
    print(f"Number of CUDA devices: {torch.cuda.current_device() + 1}")
    print(f"CUDA device name: {torch.cuda.get_device_name()}")
    print(f"Allocated memory to GPU: {torch.cuda.memory_allocated()}")



def tensors_basics():
    '''
    A tensor is a generalization of vectors and matrics and is easily understood as a multidimentional array. It is a term and seet of techniques known in machine learning in the training and operaion of deep learning models can be described in terms of tensors. In many cases tensors are used as a replacement of NumPy to use the power of GPUs.

    Tensors are a type of data structure used in linear algebra, and like vectors and matrices, you can calculate arithmetic operations with tensors.
    '''
    # create a list and make a numpy array to the list
    lst = [3, 4, 5, 6]
    arr = np.array(lst)
    print(f"Array data type: {arr.dtype}")
  

    # Convert Numpy to PyTorch Tensors
    tensors = torch.from_numpy(arr)
    print(f"torch.from_numpy = {tensors}")
    
    # Indexing similar to numpy
    print(tensors[:2])
    # Print a single value
    print(tensors[2])
    # Print a range
    print(tensors[1:4])

    # Disadvantage of from_numpy. The numpy array and tensor uses the same memory location.
    tensors[3] = 100
    print(tensors)
    # This change will also happen in the numpy array!!!
    print(arr)

    # Prevent this by using torch.tensor
    tensor_arr = torch.tensor(arr)
    print(f"torch.tensor = {tensor_arr}")
    print(arr)

    # Changing the value
    tensor_arr[3] = 20
    print(tensor_arr)
    print(arr)

    # Zeros and Ones
    zero = torch.zeros((2, 3), dtype=torch.float64)
    print(zero)

    ones = torch.ones((2, 3), dtype=torch.float64)
    print(ones)

    # Two dim arrays
    arr2 = np.arange(0, 15).reshape(5, 3)
    print(arr2)

    # Convert it into tensor
    tensor2 = torch.tensor(arr2)
    print(tensor2)

    tensor2[0][0] = 1
    print(tensor2)
    print(arr2)


    # Arithmetic Operation
    a = torch.tensor([3, 4, 5], dtype=torch.float)
    b = torch.tensor([3, 5, 6], dtype=torch.float)

    # Add
    print(f"Addition = {a + b}")

    # Resultant addition in different array
    c = torch.zeros(3) # Maintaining the output shape
    torch.add(a, b, out=c)
    print(f"c = {c}")


    # Some more operation
    res = torch.add(a, b).sum()
    print(f"Sum of the a + b = {res}")

    # Dot Products and Mult Operations
    x = torch.tensor([3, 4, 5], dtype=torch.float)
    y = torch.tensor([4, 5, 6], dtype=torch.float)
    product = x.mul(y)
    print(f"Multiplication of x and y = {product}")

    dot_prod = x.dot(y)
    print(f"Dot product of x and y = {dot_prod}") # a * x + b * y + c * z = dot_prod


    # Matrix multiplication
    x = torch.tensor([[1, 4, 2], [1, 5, 5]], dtype=torch.float)
    y = torch.tensor([[5, 7], [8, 6], [9, 11]], dtype=torch.float)

    matrix_mul = torch.matmul(x, y)
    print(matrix_mul)

    # same operation with another functions
    matrix_mul2 = torch.mm(x, y)
    print(matrix_mul2)

    matrix_mul3 = x@y 
    print(matrix_mul3)

if __name__ == "__main__":
    basic_config_check()
    print("===================================================")
    tensors_basics()



