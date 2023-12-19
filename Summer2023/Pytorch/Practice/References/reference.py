import torch
import pandas as pd
import numpy as np
import matplotlib as plt

# print("Torch: " + str(torch.__version__))
# print("Numpy: " + str(np.__version__))
# print("Pandas: " + str(pd.__version__))
# print("MatPLotLib: " + str(plt.__version__) + "\n")

#! Manually creating tensors
def manual():
    #? For 1xk dimension matrices, the 1 is not read in tensor.shape()

    tensor1 = torch.tensor([1, 2, 3])
    tensorARange = torch.arange(0, 100, 10) #* torch.arange(MIN, MAX, STEPS)
    print(f"This gives us a 1 dimensional tensor from MIN up to MAX, with intervals of STEPS {tensorARange}")


#! Tensor attributes
def attributes():
    tensor1 = torch.tensor([1, 2, 3])
    print("                 Attributes\n")
    print(f"Tensor: {tensor1}, shape: {tensor1.shape}, dtype: {tensor1.dtype}, device: {tensor1.device}, dimensions: {tensor1.ndim}")

    tensor2 = torch.tensor([[1, 2, 3],
                        [3, 4, 5]])
    print(f"Tensor: {tensor2}, shape: {tensor2.shape}, dtype: {tensor2.dtype}, device: {tensor2.device}, dimensions: {tensor2.ndim}")

    tensor3 = torch.tensor([[[1, 2, 3],
                        [3, 4, 5]],

                        [[6, 7, 8],
                        [9, 10, 11]],

                        [[1, 1, 1],
                            [1, 1, 1]]])
    print(f"Tensor: {tensor3}, shape: {tensor3.shape}, dtype: {tensor3.dtype}, device: {tensor3.device}, dimensions: {tensor3.ndim}")


#! Transposing a matrix
def transpose():
    tensor2 = torch.tensor([[1, 2, 3],
                        [3, 4, 5]])
    print("                 Transposing\n")
    print(f"Original: {tensor2} and Transposed: {tensor2.T}")
    print(f"{tensor2.T.shape}")


#! Matrix Multiplication - Dot Product
def matmul():
    tensor2 = torch.tensor([[1, 2, 3],
                        [3, 4, 5]])
    print("                 Matmul\n")
    print(tensor2.matmul(tensor2.T))
    print(f"Original: {tensor2.dtype}, New: {tensor2.to(torch.float64).dtype}")


#! To operator and addition, and types
def to_and_addition():
    tensor2 = torch.tensor([[1, 2, 3],
                        [3, 4, 5]])
    #? Matrix addition and the "to" operator (changes attributes), Floats take priority over ints
    #! Note: the "to" operation only changes the attribute temporarily
    print(f"Matrix Addition Float16 + Int64: {tensor2.to(torch.float16) + tensor2}")
    print(f"Matrix Addition Int64 + Float16: {tensor2 + tensor2.to(torch.float16)}")
    print(f"{tensor2.dtype}")


#! Creating Random Matrices
def rand():
    print("                 Random Matrices\n")
    print(f"Random Matrix: {torch.rand(2, 2, 3)}")
    print(f"Matrix Ones: {torch.ones(2, 2, 3)}")
    print(f"Matrix Zeros: {torch.zeros(2, 2, 3)}")


#! The "_like" operation, creates tensor with same dimensions
def like():
    someInput = torch.rand(4, 2, 2)
    print(f"Matrix Same size as some Input: {torch.rand_like(someInput)}")
    print(f"Matrix Zeros Same size as some Input: {torch.zeros_like(someInput)}")
    print(f"Matrix Ones Same size as some Input: {torch.ones_like(someInput)}")


#! Tensor Aggregation
def aggregation():
    #? Min, Max, Mean, Sum
    range = torch.arange(0, 100, 10)

    #* tensor.mean() only works with floats or complex dtypes
    print(f"Min: {range.min()}, Max: {range.max()}, Range: {range.to(torch.float16).mean()}, Sum: {range.sum()}")

    #? Argmin, Argmax
    print(f"Argmin: {range.argmin()}, Argmax: {range.argmax()}")


#! rehspaing, viewing, stacking, squeezing, unsqueezing, permutating
def reshape():
    print("             Reshaping: a tensor to a defined shape\n")
    t = torch.rand(2, 2, 3)
    print(t.reshape(6, 2))

    print("             Viewing: returns a view of an input tensor of a certain shape but keeps the same memory as original\n")
    print(t.view(4, 3))

    print("             Stacking: combines multiple tensors on top of each other\n")
    print(torch.stack([t, t, t, t]))

    print("             Squeeze: removes all '1' dimensions from a tensor\n")
    tens = torch.rand(1, 3, 4, 2, 6, 1, 1)
    print(tens.squeeze())

    print("             Unsqueeze: Adds a 1 dimension to a target tensor\n")
    print(tens.unsqueeze(3))

    print("             Permute: return a view of the input with dimensions swapped in a certain way\n")
    print(tens.permute(6, 5, 4, 3, 2, 1, 0))


#! Indexing
def indexing():
    x = torch.arange(1, 28).reshape(3, 3, 3)
    print(f"{x}\n")
    print(x[:, 0])
    print(x[:, :, x.shape[0] - 2:])

    #? x.shape[n] prints the magnitude of the n'th dimension: easier to find length
    #? x.shape[0] prints 1, x.shape[1] prints 2, x.shape[2] prints 3. 
    x = torch.arange(1, 7).reshape(1, 2, 3)


#! Manual seed, taking random out of randomness.
#! Makes the randomness reproducable
def manSeed():
    torch.manual_seed(42) #? Sets the random seed
    tensorA = torch.rand(3, 3)
    torch.manual_seed(42) #? Sets the random seed
    tensorB = torch.rand(3, 3)
    print(tensorA, tensorB)
    print(tensorA == tensorB)

#! Tensors on GPUS (faster computation) thanks to CUDA

indexing()