# About

For square matrix $A$ calculate matrix $B$ s.t. 

$B[i][j] = (A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1])/4$

using CUDA.

# Structure

- **matrix_mean.cu** - utilises 1 GPU.

- **matrix_mean_multigpu.cu** - utilises 2 GPUs.

- **single_gpu_2048dim.nsys-rep** - nsight profiler report for applying single-gpu algorithm to 2048x2048 matrix 

# Build 

`
nvcc <file.cu> -o <output_file>
`

# Launch

Each script accepts 2 CLI arguments. First one is obligatory. It sets matrix dimensionality. Second is optional and allows to print matrices A and B.

Launch for 4x4 matrix and print:

`
./matrix_mean 4 print
`

Launch for 2048x2048 matrix w/o printing

`
./matrix_mean 2048
`

