#include <iostream>
#include <random>
#include <cstdlib> 
#include <string_view>


template<typename T>
void printSquareMatrix(T* mat, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            std::cout << mat[i * n + j] << " ";
        std::cout << "\n";
    }
}


float* neighborAverage(const float* A, int n)
{
    // create a placeholder for a new array
    float* B = new float[n * n];

    // initialise new array with means of old array
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            float left =   (j > 0)   ? A[i * n + j - 1] : 0; 
            float right =  (j < n-1) ? A[i * n + j + 1] : 0;
            float top =    (i > 0)   ? A[(i-1) * n + j] : 0; 
            float bottom = (i < n-1) ? A[(i+1) * n + j] : 0; 

            B[i * n + j] = float(left + right + top + bottom) / 4;
        }
    }

    return B;
}


__global__ void neighborAverageMulti(const float* A, float* B, int n, int start_col, int num_cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < num_cols ){
        int gj = start_col + j; // global column index
        float left =   (gj > 0)   ? A[i * n + gj - 1] : 0; 
        float right =  (gj < n-1) ? A[i * n + gj + 1] : 0;
        float top =    (i > 0)    ? A[(i-1) * n + gj] : 0; 
        float bottom = (i < n-1)  ? A[(i+1) * n + gj] : 0; 

        B[i * n + j] = (left + right + top + bottom) / 4;
    }
}


int main(int argc, char *argv[])
{
    // get matrix size N from cli arg
    int N {atoi(argv[1])};

    // arg to print matrices    
    bool print = false;
    if (argc >= 3){
        using namespace std::literals;
        if (argv[2] == "print"sv)
            print = true;
    }

    // allocate memory for host matrices
    float* h_A = new float[N * N];
    float* h_B = new float[N * N];

    // random engine and distribution
    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_real_distribution<> dist(0.0, 100.0); // range: [0, 100]

    // initialise the source matrix parts
    for (int i = 0; i < N*N; i++){
        h_A[i] = dist(gen);
    }

    // split settings
    const int split = N / 2;
    const int cols0  = split;
    const int cols1  = N - split;

    // allocate memory on devices and copy to gpus
    float *d_A0, *d_A1, *d_B0, *d_B1;

    cudaSetDevice(0);
    cudaMalloc(&d_A0, sizeof(float) * N * N);
    cudaMalloc(&d_B0, sizeof(float) * N * cols0);
    cudaMemcpy(d_A0, h_A, sizeof(float) * N * N, cudaMemcpyHostToDevice);

    cudaSetDevice(1);
    cudaMalloc(&d_A1, sizeof(float) * N * N);
    cudaMalloc(&d_B1, sizeof(float) * N * cols1);
    cudaMemcpy(d_A1, h_A, sizeof(float) * N * N, cudaMemcpyHostToDevice);

    // thread layout on each gpu
    dim3 blockSize(16, 16);
    dim3 grid0((cols0 + blockSize.x - 1) / blockSize.x,
               (N     + blockSize.y - 1) / blockSize.y);
    dim3 grid1((cols1 + blockSize.x - 1) / blockSize.x,
               (N     + blockSize.y - 1) / blockSize.y);

    // launch kernels on each gpu
    cudaSetDevice(0);
    neighborAverageMulti<<<grid0, blockSize>>>(d_A0, d_B0, N, 0, cols0);

    cudaSetDevice(1);
    neighborAverageMulti<<<grid1, blockSize>>>(d_A1, d_B1, N, split, cols1);

    // sync both
    cudaSetDevice(0); cudaDeviceSynchronize();
    cudaSetDevice(1); cudaDeviceSynchronize();

    // copy back each matrix half
    float* h_B0 = new float[N * cols0];
    float* h_B1 = new float[N * cols1];

    cudaSetDevice(0);
    cudaMemcpy(h_B0, d_B0, sizeof(float) * N * cols0, cudaMemcpyDeviceToHost);
    cudaSetDevice(1);
    cudaMemcpy(h_B1, d_B1, sizeof(float) * N * cols1, cudaMemcpyDeviceToHost);

    // merge halfs into h_B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < cols0; j++)
            h_B[i*N + j] = h_B0[i*cols0 + j];
        for (int j = 0; j < cols1; j++)
            h_B[i*N + (split + j)] = h_B1[i*cols1 + j];
    }

    // print initial and new matrices
    if (print == true){
        std::cout << "Original matrix:\n";
        printSquareMatrix(h_A, N);
        std::cout << "New matrix:\n";
        printSquareMatrix(h_B, N);
    }

    // perform same calculations on cpu
    float* h_B_cpu = neighborAverage(h_A, N);

    // check cpu vs gpu computation errors
    float maxError = 0.0f;
    for (int i = 0; i < N * N; i++)
        maxError = fmax(maxError, fabs(h_B[i]-h_B_cpu[i]));
    std::cout << "Max error: " << maxError << std::endl;

    // delete and free
    delete[] h_A; delete[] h_B; delete[] h_B_cpu;
    delete[] h_B0; delete[] h_B1;

    cudaSetDevice(0);
    cudaFree(d_A0); cudaFree(d_B0);
    cudaSetDevice(1);
    cudaFree(d_A1); cudaFree(d_B1);
}