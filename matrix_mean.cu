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


__global__ void neighborAverage(const float* A, float* B, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n ){
        float left =   (j > 0)   ? A[i * n + j - 1] : 0; 
        float right =  (j < n-1) ? A[i * n + j + 1] : 0;
        float top =    (i > 0)   ? A[(i-1) * n + j] : 0; 
        float bottom = (i < n-1) ? A[(i+1) * n + j] : 0; 

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

    // allocate memory for host arrays
    float* h_A = new float[N * N];
    float* h_B = new float[N * N];

    // random engine and distribution
    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_real_distribution<> dist(0.0, 100.0); // range: [0, 100]

    // initialise the array
    for (int i = 0; i < N*N; i++)
            h_A[i] = dist(gen);

    // allocate memory on device
    float* d_A;
    float* d_B;
    cudaMalloc(&d_A, sizeof(float) * N * N);
    cudaMalloc(&d_B, sizeof(float) * N * N);

    // copy initial matrix to device
    cudaMemcpy(d_A, h_A, sizeof(float) * N * N, cudaMemcpyHostToDevice);

    // kernel launch thread layout
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                  (N + blockSize.y - 1) / blockSize.y);

    // launch kernel
    neighborAverage<<<gridSize, blockSize>>>(d_A, d_B, N);

    // copy new matrix from device to host
    cudaMemcpy(h_B, d_B, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
    
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
    
    // delete all matrices
    delete[] h_A;
    delete[] h_B;
    delete[] h_B_cpu;

    // free cuda memory
    cudaFree(d_A);
    cudaFree(d_B);
}