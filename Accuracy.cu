#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
// TODO: Desarrollar una mejor version en python para compararlos bien
// TODO: Documentar el codigo
// TODO: Probar un problema real con el codigo
// TODO: Optimizar rendimiento general

/*!
* \fn       __global__ void accuracyScore(float *y_pred, float *y_true,  float *accuracies, int nx, int ny);
* \brief    Compute the accuracy score
* \param    float *y_pred: array of predicted values
* \param    float *y_true: array of the target values
* \param    float *accuracies: array that stores the calculated accuracies
* \param    int nx: number of columns in y_pred and y_true
* \param    int ny: number of rows in y_pred
* \date     /////
* \author   /////
* \file     GsgpCuda.cpp
*/

__global__ void accuracyScore(float *y_pred, float *y_true,  float *accuracies, int nx, int ny)
{
    // Calculate global coordinates of the thread in the 2D block
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int tid = iy * nx + ix;

    // Check if the index is within the data bounds
    if (tid < nx * ny && ix < nx && iy < ny)
    {
        // Compare predicted and true values (TP, TN)
        if (y_pred[tid] == y_true[ix])
        {
            // Calculate the contribution to accuracy and add it to the accuracies array
            float sum = 1 / (float)nx;
            atomicAdd(&accuracies[iy], sum);
        }
    }
}

void FillingMatrices(float *matrix, int n, int m)
{
    for (int i = 0; i < n; i++)
        for (int e = 0; e < m; e++)
            matrix[i * m + e] = 1;
}
void Predictions(float *vector, int m, float num)
{
    for (int i = 0; i < m; i++)
    {
        if (i % 2 == 0)
            vector[i] = num;
        else
            vector[i] = 1;
    }
}
void VectorVacio(float *vector, int m, float num)
{
    for (int i = 0; i < m; i++)
    {
        vector[i] = num;
    }
}
void PrintVect(float *vect, int ny)
{
    printf("[");
    for (int i = 0; i < ny; i++)
    {
        if (i != ny - 1)
            printf("%f, ", vect[i]);
        else
            printf("%f", vect[i]);
    }
    printf("]\n");
}

int main()
{
    // Set up dimensions
    int ny = 1;
    int nx = 8;
    int nm = ny * nx;

    // Memory size
    int nBytesPredictions = nm * sizeof(float);
    int nBytesTargetValues = nx * sizeof(float);
    int nBytesAccuracy = ny * sizeof(float);

    // Host memory allocation
    float *predictions, *targetValues, *h_accuracy;

    h_accuracy = (float *)malloc(nBytesAccuracy);
    cudaMallocManaged((void **)&predictions, nBytesPredictions);
    cudaMallocManaged((void **)&targetValues, nBytesTargetValues);

    // Device memory allocation
    float *d_accuracy;
    cudaMalloc((void **)&d_accuracy, nBytesAccuracy);

    // Host memory initialization
    predictions[0] = 3;
    predictions[1] = 1;
    predictions[2] = 2;
    predictions[3] = 1;
    predictions[4] = 2;
    predictions[5] = 3;
    predictions[6] = 2;
    predictions[7] = 1;

    targetValues[0] = 1;
    targetValues[1] = 3;
    targetValues[2] = 2;
    targetValues[3] = 3;
    targetValues[4] = 1;
    targetValues[5] = 2;
    targetValues[6] = 3;
    targetValues[7] = 1;

    VectorVacio(h_accuracy, ny, 0);

    // Memory transfer host to device
    cudaMemcpy(d_accuracy, h_accuracy, nBytesAccuracy, cudaMemcpyHostToDevice);

    // Kernell call
    int dimx = 32;
    int dimy = 32;

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    accuracyScore<<<grid, block>>>(predictions, targetValues,d_accuracy, nx, ny);
    cudaDeviceSynchronize();

    // Memory transfer device to host
    cudaMemcpy(h_accuracy, d_accuracy, nBytesAccuracy, cudaMemcpyDeviceToHost);

    PrintVect(predictions, nx);
    PrintVect(targetValues, nx);
    PrintVect(h_accuracy, ny);

    // Reset device
    cudaDeviceReset();
}
