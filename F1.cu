#include<stdlib.h>
#include<cuda_runtime.h>
#include<stdio.h>
#include <unordered_set>
#include <unordered_map>
#include <random>






__global__ void getF1(float* TP, float* FP, float* FN, float* trueValuesByClass, int* noClasses, float* samplesPerClass, float* F1_Macro, float* F1_Weighted, int noTargetValues, int noIndividuals){
 int idx = (blockDim.x * blockDim.y * blockIdx.y * gridDim.x) + (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
 
 if (idx < *noClasses * noIndividuals){
    int individual;
    if ((idx - noIndividuals) <= 0) individual = 0; else individual = idx - noIndividuals;
    atomicAdd(&F1_Macro[individual], ( TP[idx] / (TP[idx] + 0.5 * (FP[idx] + FN[idx])) ));
    atomicAdd(&F1_Weighted[individual], ( (samplesPerClass[idx - individual * (*noClasses)] / noTargetValues) * (TP[idx] / (TP[idx] + 0.5 * (FP[idx] + FN[idx]))) ));  
 }
   
}




__global__ void getTpFpFn(float* y_true, float* y_pred, int n, int m, int noClasses, float* y_trueEachClass, float* TP, float* FP, float* FN){
    int idx = (blockDim.x * blockDim.y * blockIdx.y * gridDim.x) + (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    if (idx < m * n) {
        int individual;
        printf("\n%i - %i\n", idx, n);
        if ((idx - n) <=0)
            individual = 0;
        else
            individual = idx - n;
        
        for (int i = 0; i < noClasses; i++){
            if(y_pred[idx] == y_trueEachClass[i] && y_true[idx - individual * n] == y_trueEachClass[i]) { atomicAdd(&TP[i + individual * noClasses], 1); }
            if(y_pred[idx] == y_trueEachClass[i] && y_true[idx - individual * n] != y_trueEachClass[i]) { atomicAdd(&FP[i + individual * noClasses], 1); }
            if(y_pred[idx] != y_trueEachClass[i] && y_true[idx - individual * n] == y_trueEachClass[i]) { atomicAdd(&FN[i + individual * noClasses], 1); }
        }
    }
}


__global__ void getNoClassesKernel(float* trueValues, int* noClasses, float* temp, int* count, int* i){
 if(trueValues[threadIdx.x] == 0) *noClasses = *noClasses + 1;
 atomicAdd(count, 0);
 if(trueValues[threadIdx.x] != 0) {
     *noClasses = *noClasses + 1;
     *temp = trueValues[threadIdx.x];
 }
 atomicAdd(count, 0);
 for ( ; *i < 1; *i = *i + 1)
 {
     if(trueValues[threadIdx.x] == *temp) trueValues[threadIdx.x] = 0;
     if(trueValues[threadIdx.x] != 0) {
         *noClasses = *noClasses + 1;
         *temp = trueValues[threadIdx.x];
         *i = *i - 1;
     }
     atomicAdd(count, 0);
 }
}

__global__ void getClasses(float* trueValues, int* noClasses, float* temp, int* count, float* trueValuesByClass, int* i){
 if(trueValues[threadIdx.x] == 0) { trueValuesByClass[0] = 0; *i = *i + 1; }
 atomicAdd(count, 0);
 for ( ; *i < *noClasses; *i = *i + 1)
 {
     if(trueValues[threadIdx.x] != 0) *temp = trueValues[threadIdx.x];
     atomicAdd(count, 0);
     trueValuesByClass[*i] = *temp;
     if(trueValues[threadIdx.x] == *temp) trueValues[threadIdx.x] = 0;
     atomicAdd(count, 0);
 }
}

__global__ void getSamplesPerClass(float* trueValues, int* noClasses, int* count, float* trueValuesByClass, float* samplesPerClass){
 for (int i = 0; i < *noClasses; i++) if(trueValues[threadIdx.x] == trueValuesByClass[i]) atomicAdd(&samplesPerClass[i], 1);
}

void getNoClasses(float* y_true, int m, int& noClasses){
 std::unordered_set<int> elementosUnicos;
 for (int i = 0; i < m; i++)
     elementosUnicos.insert(y_true[i]);             // Esta parte obtiene el No. clases
 noClasses = elementosUnicos.size();
}

void getVector(float* vector, int size){
 for (int i = 0; i < size; i++){
     if(i % 2 == 0) vector[i] = 1;
     else vector[i] = 2;
 } 
}

void getMatriz(float* matriz, int rows, int columns){
    for (int i = 0; i < rows; i++)
        for (int e = 0; e < columns; e++)
            matriz[i + e] = i;
    
}

void setClasses(float* y_true, int m, int& noClasses, float* y_trueEachClass){
 int temp = y_true[0];                                                 // Esta declaracion de temp nos servira para almacenar el valor de cada clase
 for (int i = 0; i < noClasses; i++)
 {
     y_trueEachClass[i] = temp;                                        // Con este algoritmo conseguimos llenar el array con longiud [no Classes]
     for (int e = 0; e < m; e++)                                       // con el valor dado de cada clase (ej. clase 0 = 45, clase 1 = 32)
        if (y_true[e] == y_trueEachClass[i]) y_true[e] = 0;
     for (int e = 0; e < m; e++)
         if (y_true[e] != 0) { temp = y_true[e]; e = m; } 
 }
}

void F1(float* y_pred, float* y_true, int widthTrue, int rowPred){
 float* y_trueDevice, * temp, * temp_d, * valuesByClass, * valuesByClasses_d, * samplesPerClass, * samplesPerClass_d, * y_predDevice, * TP, * FP, * FN, * TP_d, * FP_d, * FN_d;
 int* count, *noClasses, * i, * count_d, *noClasses_d, * i_d;




 temp = (float*)malloc(sizeof(float));
 count = (int*)malloc(sizeof(int));
 noClasses = (int*)malloc(sizeof(int));
 i = (int*)malloc(sizeof(int));




 *temp = 0;
 *count = 0;
 *noClasses = 0;
 *i = 0;


 cudaMalloc((void**)&y_trueDevice, widthTrue * sizeof(float));
 cudaMalloc((void**)&temp_d, sizeof(float));
 cudaMalloc((void**)&count_d, sizeof(int));
 cudaMalloc((void**)&noClasses_d, sizeof(int));
 cudaMalloc((void**)&i_d, sizeof(int));




 cudaMemcpy(y_trueDevice, y_true, widthTrue * sizeof(float), cudaMemcpyHostToDevice);
 cudaMemcpy(temp_d, temp, sizeof(float), cudaMemcpyHostToDevice);
 cudaMemcpy(count_d, count, sizeof(int), cudaMemcpyHostToDevice);
 cudaMemcpy(noClasses_d, noClasses, sizeof(int), cudaMemcpyHostToDevice);
 cudaMemcpy(i_d, i, sizeof(int), cudaMemcpyHostToDevice);




 getNoClassesKernel<<<1, widthTrue>>>(y_trueDevice, noClasses_d, temp_d, count_d, i_d);
 cudaDeviceSynchronize();



 




 cudaMemcpy(noClasses, noClasses_d, sizeof(int), cudaMemcpyDeviceToHost);
 valuesByClass = (float*)malloc(*noClasses * sizeof(float));
 cudaMalloc((void**)&valuesByClasses_d, *noClasses * sizeof(float));
 cudaMemcpy(y_trueDevice, y_true, widthTrue * sizeof(float), cudaMemcpyHostToDevice);
 cudaMemcpy(i_d, i, sizeof(int), cudaMemcpyHostToDevice);
   printf("NUMERO DE CLASES: %i", *noClasses);



 getClasses<<<1, widthTrue>>>(y_trueDevice, noClasses_d, temp_d, count_d, valuesByClasses_d, i_d);
 cudaDeviceSynchronize();
 cudaMemcpy(valuesByClass, valuesByClasses_d, *noClasses * sizeof(float), cudaMemcpyDeviceToHost);








 samplesPerClass = (float*)malloc(*noClasses * sizeof(float));
 cudaMalloc((void**)&samplesPerClass_d, *noClasses * sizeof(float));
 cudaMemcpy(y_trueDevice, y_true, widthTrue * sizeof(float), cudaMemcpyHostToDevice);




 getSamplesPerClass<<<1, widthTrue>>>(y_trueDevice, noClasses_d, count_d, valuesByClasses_d, samplesPerClass_d);
 cudaDeviceSynchronize();
 cudaMemcpy(samplesPerClass, samplesPerClass_d, *noClasses * sizeof(float), cudaMemcpyDeviceToHost);








 TP = (float*)malloc(*noClasses * sizeof(float));
 FP = (float*)malloc(*noClasses * sizeof(float));
 FN = (float*)malloc(*noClasses * sizeof(float));

dim3 block(32, 32);
dim3 grid((widthTrue + block.x - 1) / block.x, (rowPred + block.y - 1) / block.y);

 cudaMalloc((void**)&TP_d, *noClasses * sizeof(float));
 cudaMalloc((void**)&FP_d, *noClasses * sizeof(float));
 cudaMalloc((void**)&FN_d, *noClasses * sizeof(float));
 cudaMalloc((void**)&y_predDevice, widthTrue * sizeof(float));
 cudaMemcpy(y_predDevice, y_pred, widthTrue * sizeof(float), cudaMemcpyHostToDevice);
 getTpFpFn<<<grid, block>>>(y_trueDevice, y_predDevice, widthTrue, rowPred, *noClasses, valuesByClasses_d, TP_d, FP_d, FN_d);
 cudaDeviceSynchronize();


 cudaMemcpy(TP, TP_d, *noClasses * sizeof(float), cudaMemcpyDeviceToHost);
 cudaMemcpy(FP, FP_d, *noClasses * sizeof(float), cudaMemcpyDeviceToHost);
 cudaMemcpy(FN, FN_d, *noClasses * sizeof(float), cudaMemcpyDeviceToHost);
 
 
 
 float* F1_Macro, * F1_Macro_d;
 float* F1_Weighted, * F1_Weighted_d;
 F1_Macro = (float*)malloc(sizeof(float));
 F1_Weighted = (float*)malloc(sizeof(float));
 cudaMalloc((void**)&F1_Macro_d, rowPred * sizeof(float));
 cudaMalloc((void**)&F1_Weighted_d, rowPred * sizeof(float));

 dim3 grid1((*noClasses + block.x - 1) / block.x, (rowPred + block.y - 1) / block.y);

 getF1<<<1, grid1>>>(TP_d, FP_d, FN_d, valuesByClasses_d, noClasses_d, samplesPerClass_d, F1_Macro_d, F1_Weighted_d, widthTrue, rowPred);
 cudaDeviceSynchronize();
 cudaMemcpy(F1_Macro, F1_Macro_d, rowPred * sizeof(float), cudaMemcpyDeviceToHost);
 cudaMemcpy(F1_Weighted, F1_Weighted_d, rowPred * sizeof(float), cudaMemcpyDeviceToHost);
 F1_Macro[0] /= *noClasses;


 printf("F1 MACRO: %f, TP_1: %f, FP_1: %f, FN_1: %f, TP_2: %f, FP_2: %f, FN_2: %f, TP_3: %f, FP_3: %f, FN_3: %f", *F1_Macro, TP[0], FP[0], FN[0], TP[1], FP[1], FN[1], TP[2], FP[2], FN[2]);
 printf("\nF1 WEIGHTED: %f\n", *F1_Weighted);


}


int main(){ 
 // float* y_pred, * y_true, * y_trueEachClass;
 float y_pred[6] = {0, 2, 1, 0, 0, 1}; float y_true [6] = {0, 1, 2, 0, 1, 2};


 F1(y_pred, y_true, 6, 1);
}