#include<stdlib.h>
#include<cuda_runtime.h>
#include<stdio.h>
#include <unordered_set>
#include <unordered_map>
#include <random>

__global__ void getNumClasses(float* true_labels, int* num_classes, float* temp_class, int* atomic_sync, int* loop_variable, int num_true_labels){
    int idx = (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    if (idx < num_true_labels)
    {
        *temp_class = 0; *num_classes = 0; *loop_variable = 0; // Initialize some variables
        atomicAdd(atomic_sync, 0); // An atomic operation is made in order to every thread in the grid knows the changes done
        // All threads in the grid check if there is some value equal to 0 in the true labels, if it's true, then the number of classes added one to its counting
        if(true_labels[idx] == 0) *num_classes = *num_classes + 1;
        atomicAdd(atomic_sync, 0); // An atomic operation is made in order to every thread in grid knows that the number of classes has increased
        if(true_labels[idx] != 0) { // Then, all threads in the grid look for a different value than 0, regardless of whichever each one find, it's stored and the number of classes increases
            *num_classes = *num_classes + 1;
            *temp_class = true_labels[idx];
        }
        atomicAdd(atomic_sync, 0); // An atomic operation is made in order to every thread in the grid knows the changes done
        for ( ; *loop_variable < 1; *loop_variable = *loop_variable + 1) // The loop ends when loop variable reaches 1
        {
            if(true_labels[idx] == *temp_class) true_labels[idx] = 0; // Using the last value different than 0, all threads look for it and rewrite its value as 0
            if(true_labels[idx] != 0) { // If there is another value different than 0, store it and loop variable decrease by 1 
            *num_classes = *num_classes + 1;
            *temp_class = true_labels[idx];
            *loop_variable = *loop_variable - 1;
        }
        atomicAdd(atomic_sync, 0); // An atomic operation is made in order to every thread in the grid knows the changes done
    }
 }
}

__global__ void getClasses(float* true_labels, int* num_classes, float* temp_class, int* atomic_sync, float* class_labels, int* loop_variable, int num_true_labels){
    int idx = (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    if (idx < num_true_labels)
    {
        *temp_class = 0; *loop_variable = 0; // Initialize some variables
        atomicAdd(atomic_sync, 0); // An atomic operation is made in order to every thread in the grid knows the changes done
        // All threads in the grid check if there is some value equal to 0 in the true labels, if it's true, 0 is going to be added to class labels array, and loop variable increase 1
        if(true_labels[threadIdx.x] == 0) { class_labels[0] = 0; *loop_variable = *loop_variable + 1; } 
        atomicAdd(atomic_sync, 0); // An atomic operation is made in order to every thread in the grid knows the changes which have been made in both true labels and class labels                                                                   
                                                                                                 
        for ( ; *loop_variable < *num_classes; *loop_variable = *loop_variable + 1) // The loop ends when loop variable reaches the number of classes - 1                      
        {
            if(true_labels[threadIdx.x] != 0) { *temp_class = true_labels[threadIdx.x]; class_labels[*loop_variable] = *temp_class; } // All threads look for any different value than 0, then it's stored in class labels array
            atomicAdd(atomic_sync, 0); // An atomic operation is made in order to every thread in the grid knows the changes done
            if(true_labels[threadIdx.x] == *temp_class) true_labels[threadIdx.x] = 0; // All threads in the grid look for the value which has already stored in class labels array, and rewrite it as 0
            atomicAdd(atomic_sync, 0); // An atomic operation is made in order to every thread in the grid knows the changes done
        }
    }
}

__global__ void getSamplesPerClass(float* true_labels, int* num_classes, int* atomic_sync, float* class_labels, float* samples_per_class, int num_true_labels){
    int idx = (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    if (idx < num_true_labels)
    {
        for (int i = 0; i < *num_classes; i++) // The loop ends when i reaches the number of classes
        if(true_labels[threadIdx.x] == class_labels[i]) atomicAdd(&samples_per_class[i], 1); // In each iteration, threads look for coincendeces between true labels and class labels arrays and count them
    }
}

__global__ void getTpFpFn(float* true_labels, float* predicted_labels, int num_true_labels, int num_individuals, int num_classes, float* class_labels, float* true_positives, float* false_positives, float* false_negatives){
    int idx = (blockDim.x * blockDim.y * blockIdx.y * gridDim.x) + (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    if (idx < num_individuals * num_true_labels) {
        int individual;
        // Every certain number of threads (num_true_labels) represents an individual, so its predictions. This operations are made in order to identify the number of individual every "group" is 
        if (idx < num_true_labels) individual = 0; else individual = ((idx - num_true_labels) / num_true_labels) + 1;
        if (idx < num_individuals * num_classes) { true_positives[idx] = 0; false_negatives[idx] = 0; false_positives[idx] = 0; }
        for (int i = 0; i < num_classes; i++){ // This loop (0 - number of classes) counts the total number of true positives, false positives and false negatives for each class
            if(predicted_labels[idx] == class_labels[i] && true_labels[idx - individual * num_true_labels] == class_labels[i]) { atomicAdd(&true_positives[i + individual * num_classes], 1); }
            if(predicted_labels[idx] == class_labels[i] && true_labels[idx - individual * num_true_labels] != class_labels[i]) { atomicAdd(&false_positives[i + individual * num_classes], 1); }
            if(predicted_labels[idx] != class_labels[i] && true_labels[idx - individual * num_true_labels] == class_labels[i]) { atomicAdd(&false_negatives[i + individual * num_classes], 1); }
        }
    }
}

__global__ void f1(float* true_positives, float* false_positives, float* false_negatives, float* class_labels, int* num_classes, float* samples_per_class, float* macro_f1_score, float* weighted_f1_score, int num_true_labels, int num_individuals){
    int idx = (blockDim.x * blockDim.y * blockIdx.y * gridDim.x) + (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
 
    if (idx < *num_classes * num_individuals)
    {
        int individual;
        // Every certain number of threads (num_true_labels) represents an individual, so its predictions. This operations are made in order to identify the number of individual every "group" is 
        if (idx < *num_classes) individual = 0; else individual = ((idx - num_true_labels) / num_true_labels) + 1; 
        // Here all threads contribute to the macro and weighted f1 score for each individual
        atomicAdd(&macro_f1_score[individual], ( true_positives[idx] / (true_positives[idx] + 0.5 * (false_positives[idx] + false_negatives[idx])) ));
        atomicAdd(&weighted_f1_score[individual], ( (samples_per_class[idx - individual * (*num_classes)] / num_true_labels) * (true_positives[idx] / (true_positives[idx] + 0.5 * (false_positives[idx] + false_negatives[idx]))) ));  
        // Since macro f1 must be divided by the number of classes, it is convenient that that operation is made by a different kernel
    }
}

__global__ void getMacro(float* macro_f1_score, int* num_classes, int num_individuals){
    int idx = (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    // Here is just made the division which was missing
    if (idx < num_individuals)
        macro_f1_score[idx] /= *num_classes;
}

// Since there are a lot of kernel calls and dinamically store statements, it's necessary to do all of this from the device and then just doing all that suff
void getF1(float* predicted_labels, float* true_labels, int num_individuals, int num_true_labels){
    // Declaration of all variables that kernles will need
    float* true_labels_d,* temp_d, * class_labels_d, * samples_per_class_d, * predicted_labels_d, * macro_f1_score_d, * weighted_f1_score_d, * true_positives_d, * false_positives_d, * false_negatives_d;
    float* macro_f1_score, * weighted_f1_score;
    int* atomic_sync_d, * num_classes_d, *loop_variable_d;
    int* num_classes;

    // Grid's dimensions
    dim3 block(32, 32);
    dim3 grid((num_true_labels + block.x - 1) / block.x);

    // Memory allocation
    cudaMalloc((void**)&true_labels_d, num_true_labels * sizeof(float));
    cudaMalloc((void**)&predicted_labels_d, num_individuals * num_true_labels * sizeof(float));
    cudaMalloc((void**)&num_classes_d, sizeof(int));
    cudaMalloc((void**)&atomic_sync_d, sizeof(int));
    cudaMalloc((void**)&loop_variable_d, sizeof(int));
    cudaMalloc((void**)&temp_d, sizeof(float));
    num_classes = (int*)malloc(sizeof(int));

    // Transfering data
    cudaMemcpy(true_labels_d, true_labels, num_true_labels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(predicted_labels_d, predicted_labels, num_individuals * num_true_labels * sizeof(float), cudaMemcpyHostToDevice);



    // Getting the total number of classes
    getNumClasses<<<grid, block>>>(true_labels_d, num_classes_d, temp_d, atomic_sync_d, loop_variable_d, num_true_labels);
    cudaDeviceSynchronize();



    // Transfering the total number of classes to host
    cudaMemcpy(num_classes, num_classes_d, sizeof(int), cudaMemcpyDeviceToHost);

    // Allocating dinamically the arrays for classes' labels, samples per class, TP, FP, FN, macro F1 and weighted F1
    cudaMalloc((void**)&class_labels_d, *num_classes * sizeof(float));
    cudaMalloc((void**)&samples_per_class_d, *num_classes * sizeof(float));
    cudaMalloc((void**)&true_positives_d, num_individuals * *num_classes * sizeof(float));
    cudaMalloc((void**)&false_negatives_d, num_individuals * *num_classes * sizeof(float));
    cudaMalloc((void**)&false_positives_d, num_individuals * *num_classes * sizeof(float));
    cudaMalloc((void**)&weighted_f1_score_d, num_individuals * sizeof(float));
    cudaMalloc((void**)&macro_f1_score_d, num_individuals * sizeof(float));

    weighted_f1_score = (float*)malloc(num_individuals * sizeof(float));
    macro_f1_score = (float*)malloc(num_individuals * sizeof(float));

    // Transfering data
    cudaMemcpy(true_labels_d, true_labels, num_true_labels * sizeof(float), cudaMemcpyHostToDevice);

    // Getting classes' labels
    getClasses<<<grid, block>>>(true_labels_d, num_classes_d, temp_d, atomic_sync_d, class_labels_d, loop_variable_d, num_true_labels);
    cudaDeviceSynchronize();



    // Transfering data
    cudaMemcpy(true_labels_d, true_labels, num_true_labels * sizeof(float), cudaMemcpyHostToDevice);

    // Getting samples per class
    getSamplesPerClass<<<grid, block>>>(true_labels_d, num_classes_d, atomic_sync_d, class_labels_d, samples_per_class_d, num_true_labels);
    cudaDeviceSynchronize();



    // Rearranging grid's dimentions
    dim3 grid2((num_true_labels + block.x - 1) / block.x, (num_individuals + block.y - 1) / block.y);

    // Getting Tp, fp, fn for each class and each individual
    getTpFpFn<<<grid2, block>>>(true_labels_d, predicted_labels_d, num_true_labels, num_individuals, *num_classes, class_labels_d, true_positives_d, false_positives_d, false_negatives_d);
    cudaDeviceSynchronize();



    // Rearranging grid's dimentions
    dim3 grid3((*num_classes + block.x - 1) / block.x, (num_individuals + block.y - 1) / block.y);

    // Getting both macro and weghted f1 | division for macro one is going to be missing
    f1<<<grid3, block>>>(true_positives_d, false_positives_d, false_negatives_d, class_labels_d, num_classes_d, samples_per_class_d, macro_f1_score_d, weighted_f1_score_d, num_true_labels, num_individuals);
    cudaDeviceSynchronize();


    // Rearranging grid's dimentions
    dim3 grid4((num_individuals + block.x - 1) / block.x);

    // Aplying the division for macro f1
    getMacro<<<grid4, block>>>(macro_f1_score_d, num_classes_d, num_individuals);


    // Transfering data back to host
    cudaMemcpy(macro_f1_score, macro_f1_score_d, num_individuals * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(weighted_f1_score, weighted_f1_score_d, num_individuals * sizeof(float), cudaMemcpyDeviceToHost);

    

    // Printing scores
    for (int i = 0; i < num_individuals; i++)
        printf("Individual %i, macro f1: %f, weighted f1: %f\n", i, macro_f1_score[i], weighted_f1_score[i]);

    // Deallocating memory for both device and host
    cudaFree(true_labels_d);
    cudaFree(predicted_labels_d);
    cudaFree(num_classes_d);
    cudaFree(macro_f1_score_d);
    cudaFree(weighted_f1_score_d);
    cudaFree(true_positives_d);
    cudaFree(false_negatives_d);
    cudaFree(false_positives_d);
    cudaFree(loop_variable_d);
    cudaFree(atomic_sync_d);
    cudaFree(samples_per_class_d);
    cudaFree(class_labels_d);
    cudaFree(temp_d);
}

int main(){
    float predicted_labels[12] = {0, 2, 1, 0, 0, 1, 1, 1, 0, 2, 2, 1}; float true_labels [6] = {0, 1, 2, 0, 1, 2};
    getF1(predicted_labels, true_labels, 2, 6);
}