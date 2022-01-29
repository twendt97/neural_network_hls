// defines for compilation behavior
// #define TEST_MLP_CRANIUM
#define TEST_TRAINING

#include "KernelHelper.hpp"
#include "Simulation.hpp"
#include <vector>
#include <algorithm>
#include <iostream>
#include <random>
#include <stdlib.h>

void extractCraniumWeights(
    NN_DataType *hiddenWeightMemory,
    NN_DataType *inputWeightMemory,
    NN_DataType *outputWeightMemory,
    NN_DataType *hiddenBiasMemory,
    NN_DataType *outputBiasMemory,
    Network *net);

int main(void)
{
    int return_value = 0;
    srand((unsigned int)2);

#ifdef TEST_MLP_CRANIUM
    // create cranium network

    // Make sure that biases and weights are in a consistent memory location
    NN_DataType *weightMemory = (NN_DataType *)malloc((N * KInput + (NumberOfHidden - 1) * N * K + NOutput * K) * sizeof(NN_DataType));
    NN_DataType *weightMemoryBram = (NN_DataType *)malloc((N * KInput + (NumberOfHidden - 1) * N * K + NOutput * K) * sizeof(NN_DataType));

    NN_DataType *biasMemory = (NN_DataType *)malloc((NumberOfHidden * N + NOutput) * sizeof(NN_DataType));
    NN_DataType *biasMemoryBram = (NN_DataType *)malloc((NumberOfHidden * N + NOutput) * sizeof(NN_DataType));
    NN_DataType *layerResultMemoryBram = (NN_DataType *)malloc(NumberOfHidden * N * sizeof(NN_DataType));

    Activation hiddenActivation[NumberOfHidden];
    size_t hiddenSize[NumberOfHidden];

    for (unsigned int i = 0; i < NumberOfHidden; i++)
    {
        hiddenActivation[i] = sigmoid;
        hiddenSize[i] = N;
    }

    Network *net = createNetwork(KInput, NumberOfHidden, hiddenSize, hiddenActivation, NOutput, sigmoid);

    extractCraniumWeights(
        &weightMemory[N * KInput],
        weightMemory,
        &weightMemory[N * KInput + (NumberOfHidden - 1) * N * K],
        biasMemory,
        &biasMemory[NumberOfHidden * N],
        net);

    NN_InputVector inputEigen;
    inputEigen.setRandom();

    Matrix *inputCran = createMatrix(1, KInput, (float *)inputEigen.data());
    forwardPass(net, inputCran);

    NN_OutputVector resultReference(getOuput(net)->data);
    NN_OutputVector result;

    std::cout << "Result reference: \n"
              << resultReference << std::endl;

    unsigned int loadParameters = 1;
    unsigned int exportLayers = 0;

    MLP(
        inputEigen.data(),
        result.data(),
        weightMemory,
        biasMemory,
        weightMemory,
        &KInput,
        &NOutput,
        &NumberOfHidden,
        &N,
        &loadParameters,
        &exportLayers);

    loadParameters = 0;
    exportLayers = 0;

    MLP(
        inputEigen.data(),
        result.data(),
        weightMemory,
        biasMemory,
        weightMemory,
        &KInput,
        &NOutput,
        &NumberOfHidden,
        &N,
        &loadParameters,
        &exportLayers);

    std::cout << "\n Result: \n"
              << result << std::endl;

    if (result.isApprox(resultReference))
    {
        return_value = 0;
    }
    else
    {
        return_value = 1;
    }

    // destroyNetwork(net);
    // free(hiddenWeightMemory);
    // free(inputWeightMemory);
    // free(outputWeightMemory);
    // free(hiddenBiasMemory);
    // free(outputBiasMemory);
    // free(tempResultsMemory);

#endif

#ifdef TEST_TRAINING
    NN_Matrix testMatrix, weightGradientsReference, weightGradients;
    NN_Vector testInputError, testInputCurResults, errorReference, error, testInputPrev;

    testMatrix.setRandom();
    testInputError.setRandom();
    testInputCurResults.setRandom();
    testInputPrev.setRandom();

    computeError(N, K, testMatrix.data(), testInputError.data(), testInputCurResults.data(), error.data());

    errorReference = (testMatrix.transpose() * testInputError).array() * testInputCurResults.unaryExpr(std::ref(sigmoidDeriv<NN_DataType>)).array();

    for(unsigned int n = 0; n < N; n++)
    {
        for(unsigned int k = 0; k < K; k++)
        {
            weightGradientsReference.data()[n * K + k] = errorReference.data()[n] * testInputPrev.data()[k];
        }
    }

    computeWeightGradient<NN_DataType, ParEntries>(N, K, error.data(), testInputPrev.data(), weightGradients.data());

    if(weightGradientsReference.isApprox(weightGradients))
        return_value = 0;
    else
        return_value = 1;

#endif

    if (return_value == 0)
        std::cout << "Test successful!!!" << std::endl
                  << std::flush;
    else
        std::cout << "Test failed!!!" << std::endl
                  << std::flush;

    return return_value;
}

/**
 * Since Cranium creates transposed matrixes compared to the implementation in
 * HLS, the matrixes need to be back transposed again.
 * Eigen does this job effortless
 * */
void extractCraniumWeights(
    NN_DataType *hiddenWeightMemory,
    NN_DataType *inputWeightMemory,
    NN_DataType *outputWeightMemory,
    NN_DataType *hiddenBiasMemory,
    NN_DataType *outputBiasMemory,
    Network *net)
{
    Eigen::Matrix<NN_DataType, KInput, N, Eigen::RowMajor> inputWeights(net->connections[0]->weights->data);
    Eigen::Matrix<NN_DataType, N, KInput, Eigen::RowMajor> inputWeightsTrans = inputWeights.transpose();
    for (unsigned int i = 0; i < N * KInput; i++)
    {
        inputWeightMemory[i] = inputWeightsTrans.data()[i];
    }

    for (unsigned int i = 0; i < NumberOfHidden; i++)
    {
        if (i < NumberOfHidden - 1)
        {
            Eigen::Matrix<NN_DataType, K, N, Eigen::RowMajor> weights(net->connections[i + 1]->weights->data);
            Eigen::Matrix<NN_DataType, N, K, Eigen::RowMajor> weightsTrans = weights.transpose();
            for (unsigned int j = 0; j < N * K; j++)
            {
                hiddenWeightMemory[i * N * K + j] = weightsTrans.data()[j];
            }
        }

        for (unsigned int j = 0; j < N; j++)
        {
            hiddenBiasMemory[i * N + j] = net->connections[i]->bias->data[j];
        }
    }

    Eigen::Matrix<NN_DataType, K, NOutput, Eigen::RowMajor> outputWeights(net->connections[NumberOfHidden]->weights->data);
    Eigen::Matrix<NN_DataType, NOutput, K, Eigen::RowMajor> outputWeightsTrans = outputWeights.transpose();
    for (unsigned int i = 0; i < NOutput * K; i++)
    {
        outputWeightMemory[i] = outputWeightsTrans.data()[i];
    }

    for (unsigned int i = 0; i < NOutput; i++)
    {
        outputBiasMemory[i] = net->connections[NumberOfHidden]->bias->data[i];
    }
}

// Union hack test for data type conversion
// union {
//     float f;
//     unsigned int i;
// } u1, u2;

// u1.i = 16;
// u2.f = u1.f;

// std::cout << u2.f << std::endl;
// std::cout << u2.i << std::endl;

//MultAdd(weights.data(), inputs.data(), bias.data(), result.data());
// MatrixVectorVectorizedDivided(
//     reinterpret_cast<Vec_t const *>(weights.data()),
//     inputs.data(),
//     reinterpret_cast<Vec_t const *>(bias.data()),
//     reinterpret_cast<Vec_t *>(result.data()));
//MatrixVectorVectorizedTree(weights.data(), inputs.data(), bias.data(), result.data());

// size_t hiddenSizes[cranNumHiddenLayers];
// Activation hiddenActivations[cranNumHiddenLayers];
// Network * craniumNetwork = createNetwork(cranFeatures, cranNumHiddenLayers,
//     createArrayHiddenSizes(hiddenSizes), createArrayHiddenActivations(hiddenActivations),
//     cranOutputs, &linear);

// Eigen::Matrix<float, cranFeatures, 1, Eigen::ColMajor> cranEigenInput;
// cranEigenInput.setRandom();
// Matrix *cranInput = createMatrix(1, cranFeatures, cranEigenInput.data());

// forwardPass(craniumNetwork, cranInput);
// Matrix * outputData = getOuput(craniumNetwork);
