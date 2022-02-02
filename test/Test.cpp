// defines for compilation behavior
//#define TEST_MLP_CRANIUM
//#define TEST_TRAINING

#include "Settings.hpp"
#include "Simulation.hpp"
#include "KernelHelper.hpp"
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
    NN_DataType *layerResultMemoryBram = (NN_DataType *)malloc((KInput + NumberOfHidden * N + NOutput) * sizeof(NN_DataType));

    Activation hiddenActivation[NumberOfHidden];
    size_t hiddenSize[NumberOfHidden];

    for (unsigned int i = 0; i < NumberOfHidden; i++)
    {
        hiddenActivation[i] = sigmoid;
        hiddenSize[i] = N;
    }

    Network *net = createNetwork(KInput, NumberOfHidden, hiddenSize, hiddenActivation, NOutput, linear);

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
    NN_Matrix testWeights, weightGradientsReference, weightGradients;
    NN_Vector testInputError, testInputCurResults, biasGradientReference, testInputPrev, biasGradient, outputError, errorReference;
    NN_OutputVector testOutput, testClasses, outputBiasGradient, outputBiasGradientReference, outputErrorReference;
    NN_OutputWeights outputWeightGradient, outputWeightGradientReference;

    testWeights.setRandom();
    testInputError.setRandom();
    testInputCurResults.setRandom();
    testInputPrev.setRandom();
    testOutput.setRandom();
    testClasses.setRandom();
    biasGradientReference.setRandom();
    weightGradientsReference.setRandom();
    biasGradient = biasGradientReference;
    weightGradients = weightGradientsReference;

    errorReference = (testWeights.transpose() * testInputError).array() * testInputCurResults.unaryExpr(std::ref(uz_mlp::sigmoidDeriv<NN_DataType>)).array();

    for (unsigned int n = 0; n < N; n++)
    {
        for (unsigned int k = 0; k < K; k++)
        {
            weightGradientsReference.data()[n * K + k] += errorReference.data()[n] * testInputPrev.data()[k];
        }
    }

    biasGradientReference += errorReference;

    uz_mlp::computeHiddenGradient<NN_DataType, ParEntries, logParEntries, streamDepth>(
        N,
        K,
        testWeights.data(),
        testInputError.data(),
        testInputCurResults.data(),
        testInputPrev.data(),
        weightGradients.data(),
        biasGradient.data(),
        outputError.data(),
        false);

    if (weightGradientsReference.isApprox(weightGradients) && biasGradientReference.isApprox(biasGradient))
        return_value = 0;
    else
    {
        return_value = 1;
        std::cout << "Hidden weight or bias gradients do not match" << std::endl << std::flush;
    }


    // Output
    outputWeightGradientReference.setRandom();
    outputBiasGradientReference.setRandom();
    outputWeightGradient = outputWeightGradientReference;
    outputBiasGradient = outputBiasGradientReference;

    outputErrorReference = testOutput - testClasses;

    for (unsigned int n = 0; n < NOutput; n++)
    {
        for (unsigned int k = 0; k < K; k++)
        {
            outputWeightGradientReference.data()[n * K + k] += outputErrorReference.data()[n] * testInputPrev.data()[k];
        }
    }

    outputBiasGradientReference += outputErrorReference;

    uz_mlp::computeOutputGradient<NN_DataType, ParEntries, streamDepth>(
        NOutput,
        K,
        testOutput.data(),
        testClasses.data(),
        testInputPrev.data(),
        outputWeightGradient.data(),
        outputBiasGradient.data(),
        outputError.data(),
        false);

    if (outputWeightGradientReference.isApprox(outputWeightGradient) 
        && outputBiasGradientReference.isApprox(outputBiasGradient))
        return_value = 0;
    else
    {
        return_value = 1;
        std::cout << "Output weight or bias gradients do not match" << std::endl << std::flush;
    }


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
