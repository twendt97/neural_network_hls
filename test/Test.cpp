// defines for compilation behavior
//#define TEST_MLP_CRANIUM
#define TEST_TRAINING

#include "Settings.hpp"
#include "Simulation.hpp"
#include "KernelHelper.hpp"
#include <vector>
#include <algorithm>
#include <iostream>
#include <random>
#include <stdlib.h>
#include <stdint.h>

#ifdef TEST_TRAINING
#include "MNIST_Extractor/include/mnist_file.h"
#endif

void extractCraniumWeights(
    NN_DataType *hiddenWeightMemory,
    NN_DataType *inputWeightMemory,
    NN_DataType *outputWeightMemory,
    NN_DataType *hiddenBiasMemory,
    NN_DataType *outputBiasMemory,
    Network *net);

/**
 * @brief Takes an array of uint8_t and scales it to a float between 0 and 1
 * 
 * @
 * */
template <typename t_DataType>
void scaleImages(
    mnist_dataset_t *dataSet,
    t_DataType *outputArray,
    unsigned int numberFeatures);

template <typename t_DataType>
void createClassesVector(
    mnist_dataset_t *classes,
    t_DataType *outputArray,
    unsigned int numberOutputs);

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

    mnist_dataset_t *mnistTrainingSet = mnist_get_dataset(trainImagesFile, trainLabelsFile);
    mnist_dataset_t *mnistTestSet = mnist_get_dataset(testImagesFile, testLabelsFile);

    NN_DataType *mnistTrainScaledImages = (NN_DataType *)malloc(mnistTrainingSet->size * MNIST_IMAGE_SIZE * sizeof(NN_DataType));
    scaleImages<NN_DataType>(mnistTrainingSet, mnistTrainScaledImages, (unsigned int)MNIST_IMAGE_SIZE);

    NN_DataType *mnistClassesVector = (NN_DataType *)malloc(mnistTrainingSet->size * MNIST_LABELS * sizeof(NN_DataType));
    createClassesVector(mnistTrainingSet, mnistClassesVector, MNIST_LABELS);

    DataSet *cranTrainingData = createDataSet(mnistTrainingSet->size, MNIST_IMAGE_SIZE, (NN_DataType **)mnistTrainScaledImages);
    DataSet *cranTrainingClasses = createDataSet(mnistTrainingSet->size, MNIST_LABELS, (NN_DataType **)mnistClassesVector);

    size_t hiddenSize[] = {16};
    Activation hiddenActivation[] = {sigmoid};
    Network *net = createNetwork(MNIST_IMAGE_SIZE, 1, hiddenSize, hiddenActivation, MNIST_LABELS, linear);

    // train network with cross-entropy loss using Mini-Batch SGD
    ParameterSet params;
    params.network = net;
    params.data = cranTrainingData;
    params.classes = cranTrainingClasses;
    params.lossFunction = MEAN_SQUARED_ERROR;
    params.batchSize = 20;
    params.learningRate = 3;
    params.searchTime = 0;
    params.regularizationStrength = 0;
    params.momentumFactor = 0;
    params.maxIters = 10000;
    params.shuffle = 1;
    params.verbose = 1;
    optimize(params);

    // test accuracy of network after training
    std::cout << "Accuracy is " << accuracy(net, cranTrainingData, cranTrainingClasses) << std::endl
              << std::flush;

    mnist_free_dataset(mnistTrainingSet);
    mnist_free_dataset(mnistTestSet);
    // free(mnistClassesVector);
    // free(mnistTrainScaledImages);
    destroyNetwork(net);
    destroyDataSet(cranTrainingData);
    destroyDataSet(cranTrainingClasses);

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
        std::cout << "Hidden weight or bias gradients do not match" << std::endl
                  << std::flush;
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

    if (outputWeightGradientReference.isApprox(outputWeightGradient) && outputBiasGradientReference.isApprox(outputBiasGradient))
        return_value = 0;
    else
    {
        return_value = 1;
        std::cout << "Output weight or bias gradients do not match" << std::endl
                  << std::flush;
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

template <typename t_DataType>
void scaleImages(
    mnist_dataset_t *dataSet,
    t_DataType *outputArray,
    unsigned int numberFeatures)
{

    for (unsigned int i = 0; i < dataSet->size; i++)
    {
        for (unsigned int j = 0; j < numberFeatures; j++)
        {
            outputArray[i * numberFeatures + j] = ((t_DataType)dataSet->images[i].pixels[j]) / 255.0;
        }
    }
}

/**
 * @brief Takes the classes as integer number between 0 and numberOutputs - 1 and returns a vector
 *        where the corresponding index number is 1 and the others are 0
 * */

template <typename t_DataType>
void createClassesVector(
    mnist_dataset_t *classes,
    t_DataType *outputArray,
    unsigned int numberOutputs)
{
    for (unsigned int i = 0; i < classes->size; i++)
    {
        for (uint8_t j = 0; j < numberOutputs; j++)
        {
            outputArray[i * numberOutputs + j] = classes->labels[i] == j ? (t_DataType)1 : (t_DataType)0;
        }
    }
}