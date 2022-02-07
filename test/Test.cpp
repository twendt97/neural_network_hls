// defines for compilation behavior
#define TEST_MLP_CRANIUM
#define TEST_TRAINING
// #define TEST_TRAINING_COMPONENTS

#include "Settings.hpp"
#include "Simulation.hpp"
#include "KernelHelper.hpp"
#include <vector>
#include <algorithm>
#include <iostream>
#include <random>
#include <stdlib.h>
#include <stdint.h>

#include "MNIST_Extractor/include/mnist/mnist_reader.hpp"

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
template <typename t_DataType, template <typename...> class Container = std::vector, template <typename...> class Sub = std::vector, typename Pixel = uint8_t, typename Label = uint8_t>
void scaleImages(
    mnist::MNIST_dataset<Container, Sub<Pixel>, Label> &dataSet,
    t_DataType *outputArray);

template <typename t_DataType, template <typename...> class Container = std::vector, template <typename...> class Sub = std::vector, typename Pixel = uint8_t, typename Label = uint8_t>
void createClasses(
    mnist::MNIST_dataset<Container, Sub<Pixel>, Label> &dataSet,
    t_DataType *outputArray,
    unsigned int numberOutputs);

template <typename t_DataType>
t_DataType **createCraniumInputArray(
    t_DataType *values,
    std::size_t rows,
    std::size_t cols);

void testBGD(
    NN_DataType *input,
    NN_DataType *weight,
    NN_DataType *bias,
    const NN_DataType *classes,
    const unsigned int *numberInputs,
    const unsigned int *numberOutputs,
    const unsigned int *numberLayers,
    const unsigned int *numberNeurons,
    const unsigned int *batchSize,
    const NN_DataType *learningRate,
    const std::size_t maxIterations);

int main(void)
{
    int return_value = 0;
    srand((unsigned int)2);

#ifdef TEST_MLP_CRANIUM
    // create cranium network

    // Make sure that biases and weights are in a consistent memory location
    NN_DataType *weightMemory = (NN_DataType *)malloc(weightBufferSize * sizeof(NN_DataType));
    NN_DataType *biasMemory = (NN_DataType *)malloc(biasBufferSize * sizeof(NN_DataType));
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

    std::cout << "Testing hardware implementation of MLP against software implemenation of Cranium..." << std::endl
              << std::flush;

    NN_InputVector inputEigen;
    inputEigen.setRandom();
    Matrix *inputCran = createMatrix(1, KInput, (float *)inputEigen.data());
    forwardPass(net, inputCran);

    NN_OutputVector resultReference(getOuput(net)->data);
    NN_OutputVector result;

    for (std::size_t i = 0; i < 2; i++)
    {
        unsigned int loadParameters = i == 0 ? 1 : 0;
        unsigned int exportLayers = 1;

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
    }

    if (result.isApprox(resultReference))
    {
        std::cout << "Hardware produces the same results as software" << std::endl
                  << std::flush;
    }
    else
    {
        std::cout << "Hardware and software results differ" << std::endl
                  << std::flush;
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
    NN_DataType *weightReference = (NN_DataType *)malloc(weightBufferSize * sizeof(NN_DataType));
    NN_DataType *biasReference = (NN_DataType *)malloc(biasBufferSize * sizeof(NN_DataType));

    std::cout << "Extracting MNIST dataset..." << std::endl
              << std::flush;
    auto mnistDataSet = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("/home/thilo/master_thesis_code/uz_neural_network_hls_refactor/MNIST_Extractor", 2000, 2000);
    std::size_t numberImages, imageSize, numberLabels;
    numberImages = mnistDataSet.training_images.size();
    numberLabels = mnistDataSet.training_labels.size();
    assert(numberImages == numberLabels);
    imageSize = mnistDataSet.training_images.data()->size();

    NN_DataType *mnistTrainScaledImages = (NN_DataType *)malloc(numberImages * imageSize * sizeof(NN_DataType));
    assert(mnistTrainScaledImages != NULL);
    scaleImages<NN_DataType>(mnistDataSet, mnistTrainScaledImages);

    NN_DataType *mnistClassesVector = (NN_DataType *)malloc(numberLabels * numberOutputs * sizeof(NN_DataType));
    assert(mnistClassesVector != NULL);
    createClasses<NN_DataType>(mnistDataSet, mnistClassesVector, numberOutputs);

    NN_DataType **craniumInputImages = createCraniumInputArray<NN_DataType>(mnistTrainScaledImages, numberImages, imageSize);
    NN_DataType **craniumClasses = createCraniumInputArray<NN_DataType>(mnistClassesVector, numberLabels, numberOutputs);

    DataSet *cranTrainingData = createDataSet(numberImages, imageSize, craniumInputImages);
    DataSet *cranTrainingClasses = createDataSet(numberLabels, numberOutputs, craniumClasses);

    std::cout << "Running optimzation with Cranium..." << std::endl
              << std::flush;
    // train network with cross-entropy loss using Mini-Batch SGD
    ParameterSet params;
    params.network = net;
    params.data = cranTrainingData;
    params.classes = cranTrainingClasses;
    params.lossFunction = MEAN_SQUARED_ERROR;
    params.batchSize = batchSize;
    params.learningRate = learningRate;
    params.searchTime = 0;
    params.regularizationStrength = 0;
    params.momentumFactor = 0;
    params.maxIters = maxIters;
    params.shuffle = 0;
    params.verbose = 0;
    optimize(params);

    // test accuracy of network after training
    std::cout << "Accuracy is of Cranium training is " << accuracy(net, cranTrainingData, cranTrainingClasses) << std::endl
             << std::flush;

    extractCraniumWeights(
        &weightReference[N * KInput],
        weightReference,
        &weightReference[N * KInput + (NumberOfHidden - 1) * N * K],
        biasReference,
        &biasReference[NumberOfHidden * N],
        net);

    std::cout << "Running optimization with hardware implementation..." << std::endl << std::flush;

    testBGD(
        mnistTrainScaledImages,
        weightMemory,
        biasMemory,
        mnistClassesVector,
        &KInput,
        &NOutput,
        &NumberOfHidden,
        &N,
        &batchSize,
        &learningRate,
        maxIters);

    for (size_t i = 0; i < weightBufferSize; i++)
    {
        if(abs(weightReference[i] - weightMemory[i]) > precision)
        {
            std::cout << "Weights after optimization differ" << std::endl << std::flush;
            return_value = 2;
            break;
        }
    }

    for (size_t i = 0; i < biasBufferSize; i++)
    {
        if(abs(biasReference[i] - biasMemory[i]) > precision)
        {
            std::cout << "Bias after optimization differ" << std::endl << std::flush;
            return_value = 3;
            break;
        }
    }
    
    
    

    destroyNetwork(net);
    free(mnistTrainScaledImages);
    free(mnistClassesVector);
    free(craniumInputImages);
    free(craniumClasses);

#endif

#ifdef TEST_TRAINING_COMPONENTS

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

template <typename t_DataType, template <typename...> class Container = std::vector, template <typename...> class Sub = std::vector, typename Pixel = uint8_t, typename Label = uint8_t>
void scaleImages(
    mnist::MNIST_dataset<Container, Sub<Pixel>, Label> &dataSet,
    t_DataType *outputArray)
{
    std::size_t numberElements, numberFeatures;
    numberElements = dataSet.test_images.size();
    numberFeatures = dataSet.test_images.data()->size();
    for (std::size_t i = 0; i < numberElements; i++)
    {
        for (std::size_t j = 0; j < numberFeatures; j++)
        {
            outputArray[i * numberFeatures + j] = ((t_DataType)dataSet.training_images[i].data()[j]) / 255.0;
        }
    }
}

/**
 * @brief Takes the classes as integer number between 0 and numberOutputs - 1 and returns a vector
 *        where the corresponding index number is 1 and the others are 0
 * */

template <typename t_DataType, template <typename...> class Container = std::vector, template <typename...> class Sub = std::vector, typename Pixel = uint8_t, typename Label = uint8_t>
void createClasses(
    mnist::MNIST_dataset<Container, Sub<Pixel>, Label> &dataSet,
    t_DataType *outputArray,
    unsigned int numberOutputs)
{
    std::size_t numberEntries = dataSet.training_labels.size();

    for (std::size_t i = 0; i < numberEntries; i++)
    {
        for (Label j = 0; j < numberOutputs; j++)
        {
            outputArray[i * numberOutputs + j] = dataSet.training_labels[i] == j ? (t_DataType)1 : (t_DataType)0;
        }
    }
}

template <typename t_DataType>
t_DataType **createCraniumInputArray(
    t_DataType *values,
    std::size_t rows,
    std::size_t cols)
{
    t_DataType **craniumInput = (t_DataType **)malloc(rows * sizeof(t_DataType *));

    for (std::size_t i = 0; i < rows; i++)
    {
        craniumInput[i] = &values[i * cols];
    }

    return craniumInput;
}

void testBGD(
    NN_DataType *input,
    NN_DataType *weight,
    NN_DataType *bias,
    const NN_DataType *classes,
    const unsigned int *numberInputs,
    const unsigned int *numberOutputs,
    const unsigned int *numberLayers,
    const unsigned int *numberNeurons,
    const unsigned int *batchSize,
    const NN_DataType *learningRate,
    const std::size_t maxIterations)
{
    const std::size_t layerBufferSize = *numberInputs + *numberLayers * *numberNeurons + *numberOutputs;
    const unsigned int exportLayers = 1;
    std::size_t iter = 0;
    NN_DataType *mlpLayerResults = (NN_DataType *)malloc(*batchSize * layerBufferSize * sizeof(NN_DataType));
    NN_DataType *outputBuffer = (NN_DataType *)malloc(*numberOutputs * sizeof(NN_DataType));

    while (iter < maxIterations)
    {
        unsigned int loadParametersBgd = iter == 0 ? (unsigned int)1 : (unsigned int)0;
        for (size_t i = 0; i < *batchSize; i++)
        {
            unsigned int loadParametersMlp = i == 0 ? (unsigned int)1 : (unsigned int)0;
            MLP(
                &input[*numberInputs * (iter + i)],
                outputBuffer,
                weight,
                bias,
                &mlpLayerResults[i * layerBufferSize],
                numberInputs,
                numberOutputs,
                numberLayers,
                numberNeurons,
                &loadParametersMlp,
                &exportLayers);
        }

        BGD(
            mlpLayerResults,
            classes,
            weight,
            bias,
            weight,
            bias,
            numberInputs,
            numberOutputs,
            numberLayers,
            numberNeurons,
            &loadParametersBgd,
            batchSize,
            learningRate);

        iter++;
    }

    free(mlpLayerResults);
    free(outputBuffer);
}