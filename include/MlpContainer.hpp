#pragma once
#include "Settings.hpp"
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include "../Cranium/src/cranium.h"
#include <iostream>

void MLP(
    NN_DataType *input,
    NN_DataType *output,
    const NN_DataType *axiWeightInput,
    const NN_DataType *axiBiasInput,
    NN_DataType *axiLayerOutput,
    const unsigned int *numberInputs,
    const unsigned int *numberOutputs,
    const unsigned int *numberLayers,
    const unsigned int *numberNeurons,
    const unsigned int *loadParameters,
    const unsigned int *exportLayers);

class MlpContainer
{
private:
    bool memorySelfOrganized;
    void updateBufferSizes(void);
    Network *createReferenceImplementation(void);
    void copyParametersFromReference(NN_DataType *weightAddress, NN_DataType *biasAddress);

private:
    NN_DataType *weightMemory;
    NN_DataType *biasMemory;
    NN_DataType *inputAddress;
    NN_DataType *outputAddress;
    NN_DataType *layerOutputAddress;

    std::size_t numberNeurons;
    std::size_t numberHiddenLayers;
    std::size_t numberInputs;
    std::size_t numberOutputs;
    std::size_t weightBufferSize;
    std::size_t biasBufferSize;
    std::size_t layerBufferSize;

    unsigned int loadParameters;
    unsigned int exportLayers;

    Network *referenceImplementation;

public:
    MlpContainer(
        std::size_t numberNeurons,
        std::size_t numberHiddenLayers,
        std::size_t numberInputs,
        std::size_t numberOutputs);

    MlpContainer(
        std::size_t numberNeurons,
        std::size_t numberHiddenLayers,
        std::size_t numberInputs,
        std::size_t numberOutputs,
        NN_DataType *weightAddress,
        NN_DataType *biasAddress,
        NN_DataType *inputAddress,
        NN_DataType *outputAddress,
        NN_DataType *layerOutputAddress);
    ~MlpContainer();

    void feedForward();

    void setInputAddress(NN_DataType *address);
    void setLayerBufferAddress(NN_DataType *address);
    void setLoadParameters(bool value);
    void setExportLayers(bool value);

    NN_DataType *getWeightAddress(void);
    NN_DataType *getBiasAddress(void);
    NN_DataType *getInputAddress(void);
    NN_DataType *getLayerBufferAddress(void);
    NN_DataType *getOutputAddress(void);
    std::size_t getWeightBufferSize(void);
    std::size_t getBiasBufferSize(void);
    std::size_t getLayerBufferSize(void);
    Network *getReferenceImplementation(void);
    bool getLoadParameter(void);
    bool getExportLayer(void);
    
    bool testHwAgainstReference(NN_DataType precision = 1e-3);
    bool hwParametersEqualReference(NN_DataType precision = 1e-3);
};

MlpContainer::MlpContainer(
    std::size_t numberNeurons,
    std::size_t numberHiddenLayers,
    std::size_t numberInputs,
    std::size_t numberOutputs)
{
    this->numberNeurons = numberNeurons;
    this->numberHiddenLayers = numberHiddenLayers;
    this->numberInputs = numberInputs;
    this->numberOutputs = numberOutputs;
    this->loadParameters = 1;
    this->exportLayers = 0;
    this->updateBufferSizes();
    this->weightMemory = (NN_DataType *)std::malloc(weightBufferSize * sizeof(NN_DataType));
    this->biasMemory = (NN_DataType *)std::malloc(biasBufferSize * sizeof(NN_DataType));
    this->layerOutputAddress = (NN_DataType *)std::malloc(layerBufferSize * sizeof(NN_DataType));
    this->inputAddress = (NN_DataType *)std::malloc(numberInputs * sizeof(NN_DataType));
    this->outputAddress = (NN_DataType *)std::malloc(numberOutputs * sizeof(NN_DataType));
    this->memorySelfOrganized = true;
    this->referenceImplementation = this->createReferenceImplementation();
    this->copyParametersFromReference(this->weightMemory, this->biasMemory);
}

MlpContainer::MlpContainer(
    std::size_t numberNeurons,
    std::size_t numberHiddenLayers,
    std::size_t numberInputs,
    std::size_t numberOutputs,
    NN_DataType *weightAddress,
    NN_DataType *biasAddress,
    NN_DataType *inputAddress,
    NN_DataType *outputAddress,
    NN_DataType *layerOutputAddress) : numberNeurons(numberNeurons),
                                       numberHiddenLayers(numberHiddenLayers),
                                       numberInputs(numberInputs),
                                       numberOutputs(numberOutputs),
                                       weightMemory(weightAddress),
                                       biasMemory(biasAddress),
                                       inputAddress(inputAddress),
                                       outputAddress(outputAddress),
                                       layerOutputAddress(layerOutputAddress)
{
    this->updateBufferSizes();
    this->memorySelfOrganized = false;
    this->loadParameters = 1;
    this->exportLayers = 1;
    this->referenceImplementation = this->createReferenceImplementation();
    this->copyParametersFromReference(this->weightMemory, this->biasMemory);
}

MlpContainer::~MlpContainer()
{
    if (memorySelfOrganized)
    {
        std::free(weightMemory);
        std::free(biasMemory);
        std::free(inputAddress);
        std::free(outputAddress);
        std::free(layerOutputAddress);
    }
    destroyNetwork(this->referenceImplementation);
}

void MlpContainer::feedForward()
{
    unsigned int inputs = this->numberInputs;
    unsigned int outputs = this->numberOutputs;
    unsigned int hidden = this->numberHiddenLayers;
    unsigned int neurons = this->numberNeurons;
    unsigned int loadParam = this->loadParameters;
    unsigned int exportLayers = this->exportLayers;
    MLP(
        this->inputAddress,
        this->outputAddress,
        this->weightMemory,
        this->biasMemory,
        this->layerOutputAddress,
        &inputs,
        &outputs,
        &hidden,
        &neurons,
        &loadParam,
        &exportLayers);
}

bool MlpContainer::testHwAgainstReference(NN_DataType precision)
{
    bool returnValue = true;
    float *referenceInputData = (float *)malloc(this->numberInputs * sizeof(float));
    NN_DataType *originalInput;
    if (this->memorySelfOrganized == false)
    {
        NN_DataType *originalInput = this->inputAddress;
        this->inputAddress = (NN_DataType *)malloc(this->numberInputs * sizeof(NN_DataType));
    }

    for (std::size_t i = 0; i < this->numberInputs; i++)
    {
        this->inputAddress[i] = ((NN_DataType)(rand() % 100)) / 100.0;
        referenceInputData[i] = (float)this->inputAddress[i];
    }
    Matrix *referenceInput = createMatrix(1, this->numberInputs, referenceInputData);
    forwardPass(this->referenceImplementation, referenceInput);
    this->feedForward();

    Matrix *referenceOutput = getOuput(this->referenceImplementation);
    for (std::size_t i = 0; i < this->numberOutputs; i++)
    {
        assert(abs(referenceOutput->data[i] - this->outputAddress[i]) <= precision);
        if (abs(referenceOutput->data[i] - this->outputAddress[i]) > precision)
            returnValue = false;
    }
    destroyMatrix(referenceInput);

    if (this->memorySelfOrganized == false)
    {
        free(this->inputAddress);
        this->inputAddress = originalInput;
    }

    if (returnValue)
        std::cout << "HW produces the same results as the reference implementation" << std::endl
                  << std::flush;
    else
        std::cout << "HW and reference results differ" << std::endl
                  << std::flush;

    return returnValue;
}

void MlpContainer::setInputAddress(NN_DataType *address)
{
    assert(this->memorySelfOrganized == false);
    this->inputAddress = address;
}

void MlpContainer::setLayerBufferAddress(NN_DataType *address)
{
    assert(this->memorySelfOrganized == false);
    this->layerOutputAddress = address;
}

void MlpContainer::setLoadParameters(bool value)
{
    if (value)
        this->loadParameters = 1;
    else
        this->loadParameters = 0;
}

void MlpContainer::setExportLayers(bool value)
{
    if (value)
        this->exportLayers = 1;
    else
        this->exportLayers = 0;
}

NN_DataType *MlpContainer::getInputAddress(void)
{
    return this->inputAddress;
}

NN_DataType *MlpContainer::getLayerBufferAddress(void)
{
    return this->inputAddress;
}

NN_DataType *MlpContainer::getWeightAddress(void)
{
    return this->weightMemory;
}

NN_DataType *MlpContainer::getBiasAddress(void)
{
    return this->biasMemory;
}

NN_DataType *MlpContainer::getOutputAddress(void)
{
    return this->outputAddress;
}

Network *MlpContainer::getReferenceImplementation(void)
{
    return this->referenceImplementation;
}

std::size_t MlpContainer::getWeightBufferSize(void)
{
    return this->weightBufferSize;
}

std::size_t MlpContainer::getBiasBufferSize(void)
{
    return this->biasBufferSize;
}

std::size_t MlpContainer::getLayerBufferSize(void)
{
    return this->layerBufferSize;
}

// private functions

void MlpContainer::updateBufferSizes(void)
{
    this->weightBufferSize =
        (numberInputs + (numberHiddenLayers - 1) * numberNeurons + numberOutputs) * numberNeurons;
    this->biasBufferSize = numberHiddenLayers * numberNeurons + numberOutputs;
    this->layerBufferSize = numberInputs + biasBufferSize;
}

Network *MlpContainer::createReferenceImplementation(void)
{
    std::size_t hiddenSizes[this->numberHiddenLayers];
    Activation hiddenActivation[this->numberHiddenLayers];
    for (std::size_t i = 0; i < this->numberHiddenLayers; i++)
    {
        hiddenSizes[i] = this->numberNeurons;
#ifdef ACTIVATION_RELU
        hiddenActivation[i] = relu;
#else
        hiddenActivation[i] = sigmoid;
#endif
    }

    Network *net = createNetwork(this->numberInputs, this->numberHiddenLayers, hiddenSizes, hiddenActivation, this->numberOutputs, linear);
    return net;
}

void MlpContainer::copyParametersFromReference(NN_DataType *weightAddress, NN_DataType *biasAddress)
{
    Network *ref = this->referenceImplementation;
    const std::size_t inputs = this->numberInputs;
    const std::size_t outputs = this->numberOutputs;
    const std::size_t hidden = this->numberHiddenLayers;
    const std::size_t neurons = this->numberNeurons;
    NN_DataType *weightPointer = weightAddress;
    NN_DataType *biasPointer = biasAddress;

    Matrix *inputWeights = createMatrixZeroes(neurons, inputs);
    transposeInto(ref->connections[0]->weights, inputWeights);
    for (unsigned int i = 0; i < neurons * inputs; i++)
    {
        weightPointer[i] = inputWeights->data[i];
    }
    weightPointer += neurons * inputs;
    destroyMatrix(inputWeights);

    Matrix *hiddenWeights = createMatrixZeroes(neurons, neurons);
    for (std::size_t i = 0; i < hidden; i++)
    {
        if (i < hidden - 1)
        {
            transposeInto(ref->connections[i + 1]->weights, hiddenWeights);
            for (unsigned int j = 0; j < neurons * neurons; j++)
            {
                weightPointer[j] = hiddenWeights->data[j];
            }
            weightPointer += neurons * neurons;
        }

        for (size_t j = 0; j < neurons; j++)
        {
            biasPointer[j] = ref->connections[i]->bias->data[j];
        }
        biasPointer += neurons;
    }
    destroyMatrix(hiddenWeights);

    Matrix *outputWeights = createMatrixZeroes(outputs, neurons);
    transposeInto(ref->connections[hidden]->weights, outputWeights);
    for (std::size_t i = 0; i < outputs * neurons; i++)
    {
        weightPointer[i] = outputWeights->data[i];
    }
    destroyMatrix(outputWeights);

    for (std::size_t i = 0; i < outputs; i++)
    {
        biasPointer[i] = ref->connections[hidden]->bias->data[i];
    }
}

bool MlpContainer::hwParametersEqualReference(NN_DataType precision)
{
    bool returnValue = true;
    NN_DataType *refWeights = (NN_DataType *)malloc(this->weightBufferSize * sizeof(NN_DataType));
    NN_DataType *refBias = (NN_DataType *)malloc(this->biasBufferSize * sizeof(NN_DataType));
    this->copyParametersFromReference(refWeights, refBias);
    for (std::size_t i = 0; i < this->weightBufferSize; i++)
    {
        assert(abs(refWeights[i] - this->weightMemory[i]) <= precision);
        if (abs(refWeights[i] - this->weightMemory[i]) > precision)
        {
            returnValue = false;
            break;
        }
    }

    for (std::size_t i = 0; i < this->biasBufferSize; i++)
    {
        assert(abs(refBias[i] - this->biasMemory[i]) <= precision);
        if (abs(refBias[i] - this->biasMemory[i]) > precision)
        {
            returnValue = false;
            break;
        }
    }

    if (returnValue)
        std::cout << "Parameters of the HW implementation equal the reference implementation" << std::endl
                  << std::flush;
    else
        std::cout << "Parameters of the HW implementation differ from the reference implementation" << std::endl
                  << std::flush;

    return returnValue;
}