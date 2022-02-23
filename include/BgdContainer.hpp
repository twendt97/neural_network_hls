#pragma once
#include "Settings.hpp"
#include "MlpContainer.hpp"
#include <cstddef>
#include <cstdlib>

void BGD(
    const NN_DataType *axiMlpResultsInput,
    const NN_DataType *axiClassesInput,
    const NN_DataType *axiWeightInput,
    const NN_DataType *axiBiasInput,
    NN_DataType *axiWeightOutput,
    NN_DataType *axiBiasOutput,
    const unsigned int *numberInputs,
    const unsigned int *numberOutputs,
    const unsigned int *numberLayers,
    const unsigned int *numberNeurons,
    const unsigned int *loadParameters,
    const unsigned int *batchSize,
    const NN_DataType *learningRate);

class BgdContainer
{
private:
    MlpContainer &mlpInstance;
    NN_DataType *batchBuffer;

public:
    std::size_t batchSize;
    NN_DataType learningRate;

    bool loadParameters;

public:
    BgdContainer(MlpContainer &mlpInstance, std::size_t batchSize);
    ~BgdContainer();
};

BgdContainer::~BgdContainer()
{
    free(batchBuffer);
}