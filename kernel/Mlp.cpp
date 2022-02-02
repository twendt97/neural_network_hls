#include "KernelHelper.hpp"
#include "Settings.hpp"
#include <string.h>

using namespace uz_mlp;

/**
 * @param numberLayers = Number of hidden layers
 * */

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
    const unsigned int *exportLayers)
{
#pragma HLS INTERFACE m_axi port = input offset = slave bundle = axi_read max_read_burst_length = 128 depth = 2 * KInput
#pragma HLS INTERFACE m_axi port = output offset = slave bundle = axi_write max_write_burst_length = 128 depth = 2 * NOutput
#pragma HLS INTERFACE m_axi port = axiWeightInput offset = slave bundle = axi_read max_read_burst_length = 128 depth = 2 * (KInput * N + (NumberOfHidden - 1) * N * K + NOutput * K)
#pragma HLS INTERFACE m_axi port = axiBiasInput offset = slave bundle = axi_read max_read_burst_length = 128 depth = 2 * (NumberOfHidden * N + NOutput)
#pragma HLS INTERFACE m_axi port = axiLayerOutput offset = slave bundle = axi_write max_write_burst_length = 128 depth = 2 * (KInput + NumberOfHidden * N + NOutput)
#pragma HLS INTERFACE s_axilite port = return bundle = configuration
#pragma HLS INTERFACE s_axilite port = numberInputs bundle = configuration
#pragma HLS INTERFACE s_axilite port = numberOutputs bundle = configuration
#pragma HLS INTERFACE s_axilite port = numberLayers bundle = configuration
#pragma HLS INTERFACE s_axilite port = numberNeurons bundle = configuration
#pragma HLS INTERFACE s_axilite port = loadParameters bundle = configuration
#pragma HLS INTERFACE s_axilite port = exportLayers bundle = configuration

    // the size of theses buffers determine the maximum size of the neural network to be processed
    static NN_DataType bramWeight[KInput * N + (NumberOfHidden - 1) * N * K + NOutput * K];
    static NN_DataType bramBias[NumberOfHidden * N + NOutput];
    NN_DataType inputData[KInput];
    NN_DataType layerBuffer0[N], layerBuffer1[N];
    NN_DataType bramLayerResults[KInput + NumberOfHidden * N + NOutput];

    if (*loadParameters != 0)
    {
        unsigned int valuesToCopy;
        // weights
        valuesToCopy = *numberInputs * *numberNeurons;
        valuesToCopy += (*numberLayers - 1) * *numberNeurons * *numberNeurons;
        valuesToCopy += *numberOutputs * *numberNeurons;

        memcpy(bramWeight, axiWeightInput, valuesToCopy * sizeof(NN_DataType));

        // biases
        valuesToCopy = *numberLayers * *numberNeurons;
        valuesToCopy += *numberOutputs;
        memcpy(bramBias, axiBiasInput, valuesToCopy * sizeof(NN_DataType));
    }

    memcpy(inputData, input, *numberInputs * sizeof(NN_DataType));

    if (*exportLayers != 0)
        copyArray<NN_DataType, 1>(inputData, bramLayerResults, *numberInputs);

    inputLayer<NN_DataType, ParEntriesInput, logParEntriesInput>(
        bramWeight,
        inputData,
        bramBias,
        layerBuffer0,
        *numberNeurons,
        *numberInputs);

    // only write to bramLayerResults to get a write only one port interface and not dual port
    if (*exportLayers != 0)
        copyArray<NN_DataType, 1>(layerBuffer0, &bramLayerResults[*numberInputs], *numberNeurons);

    copyArray<NN_DataType, 1>(layerBuffer0, layerBuffer1, *numberNeurons);

HIDDEN:
    for (unsigned int i = 0; i < *numberLayers - 1; i++)
    {
        processLayer<NN_DataType, ParEntries, logParEntries>(
            &bramWeight[(*numberInputs + i * *numberNeurons) * *numberNeurons],
            layerBuffer1,
            &bramBias[(i + 1) * *numberNeurons],
            layerBuffer0,
            *numberNeurons,
            *numberNeurons);

        if (*exportLayers != 0)
            copyArray<NN_DataType, 1>(layerBuffer0, &bramLayerResults[*numberInputs + (i + 1) * *numberNeurons], *numberNeurons);

        copyArray<NN_DataType, 1>(layerBuffer0, layerBuffer1, *numberNeurons);
    }
    outputLayer<NN_DataType, ParEntriesOutput, logParEntriesOutput>(
        &bramWeight[(*numberInputs + (*numberLayers - 1) * *numberNeurons) * *numberNeurons],
        layerBuffer1,
        &bramBias[*numberLayers * *numberNeurons],
        layerBuffer0,
        *numberOutputs,
        *numberNeurons);

    if (*exportLayers != 0)
    {
        copyArray<NN_DataType, 1>(layerBuffer0, &bramLayerResults[*numberInputs + *numberLayers * *numberNeurons], *numberOutputs);
        memcpy(bramLayerResults, axiLayerOutput, (*numberInputs + *numberNeurons * *numberLayers + *numberOutputs) * sizeof(NN_DataType));
    }

    memcpy(output, layerBuffer0, *numberOutputs * sizeof(NN_DataType));
}
