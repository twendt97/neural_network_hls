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
#pragma HLS INTERFACE m_axi port = input offset = slave bundle = read max_read_burst_length = 128 depth = simNumberInputs
#pragma HLS INTERFACE m_axi port = output offset = slave bundle = write max_write_burst_length = 128 depth = simNumberOutputs
#pragma HLS INTERFACE m_axi port = axiWeightInput offset = slave bundle = read max_read_burst_length = 128 depth = axiWeightDepth
#pragma HLS INTERFACE m_axi port = axiBiasInput offset = slave bundle = read max_read_burst_length = 128 depth = axiBiasDepth
#pragma HLS INTERFACE m_axi port = axiLayerOutput offset = slave bundle = write max_write_burst_length = 128 depth = axiLayerResultsDepth
//#pragma HLS INTERFACE s_axilite port = return
#pragma HLS INTERFACE ap_ctrl_hs port = return
#pragma HLS INTERFACE s_axilite port = numberInputs
#pragma HLS INTERFACE s_axilite port = numberOutputs
#pragma HLS INTERFACE s_axilite port = numberLayers
#pragma HLS INTERFACE s_axilite port = numberNeurons
#pragma HLS INTERFACE s_axilite port = loadParameters
#pragma HLS INTERFACE s_axilite port = exportLayers

    // the size of theses buffers determine the maximum size of the neural network to be processed
    static NN_DataType bramWeight[weightBufferSize];
    static NN_DataType bramBias[biasBufferSize];
    NN_DataType inputData[hwNumberInputs];
    NN_DataType layerBuffer0[layerBufferSize], layerBuffer1[layerBufferSize];
    NN_DataType bramLayerResults[layerResultsBufferSize];

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
        copyArray<NN_DataType, ParEntries>(inputData, bramLayerResults, *numberInputs);

    processLayer<NN_DataType, ParEntries, logParEntries>(
        bramWeight,
        inputData,
        bramBias,
        layerBuffer0,
        *numberNeurons,
        *numberInputs);

    // only write to bramLayerResults to get a write only one port interface and not dual port
    if (*exportLayers != 0)
        copyArray<NN_DataType, ParEntries>(layerBuffer0, &bramLayerResults[*numberInputs], *numberNeurons);

    copyArray<NN_DataType, ParEntries>(layerBuffer0, layerBuffer1, *numberNeurons);

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
            copyArray<NN_DataType, ParEntries>(layerBuffer0, &bramLayerResults[*numberInputs + (i + 1) * *numberNeurons], *numberNeurons);

        copyArray<NN_DataType, ParEntries>(layerBuffer0, layerBuffer1, *numberNeurons);
    }
    outputLayer<NN_DataType, ParEntries, logParEntries>(
        &bramWeight[(*numberInputs + (*numberLayers - 1) * *numberNeurons) * *numberNeurons],
        layerBuffer1,
        &bramBias[*numberLayers * *numberNeurons],
        layerBuffer0,
        *numberOutputs,
        *numberNeurons);

    if (*exportLayers != 0)
    {
        copyArray<NN_DataType, ParEntries>(layerBuffer0, &bramLayerResults[*numberInputs + *numberLayers * *numberNeurons], *numberOutputs);
        memcpy(axiLayerOutput, bramLayerResults, (*numberInputs + *numberNeurons * *numberLayers + *numberOutputs) * sizeof(NN_DataType));
    }

    memcpy(output, layerBuffer0, *numberOutputs * sizeof(NN_DataType));
}
