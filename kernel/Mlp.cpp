#include "KernelHelper.hpp"
#include <string.h>

#ifndef __SYNTHESIS__
#include <iostream>
#include "Simulation.hpp"
#endif

using namespace uz_mlp;

/**
 * @param numberLayers = Number of hidden layers
 * */

void MLP(
    NN_DataType *input,
    NN_DataType *output,
    const NN_DataType *axiWeightInput,
    const NN_DataType *axiBiasInput,
    NN_DataType bramLayerResults[NumberOfHidden * N + NOutput],
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
#pragma HLS INTERFACE bram port = bramLayerResults
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

#pragma HLS ARRAY_PARTITION variable = bramWeight cyclic factor = 8

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

    NN_DataType inputData[KInput];
    NN_DataType layerBuffer0[N], layerBuffer1[N];
    memcpy(inputData, input, *numberInputs * sizeof(NN_DataType));


    ProcessLayer<NN_DataType, 1, 0>(bramWeight, inputData, bramBias, layerBuffer0, *numberNeurons, *numberInputs);
    

    // only write to bramLayerResults to get a write only one port interface and not dual port
    if (*exportLayers != 0)
        CopyArray<NN_DataType>(layerBuffer0, bramLayerResults, *numberNeurons);

    CopyArray<NN_DataType>(layerBuffer0, layerBuffer1, *numberNeurons);

HIDDEN:
    for (unsigned int i = 0; i < *numberLayers - 1; i++)
    {
        ProcessLayer<NN_DataType, ParEntries, logParEntries>(
            &bramWeight[(*numberInputs + i * *numberNeurons) * *numberNeurons],
            layerBuffer1,
            &bramBias[(i + 1) * *numberNeurons],
            layerBuffer0,
            *numberNeurons,
            *numberNeurons);

        if (*exportLayers != 0)
            CopyArray<NN_DataType>(layerBuffer0, &bramLayerResults[(i + 1) * *numberNeurons], *numberNeurons);

        CopyArray<NN_DataType>(layerBuffer0, layerBuffer1, *numberNeurons);
    }
    ProcessLayer<NN_DataType, 1, 0>(
        &bramWeight[(*numberInputs + (*numberLayers - 1) * *numberNeurons) * *numberNeurons],
        layerBuffer1,
        &bramBias[*numberLayers * *numberNeurons],
        layerBuffer0,
        *numberOutputs,
        *numberNeurons);
    
    if (*exportLayers != 0)
        CopyArray<NN_DataType>(layerBuffer0, &bramLayerResults[*numberLayers * *numberNeurons], *numberOutputs);
    

    memcpy(output, layerBuffer0, *numberOutputs * sizeof(NN_DataType));
}
