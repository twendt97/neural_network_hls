#include "KernelHelper.hpp"
#include "Vitis_Libraries/blas/L1/include/hw/xf_blas.hpp"
#include "ap_int.h"
#include "hls_stream.h"
#include "hls_math.h"
#include <string.h>

#ifndef __SYNTHESIS__
#include <iostream>
#include "Simulation.hpp"
#endif

using namespace xf::blas;

template <typename t_DataType>
t_DataType relu(t_DataType x)
{
    if (x > 0)
        return x;
    else
        return 0;
}

template <typename t_DataType>
t_DataType sigmoid(t_DataType x)
{
    t_DataType l_exp = hls::expf(-x);
    return 1.0f / (1.0f + l_exp);
}

void FcnActivation(
    hls::stream<typename WideType<NN_DataType, 1>::t_TypeInt> &inputStream,
    hls::stream<typename WideType<NN_DataType, 1>::t_TypeInt> &outputStream,
    unsigned int p_n,
    NN_DataType (*activationFunction)(NN_DataType))
{
    for (unsigned int i = 0; i < p_n; i++)
    {
#pragma HLS PIPELINE
        WideType<NN_DataType, 1> l_val = inputStream.read();
        WideType<NN_DataType, 1> l_valOut;
        l_valOut[0] = activationFunction(l_val[0]);
        outputStream.write(l_valOut);
    }
}

void MyGemv(
    NN_DataType *weights,
    NN_DataType *input,
    NN_DataType *bias,
    NN_DataType *outputPort,
    unsigned int p_n,
    unsigned int p_k)
{
#pragma HLS DATAFLOW
    // Stream that holds ParEntries operands
    hls::stream<typename WideType<NN_DataType, 1 << logParEntries>::t_TypeInt> l_strWeights("Weights");
    hls::stream<typename WideType<NN_DataType, 1 << logParEntries>::t_TypeInt> l_strInput("Input");
    // Stream that holds exactly one operand.
    // This is fed to a function that assembles a vector from the incoming entries
    hls::stream<typename WideType<NN_DataType, 1>::t_TypeInt> l_strOutput("Output");
    hls::stream<typename WideType<NN_DataType, 1>::t_TypeInt> l_strMv("Matrix Vector Result");
    hls::stream<typename WideType<NN_DataType, 1>::t_TypeInt> l_strBias("Bias");
#pragma HLS DATAFLOW
    gem2Stream<NN_DataType, ParEntries>(p_n, p_k, weights, l_strWeights);
    vec2GemStream<NN_DataType, ParEntries>(p_n, p_k, input, l_strInput);
    readVec2Stream<NN_DataType, 1>(bias, p_n, l_strBias);
    gemv<NN_DataType, logParEntries>(p_n, p_k, (NN_DataType)1, l_strWeights, l_strInput, (NN_DataType)1, l_strBias, l_strMv);
    FcnActivation(l_strMv, l_strOutput, p_n, sigmoid);
    writeStream2Vec<NN_DataType, 1>(l_strOutput, p_n, outputPort);
}

void CopyLayers(
    NN_DataType *input,
    NN_DataType *output,
    unsigned int size)
{
COPY_LAYERS:
    for (unsigned int i = 0; i < size; i++)
    {
#pragma HLS PIPELINE
        output[i] = input[i];
    }
}

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

    MyGemv(bramWeight, inputData, bramBias, layerBuffer0, *numberNeurons, *numberInputs);

    // only write to bramLayerResults to get a write only one port interface and not dual port
    if (*exportLayers != 0)
        CopyLayers(layerBuffer0, bramLayerResults, *numberNeurons);

    CopyLayers(layerBuffer0, layerBuffer1, *numberNeurons);

HIDDEN:
    for (unsigned int i = 0; i < *numberLayers - 1; i++)
    {
        MyGemv(
            &bramWeight[(*numberInputs + i * *numberNeurons) * *numberNeurons],
            layerBuffer1,
            &bramBias[(i + 1) * *numberNeurons],
            layerBuffer0,
            *numberNeurons,
            *numberNeurons);

        if (*exportLayers != 0)
            CopyLayers(layerBuffer0, &bramLayerResults[(i + 1) * *numberNeurons], *numberNeurons);

        CopyLayers(layerBuffer0, layerBuffer1, *numberNeurons);
    }
    MyGemv(
        &bramWeight[(*numberInputs + (*numberLayers - 1) * *numberNeurons) * *numberNeurons],
        layerBuffer1,
        &bramBias[*numberLayers * *numberNeurons],
        layerBuffer0,
        *numberOutputs,
        *numberNeurons);
    
    if (*exportLayers != 0)
        CopyLayers(layerBuffer0, &bramLayerResults[*numberLayers * *numberNeurons], *numberOutputs);
    

    memcpy(output, layerBuffer0, *numberOutputs * sizeof(NN_DataType));
}
