/******************************************************************************
 * Copyright 2022 Thilo Wendt
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 ******************************************************************************/

#include "KernelHelper.hpp"
#include "Settings.hpp"
#include <string.h>

using namespace uz_mlp;

/**
 * @param numberLayers = Number of hidden layers
 * */

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
    const NN_DataType *learningRate)
{
#pragma HLS INTERFACE m_axi port = axiMlpResultsInput offset = slave bundle = read max_read_burst_length = 128 depth = axiBatchResultsDepth
#pragma HLS INTERFACE m_axi port = axiClassesInput offset = slave bundle = read max_read_burst_length = 128 depth = simBatchSize * simNumberOutputs
#pragma HLS INTERFACE m_axi port = axiWeightInput offset = slave bundle = read max_read_burst_length = 128 depth = axiWeightDepth
#pragma HLS INTERFACE m_axi port = axiBiasInput offset = slave bundle = read max_read_burst_length = 128 depth = axiBiasDepth
#pragma HLS INTERFACE m_axi port = axiWeightOutput offset = slave bundle = write max_write_burst_length = 128 depth = axiWeightDepth
#pragma HLS INTERFACE m_axi port = axiBiasOutput offset = slave bundle = write max_write_burst_length = 128 depth = axiBiasDepth
//#pragma HLS INTERFACE s_axilite port = return
#pragma HLS INTERFACE ap_ctrl_hs port = return
#pragma HLS INTERFACE s_axilite port = numberInputs
#pragma HLS INTERFACE s_axilite port = numberOutputs
#pragma HLS INTERFACE s_axilite port = numberLayers
#pragma HLS INTERFACE s_axilite port = numberNeurons
#pragma HLS INTERFACE s_axilite port = loadParameters
#pragma HLS INTERFACE s_axilite port = batchSize
#pragma HLS INTERFACE s_axilite port = learningRate

    static NN_DataType bramWeight[weightBufferSize];
    static NN_DataType bramBias[biasBufferSize];
    static NN_DataType bramWeightGradientAvg[weightBufferSize];
    static NN_DataType bramBiasGradientAvg[biasBufferSize];
    NN_DataType bramClasses[hwMaxBatchSize * hwNumberOutputs];
    NN_DataType bramMlpResults[hwMaxBatchSize * layerResultsBufferSize];
    NN_DataType bramError0[layerBufferSize], bramError1[layerBufferSize];
    NN_DataType *currentResults, *currentClasses;

    if (*loadParameters != 0)
    {
        unsigned int valuesToLoad;
        // weights
        valuesToLoad = *numberInputs * *numberNeurons;
        valuesToLoad += (*numberLayers - 1) * *numberNeurons * *numberNeurons;
        valuesToLoad += *numberOutputs * *numberNeurons;

        memcpy(bramWeight, axiWeightInput, valuesToLoad * sizeof(NN_DataType));

        // biases
        valuesToLoad = *numberLayers * *numberNeurons;
        valuesToLoad += *numberOutputs;
        memcpy(bramBias, axiBiasInput, valuesToLoad * sizeof(NN_DataType));
    }

    // copy volatile parameters
    memcpy(bramClasses, axiClassesInput, *batchSize * *numberOutputs * sizeof(NN_DataType));
    memcpy(
        bramMlpResults,
        axiMlpResultsInput,
        *batchSize * (*numberInputs + *numberLayers * *numberNeurons + *numberOutputs) * sizeof(NN_DataType));

    // calculate each iteration of backpropagation

    for (unsigned int i = 0; i < *batchSize; i++)
    {

        currentResults = &bramMlpResults[i * (*numberInputs + *numberNeurons * *numberLayers + *numberOutputs)];
        currentClasses = &bramClasses[i * *numberOutputs];
        bool initZero = (i == 0);

        computeOutputGradient<NN_DataType, ParEntries, streamDepth>(
            *numberOutputs,
            *numberNeurons,
            &currentResults[*numberInputs + *numberLayers * *numberNeurons],
            currentClasses,
            &currentResults[*numberInputs + (*numberLayers - 1) * *numberNeurons],
            &bramWeightGradientAvg[(*numberInputs + (*numberLayers - 1) * *numberNeurons) * *numberNeurons],
            &bramBiasGradientAvg[*numberLayers * *numberNeurons],
            bramError0,
            initZero);

        copyArray<NN_DataType, ParEntries>(
            bramError0,
            bramError1,
            *numberOutputs);

        for (int layer = *numberLayers - 1; layer > 0; layer--)
        {
            unsigned int p_n = layer < *numberLayers - 1 ? *numberNeurons : *numberOutputs;
            // unsigned int outputsPrev = layer > 0 ? *numberNeurons : *numberInputs;
            computeHiddenGradient<NN_DataType, ParEntries, logParEntries, streamDepth>(
                p_n,
                *numberNeurons,
                *numberNeurons,
                &bramWeight[(*numberInputs + layer * *numberNeurons) * *numberNeurons],
                bramError1,
                &currentResults[*numberInputs + layer * *numberNeurons],
                &currentResults[*numberInputs + (layer - 1) * *numberNeurons],
                &bramWeightGradientAvg[(*numberInputs + (layer - 1) * *numberNeurons) * *numberNeurons],
                &bramBiasGradientAvg[layer * *numberNeurons],
                bramError0,
                initZero);

            copyArray<NN_DataType, ParEntries>(
                bramError0,
                bramError1,
                *numberNeurons);
        }

        unsigned int l_n = *numberLayers > 1 ? *numberNeurons : *numberOutputs;
        computeHiddenGradient<NN_DataType, ParEntries, logParEntries, streamDepth>(
            l_n,
            *numberNeurons,
            *numberInputs,
            &bramWeight[*numberNeurons * *numberInputs],
            bramError1,
            &currentResults[*numberInputs],
            currentResults,
            bramWeightGradientAvg,
            bramBiasGradientAvg,
            bramError0,
            initZero);
    }

    updateParameter<NN_DataType, ParEntries>(
        bramWeight,
        bramBias,
        bramWeightGradientAvg,
        bramBiasGradientAvg,
        *learningRate,
        (NN_DataType)*batchSize,
        (*numberInputs + (*numberLayers - 1) * *numberNeurons + *numberOutputs) * *numberNeurons,
        *numberLayers * *numberNeurons + *numberOutputs);

    unsigned int valuesToStore;
    // weights
    valuesToStore = *numberInputs * *numberNeurons;
    valuesToStore += (*numberLayers - 1) * *numberNeurons * *numberNeurons;
    valuesToStore += *numberOutputs * *numberNeurons;

    memcpy(axiWeightOutput, bramWeight, valuesToStore * sizeof(NN_DataType));

    // biases
    valuesToStore = *numberLayers * *numberNeurons;
    valuesToStore += *numberOutputs;
    memcpy(axiBiasOutput, bramBias, valuesToStore * sizeof(NN_DataType));
}