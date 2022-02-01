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
    const unsigned int *numberSamples,
    const unsigned int *loadParameters,
    const unsigned int *batchSize,
    const NN_DataType *learningRate)
{
    static NN_DataType bramWeight[KInput * N + (NumberOfHidden - 1) * N * K + NOutput * K];
    static NN_DataType bramBias[NumberOfHidden * N + NOutput];
    static NN_DataType bramWeightGradientAvg[KInput * N + (NumberOfHidden - 1) * N * K + NOutput * K];
    static NN_DataType bramBiasGradientAvg[NumberOfHidden * N + NOutput];
    NN_DataType bramClasses[maxSamples * NOutput];
    NN_DataType bramMlpResults[maxSamples * (NumberOfHidden * N + NOutput + KInput)];
    NN_DataType bramError[N];
    NN_DataType *currentResults, *currentClasses;
    hls::stream<typename xf::blas::WideType<NN_DataType, ParEntries>::t_TypeInt> prevLayerOutputStream;

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
    memcpy(bramClasses, axiClassesInput, *numberSamples * sizeof(NN_DataType));
    memcpy(
        bramMlpResults,
        axiMlpResultsInput,
        *numberSamples * (*numberInputs + *numberLayers * *numberNeurons + *numberOutputs) * sizeof(NN_DataType));

    // calculate each iteration of backpropagation

    for (unsigned int i = 0; i < *batchSize; i++)
    {

        currentResults = bramMlpResults;
        currentClasses = bramClasses;

        computeOutputGradient<NN_DataType, ParEntriesOutput, streamDepth>(
            *numberOutputs,
            *numberNeurons,
            &currentResults[*numberLayers * *numberNeurons],
            currentClasses,
            &currentResults[(*numberLayers - 1) * *numberNeurons],
            &bramWeightGradientAvg[(*numberInputs + (*numberLayers - 1) * *numberNeurons) * *numberNeurons],
            &bramBiasGradientAvg[*numberLayers * *numberNeurons],
            bramError,
            false);

        copyArray<NN_DataType, ParEntriesOutput>(
            &bramBiasGradientAvg[*numberLayers * *numberNeurons],
            bramError,
            *numberOutputs);

        for (int layer = *numberLayers - 1; layer > 0; layer--)
        {
            unsigned int p_n = layer < *numberLayers - 1 ? *numberNeurons : *numberOutputs;
            computeHiddenGradient<NN_DataType, ParEntries, logParEntries, streamDepth>(
                p_n,
                *numberNeurons,
                &bramWeight[(*numberInputs + (layer - 1) * *numberNeurons) * *numberNeurons],
                bramError,
                &currentResults[layer * *numberNeurons],
                &currentResults[(layer - 1) * *numberNeurons],
                &bramWeightGradientAvg[(*numberInputs + (layer - 1) * *numberNeurons) * *numberNeurons],
                &bramBiasGradientAvg[layer * *numberNeurons],
                bramError,
                false);

            copyArray<NN_DataType, ParEntriesOutput>(
                &bramBiasGradientAvg[layer * *numberNeurons],
                bramError,
                *numberNeurons);
        }

        computeHiddenGradient<NN_DataType, ParEntriesInput, logParEntriesInput, streamDepth>(
            *numberNeurons,
            *numberInputs,
            bramWeight,
            bramError,
            &currentResults[*numberInputs],
            currentResults,
            bramWeightGradientAvg,
            bramBiasGradientAvg,
            bramError,
            false);
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