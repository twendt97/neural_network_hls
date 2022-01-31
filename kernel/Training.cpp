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
    const unsigned int *loadParameters)
{
    static NN_DataType bramWeight[KInput * N + (NumberOfHidden - 1) * N * K + NOutput * K];
    static NN_DataType bramBias[NumberOfHidden * N + NOutput];
    static NN_DataType bramWeightGradient[KInput * N + (NumberOfHidden - 1) * N * K + NOutput * K];
    static NN_DataType bramBiasGradient[NumberOfHidden * N + NOutput];
    static NN_DataType bramWeightGradientAvg[KInput * N + (NumberOfHidden - 1) * N * K + NOutput * K];
    static NN_DataType bramBiasGradientAvg[NumberOfHidden * N + NOutput];
    NN_DataType bramClasses[maxSamples * NOutput];
    NN_DataType bramMlpResults[maxSamples * (NumberOfHidden * N + NOutput + KInput)];
    NN_DataType bramError[N];
    NN_DataType *currentResults, *currentClasses;
    hls::stream<typename xf::blas::WideType<NN_DataType, ParEntries>::t_TypeInt> prevLayerOutputStream;

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

    // copy volatile parameters
    memcpy(bramClasses, axiClassesInput, *numberSamples * sizeof(NN_DataType));
    memcpy(
        bramMlpResults,
        axiMlpResultsInput,
        *numberSamples * (*numberInputs + *numberLayers * *numberNeurons + *numberOutputs) * sizeof(NN_DataType));

    // calculate each iteration of backpropagation

    currentResults = bramMlpResults;
    currentClasses = bramClasses;

    computeOutputGradient<NN_DataType, ParEntriesOutput, streamDepth>(
        *numberOutputs,
        *numberNeurons,
        &currentResults[*numberLayers * *numberNeurons],
        currentClasses,
        &currentResults[(*numberLayers - 1) * *numberNeurons],
        &bramWeightGradient[(*numberInputs + (*numberLayers - 1) * *numberNeurons) * *numberNeurons],
        &bramBiasGradient[*numberLayers * *numberNeurons]);

    copyArray<NN_DataType, ParEntriesOutput>(
        &bramBiasGradient[*numberLayers * *numberNeurons],
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
            &bramWeightGradient[(*numberInputs + (layer - 1) * *numberNeurons) * *numberNeurons],
            &bramBiasGradient[layer * *numberNeurons]);

        copyArray<NN_DataType, ParEntriesOutput>(
            &bramBiasGradient[layer * *numberNeurons],
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
        bramWeightGradient,
        bramBiasGradient);
}