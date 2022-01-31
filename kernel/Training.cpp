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
    NN_DataType bramError[NumberOfHidden * N + NOutput];
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

    {
#pragma HLS DATAFLOW

        xf::blas::readVec2Stream<NN_DataType, ParEntries>(
            &currentResults[(*numberLayers - 1) * *numberNeurons],
            *numberNeurons,
            prevLayerOutputStream);

        computeGradient<NN_DataType, ParEntries>(
            *numberOutputs,
            *numberNeurons,
            errorStream,
            prevLayerOutputStream,
            &bramWeightGradient[(*numberInputs + (*numberLayers - 1) * *numberNeurons) * *numberNeurons],
            &bramBiasGradient[*numberLayers * *numberNeurons]);
    }

    // for (unsigned int layer = *numberLayers; layer > 0; layer--)
    // {

    //     NN_DataType *resultVector = network->layers[layer];
    //     NN_DataType + *con = network->connections[layer - 1];
    //     // calculate error term for hidden layer
    //     int hiddenLayer = layer - 1;
    //     transposeInto(network->connections[layer]->weights, WTi[hiddenLayer]);
    //     multiplyInto(errori[layer + 1], WTi[hiddenLayer], errorLastTi[hiddenLayer]);
    //     copyValuesInto(con->to->input, fprimei[hiddenLayer]);
    //     float (*derivative)(float) = activationDerivative(con->to->activation);
    //     for (j = 0; j < fprimei[hiddenLayer]->cols; j++)
    //     {
    //         fprimei[hiddenLayer]->data[j] = derivative(fprimei[hiddenLayer]->data[j]);
    //     }
    //     hadamardInto(errorLastTi[hiddenLayer], fprimei[hiddenLayer], errori[layer]);

    //     // calculate dWi and dbi
    //     transposeInto(con->from->input, inputTi[hiddenLayer]);
    //     multiplyInto(inputTi[hiddenLayer], errori[layer], dWi[layer - 1]);
    //     copyValuesInto(errori[layer], dbi[layer - 1]);
    // }
}