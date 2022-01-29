#pragma once

using NN_DataType = float;

// parameters
constexpr unsigned N = 32; //  Rows of A aka weight Matrix
constexpr unsigned K = N;  //  Cols of A and Rows of Input Vector
constexpr unsigned W = 4;  //  Replication factor in width aka vector type width
constexpr unsigned D = 4;  //  Partitioning factor of A
constexpr unsigned P = 4;  // Partition factor for internal buffers
constexpr unsigned PInput = 4;
constexpr unsigned POutput = 2;
constexpr unsigned WInput = 4;
constexpr unsigned DOutput = 4;

constexpr unsigned KInput = 8;
constexpr unsigned NOutput = 8;
constexpr unsigned NumberOfHidden = 4;

constexpr unsigned ParEntries = 8;
constexpr unsigned logParEntries = 3;

// Training
constexpr unsigned maxSamples = 10;

//void MyGemv(NN_DataType *weights, NN_DataType *input, const NN_DataType *bias, NN_DataType *output);
void MyGemv(
    NN_DataType *weights,
    NN_DataType *input,
    NN_DataType *bias,
    NN_DataType *output,
    unsigned int p_n,
    unsigned int p_k);

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
    const unsigned int *exportLayers);

void computeError(
    unsigned int p_n,
    unsigned int p_k,
    NN_DataType *weights,
    NN_DataType *inputError,
    NN_DataType *outputCurrentLayer,
    NN_DataType *outputError);

template <typename t_DataType, unsigned int t_ParEntries>
void computeWeightGradient(
    unsigned int p_n,
    unsigned int p_k,
    t_DataType *currentError,
    t_DataType *outputPrevLayer,
    t_DataType *weightGradient);

template <typename t_DataType>
t_DataType sigmoidDeriv(t_DataType sigmoidInput)
{
    return sigmoidInput * (1 - sigmoidInput);
}