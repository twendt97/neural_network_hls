#pragma once

#include <Eigen/Dense>
#include "Cranium/src/cranium.h"
#include "Settings.hpp"

using NN_Matrix = Eigen::Matrix<NN_DataType, N, K, Eigen::RowMajor>;
using NN_Vector = Eigen::Matrix<NN_DataType, N, 1, Eigen::ColMajor>;
using NN_InputWeights = Eigen::Matrix<NN_DataType, N, KInput, Eigen::RowMajor>;
using NN_OutputWeights = Eigen::Matrix<NN_DataType, NOutput, K, Eigen::RowMajor>;
using NN_InputVector = Eigen::Matrix<NN_DataType, KInput, 1, Eigen::ColMajor>;
using NN_OutputVector = Eigen::Matrix<NN_DataType, NOutput, 1, Eigen::ColMajor>;

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
    const unsigned int *exportLayers);

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
    const NN_DataType *learningRate);