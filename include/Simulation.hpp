#pragma once

#include <Eigen/Dense>
#include "Cranium/src/cranium.h"

// cranium network defines
constexpr size_t cranFeatures = 5;
constexpr size_t cranNumHiddenLayers = 5;
constexpr size_t cranHiddenSize = 64;
constexpr size_t cranOutputs = 8;
constexpr Activation defaultActivation = &linear;
constexpr float precTreshold = 1.0e-1;

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
    NN_DataType bramLayerResults[NumberOfHidden * N + NOutput],
    const unsigned int *numberInputs,
    const unsigned int *numberOutputs,
    const unsigned int *numberLayers,
    const unsigned int *numberNeurons,
    const unsigned int *loadParameters,
    const unsigned int *exportLayers);