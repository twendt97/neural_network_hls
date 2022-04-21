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

#pragma once
#include <cstddef>

using NN_DataType = float;

// defines
// if ACTIVATION_RELU is defined, the relu function is used as activation
// otherwise sigmoid is applied
#define ACTIVATION_RELU

// parameters

// The following parameters are settings for the simulation
// Neural network
constexpr std::size_t simNumberNeurons = 64;
constexpr std::size_t simNumberInputs = 16;
constexpr std::size_t simNumberOutputs = 16;
constexpr std::size_t simNumberHidden = 3;

// Training
constexpr unsigned simBatchSize = 1;
constexpr NN_DataType learningRate = 0.5;
constexpr unsigned epochs = 2;

// parameters for MLP testbench
constexpr std::size_t axiWeightDepth = simNumberNeurons * (simNumberInputs + (simNumberHidden - 1) * simNumberNeurons + simNumberOutputs);
constexpr std::size_t axiBiasDepth = simNumberHidden * simNumberNeurons + simNumberOutputs;
constexpr std::size_t axiLayerResultsDepth = axiBiasDepth + simNumberInputs;
constexpr std::size_t axiBatchResultsDepth = simBatchSize * axiLayerResultsDepth;

// The following parameters determine the amount of BRAM ressources consumed by the kernels
// In order to fulfill the requirements of the application the following constraints must be satisfied
// for the expected target application
// However, the actual dimensions of the network can be set during runtime but the BRAM ressources are
// the limiting factor
// weightBufferSize >= NUMBER_NEURONS * (NUMBER_INPUTS + (NUMBER_HIDDEN - 1) * NUMBER_NEURONS + NUMBER_OUTPUTS)
// biasBufferSize >= NUMBER_HIDDEN * NUMBER_NEURONS + NUMBER_OUTPUTS
// hwMaxBatchSize >= expected batch size
// layerBufferSize >= NUMBER_NEURONS
// layerResultsBufferSize >= biasBufferSize + NUMBER_INPUTS

constexpr std::size_t hwNumberOutputs = 32;
constexpr std::size_t hwNumberInputs = 32;
constexpr std::size_t hwNumberNeurons = 128;
constexpr std::size_t hwNumberHiddenLayers = 3;
constexpr std::size_t weightBufferSize = hwNumberNeurons * (hwNumberInputs + (hwNumberHiddenLayers - 1) * hwNumberNeurons + hwNumberOutputs);
constexpr std::size_t biasBufferSize = hwNumberHiddenLayers * hwNumberNeurons + hwNumberOutputs;

constexpr std::size_t layerBufferSize = hwNumberNeurons;
constexpr std::size_t layerResultsBufferSize = biasBufferSize + hwNumberInputs;
constexpr std::size_t hwMaxBatchSize = 20;

constexpr unsigned ParEntries = 16;
constexpr unsigned logParEntries = 4;
constexpr unsigned streamDepth = 2;

// General
// constexpr char *projectPathString = "/home/thilo/master_thesis_code/uz_neural_network_hls_refactor/";