#pragma once
#include <cstddef>

using NN_DataType = float;

// defines
// if ACTIVATION_RELU is defined, the relu function is used as activation
// otherwise sigmoid is applied
//#define ACTIVATION_RELU

// parameters

// The following parameters are settings for the simulation
// Neural network
constexpr std::size_t NUMBER_NEURONS = 4;
constexpr std::size_t NUMBER_INPUTS = 4;
constexpr std::size_t NUMBER_OUTPUTS = 4;
constexpr std::size_t NUMBER_HIDDEN = 1;
// Training
constexpr unsigned batchSize = 16;
constexpr NN_DataType learningRate = 0.5;
constexpr unsigned epochs = 2000;

// The following parameters determine the amount of BRAM ressources consumed by the kernels
// In order to fulfill the requirements of the application the following constraints must be satisfied
// for the expected target application
// However, the actual dimensions of the network can be set during runtime but the BRAM ressources are
// the limiting factor
// weightBufferSize >= NUMBER_NEURONS * (NUMBER_INPUTS + (NUMBER_HIDDEN - 1) * NUMBER_NEURONS + NUMBER_OUTPUTS)
// biasBufferSize >= NUMBER_HIDDEN * NUMBER_NEURONS + NUMBER_OUTPUTS
// maxSamples >= expected batch size
// layerBufferSize >= NUMBER_NEURONS
// layerResultsBufferSize >= biasBufferSize + NUMBER_INPUTS

constexpr std::size_t hwNumberOutputs = 8;
constexpr std::size_t hwNumberInputs = 8;
constexpr std::size_t hwNumberNeurons = 64;
constexpr std::size_t hwNumberHiddenLayers = 4;
constexpr std::size_t weightBufferSize = hwNumberNeurons * (hwNumberInputs + (hwNumberHiddenLayers - 1) * hwNumberNeurons + hwNumberOutputs);
constexpr std::size_t biasBufferSize = hwNumberHiddenLayers * hwNumberNeurons + hwNumberOutputs;

constexpr std::size_t layerBufferSize = hwNumberNeurons;
constexpr std::size_t layerResultsBufferSize = biasBufferSize + hwNumberInputs;
constexpr std::size_t maxSamples = 20;

constexpr unsigned ParEntries = 4;
constexpr unsigned logParEntries = 2;
constexpr unsigned streamDepth = 16;

// General
// constexpr char *projectPathString = "/home/thilo/master_thesis_code/uz_neural_network_hls_refactor/";