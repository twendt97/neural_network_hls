#pragma once
#include <cstddef>

using NN_DataType = float;

// parameters
constexpr unsigned N = 16; //  Rows of A aka weight Matrix
constexpr unsigned K = N;  //  Cols of A and Rows of Input Vector

constexpr unsigned KInput = 8;
constexpr unsigned NOutput = 4;
constexpr unsigned NumberOfHidden = 2;
constexpr unsigned weightBufferSize = N * KInput + (NumberOfHidden - 1) * N * K + NOutput * K;
constexpr unsigned biasBufferSize = NumberOfHidden * N + NOutput;

constexpr std::size_t NUMBER_NEURONS = 4;
constexpr std::size_t NUMBER_INPUTS = 4;
constexpr std::size_t NUMBER_OUTPUTS = 2;
constexpr std::size_t NUMBER_HIDDEN = 1;


// Cosim only works if the paralellism for input and hidden layers
// is the same
constexpr unsigned ParEntries = 2;
constexpr unsigned logParEntries = 1;
constexpr unsigned streamDepth = 16;

// Training
constexpr unsigned maxSamples = 20;
constexpr unsigned batchSize = 16;
constexpr NN_DataType learningRate = 0.5;
constexpr unsigned epochs = 2000;

// General
// constexpr char *projectPathString = "/home/thilo/master_thesis_code/uz_neural_network_hls_refactor/";