#pragma once

using NN_DataType = float;

// parameters
constexpr unsigned N = 4; //  Rows of A aka weight Matrix
constexpr unsigned K = N;  //  Cols of A and Rows of Input Vector

constexpr unsigned KInput = 4;
constexpr unsigned NOutput = 2;
constexpr unsigned NumberOfHidden = 2;
constexpr unsigned weightBufferSize = N * KInput + (NumberOfHidden - 1) * N * K + NOutput * K;
constexpr unsigned biasBufferSize = NumberOfHidden * N + NOutput;

// Cosim only works if the paralellism for input and hidden layers
// is the same
constexpr unsigned ParEntries = 16;
constexpr unsigned logParEntries = 4;
constexpr unsigned streamDepth = 16;

// Training
constexpr unsigned maxSamples = 20;
constexpr unsigned batchSize = 4;
constexpr NN_DataType learningRate = 0.5;
constexpr unsigned maxIters = 2000;

constexpr NN_DataType precision = 1e-3;

// General
constexpr char *projectPathString = "/home/thilo/master_thesis_code/uz_neural_network_hls_refactor/";