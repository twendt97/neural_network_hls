#pragma once

using NN_DataType = float;

// parameters
constexpr unsigned N = 16; //  Rows of A aka weight Matrix
constexpr unsigned K = N;  //  Cols of A and Rows of Input Vector

constexpr unsigned KInput = 784;
constexpr unsigned NOutput = 16;
constexpr unsigned NumberOfHidden = 4;
constexpr unsigned weightBufferSize = N * KInput + (NumberOfHidden - 1) * N * K + NOutput * K;
constexpr unsigned biasBufferSize = NumberOfHidden * N + NOutput;

// Cosim only works if the paralellism for input and hidden layers
// is the same
constexpr unsigned ParEntries = 16;
constexpr unsigned logParEntries = 4;
constexpr unsigned streamDepth = 16;

// Training
constexpr unsigned maxSamples = 20;
constexpr unsigned batchSize = 20;
constexpr NN_DataType learningRate = 3;
constexpr unsigned maxIters = 100;

constexpr NN_DataType precision = 1e-3;