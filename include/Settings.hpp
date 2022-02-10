#pragma once

using NN_DataType = float;

// parameters
constexpr unsigned N = 16; //  Rows of A aka weight Matrix
constexpr unsigned K = N;  //  Cols of A and Rows of Input Vector

constexpr unsigned KInput = 784;
constexpr unsigned NOutput = 10;
constexpr unsigned NumberOfHidden = 4;
constexpr unsigned weightBufferSize = N * KInput + (NumberOfHidden - 1) * N * K + NOutput * K;
constexpr unsigned biasBufferSize = NumberOfHidden * N + NOutput;

// Cosim only works if the paralellism for input and hidden layers
// is the same
constexpr unsigned ParEntries = 2;
constexpr unsigned logParEntries = 1;
constexpr unsigned ParEntriesInput = ParEntries;
constexpr unsigned logParEntriesInput = logParEntries;
constexpr unsigned ParEntriesOutput = 2;
constexpr unsigned logParEntriesOutput = 1;
constexpr unsigned streamDepth = 16;

// Training
constexpr unsigned maxSamples = 20;
constexpr unsigned numberOutputs = 10;
constexpr unsigned batchSize = 1;
constexpr NN_DataType learningRate = 3;
constexpr unsigned maxIters = 1;

constexpr NN_DataType precision = 1e-3;