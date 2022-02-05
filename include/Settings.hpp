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

// Cosim only works if the paralellism for input and hidden layers
// is the same
constexpr unsigned ParEntries = 8;
constexpr unsigned logParEntries = 3;
constexpr unsigned ParEntriesInput = ParEntries;
constexpr unsigned logParEntriesInput = logParEntries;
constexpr unsigned ParEntriesOutput = 2;
constexpr unsigned logParEntriesOutput = 1;
constexpr unsigned streamDepth = 16;

// Training
constexpr unsigned maxSamples = 10;
constexpr unsigned numberOutputs = 10;