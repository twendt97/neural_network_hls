#pragma once

#include <Eigen/Dense>
#include "Settings.hpp"

#include "MlpContainer.hpp"
#include "BgdContainer.hpp"

using NN_Matrix = Eigen::Matrix<NN_DataType, N, K, Eigen::RowMajor>;
using NN_Vector = Eigen::Matrix<NN_DataType, N, 1, Eigen::ColMajor>;
using NN_InputWeights = Eigen::Matrix<NN_DataType, N, KInput, Eigen::RowMajor>;
using NN_OutputWeights = Eigen::Matrix<NN_DataType, NOutput, K, Eigen::RowMajor>;
using NN_InputVector = Eigen::Matrix<NN_DataType, KInput, 1, Eigen::ColMajor>;
using NN_OutputVector = Eigen::Matrix<NN_DataType, NOutput, 1, Eigen::ColMajor>;
