#pragma once

#include <Eigen/Dense>
#include "Cranium/src/cranium.h"
#include "Settings.hpp"
#include <math.h>
#include <iostream>

#include "MlpContainer.hpp"
#include "BgdContainer.hpp"

using NN_Matrix = Eigen::Matrix<NN_DataType, N, K, Eigen::RowMajor>;
using NN_Vector = Eigen::Matrix<NN_DataType, N, 1, Eigen::ColMajor>;
using NN_InputWeights = Eigen::Matrix<NN_DataType, N, KInput, Eigen::RowMajor>;
using NN_OutputWeights = Eigen::Matrix<NN_DataType, NOutput, K, Eigen::RowMajor>;
using NN_InputVector = Eigen::Matrix<NN_DataType, KInput, 1, Eigen::ColMajor>;
using NN_OutputVector = Eigen::Matrix<NN_DataType, NOutput, 1, Eigen::ColMajor>;

template <typename t_DataType>
void createXorTrainingData(t_DataType *trainingData, t_DataType *trainingClasses, std::size_t numberInputs, std::size_t numberOutputs)
{
    const std::size_t numberSamples = pow(2, numberInputs);
    // create truth table
    for (std::size_t i = 0; i < numberInputs; i++)
    {
        std::size_t range = numberSamples / pow(2, i + 1);
        t_DataType value = 1;
        for (std::size_t j = 0; j < numberSamples; j++)
        {
            if ((j % range) == 0)
                value = value != 0 ? 0 : 1;
            trainingData[j * numberInputs + i] = value;
        }
    }

    for (std::size_t i = 0; i < numberSamples; i++)
    {
        std::size_t numberOnes = 0;
        for (std::size_t j = 0; j < numberInputs; j++)
        {
            if (trainingData[i * numberInputs + j] != 0)
                numberOnes++;
            std::cout << trainingData[i * numberInputs + j];
        }
        if (numberOnes == 1)
        {
            trainingClasses[i * numberOutputs] = 1;
            trainingClasses[i * numberOutputs + 1] = 0;
        }
        else
        {
            trainingClasses[i * numberOutputs] = 0;
            trainingClasses[i * numberOutputs + 1] = 1;
        }
        std::cout << "  " << trainingClasses[i * numberOutputs] << trainingClasses[i * numberOutputs + 1] << std::endl
                  << std::flush;
        for (std::size_t j = 2; j < numberOutputs; j++)
        {
            trainingClasses[i * numberOutputs + j] = 0;
        }
    }
}

template <typename t_DataType>
void destroyTrainingData(t_DataType *trainingData, t_DataType *trainingClasses)
{
    free(trainingData);
    free(trainingClasses);
}
