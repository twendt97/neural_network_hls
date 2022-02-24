// defines for compilation behavior
#define TEST_MLP_CRANIUM
#define TEST_TRAINING
// #define EXPORT_MNIST
// #define TEST_TRAINING_COMPONENTS

#include "Settings.hpp"
#include "Simulation.hpp"
#include <vector>
#include <iostream>
#include <random>
#include <cstddef>
#include <fstream>

#include "MNIST_Extractor/include/mnist/mnist_reader.hpp"

/**
 * @brief Takes an array of uint8_t and scales it to a float between 0 and 1
 *
 * @
 * */
template <typename t_DataType, template <typename...> class Container = std::vector, template <typename...> class Sub = std::vector, typename Pixel = uint8_t, typename Label = uint8_t>
void scaleImages(
    mnist::MNIST_dataset<Container, Sub<Pixel>, Label> &dataSet,
    t_DataType *outputArray);

template <typename t_DataType, template <typename...> class Container = std::vector, template <typename...> class Sub = std::vector, typename Pixel = uint8_t, typename Label = uint8_t>
void createClasses(
    mnist::MNIST_dataset<Container, Sub<Pixel>, Label> &dataSet,
    t_DataType *outputArray,
    unsigned int numberOutputs);

int main(void)
{
    int return_value = 0;
    srand((unsigned int)2);

#ifdef TEST_MLP_CRANIUM
    MlpContainer *mlp = new MlpContainer(NUMBER_NEURONS, NUMBER_HIDDEN, NUMBER_INPUTS, NUMBER_OUTPUTS);
    mlp->testHwAgainstReference(1e-3);
    delete mlp;

#endif

#ifdef TEST_TRAINING

    // std::cout << "Extracting MNIST dataset..." << std::endl
    //           << std::flush;
    // auto mnistDataSet = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(projectPath + "MNIST_Extractor", 5, 5);
    // std::size_t numberImages, imageSize, numberLabels;
    // numberImages = mnistDataSet.training_images.size();
    // numberLabels = mnistDataSet.training_labels.size();
    // assert(numberImages == numberLabels);
    // imageSize = mnistDataSet.training_images.data()->size();

    // NN_DataType *mnistTrainScaledImages = (NN_DataType *)malloc(numberImages * imageSize * sizeof(NN_DataType));
    // assert(mnistTrainScaledImages != NULL);
    // scaleImages<NN_DataType>(mnistDataSet, mnistTrainScaledImages);

    // NN_DataType *mnistClassesVector = (NN_DataType *)malloc(numberLabels * NOutput * sizeof(NN_DataType));
    // assert(mnistClassesVector != NULL);
    // createClasses<NN_DataType>(mnistDataSet, mnistClassesVector, NOutput);

    BgdContainer *bgd = new BgdContainer(NUMBER_NEURONS, NUMBER_HIDDEN, NUMBER_INPUTS, NUMBER_OUTPUTS, batchSize, epochs, learningRate);
    bgd->optimizeAndTest();
    delete bgd;

#ifdef EXPORT_MNIST
    std::ofstream mnistCsv;
    mnistCsv.open(projectPath + "export/mnist_data.csv");
    for (size_t i = 0; i < numberImages * imageSize; i++)
    {
        mnistCsv << mnistTrainScaledImages[i];
        if (i < numberImages * imageSize - 1)
            mnistCsv << ",";
    }

    mnistCsv << std::flush;
    mnistCsv.close();
    mnistCsv.open(projectPath + "export/mnist_labels.csv");
    for (size_t i = 0; i < numberLabels * NOutput; i++)
    {
        mnistCsv << mnistClassesVector[i];
        if (i < numberLabels * NOutput - 1)
            mnistCsv << ",";
    }

    mnistCsv << std::flush;
    mnistCsv.close();
#endif

#endif

    if (return_value == 0)
        std::cout << "Test successful!!!" << std::endl
                  << std::flush;
    else
        std::cout << "Test failed!!!" << std::endl
                  << std::flush;

    return return_value;
}

template <typename t_DataType, template <typename...> class Container = std::vector, template <typename...> class Sub = std::vector, typename Pixel = uint8_t, typename Label = uint8_t>
void scaleImages(
    mnist::MNIST_dataset<Container, Sub<Pixel>, Label> &dataSet,
    t_DataType *outputArray)
{
    std::size_t numberElements, numberFeatures;
    numberElements = dataSet.test_images.size();
    numberFeatures = dataSet.test_images.data()->size();
    for (std::size_t i = 0; i < numberElements; i++)
    {
        for (std::size_t j = 0; j < numberFeatures; j++)
        {
            outputArray[i * numberFeatures + j] = ((t_DataType)dataSet.training_images[i].data()[j]) / 255.0;
        }
    }
}

/**
 * @brief Takes the classes as integer number between 0 and numberOutputs - 1 and returns a vector
 *        where the corresponding index number is 1 and the others are 0
 * */

template <typename t_DataType, template <typename...> class Container = std::vector, template <typename...> class Sub = std::vector, typename Pixel = uint8_t, typename Label = uint8_t>
void createClasses(
    mnist::MNIST_dataset<Container, Sub<Pixel>, Label> &dataSet,
    t_DataType *outputArray,
    unsigned int numberOutputs)
{
    std::size_t numberEntries = dataSet.training_labels.size();

    for (std::size_t i = 0; i < numberEntries; i++)
    {
        for (Label j = 0; j < numberOutputs; j++)
        {
            outputArray[i * numberOutputs + j] = dataSet.training_labels[i] == j ? (t_DataType)1 : (t_DataType)0;
        }
    }
}