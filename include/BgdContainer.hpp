#pragma once
#include "Settings.hpp"
#include "MlpContainer.hpp"
#include <cstddef>
#include <cstdlib>

void BGD(
    const NN_DataType *axiMlpResultsInput,
    const NN_DataType *axiClassesInput,
    const NN_DataType *axiWeightInput,
    const NN_DataType *axiBiasInput,
    NN_DataType *axiWeightOutput,
    NN_DataType *axiBiasOutput,
    const unsigned int *numberInputs,
    const unsigned int *numberOutputs,
    const unsigned int *numberLayers,
    const unsigned int *numberNeurons,
    const unsigned int *loadParameters,
    const unsigned int *batchSize,
    const NN_DataType *learningRate);

class BgdContainer
{
private:
    MlpContainer *mlpInstance;

    NN_DataType *batchBuffer;
    NN_DataType *weightMemory;
    NN_DataType *biasMemory;
    NN_DataType *outputAddress;

    NN_DataType *trainingData;
    NN_DataType *trainingClasses;

    std::size_t numberNeurons;
    std::size_t numberHiddenLayers;
    std::size_t numberInputs;
    std::size_t numberOutputs;

    std::size_t weightBufferSize;
    std::size_t biasBufferSize;
    std::size_t layerBufferSize;
    std::size_t batchBufferSize;

    std::size_t numberTrainingSamples;
    std::size_t epochs;
    std::size_t batchSize;
    NN_DataType learningRate;

    void updateBufferSizes(void);
    void createTruthTableAsTrainingData(void);
    void xorTruthTableToClasses(void);
    void optimizeReference(void);
    void optimizeHardware(void);
    float hardwareAccuracy(void);
    std::size_t hardwarePrediction(void);

public:
    BgdContainer(
        std::size_t numberNeurons,
        std::size_t numberHiddenLayers,
        std::size_t numberInputs,
        std::size_t numberOutputs,
        std::size_t batchSize,
        std::size_t epochs,
        NN_DataType learningRate);

    ~BgdContainer();

    bool optimizeAndTest(void);
};

BgdContainer::~BgdContainer()
{
    free(this->batchBuffer);
    free(this->biasMemory);
    free(this->weightMemory);
    free(this->outputAddress);
    free(this->trainingClasses);
    free(this->trainingData);
    delete this->mlpInstance;
}

BgdContainer::BgdContainer(
    std::size_t numberNeurons,
    std::size_t numberHiddenLayers,
    std::size_t numberInputs,
    std::size_t numberOutputs,
    std::size_t batchSize,
    std::size_t epochs,
    NN_DataType learningRate) : numberNeurons(numberNeurons),
                                numberHiddenLayers(numberHiddenLayers),
                                numberInputs(numberInputs),
                                numberOutputs(numberOutputs),
                                batchSize(batchSize),
                                epochs(epochs),
                                learningRate(learningRate)
{
    this->updateBufferSizes();
    this->weightMemory = (NN_DataType *)malloc(this->weightBufferSize * sizeof(NN_DataType));
    this->biasMemory = (NN_DataType *)malloc(this->biasBufferSize * sizeof(NN_DataType));
    this->batchBuffer = (NN_DataType *)malloc(this->batchBufferSize * sizeof(NN_DataType));
    this->outputAddress = (NN_DataType *)malloc(this->numberOutputs * sizeof(NN_DataType));
    // just allocate some memory so it can be freed later
    this->trainingData = (NN_DataType *)malloc(this->numberInputs * sizeof(NN_DataType));
    this->trainingClasses = (NN_DataType *)malloc(sizeof(NN_DataType));
    this->mlpInstance = new MlpContainer(
        this->numberNeurons,
        this->numberHiddenLayers,
        this->numberInputs,
        this->numberOutputs,
        this->weightMemory,
        this->biasMemory,
        this->trainingData,
        this->outputAddress,
        this->batchBuffer);
}

void BgdContainer::updateBufferSizes(void)
{
    this->weightBufferSize =
        (numberInputs + (numberHiddenLayers - 1) * numberNeurons + numberOutputs) * numberNeurons;
    this->biasBufferSize = numberHiddenLayers * numberNeurons + numberOutputs;
    this->layerBufferSize = numberInputs + biasBufferSize;
    this->batchBufferSize = this->layerBufferSize * this->batchSize;
}

void BgdContainer::createTruthTableAsTrainingData(void)
{
    this->numberTrainingSamples = pow(2, this->numberInputs);
    assert(this->numberTrainingSamples % this->batchSize == 0);
    free(this->trainingData);
    this->trainingData = (NN_DataType *)malloc(this->numberTrainingSamples * this->numberInputs * sizeof(NN_DataType));
    this->mlpInstance->setInputAddress(this->trainingData);
    // create truth table
    for (std::size_t i = 0; i < this->numberInputs; i++)
    {
        std::size_t range = this->numberTrainingSamples / pow(2, i + 1);
        NN_DataType value = 1;
        for (std::size_t j = 0; j < this->numberTrainingSamples; j++)
        {
            if ((j % range) == 0)
                value = value != 0 ? 0 : 1;
            this->trainingData[j * numberInputs + i] = value;
        }
    }
}

void BgdContainer::xorTruthTableToClasses(void)
{
    free(this->trainingClasses);
    this->trainingClasses = (NN_DataType *)malloc(this->numberTrainingSamples * this->numberOutputs * sizeof(NN_DataType));
    for (std::size_t i = 0; i < this->numberTrainingSamples; i++)
    {
        std::size_t numberOnes = 0;
        for (std::size_t j = 0; j < numberInputs; j++)
        {
            if (trainingData[i * numberInputs + j] != 0)
                numberOnes++;
        }
        if (numberOnes == 1)
        {
            this->trainingClasses[i * this->numberOutputs] = 1;
            this->trainingClasses[i * this->numberOutputs + 1] = 0;
        }
        else
        {
            this->trainingClasses[i * this->numberOutputs] = 0;
            this->trainingClasses[i * this->numberOutputs + 1] = 1;
        }

        // fill remaining output entries with 0
        for (std::size_t j = 2; j < this->numberOutputs; j++)
        {
            this->trainingClasses[i * this->numberOutputs + j] = 0;
        }
    }
}

void BgdContainer::optimizeReference(void)
{
    NN_DataType **craniumTrainingData = (NN_DataType **)malloc(this->numberTrainingSamples * sizeof(NN_DataType *));
    NN_DataType **craniumClassesData = (NN_DataType **)malloc(this->numberTrainingSamples * sizeof(NN_DataType *));

    // copy training data from class because cranium will free the allocated memory
    // when calling the function destroyDataSet(...)
    for (std::size_t i = 0; i < this->numberTrainingSamples; i++)
    {
        craniumTrainingData[i] = (NN_DataType *)malloc(this->numberInputs * sizeof(NN_DataType));
        for (std::size_t j = 0; j < this->numberInputs; j++)
        {
            craniumTrainingData[i][j] = this->trainingData[i * this->numberInputs + j];
        }
        craniumClassesData[i] = (NN_DataType *)malloc(this->numberOutputs * sizeof(NN_DataType));
        for (std::size_t j = 0; j < this->numberOutputs; j++)
        {
            craniumClassesData[i][j] = this->trainingClasses[i * this->numberOutputs + j];
        }
    }

    DataSet *cranTrainingDataSet = createDataSet(this->numberTrainingSamples, this->numberInputs, craniumTrainingData);
    DataSet *cranTrainingClassesSet = createDataSet(this->numberTrainingSamples, this->numberOutputs, craniumClassesData);
    ParameterSet params = {
        .network = this->mlpInstance->getReferenceImplementation(),
        .data = cranTrainingDataSet,
        .classes = cranTrainingClassesSet,
        .lossFunction = MEAN_SQUARED_ERROR,
        .batchSize = this->batchSize,
        .learningRate = this->learningRate,
        .searchTime = 0,
        .regularizationStrength = 0,
        .momentumFactor = 0,
        .maxIters = (int)(this->epochs * (this->numberTrainingSamples / this->batchSize)),
        .shuffle = 0,
        .verbose = 1};
    optimize(params);

    // test accuracy of network after training
    std::cout << "Accuracy of Cranium training is "
              << accuracy(this->mlpInstance->getReferenceImplementation(), cranTrainingDataSet, cranTrainingClassesSet)
              << std::endl
              << std::flush;
    destroyDataSet(cranTrainingDataSet);
    destroyDataSet(cranTrainingClassesSet);
}

void BgdContainer::optimizeHardware(void)
{
    unsigned int inputs = this->numberInputs;
    unsigned int outputs = this->numberOutputs;
    unsigned int hidden = this->numberHiddenLayers;
    unsigned int neurons = this->numberNeurons;
    unsigned int batchSize = this->batchSize;
    NN_DataType learningRate = this->learningRate;

    this->mlpInstance->setExportLayers(true);
    for (std::size_t i = 0; i < this->epochs; i++)
    {
        unsigned int loadParametersBgd = i == 0 ? (unsigned int)1 : (unsigned int)0;
        for (std::size_t j = 0; j < this->numberTrainingSamples / this->batchSize; j++)
        {
            this->mlpInstance->setLoadParameters(true);
            for (std::size_t k = 0; k < this->batchSize; k++)
            {
                this->mlpInstance->setInputAddress(&this->trainingData[this->numberInputs * (j * this->batchSize + k)]);
                this->mlpInstance->setLayerBufferAddress(&this->batchBuffer[this->layerBufferSize * k]);
                this->mlpInstance->feedForward();
                this->mlpInstance->setLoadParameters(false);
            }
            BGD(
                this->batchBuffer,
                &this->trainingClasses[j * this->numberOutputs],
                this->weightMemory,
                this->biasMemory,
                this->weightMemory,
                this->biasMemory,
                &inputs,
                &outputs,
                &hidden,
                &neurons,
                &loadParametersBgd,
                &batchSize,
                &learningRate);
        }
    }

    std::cout << "Accuracy of hardware training is " << hardwareAccuracy() << std::endl
              << std::flush;
}

std::size_t BgdContainer::hardwarePrediction(void)
{
    std::size_t max = 0;
    for (std::size_t i = 0; i < this->numberOutputs; i++)
    {
        if (this->outputAddress[i] > this->outputAddress[max])
            max = i;
    }
    return max;
}

float BgdContainer::hardwareAccuracy(void)
{
    std::size_t prediction = 0;
    float numCorrect = 0;
    this->mlpInstance->setExportLayers(false);
    this->mlpInstance->setLoadParameters(true);
    for (std::size_t i = 0; i < this->numberTrainingSamples; i++)
    {
        this->mlpInstance->setInputAddress(&this->trainingData[i * this->numberInputs]);
        this->mlpInstance->feedForward();
        this->mlpInstance->setLoadParameters(false);
        prediction = this->hardwarePrediction();
        if (this->trainingClasses[i * this->numberOutputs + prediction] == 1)
            numCorrect++;
    }
    return (numCorrect / this->numberTrainingSamples);
}

bool BgdContainer::optimizeAndTest(void)
{
    this->createTruthTableAsTrainingData();
    this->xorTruthTableToClasses();
    std::cout << "Starting optimization of the reference implementation..." << std::endl << std::flush;
    this->optimizeReference();
    std::cout << "Starting hardware optimization..." << std::endl << std::flush;
    this->optimizeHardware();
    return this->mlpInstance->hwParametersEqualReference();
}