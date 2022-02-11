#pragma once

#include "Vitis_Libraries/blas/L1/include/hw/xf_blas.hpp"
#include "ap_int.h"
#include "hls_stream.h"
#include "hls_math.h"

namespace uz_mlp
{
    // Activation functions

    template <typename t_DataType>
    t_DataType relu(t_DataType x)
    {
        if (x > 0)
            return x;
        else
            return 0;
    }

    template <typename t_DataType>
    t_DataType sigmoid(t_DataType x)
    {
        t_DataType l_exp = hls::expf(-x);
        return 1.0f / (1.0f + l_exp);
    }

    template <typename t_DataType>
    t_DataType sigmoidDeriv(t_DataType sigmoidInput)
    {
        return sigmoidInput * (1 - sigmoidInput);
    }

    template <typename t_DataType>
    t_DataType reluDeriv(t_DataType reluInput)
    {
        return reluInput > 0 ? 1 : 0;
    }

    template <typename t_DataType, unsigned int t_ParEntries>
    void applyFunction(
        hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> &p_in,
        hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> &p_out,
        unsigned int p_n,
        t_DataType (*function)(t_DataType))
    {
#ifndef __SYNTHESIS__
        assert((p_n % t_ParEntries) == 0);
#endif
        unsigned int l_parBlocks = p_n / t_ParEntries;

        for (unsigned int i = 0; i < l_parBlocks; i++)
        {
#pragma HLS PIPELINE
            xf::blas::WideType<t_DataType, t_ParEntries> l_in, l_out;
            l_in = p_in.read();
            for (unsigned int j = 0; j < t_ParEntries; j++)
            {
                l_out[j] = function(l_in[j]);
            }
            p_out.write(l_out);
        }
    }

    // Layer processing

    template <typename t_DataType, unsigned int t_ParEntries, unsigned int t_logParEntries>
    void processLayer(
        t_DataType *p_weights,
        t_DataType *p_input,
        t_DataType *p_bias,
        t_DataType *p_output,
        unsigned int p_n,
        unsigned int p_k)
    {
#pragma HLS DATAFLOW
        // Stream that holds ParEntries operands
        hls::stream<typename xf::blas::WideType<t_DataType, 1 << t_logParEntries>::t_TypeInt> l_strWeights("Weights");
        hls::stream<typename xf::blas::WideType<t_DataType, 1 << t_logParEntries>::t_TypeInt> l_strInput("Input");
        // Stream that holds exactly one operand.
        // This is fed to a function that assembles a vector from the incoming entries
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> l_strOutput("Output");
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> l_strMv("Matrix Vector Result");
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> l_strBias("Bias");
#pragma HLS DATAFLOW
        xf::blas::gem2Stream<t_DataType, t_ParEntries>(p_n, p_k, p_weights, l_strWeights);
        xf::blas::vec2GemStream<t_DataType, t_ParEntries>(p_n, p_k, p_input, l_strInput);
        xf::blas::readVec2Stream<t_DataType, 1>(p_bias, p_n, l_strBias);
        xf::blas::gemv<t_DataType, t_logParEntries>(p_n, p_k, (t_DataType)1, l_strWeights, l_strInput, (t_DataType)1, l_strBias, l_strMv);
        applyFunction<t_DataType, 1>(l_strMv, l_strOutput, p_n, uz_mlp::sigmoid<t_DataType>);
        xf::blas::writeStream2Vec<t_DataType, 1>(l_strOutput, p_n, p_output);
    }

    template <typename t_DataType, unsigned int t_ParEntries, unsigned int t_logParEntries>
    void outputLayer(
        t_DataType *p_weights,
        t_DataType *p_input,
        t_DataType *p_bias,
        t_DataType *p_output,
        unsigned int p_n,
        unsigned int p_k)
    {
#pragma HLS DATAFLOW
        // Stream that holds ParEntries operands
        hls::stream<typename xf::blas::WideType<t_DataType, 1 << t_logParEntries>::t_TypeInt> l_strWeights("Weights");
        hls::stream<typename xf::blas::WideType<t_DataType, 1 << t_logParEntries>::t_TypeInt> l_strInput("Input");
        // Stream that holds exactly one operand.
        // This is fed to a function that assembles a vector from the incoming entries
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> l_strMv("Matrix Vector Result");
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> l_strBias("Bias");
#pragma HLS DATAFLOW
        xf::blas::gem2Stream<t_DataType, t_ParEntries>(p_n, p_k, p_weights, l_strWeights);
        xf::blas::vec2GemStream<t_DataType, t_ParEntries>(p_n, p_k, p_input, l_strInput);
        xf::blas::readVec2Stream<t_DataType, 1>(p_bias, p_n, l_strBias);
        xf::blas::gemv<t_DataType, t_logParEntries>(p_n, p_k, (t_DataType)1, l_strWeights, l_strInput, (t_DataType)1, l_strBias, l_strMv);
        xf::blas::writeStream2Vec<t_DataType, 1>(l_strMv, p_n, p_output);
    }

    template <typename t_DataType, unsigned int t_ParEntries>
    void copyArray(
        t_DataType *p_input,
        t_DataType *p_output,
        unsigned int size)
    {
#ifndef __SYNTHESIS__
        assert((size % t_ParEntries) == 0);
#endif
        unsigned int l_parBlocks = size / t_ParEntries;
    COPY_ARRAY:
        for (unsigned int i = 0; i < l_parBlocks; i++)
        {
#pragma HLS PIPELINE
            for (unsigned int j = 0; j < t_ParEntries; j++)
            {
#pragma HLS UNROLL
                p_output[i * t_ParEntries + j] = p_input[i * t_ParEntries + j];
            }
        }
    }

    // Training

    /**
     * @brief gem2Stream function that moves row-major matrix from memory to stream and
     *        performs on the fly transposition
     *
     * @tparam t_DataType the data type of the matrix entries
     * @tparam t_ParEntries number of parallelly processed entries in the matrix
     *
     * @param p_n number of rows in p_input matrix
     * @param p_k number of cols in p_input matrix
     * @param p_in a p_n x p_k matrix with on-chip row-major storage
     * @param p_out output stream
     */
    template <typename t_DataType, unsigned int t_ParEntries>
    void gem2StreamTranspose(unsigned int p_n,
                             unsigned int p_k,
                             t_DataType *p_in,
                             hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> &p_out)
    {
#ifndef __SYNTHESIS__
        assert((p_n % t_ParEntries) == 0);
#endif
        unsigned int l_parBlocksPerCol = p_n / t_ParEntries;
        for (unsigned int i = 0; i < p_k; ++i)
        {
            for (unsigned int k = 0; k < l_parBlocksPerCol; ++k)
            {
#pragma HLS PIPELINE
                xf::blas::WideType<t_DataType, t_ParEntries> l_val;
                for (unsigned int j = 0; j < t_ParEntries; ++j)
                {
                    l_val[j] = p_in[(j + k * t_ParEntries) * p_k + i];
                }
                p_out.write(l_val);
            }
        }
    } // end gem2StreamTranspose

    /**
     * @brief Produce equivelant output to xf::blas::readVec2Stream but the output contains only zeroes
     *
     * @tparam t_DataType the data type of the matrix entries
     * @tparam t_ParEntries number of parallelly processed entries in the matrix
     *
     * @param p_n number of entries in a vectpr
     * @param p_out output stream
     * */
    template <typename t_DataType, unsigned int t_ParEntries>
    void streamZero(
        unsigned int p_n,
        hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> &p_out)
    {
#ifndef __SYNTHESIS__
        assert((p_n % t_ParEntries) == 0);
#endif
        unsigned int l_parBlocks = p_n / t_ParEntries;
        for (unsigned int i = 0; i < l_parBlocks; ++i)
        {
#pragma HLS PIPELINE
            xf::blas::WideType<t_DataType, t_ParEntries> l_val;
            for (unsigned int j = 0; j < t_ParEntries; ++j)
            {
                l_val[j] = (t_DataType)0;
            }
            p_out.write(l_val);
        }
    }

    /**
     * @brief Compute the hadamard product of x and y
     *
     * @tparam t_DataType the data type of the vector entries
     * @tparam t_ParEntries number of parallelly processed entries
     *
     * @param p_n Number of entries in the vectors x and y
     * @param p_x Input stream of x
     * @param p_y Input stream of y
     * @param p_out output stream with results
     * */
    template <typename t_DataType, unsigned int t_ParEntries>
    void hadamardProduct(
        unsigned int p_n,
        hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> &p_x,
        hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> &p_y,
        hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> &p_out)
    {
#ifndef __SYNTHESIS__
        assert((p_n % t_ParEntries) == 0);
#endif
        unsigned int l_parBlocks = p_n / t_ParEntries;

        for (unsigned int i = 0; i < l_parBlocks; i++)
        {
#pragma HLS PIPELINE
            xf::blas::WideType<t_DataType, t_ParEntries> l_x, l_y, l_result;
            l_x = p_x.read();
            l_y = p_y.read();
            for (unsigned int j = 0; j < t_ParEntries; j++)
            {
                l_result[j] = l_x[j] * l_y[j];
            }
            p_out.write(l_result);
        }
    }

    /**
     * @brief Compute the error of the results given in p_outputCurrentLayer
     *
     * @tparam t_DataType the data type of the vector entries
     * @tparam t_ParEntries number of parallelly processed entries
     * @tparam t_logParEntries log2 of t_ParEntries
     *
     *
     * @param p_n Number of rows in weight matrix and number of rows in error vector of latter layer
     * @param p_k Number of cols in weight matrix, number of rows of the output error vector and number
     *            of rows in output vector of current layer
     * @param p_weights Weight matrix between the current layer and the latter layer
     * @param p_latterError Error of the latter layer
     * @param p_outputCurrentLayer Output of the current layer
     * @param p_outputError Stream with the values of the error vector
     *
     * */
    template <typename t_DataType, unsigned int t_ParEntries, unsigned int t_logParEntries>
    void computeHiddenError(
        unsigned int p_n,
        unsigned int p_k,
        t_DataType *p_weights,
        t_DataType *p_latterError,
        t_DataType *p_outputCurrentLayer,
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> &p_outputError)
    {
#pragma HLS DATAFLOW
        hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> l_strWeights("Transposed weights");
        hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> l_strLatterError("Input error");
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> l_strZero("Zero stream");
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> l_strMv("Matrix Vector product");
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> l_strOutputCurLayer("Output current layer");
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> l_strActivDeriv("Activation Derivative");
        // hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> l_strOutputError("Final error");
#pragma HLS DATAFLOW
        uz_mlp::gem2StreamTranspose<t_DataType, t_ParEntries>(p_n, p_k, p_weights, l_strWeights);
        xf::blas::vec2GemStream<t_DataType, t_ParEntries>(p_k, p_n, p_latterError, l_strLatterError);
        uz_mlp::streamZero<t_DataType, 1>(p_k, l_strZero);
        xf::blas::gemv<t_DataType, t_logParEntries>(p_k, p_n, (t_DataType)1, l_strWeights, l_strLatterError, (t_DataType)1, l_strZero, l_strMv);
        xf::blas::readVec2Stream<t_DataType, 1>(p_outputCurrentLayer, p_k, l_strOutputCurLayer);
        uz_mlp::applyFunction<t_DataType, 1>(l_strOutputCurLayer, l_strActivDeriv, p_k, uz_mlp::sigmoidDeriv<t_DataType>);
        uz_mlp::hadamardProduct<t_DataType, 1>(p_k, l_strMv, l_strActivDeriv, p_outputError);
        // xf::blas::writeStream2Vec<t_DataType, 1>(l_strOutputError, p_k, outputError);
    }

    /**
     * @brief Compute the error of the outputs of the neural network which happens to be
     *        p_results - p_classes
     *
     * @tparam t_DataType Data type of the entries to be processed
     * @tparam t_ParEntries Width of the output stream
     *
     * @param p_n Number of entries in p_results and p_classes
     * @param p_results Pointer to the results of the output layer
     * @param p_classes Pointer to the the expected results of the output layer
     * @param p_out Output stream
     * */
    template <typename t_DataType, unsigned int t_ParEntries>
    void computeOutputError(
        unsigned int p_n,
        t_DataType *p_results,
        t_DataType *p_classes,
        hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> &p_out)
    {
#ifndef __SYNTHESIS__
        assert((p_n % t_ParEntries) == 0);
#endif
        unsigned int l_parBlocks = p_n / t_ParEntries;
        // error aka partial bias derivative of output layer
        for (unsigned int i = 0; i < l_parBlocks; i++)
        {
#pragma HLS PIPELINE
            xf::blas::WideType<t_DataType, t_ParEntries> l_error;
            for (unsigned int j = 0; j < t_ParEntries; j++)
            {
                l_error[j] = p_results[i * t_ParEntries + j] - p_classes[i * t_ParEntries + j];
            }
            p_out.write(l_error);
        }
    }

    /**
     * @brief Compute the gradient of every element in the weight matrix
     *
     * @tparam t_DataType Datatype of the vectors
     * @tparam t_ParEntries Number of parallel processed entries of the outputPrevLayer vector
     *
     * @param p_n number of rows of the output matrix and number of entries in the p_currentErrorInput vector
     * @param p_k number of cols of the output matrix and number of entries in the outputPrevLayer vector
     * @param p_currentErrorInput Error vector of the current layer
     * @param outputPrevLayer Output of the previous layer
     *
     * */
    template <typename t_DataType, unsigned int t_ParEntries>
    void computeGradient(
        unsigned int p_n,
        unsigned int p_k,
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> &p_currentErrorInput,
        hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> &p_outputPrevLayer,
        hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> &p_weightGradient,
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> &p_biasGradient,
        t_DataType *p_currentErrorOutput)
    {
#ifndef __SYNTHESIS__
        assert((p_k % t_ParEntries) == 0);
#endif
        for (unsigned int n = 0; n < p_n; n++)
        {
            xf::blas::WideType<t_DataType, 1> l_currentError = p_currentErrorInput.read();
            p_currentErrorOutput[n] = l_currentError[0];
            p_biasGradient.write(l_currentError);
            for (unsigned int k = 0; k < p_k / t_ParEntries; k++)
            {
#pragma HLS PIPELINE
                xf::blas::WideType<t_DataType, t_ParEntries> l_outputPrevLayer = p_outputPrevLayer.read();
                xf::blas::WideType<t_DataType, t_ParEntries> l_weightGradient;
                for (unsigned int j = 0; j < t_ParEntries; j++)
                {
                    l_weightGradient[j] = l_outputPrevLayer[j] * l_currentError[0];
                    // p_weightGradient[n * p_k + k * t_ParEntries + j] = l_outputPrevLayer[j] * l_currentError[0];
                }
                p_weightGradient.write(l_weightGradient);
            }
        }
    }

    /**
     * @brief Compute p_accumulate += p_values
     *
     * @tparam t_DataType Datatype of the vectors
     * @tparam t_ParEntries Number of parallel processed entries
     *
     * @param p_accumulator Array from which the values are read from and written to
     * @param p_values Input stream with values that are added to p_accumulator
     * @param p_size Size of p_accumulator and number of values fed by p_values
     * @param p_initZero If true p_accumulator is initialized with zeroes otherwise the above mentioned
     *        computation is performed
     *
     * */
    template <typename t_DataType, unsigned int t_ParEntries>
    void accumulate(
        t_DataType *p_accumulator,
        hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> &p_values,
        unsigned int p_size,
        bool p_initZero)
    {
#ifndef __SYNTHESIS__
        assert((p_size % t_ParEntries) == 0);
#endif
        unsigned int l_parBlocks = p_size / t_ParEntries;
        for (unsigned int i = 0; i < l_parBlocks; i++)
        {
#pragma HLS PIPELINE
            xf::blas::WideType<t_DataType, t_ParEntries> l_values = p_values.read();
            for (unsigned int j = 0; j < t_ParEntries; j++)
            {
                t_DataType l_accumulator = p_initZero == true ? (t_DataType)0 : p_accumulator[i * t_ParEntries + j];
                p_accumulator[i * t_ParEntries + j] = l_accumulator + l_values[j];
            }
        }
    }

    /**
     * @brief Compute the gradients for weights and biases for the output layer
     *
     * @tparam t_DataType Datatype of the vectors
     * @tparam t_ParEntries Number of parallel processed entries of the outputPrevLayer vector
     * @tparam t_StreamDepth Depth of the FIFO between the error and the gradient compuatation engine
     *
     * @param p_n number of rows of p_weightGradientAvg and number of entries in the output vector of the MLP and
     *            number of entries in p_biasGradientAvg
     * @param p_k number of cols of p_weightGradientAvg, number of entries in the p_outputPrevLayer vector
     * @param p_outputPrevLayer Output of the last hidden layer
     * @param p_weightGradientAvg Matrix with gradients of the weights
     * @param p_biasGradientAvg Vector with the gradients of the bias
     *
     * */
    template <typename t_DataType, unsigned int t_ParEntries, unsigned int t_StreamDepth>
    void computeOutputGradient(
        unsigned int p_n,
        unsigned int p_k,
        t_DataType *p_results,
        t_DataType *p_classes,
        t_DataType *p_outputPrevLayer,
        t_DataType *p_weightGradientAvg,
        t_DataType *p_biasGradientAvg,
        t_DataType *p_error,
        bool p_initZero)
    {
#pragma HLS DATAFLOW
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> l_outputErrorStream("Output error");
#pragma HLS stream variable = l_outputErrorStream depth = t_StreamDepth
        hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> l_outputPrevLayer("Output prev layer");
        hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> l_weightGradient("Weight Gradient");
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> l_biasGradient("Bias Gradient");

#pragma HLS DATAFLOW
        uz_mlp::computeOutputError<t_DataType, 1>(
            p_n,
            p_results,
            p_classes,
            l_outputErrorStream);

        xf::blas::vec2GemStream<t_DataType, t_ParEntries>(
            p_n,
            p_k,
            p_outputPrevLayer,
            l_outputPrevLayer);

        uz_mlp::computeGradient<t_DataType, t_ParEntries>(
            p_n,
            p_k,
            l_outputErrorStream,
            l_outputPrevLayer,
            l_weightGradient,
            l_biasGradient,
            p_error);

        uz_mlp::accumulate<t_DataType, t_ParEntries>(
            p_weightGradientAvg,
            l_weightGradient,
            p_n * p_k,
            p_initZero);

        uz_mlp::accumulate<t_DataType, 1>(
            p_biasGradientAvg,
            l_biasGradient,
            p_n,
            p_initZero);
    }

    /**
     * @brief Compute the weight and bias gradient for the matrixes between p_outputCurrentLayer and the latter layer
     *
     * @tparam t_DataType the data type of the vector entries
     * @tparam t_ParEntries number of parallelly processed entries
     * @tparam t_logParEntries log2 of t_ParEntries
     * @tparam t_StreamDepth Depth of the FIFO between the error and the gradient compuatation engine
     *
     *
     * @param p_n Number of rows in weight matrix and number of rows in error vector of latter layer
     * @param p_k Number of cols in weight matrix, number of rows of the output error vector and number
     *            of rows in output vector of current layer
     * @param p_weights Weight matrix between the current layer and the latter layer
     * @param p_latterError Error of the latter layer
     * @param p_outputCurrentLayer Output of the current layer
     * @param p_outputPrevLayer Output of the previous layer
     * @param p_weightGradientAvg Matrix with gradients of the weights
     * @param p_biasGradientAvg Vector with the gradients of the bias
     *
     * */
    template <typename t_DataType, unsigned int t_ParEntries, unsigned int t_logParEntries, unsigned int t_StreamDepth = 2>
    void computeHiddenGradient(
        unsigned int p_n,
        unsigned int p_k,
        unsigned int p_numberOutputsPrev,
        t_DataType *p_weights,
        t_DataType *p_latterError,
        t_DataType *p_outputCurrentLayer,
        t_DataType *p_outputPrevLayer,
        t_DataType *p_weightGradientAvg,
        t_DataType *p_biasGradientAvg,
        t_DataType *p_error,
        bool p_initZero)
    {
#pragma HLS DATAFLOW
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> l_errorStream("Error current layer");
#pragma HLS stream variable = l_outputErrorStream depth = t_StreamDepth
        hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> l_outputPrevLayer("Output prev layer");
        hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> l_weightGradient("Weight Gradient");
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> l_biasGradient("Bias Gradient");

#pragma HLS DATAFLOW
        uz_mlp::computeHiddenError<t_DataType, t_ParEntries, t_logParEntries>(
            p_n,
            p_k,
            p_weights,
            p_latterError,
            p_outputCurrentLayer,
            l_errorStream);

        xf::blas::vec2GemStream<t_DataType, t_ParEntries>(
            p_k,
            p_numberOutputsPrev,
            p_outputPrevLayer,
            l_outputPrevLayer);

        uz_mlp::computeGradient<t_DataType, t_ParEntries>(
            p_k,
            p_numberOutputsPrev,
            l_errorStream,
            l_outputPrevLayer,
            l_weightGradient,
            l_biasGradient,
            p_error);

        uz_mlp::accumulate<t_DataType, t_ParEntries>(
            p_weightGradientAvg,
            l_weightGradient,
            p_k * p_numberOutputsPrev,
            p_initZero);

        uz_mlp::accumulate<t_DataType, 1>(
            p_biasGradientAvg,
            l_biasGradient,
            p_k,
            p_initZero);
    }

    template <typename t_DataType, unsigned int t_ParEntries>
    void updateParameter(
        t_DataType *p_weights,
        t_DataType *p_bias,
        t_DataType *p_weightGradient,
        t_DataType *p_biasGradient,
        t_DataType p_learningRate,
        t_DataType p_batchSize,
        unsigned int p_weightSize,
        unsigned int p_biasSize)
    {
#pragma HLS DATAFLOW
        hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> l_weightGradient("Weight gradient sum"), l_weightGradientScal("Weight gradient avg");
        hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> l_biasGradient("Bias gradient sum"), l_biasGradientScal("Bias gradient avg");

        t_DataType l_multiplicator = (t_DataType)-1 * p_learningRate / p_batchSize;

#pragma HLS DATAFLOW
        xf::blas::readVec2Stream<t_DataType, t_ParEntries>(p_weightGradient, p_weightSize, l_weightGradient);
        xf::blas::scal<t_DataType, t_ParEntries>(
            p_weightSize,
            l_multiplicator,
            l_weightGradient,
            l_weightGradientScal);

        xf::blas::readVec2Stream<t_DataType, t_ParEntries>(p_biasGradient, p_biasSize, l_biasGradient);
        xf::blas::scal<t_DataType, t_ParEntries>(
            p_biasSize,
            l_multiplicator,
            l_biasGradient,
            l_biasGradientScal);

        uz_mlp::accumulate<t_DataType, t_ParEntries>(
            p_weights,
            l_weightGradientScal,
            p_weightSize,
            false);

        uz_mlp::accumulate<t_DataType, t_ParEntries>(
            p_bias,
            l_biasGradientScal,
            p_biasSize,
            false);
    }

} // end namespace uz_mlp