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
    void ApplyFunction(
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
    void ProcessLayer(
        t_DataType *weights,
        t_DataType *input,
        t_DataType *bias,
        t_DataType *output,
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
        xf::blas::gem2Stream<t_DataType, t_ParEntries>(p_n, p_k, weights, l_strWeights);
        xf::blas::vec2GemStream<t_DataType, t_ParEntries>(p_n, p_k, input, l_strInput);
        xf::blas::readVec2Stream<t_DataType, 1>(bias, p_n, l_strBias);
        xf::blas::gemv<t_DataType, t_logParEntries>(p_n, p_k, (t_DataType)1, l_strWeights, l_strInput, (t_DataType)1, l_strBias, l_strMv);
        ApplyFunction<t_DataType, 1>(l_strMv, l_strOutput, p_n, uz_mlp::sigmoid<t_DataType>);
        xf::blas::writeStream2Vec<t_DataType, 1>(l_strOutput, p_n, output);
    }

    template <typename t_DataType, unsigned int t_ParEntries, unsigned int t_logParEntries>
    void InputLayer(
        t_DataType *weights,
        t_DataType *input,
        t_DataType *bias,
        t_DataType *output,
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
        xf::blas::gem2Stream<t_DataType, t_ParEntries>(p_n, p_k, weights, l_strWeights);
        xf::blas::vec2GemStream<t_DataType, t_ParEntries>(p_n, p_k, input, l_strInput);
        xf::blas::readVec2Stream<t_DataType, 1>(bias, p_n, l_strBias);
        xf::blas::gemv<t_DataType, t_logParEntries>(p_n, p_k, (t_DataType)1, l_strWeights, l_strInput, (t_DataType)1, l_strBias, l_strMv);
        ApplyFunction<t_DataType, 1>(l_strMv, l_strOutput, p_n, uz_mlp::sigmoid<t_DataType>);
        xf::blas::writeStream2Vec<t_DataType, 1>(l_strOutput, p_n, output);
    }

    template <typename t_DataType, unsigned int t_ParEntries, unsigned int t_logParEntries>
    void OutputLayer(
        t_DataType *weights,
        t_DataType *input,
        t_DataType *bias,
        t_DataType *output,
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
        xf::blas::gem2Stream<t_DataType, t_ParEntries>(p_n, p_k, weights, l_strWeights);
        xf::blas::vec2GemStream<t_DataType, t_ParEntries>(p_n, p_k, input, l_strInput);
        xf::blas::readVec2Stream<t_DataType, 1>(bias, p_n, l_strBias);
        xf::blas::gemv<t_DataType, t_logParEntries>(p_n, p_k, (t_DataType)1, l_strWeights, l_strInput, (t_DataType)1, l_strBias, l_strMv);
        xf::blas::writeStream2Vec<t_DataType, 1>(l_strMv, p_n, output);
    }

    template <typename t_DataType, unsigned int t_ParEntries>
    void CopyArray(
        t_DataType *input,
        t_DataType *output,
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
                output[i * t_ParEntries + j] = input[i * t_ParEntries + j];
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
 * @param p_n number of rows in input matrix
 * @param p_k number of cols in input matrix
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
    void streamZero(unsigned int p_n, hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> &p_out)
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
 * p_n = number of rows in weight matrix and number of rows in error vector of latter layer
 * p_k = number of cols in weight matrix, number of rows of the output error vector and number
 *       of rows in output vector of current layer
 * 
 * */
    template <typename t_DataType, unsigned int t_ParEntries, unsigned int t_logParEntries>
    void computeError(
        unsigned int p_n,
        unsigned int p_k,
        t_DataType *weights,
        t_DataType *inputError,
        t_DataType *outputCurrentLayer,
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> &strOutputError)
    {
#pragma HLS DATAFLOW
        hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> l_strWeights("Transposed weights");
        hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> l_strInputError("Input error");
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> l_strZero("Zero stream");
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> l_strMv("Matrix Vector product");
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> l_strOutputCurLayer("Output current layer");
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> l_strActivDeriv("Activation Derivative");
        //hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> l_strOutputError("Final error");
#pragma HLS DATAFLOW
        uz_mlp::gem2StreamTranspose<t_DataType, t_ParEntries>(p_n, p_k, weights, l_strWeights);
        xf::blas::vec2GemStream<t_DataType, t_ParEntries>(p_k, p_n, inputError, l_strInputError);
        uz_mlp::streamZero<t_DataType, 1>(p_k, l_strZero);
        xf::blas::gemv<t_DataType, t_logParEntries>(p_k, p_n, (t_DataType)1, l_strWeights, l_strInputError, (t_DataType)1, l_strZero, l_strMv);
        xf::blas::readVec2Stream<t_DataType, 1>(outputCurrentLayer, p_k, l_strOutputCurLayer);
        uz_mlp::ApplyFunction<t_DataType, 1>(l_strOutputCurLayer, l_strActivDeriv, p_k, uz_mlp::sigmoidDeriv<t_DataType>);
        uz_mlp::hadamardProduct<t_DataType, 1>(p_k, l_strMv, l_strActivDeriv, strOutputError);
        //xf::blas::writeStream2Vec<t_DataType, 1>(l_strOutputError, p_k, outputError);
    }

    /**
 * @brief Compute the gradient of every element in the weight matrix
 * 
 * @tparam t_DataType Datatype of the vectors
 * @tparam t_ParEntries Number of parallel processed entries of the outputPrevLayer vector
 * 
 * @param p_n number of rows of the output matrix and number of entries in the currentError vector
 * @param p_k number of cols of the output matrix and number of entries in the outputPrevLayer vector
 * @param currentError Error vector of the current layer
 * @param outputPrevLayer Output of the previous layer
 * 
 * */
    template <typename t_DataType, unsigned int t_ParEntries>
    void computeGradient(
        unsigned int p_n,
        unsigned int p_k,
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> &currentError,
        hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> &outputPrevLayer,
        t_DataType *weightGradient,
        t_DataType *biasGradient)
    {
#ifndef __SYNTHESIS__
        assert((p_k % t_ParEntries) == 0);
#endif
        for (unsigned int n = 0; n < p_n; n++)
        {
            xf::blas::WideType<t_DataType, 1> l_currentError = currentError.read();
            biasGradient[n] = l_currentError[0];
            for (unsigned int k = 0; k < p_k / t_ParEntries; k++)
            {
#pragma HLS PIPELINE
                xf::blas::WideType<t_DataType, t_ParEntries> l_outputPrevLayer = outputPrevLayer.read();
                for (unsigned int j = 0; j < t_ParEntries; j++)
                {
                    weightGradient[n * p_k + k * t_ParEntries + j] = l_outputPrevLayer[j] * l_currentError[0];
                }
            }
        }
    }

} // end namespace uz_mlp