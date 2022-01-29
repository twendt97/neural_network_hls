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
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> &p_in,
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> &p_out,
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

    template <typename t_DataType>
    void CopyArray(
        t_DataType *input,
        t_DataType *output,
        unsigned int size)
    {
    COPY_ARRAY:
        for (unsigned int i = 0; i < size; i++)
        {
#pragma HLS PIPELINE
            output[i] = input[i];
        }
    }

} // end namespace uz_mlp