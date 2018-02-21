/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#include <maths/CSignal.h>

#include <core/CLogger.h>

#include <maths/CIntegerTools.h>

#include <boost/math/constants/constants.hpp>

#include <algorithm>
#include <cmath>

namespace ml
{
namespace maths
{

namespace
{

using TComplex = std::complex<double>;
using TComplexVec = std::vector<TComplex>;
using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

//! Scale \p f by \p scale.
void scale(double scale, TComplexVec &f)
{
    for (std::size_t i = 0u; i < f.size(); ++i)
    {
        f[i] *= scale;
    }
}

//! Compute the radix 2 FFT of \p f in-place.
void radix2fft(TComplexVec &f)
{
    // Perform the appropriate permutation of f(x) by swapping
    // each i in [0, N] with its bit reversal.

    uint64_t bits = CIntegerTools::nextPow2(f.size()) - 1;
    for (uint64_t i = 0; i < f.size(); ++i)
    {
        uint64_t j = CIntegerTools::reverseBits(i) >> (64 - bits);
        if (j > i)
        {
            LOG_TRACE(j << " -> " << i);
            std::swap(f[i], f[j]);
        }
    }

    // Apply the twiddle factors.

    for (std::size_t stride = 1; stride < f.size(); stride <<= 1)
    {
        for (std::size_t k = 0u; k < stride; ++k)
        {
            double t = boost::math::double_constants::pi * static_cast<double>(k)
                                                         / static_cast<double>(stride);
            TComplex w(std::cos(t), std::sin(t));
            for (std::size_t start = k; start < f.size(); start += 2 * stride)
            {
                TComplex fs = f[start];
                TComplex tw = w * f[start + stride];
                f[start] = fs + tw;
                f[start + stride] = fs - tw;
            }
        }
    }

    std::reverse(f.begin() + 1, f.end());
}

}

void CSignal::conj(TComplexVec &f)
{
    for (std::size_t i = 0u; i < f.size(); ++i)
    {
        f[i] = std::conj(f[i]);
    }
}

void CSignal::hadamard(const TComplexVec &fx, TComplexVec &fy)
{
    for (std::size_t i = 0u; i < fx.size(); ++i)
    {
        fy[i] *= fx[i];
    }
}

void CSignal::fft(TComplexVec &f)
{
    std::size_t n = f.size();
    std::size_t p = CIntegerTools::nextPow2(n);
    std::size_t m = 1 << p;

    LOG_TRACE("n = " << n << ", m = " << m);

    if ((m >> 1) == n)
    {
        radix2fft(f);
    }
    else
    {
        // We use Bluestein's trick to reformulate as a convolution
        // which can be computed by padding to a power of 2.

        LOG_TRACE("Using Bluestein's trick");

        m = 2 * n - 1;
        p = CIntegerTools::nextPow2(m);
        m = 1 << p;

        TComplexVec chirp;
        chirp.reserve(n);
        TComplexVec a(m, TComplex(0.0));
        TComplexVec b(m, TComplex(0.0));

        chirp.emplace_back(1.0, 0.0);
        a[0] = f[0] * chirp[0];
        b[0] = chirp[0];
        for (std::size_t i = 1u; i < n; ++i)
        {
            double t = boost::math::double_constants::pi * static_cast<double>(i * i)
                                                         / static_cast<double>(n);
            chirp.emplace_back(std::cos(t), std::sin(t));
            a[i] = f[i] * std::conj(chirp[i]);
            b[i] = b[m - i] = chirp[i];
        }

        fft(a);
        fft(b);
        hadamard(a, b);
        ifft(b);

        for (std::size_t i = 0u; i < n; ++i)
        {
            f[i] = std::conj(chirp[i]) * b[i];
        }
    }
}

void CSignal::ifft(TComplexVec &f)
{
    conj(f);
    fft(f);
    conj(f);
    scale(1.0 / static_cast<double>(f.size()), f);
}

double CSignal::autocorrelation(std::size_t offset, const TFloatMeanAccumulatorVec &values)
{
    return autocorrelation(offset, TFloatMeanAccumulatorCRng(values, 0, values.size()));
}

double CSignal::autocorrelation(std::size_t offset, TFloatMeanAccumulatorCRng values)
{
    std::size_t n = values.size();

    TMeanVarAccumulator moments;
    for (const auto &value : values)
    {
        if (CBasicStatistics::count(value) > 0.0)
        {
            moments.add(CBasicStatistics::mean(value));
        }
    }

    TMeanAccumulator autocorrelation;
    double mean = CBasicStatistics::mean(moments);
    for (std::size_t i = 0u; i < values.size(); ++i)
    {
        std::size_t j = (i + offset) % n;
        double ni = CBasicStatistics::count(values[i]);
        double nj = CBasicStatistics::count(values[j]);
        if (ni > 0.0 && nj > 0.0)
        {
            autocorrelation.add(  (CBasicStatistics::mean(values[i]) - mean)
                                * (CBasicStatistics::mean(values[j]) - mean));
        }
    }

    return CBasicStatistics::mean(autocorrelation) == CBasicStatistics::variance(moments) ?
           1.0 : CBasicStatistics::mean(autocorrelation) / CBasicStatistics::variance(moments);
}

void CSignal::autocorrelations(const TFloatMeanAccumulatorVec &values, TDoubleVec &result)
{
    if (values.empty())
    {
        return;
    }

    std::size_t n = values.size();

    TMeanVarAccumulator moments;
    for (const auto &value : values)
    {
        if (CBasicStatistics::count(value) > 0.0)
        {
            moments.add(CBasicStatistics::mean(value));
        }
    }
    double mean = CBasicStatistics::mean(moments);
    double variance = CBasicStatistics::variance(moments);

    TComplexVec f;
    f.reserve(n);
    for (std::size_t i = 0u; i < n; ++i)
    {
        std::size_t j = i;
        for (/**/; j < n && CBasicStatistics::count(values[j]) == 0; ++j);
        if (i != j)
        {
            // Infer missing values by linearly interpolating.
            if (j == n)
            {
                f.resize(n, TComplex(0.0, 0.0));
                break;
            }
            else if (i == 0)
            {
                f.resize(j - 1, TComplex(0.0, 0.0));
            }
            else
            {
                for (std::size_t k = i; k < j; ++k)
                {
                    double alpha = static_cast<double>(k - i + 1) / static_cast<double>(j - i + 1);
                    double real  = CBasicStatistics::mean(values[j]) - mean;
                    f.push_back((1.0 - alpha) * f[i-1] + alpha * TComplex(real, 0.0));
                }
            }
            i = j;
        }
        f.emplace_back(CBasicStatistics::mean(values[i]) - mean, 0.0);
    }

    fft(f);
    TComplexVec fConj(f);
    conj(fConj);
    hadamard(fConj, f);
    ifft(f);

    result.reserve(n);
    for (std::size_t i = 1u; i < n; ++i)
    {
        result.push_back(f[i].real() / variance / static_cast<double>(n));
    }
}

}
}
