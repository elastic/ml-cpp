/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CSignal_h
#define INCLUDED_ml_maths_CSignal_h

#include <core/CVectorRange.h>

#include <maths/CBasicStatistics.h>
#include <maths/ImportExport.h>

#include <complex>
#include <vector>

namespace ml {
namespace maths {

//! \brief Useful functions from signal processing.
class MATHS_EXPORT CSignal {
public:
    using TDoubleVec = std::vector<double>;
    using TComplex = std::complex<double>;
    using TComplexVec = std::vector<TComplex>;
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
    using TFloatMeanAccumulatorCRng = core::CVectorRange<const TFloatMeanAccumulatorVec>;

public:
    //! Compute the conjugate of \p f.
    static void conj(TComplexVec& f);

    //! Compute the Hadamard product of \p fx and \p fy.
    static void hadamard(const TComplexVec& fx, TComplexVec& fy);

    //! Cooley-Tukey fast DFT transform implementation.
    //!
    //! \note This is a simple implementation radix 2 DIT which uses the chirp-z
    //! idea to handle the case that the length of \p fx is not a power of 2. As
    //! such it is definitely not a highly optimized FFT implementation. It should
    //! be sufficiently fast for our needs.
    static void fft(TComplexVec& f);

    //! This uses conjugate of the conjugate of the series is the inverse DFT trick
    //! to compute this using fft.
    static void ifft(TComplexVec& f);

    //! Compute the discrete cyclic autocorrelation of \p values for the offset
    //! \p offset.
    //!
    //! This is just
    //! <pre class="fragment">
    //!   \f$\(\frac{1}{(n-k)\sigma^2}\sum_{k=1}^{n-k}{(f(k) - \mu)(f(k+p) - \mu)}\)\f$
    //! </pre>
    //!
    //! \param[in] offset The offset as a distance in \p values.
    //! \param[in] values The values for which to compute the autocorrelation.
    static double autocorrelation(std::size_t offset, const TFloatMeanAccumulatorVec& values);

    //! Compute the discrete cyclic autocorrelation of \p values for the offset
    //! \p offset.
    //!
    //! \note Implementation for vector ranges.
    static double autocorrelation(std::size_t offset, TFloatMeanAccumulatorCRng values);

    //! Get linear autocorrelations for all offsets up to the length of \p values.
    //!
    //! \param[in] values The values for which to compute autocorrelation.
    //! \param[in] result Filled in with the autocorrelations of \p values for
    //! offsets 1, 2, ..., length \p values - 1.
    static void autocorrelations(const TFloatMeanAccumulatorVec& values, TDoubleVec& result);
};
}
}

#endif // INCLUDED_ml_maths_CSignal_h
