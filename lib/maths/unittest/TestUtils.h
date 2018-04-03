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

#ifndef INCLUDED_ml_TestUtils_h
#define INCLUDED_ml_TestUtils_h

#include <core/CSmallVector.h>
#include <core/CoreTypes.h>

#include <maths/CLinearAlgebra.h>
#include <maths/CMultivariatePrior.h>
#include <maths/CPrior.h>
#include <maths/Constants.h>

#include <cmath>
#include <cstddef>

namespace ml {
namespace handy_typedefs {
using TDouble1Vec = core::CSmallVector<double, 1>;
using TDouble4Vec = core::CSmallVector<double, 4>;
using TDouble10Vec = core::CSmallVector<double, 10>;
using TDouble4Vec1Vec = core::CSmallVector<TDouble4Vec, 1>;
using TDouble10Vec1Vec = core::CSmallVector<TDouble10Vec, 1>;
using TDouble10Vec4Vec = core::CSmallVector<TDouble10Vec, 4>;
using TDouble10Vec10Vec = core::CSmallVector<TDouble10Vec, 10>;
using TDouble10Vec4Vec1Vec = core::CSmallVector<TDouble10Vec4Vec, 1>;
using TVector2 = maths::CVectorNx1<double, 2>;
using TVector2Vec = std::vector<TVector2>;
using TVector2VecVec = std::vector<TVector2Vec>;
using TMatrix2 = maths::CSymmetricMatrixNxN<double, 2>;
using TMatrix2Vec = std::vector<TMatrix2>;
using TVector3 = maths::CVectorNx1<double, 3>;
using TMatrix3 = maths::CSymmetricMatrixNxN<double, 3>;
using TGenerator = double (*)(core_t::TTime);
using TGeneratorVec = std::vector<TGenerator>;
}

//! \brief A set of test and utility functions for use in testing only.
//!
//! DESCRIPTION:\n
//! This is a mix in interface for use within the testing framework.
class CPriorTestInterface {
public:
    using TDoubleDoublePr = std::pair<double, double>;
    using TDoubleDoublePr1Vec = core::CSmallVector<TDoubleDoublePr, 1>;
    using TWeightStyleVec = maths_t::TWeightStyleVec;
    using TWeights = maths::CConstantWeights;

public:
    explicit CPriorTestInterface(maths::CPrior& prior);

    //! Wrapper which takes care of weights.
    void addSamples(const handy_typedefs::TDouble1Vec& samples);

    //! Wrapper which takes care of weights.
    maths_t::EFloatingPointErrorStatus jointLogMarginalLikelihood(const handy_typedefs::TDouble1Vec& samples, double& result) const;

    //! Wrapper which takes care of weights.
    bool minusLogJointCdf(const handy_typedefs::TDouble1Vec& samples, double& lowerBound, double& upperBound) const;

    //! Wrapper which takes care of weights.
    bool minusLogJointCdfComplement(const handy_typedefs::TDouble1Vec& samples, double& lowerBound, double& upperBound) const;

    //! Wrapper which takes care of weights.
    bool probabilityOfLessLikelySamples(maths_t::EProbabilityCalculation calculation,
                                        const handy_typedefs::TDouble1Vec& samples,
                                        double& lowerBound,
                                        double& upperBound) const;

    //! A wrapper around weighted compute anomaly scores which uses unit
    //! weights for all samples.
    bool anomalyScore(maths_t::EProbabilityCalculation calculation, const handy_typedefs::TDouble1Vec& samples, double& result) const;

    //! Calculate an anomaly score for a collection of independent samples
    //! from the variable.
    //!
    //! \param[in] calculation The style of the probability calculation
    //! (see maths_t::EProbabilityCalculation for details).
    //! \param[in] weightStyle Controls the interpretation of the weight that
    //! is associated with each sample. See maths_t::ESampleWeightStyle for
    //! more details.
    //! \param[in] samples A collection of samples of the variable.
    //! Each pair is the sample and weight, i.e. \f$(x_i, \gamma_i)\f$ where
    //! \f$x_i\f$ is \f$i^{th}\f$ sample and \f$\gamma_i\f$ is the weight of
    //! that sample.
    //! \param[out] result Filled in with the total anomaly score of \p samples.
    bool anomalyScore(maths_t::EProbabilityCalculation calculation,
                      maths_t::ESampleWeightStyle weightStyle,
                      const TDoubleDoublePr1Vec& samples,
                      double& result) const;

    //! This is a slow method that uses numerical root finding to compute
    //! the quantile so ***only*** use this for testing.
    //!
    //! \param[in] percentage The desired quantile expressed as a percentage.
    //! \param[in] eps The tolerated error in the quantile: if it could be
    //! calculated, \p result will be no further than \p eps away from
    //! the exact quantile.
    //! \param[out] result Filled in with the quantile if it could be found.
    //! \note Since this is for testing purposes only it is not especially
    //! robust. For example, it won't handle a normal with mean of \f$10^8\f$
    //! and standard deviation of \f$10^{-8}\f$ particularly well.
    bool marginalLikelihoodQuantileForTest(double percentage, double eps, double& result) const;

    //! This is a slow method that uses numerical integration to compute
    //! the mean so ***only*** use this for testing.
    //!
    //! \param[out] result Filled in with the mean if it could be found.
    //! \note This makes use of marginalLikelihoodQuantile and suffers
    //! the same limitations.
    bool marginalLikelihoodMeanForTest(double& result) const;

    //! This is a slow method that uses numerical integration to compute
    //! the variance so ***only*** use this for testing.
    //!
    //! \param[out] result Filled in with the variance if it could be
    //! found.
    //! \note This makes use of marginalLikelihoodQuantile and suffers
    //! the same limitations.
    bool marginalLikelihoodVarianceForTest(double& result) const;

protected:
    maths::CPrior* m_Prior;
};

//! \brief A mix in of test interface which brings the necessary functions
//! into scope and implements value semantics.
//!
//! IMPLMENTATION:\n
//! This is variant of the curiously recurring template pattern to mix
//! in some interface for test purposes only.
//!
//! Note that this also uses double inheritance, contravening the coding
//! standards, because it's the cleanest way to implement this functionality.
//! DON'T use this elsewhere.
template<typename PRIOR>
class CPriorTestInterfaceMixin : public PRIOR, public CPriorTestInterface {
public:
    using CPriorTestInterface::addSamples;
    using CPriorTestInterface::jointLogMarginalLikelihood;
    using CPriorTestInterface::minusLogJointCdf;
    using CPriorTestInterface::minusLogJointCdfComplement;
    using CPriorTestInterface::probabilityOfLessLikelySamples;
    using PRIOR::addSamples;
    using PRIOR::jointLogMarginalLikelihood;
    using PRIOR::minusLogJointCdf;
    using PRIOR::minusLogJointCdfComplement;
    using PRIOR::probabilityOfLessLikelySamples;

public:
    CPriorTestInterfaceMixin(const PRIOR& prior) : PRIOR(prior), CPriorTestInterface(static_cast<maths::CPrior&>(*this)) {}

    CPriorTestInterfaceMixin(const CPriorTestInterfaceMixin& other)
        : PRIOR(static_cast<const PRIOR&>(other)), CPriorTestInterface(static_cast<maths::CPrior&>(*this)) {}

    virtual ~CPriorTestInterfaceMixin(void) {}

    //! Swap the contents efficiently.
    void swap(CPriorTestInterfaceMixin& other) { this->PRIOR::swap(other); }

    //! Clone the object.
    virtual CPriorTestInterfaceMixin* clone(void) const { return new CPriorTestInterfaceMixin(*this); }
};

//! \brief Kernel for checking normalization with CPrior::expectation.
class C1dUnitKernel {
public:
    bool operator()(double /*x*/, double& result) const {
        result = 1.0;
        return true;
    }
};

//! \brief Kernel for computing the variance with CPrior::expectation.
class CVarianceKernel {
public:
    CVarianceKernel(double mean) : m_Mean(mean) {}

    bool operator()(double x, double& result) const {
        result = (x - m_Mean) * (x - m_Mean);
        return true;
    }

private:
    double m_Mean;
};

//! \brief A constant unit kernel.
template<std::size_t N>
class CUnitKernel {
public:
    CUnitKernel(const maths::CMultivariatePrior& prior) : m_Prior(&prior), m_X(1) {}

    bool operator()(const maths::CVectorNx1<double, N>& x, double& result) const {
        m_X[0].assign(x.begin(), x.end());
        m_Prior->jointLogMarginalLikelihood(maths::CConstantWeights::COUNT, m_X, SINGLE_UNIT, result);
        result = std::exp(result);
        return true;
    }

private:
    static handy_typedefs::TDouble10Vec4Vec1Vec SINGLE_UNIT;

private:
    const maths::CMultivariatePrior* m_Prior;
    mutable handy_typedefs::TDouble10Vec1Vec m_X;
};

template<std::size_t N>
handy_typedefs::TDouble10Vec4Vec1Vec CUnitKernel<N>::SINGLE_UNIT(1,
                                                                 handy_typedefs::TDouble10Vec4Vec(1, handy_typedefs::TDouble10Vec(N, 1.0)));

//! \brief The kernel for computing the mean of a multivariate prior.
template<std::size_t N>
class CMeanKernel {
public:
    CMeanKernel(const maths::CMultivariatePrior& prior) : m_Prior(&prior), m_X(1) {}

    bool operator()(const maths::CVectorNx1<double, N>& x, maths::CVectorNx1<double, N>& result) const {
        m_X[0].assign(x.begin(), x.end());
        double likelihood;
        m_Prior->jointLogMarginalLikelihood(maths::CConstantWeights::COUNT, m_X, SINGLE_UNIT, likelihood);
        likelihood = std::exp(likelihood);
        result = x * likelihood;
        return true;
    }

private:
    static handy_typedefs::TDouble10Vec4Vec1Vec SINGLE_UNIT;

private:
    const maths::CMultivariatePrior* m_Prior;
    mutable handy_typedefs::TDouble10Vec1Vec m_X;
};

template<std::size_t N>
handy_typedefs::TDouble10Vec4Vec1Vec CMeanKernel<N>::SINGLE_UNIT(1,
                                                                 handy_typedefs::TDouble10Vec4Vec(1, handy_typedefs::TDouble10Vec(N, 1.0)));

//! \brief The kernel for computing the variance of a multivariate prior.
template<std::size_t N>
class CCovarianceKernel {
public:
    CCovarianceKernel(const maths::CMultivariatePrior& prior, const maths::CVectorNx1<double, N>& mean)
        : m_Prior(&prior), m_Mean(mean), m_X(1) {}

    bool operator()(const maths::CVectorNx1<double, N>& x, maths::CSymmetricMatrixNxN<double, N>& result) const {
        m_X[0].assign(x.begin(), x.end());
        double likelihood;
        m_Prior->jointLogMarginalLikelihood(maths::CConstantWeights::COUNT, m_X, SINGLE_UNIT, likelihood);
        likelihood = std::exp(likelihood);
        result = (x - m_Mean).outer() * likelihood;
        return true;
    }

private:
    static handy_typedefs::TDouble10Vec4Vec1Vec SINGLE_UNIT;

private:
    const maths::CMultivariatePrior* m_Prior;
    maths::CVectorNx1<double, N> m_Mean;
    mutable handy_typedefs::TDouble10Vec1Vec m_X;
};

template<std::size_t N>
handy_typedefs::TDouble10Vec4Vec1Vec
    CCovarianceKernel<N>::SINGLE_UNIT(1, handy_typedefs::TDouble10Vec4Vec(1, handy_typedefs::TDouble10Vec(N, 1.0)));

//! A constant function.
double constant(core_t::TTime time);

//! A linear ramp.
double ramp(core_t::TTime time);

//! A Markov process.
double markov(core_t::TTime time);

//! Smooth daily periodic.
double smoothDaily(core_t::TTime time);

//! Smooth weekly periodic.
double smoothWeekly(core_t::TTime time);

//! Spikey daily periodic.
double spikeyDaily(core_t::TTime time);

//! Spikey weekly periodic.
double spikeyWeekly(core_t::TTime time);

//! Weekday/weekend periodic.
double weekends(core_t::TTime time);

//! Scales time input to \p generator.
double scale(double scale, core_t::TTime time, handy_typedefs::TGenerator generator);
}

#endif // INCLUDED_ml_TestUtils_h
