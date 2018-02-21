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

#include <maths/CLinearAlgebra.h>
#include <maths/CMultivariatePrior.h>
#include <maths/Constants.h>
#include <maths/CPrior.h>

#include <cstddef>

namespace ml
{
namespace handy_typedefs
{
typedef core::CSmallVector<double, 1> TDouble1Vec;
typedef core::CSmallVector<double, 4> TDouble4Vec;
typedef core::CSmallVector<double, 10> TDouble10Vec;
typedef core::CSmallVector<TDouble4Vec, 1> TDouble4Vec1Vec;
typedef core::CSmallVector<TDouble10Vec, 1> TDouble10Vec1Vec;
typedef core::CSmallVector<TDouble10Vec, 4> TDouble10Vec4Vec;
typedef core::CSmallVector<TDouble10Vec, 10> TDouble10Vec10Vec;
typedef core::CSmallVector<TDouble10Vec4Vec, 1> TDouble10Vec4Vec1Vec;
typedef maths::CVectorNx1<double, 2> TVector2;
typedef std::vector<TVector2> TVector2Vec;
typedef std::vector<TVector2Vec> TVector2VecVec;
typedef maths::CSymmetricMatrixNxN<double, 2> TMatrix2;
typedef std::vector<TMatrix2> TMatrix2Vec;
typedef maths::CVectorNx1<double, 3> TVector3;
typedef maths::CSymmetricMatrixNxN<double, 3> TMatrix3;
}

//! \brief A set of test and utility functions for use in testing only.
//!
//! DESCRIPTION:\n
//! This is a mix in interface for use within the testing framework.
class CPriorTestInterface
{
    public:
        typedef std::pair<double, double> TDoubleDoublePr;
        typedef core::CSmallVector<TDoubleDoublePr, 1> TDoubleDoublePr1Vec;
        typedef maths_t::TWeightStyleVec TWeightStyleVec;
        typedef maths::CConstantWeights TWeights;

    public:
        explicit CPriorTestInterface(maths::CPrior &prior);

        //! Wrapper which takes care of weights.
        void addSamples(const handy_typedefs::TDouble1Vec &samples);

        //! Wrapper which takes care of weights.
        maths_t::EFloatingPointErrorStatus
            jointLogMarginalLikelihood(const handy_typedefs::TDouble1Vec &samples,
                                       double &result) const;

        //! Wrapper which takes care of weights.
        bool minusLogJointCdf(const handy_typedefs::TDouble1Vec &samples,
                              double &lowerBound,
                              double &upperBound) const;

        //! Wrapper which takes care of weights.
        bool minusLogJointCdfComplement(const handy_typedefs::TDouble1Vec &samples,
                                        double &lowerBound,
                                        double &upperBound) const;

        //! Wrapper which takes care of weights.
        bool probabilityOfLessLikelySamples(maths_t::EProbabilityCalculation calculation,
                                            const handy_typedefs::TDouble1Vec &samples,
                                            double &lowerBound,
                                            double &upperBound) const;

        //! A wrapper around weighted compute anomaly scores which uses unit
        //! weights for all samples.
        bool anomalyScore(maths_t::EProbabilityCalculation calculation,
                          const handy_typedefs::TDouble1Vec &samples,
                          double &result) const;

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
                          const TDoubleDoublePr1Vec &samples,
                          double &result) const;

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
        bool marginalLikelihoodQuantileForTest(double percentage,
                                               double eps,
                                               double &result) const;

        //! This is a slow method that uses numerical integration to compute
        //! the mean so ***only*** use this for testing.
        //!
        //! \param[out] result Filled in with the mean if it could be found.
        //! \note This makes use of marginalLikelihoodQuantile and suffers
        //! the same limitations.
        bool marginalLikelihoodMeanForTest(double &result) const;

        //! This is a slow method that uses numerical integration to compute
        //! the variance so ***only*** use this for testing.
        //!
        //! \param[out] result Filled in with the variance if it could be
        //! found.
        //! \note This makes use of marginalLikelihoodQuantile and suffers
        //! the same limitations.
        bool marginalLikelihoodVarianceForTest(double &result) const;

    protected:
        maths::CPrior *m_Prior;
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
class CPriorTestInterfaceMixin : public PRIOR, public CPriorTestInterface
{
    public:
        using PRIOR::addSamples;
        using PRIOR::jointLogMarginalLikelihood;
        using PRIOR::minusLogJointCdf;
        using PRIOR::minusLogJointCdfComplement;
        using PRIOR::probabilityOfLessLikelySamples;
        using CPriorTestInterface::addSamples;
        using CPriorTestInterface::jointLogMarginalLikelihood;
        using CPriorTestInterface::minusLogJointCdf;
        using CPriorTestInterface::minusLogJointCdfComplement;
        using CPriorTestInterface::probabilityOfLessLikelySamples;

    public:
        CPriorTestInterfaceMixin(const PRIOR &prior) :
            PRIOR(prior),
            CPriorTestInterface(static_cast<maths::CPrior&>(*this)),
            m_Offset(0.0)
        {
        }

        CPriorTestInterfaceMixin(const CPriorTestInterfaceMixin &other) :
            PRIOR(static_cast<const PRIOR&>(other)),
            CPriorTestInterface(static_cast<maths::CPrior&>(*this)),
            m_Offset(0.0)
        {
        }

        virtual ~CPriorTestInterfaceMixin(void) {}

        //! Swap the contents efficiently.
        void swap(CPriorTestInterfaceMixin &other)
        {
            this->PRIOR::swap(other);
        }

        //! Overload assignment.
        CPriorTestInterfaceMixin &operator=(const CPriorTestInterfaceMixin &other)
        {
            if (this != &other)
            {
                // This intentionally slices! We don't want to copy the
                // CPriorTestInterface state.
                static_cast<PRIOR&>(*this) = static_cast<const PRIOR&>(other);
                m_Offset = other.m_Offset;
            }
            return *this;
        }

        //! Clone the object.
        virtual CPriorTestInterfaceMixin *clone(void) const
        {
            return new CPriorTestInterfaceMixin(*this);
        }

        //! Set the offset margin.
        void setOffset(double offset)
        {
            m_Offset = offset;
        }

    private:
        //! Override to zero for nearly all testing.
        virtual double offsetMargin(void) const
        {
            return m_Offset;
        }

        double m_Offset;
};


//! \brief Kernel for checking normalization with CPrior::expectation.
class C1dUnitKernel
{
    public:
        bool operator()(double /*x*/, double &result) const
        {
            result = 1.0;
            return true;
        }
};

//! \brief Kernel for computing the variance with CPrior::expectation.
class CVarianceKernel
{
    public:
        CVarianceKernel(double mean) : m_Mean(mean) {}

        bool operator()(double x, double &result) const
        {
            result = (x - m_Mean) * (x - m_Mean);
            return true;
        }

    private:
        double m_Mean;
};

template<std::size_t N>
class CUnitKernel
{
    public:
        CUnitKernel(const maths::CMultivariatePrior &prior) :
                m_Prior(&prior),
                m_X(1)
        {}

        bool operator()(const maths::CVectorNx1<double, N> &x, double &result) const
        {
            m_X[0].assign(x.begin(), x.end());
            m_Prior->jointLogMarginalLikelihood(maths::CConstantWeights::COUNT, m_X, SINGLE_UNIT, result);
            result = ::exp(result);
            return true;
        }

    private:
        static handy_typedefs::TDouble10Vec4Vec1Vec SINGLE_UNIT;

    private:
        const maths::CMultivariatePrior *m_Prior;
        mutable handy_typedefs::TDouble10Vec1Vec m_X;
};

template<std::size_t N>
handy_typedefs::TDouble10Vec4Vec1Vec CUnitKernel<N>::SINGLE_UNIT(1, handy_typedefs::TDouble10Vec4Vec(1, handy_typedefs::TDouble10Vec(N, 1.0)));


template<std::size_t N>
class CMeanKernel
{
    public:
        CMeanKernel(const maths::CMultivariatePrior &prior) :
                m_Prior(&prior),
                m_X(1)
        {}

        bool operator()(const maths::CVectorNx1<double, N> &x,
                        maths::CVectorNx1<double, N> &result) const
        {
            m_X[0].assign(x.begin(), x.end());
            double likelihood;
            m_Prior->jointLogMarginalLikelihood(maths::CConstantWeights::COUNT, m_X, SINGLE_UNIT, likelihood);
            likelihood = ::exp(likelihood);
            result = x * likelihood;
            return true;
        }

    private:
        static handy_typedefs::TDouble10Vec4Vec1Vec SINGLE_UNIT;

    private:
        const maths::CMultivariatePrior *m_Prior;
        mutable handy_typedefs::TDouble10Vec1Vec m_X;
};

template<std::size_t N>
handy_typedefs::TDouble10Vec4Vec1Vec CMeanKernel<N>::SINGLE_UNIT(1, handy_typedefs::TDouble10Vec4Vec(1, handy_typedefs::TDouble10Vec(N, 1.0)));


template<std::size_t N>
class CCovarianceKernel
{
    public:
        CCovarianceKernel(const maths::CMultivariatePrior &prior,
                          const maths::CVectorNx1<double, N> &mean) :
                m_Prior(&prior),
                m_Mean(mean),
                m_X(1)
        {}

        bool operator()(const maths::CVectorNx1<double, N> &x,
                        maths::CSymmetricMatrixNxN<double, N> &result) const
        {
            m_X[0].assign(x.begin(), x.end());
            double likelihood;
            m_Prior->jointLogMarginalLikelihood(maths::CConstantWeights::COUNT, m_X, SINGLE_UNIT, likelihood);
            likelihood = ::exp(likelihood);
            result = (x - m_Mean).outer() * likelihood;
            return true;
        }

    private:
        static handy_typedefs::TDouble10Vec4Vec1Vec SINGLE_UNIT;

    private:
        const maths::CMultivariatePrior *m_Prior;
        maths::CVectorNx1<double, N> m_Mean;
        mutable handy_typedefs::TDouble10Vec1Vec m_X;
};

template<std::size_t N>
handy_typedefs::TDouble10Vec4Vec1Vec CCovarianceKernel<N>::SINGLE_UNIT(1, handy_typedefs::TDouble10Vec4Vec(1, handy_typedefs::TDouble10Vec(N, 1.0)));

}

#endif // INCLUDED_ml_TestUtils_h
