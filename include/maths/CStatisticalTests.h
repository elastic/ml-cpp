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

#ifndef INCLUDED_ml_maths_CStatisticalTests_h
#define INCLUDED_ml_maths_CStatisticalTests_h

#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/ImportExport.h>

#include <cstddef>
#include <vector>

#include <stdint.h>

namespace ml
{
namespace core
{
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths
{

//! \brief A collection of statistical tests and test statistics.
class MATHS_EXPORT CStatisticalTests
{
    public:
        typedef std::vector<uint16_t> TUInt16Vec;
        typedef std::vector<double> TDoubleVec;

    public:
        //! Get the significance of a left tail F-test for \p x when
        //! the test statistic has \p d1 and \p d2 degrees of freedom
        //! under the null hypothesis.
        static double leftTailFTest(double x, double d1, double d2);

        //! Get the significance of a right tail F-test for \p x when
        //! the test statistic has \p d1 and \p d2 degrees of freedom
        //! under the null hypothesis.
        static double rightTailFTest(double x, double d1, double d2);

        //! A two sample Kolmogorov-Smirnov test.
        //!
        //! This computes the test significance for rejecting the
        //! null hypothesis that \p x and \p y are samples from the
        //! same distribution. The smaller the significance the
        //! more likely that \p x and \p y come from different
        //! distributions.
        //!
        //! \note This is based on the implementation in Numerical
        //! Recipes in C.
        static double twoSampleKS(TDoubleVec x, TDoubleVec y);

        //! \brief Implements the Cramer-von Mises criterion.
        //!
        //! DESCRIPTION:\n
        //! The Cramer-von Mises test is a non-parameteric goodness
        //! of fit test for the values of random variable compared
        //! to some estimated probability density function. In
        //! particular, the statistic is:
        //! <pre class="fragment">
        //!   \f$\displaystyle T = n\omega^2 = \frac{1}{12n}+\sum_{i=1}^n{\frac{2i-1}{2n}-F(x_{(i)})}\f$
        //! </pre>
        //! Here, \f$x_{(i)}\f$ are the \f$n\f$ order statistics of
        //! a collection of \f$n\f$ samples of the random variable
        //! under test and \f$F(.)\f$ is the estimated cumulative
        //! density function. Under the null hypothesis, that the
        //! random variable is distributed according to \f$F(.)\f$,
        //! the distribution of the values of \f$T\f$ are independent
        //! of the form of the distribution function and can be
        //! tabulated for different p-values and sample counts. For
        //! large count the values are approximately independent of
        //! count. We tabulate the values and interpolate the table.
        //!
        //! \see http://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93von_Mises_criterion
        //! for more information on this test statistic.
        class MATHS_EXPORT CCramerVonMises
        {
            public:
                //! Enumeration of the p values for which the test
                //! statistic value is tabulated.
                static const double P_VALUES[16];
                //! Enumeration of the count, of the values used in
                //! the test statistic, for which the value of the
                //! test statistic is tabulated.
                static const std::size_t N[13];
                //! The tabulated values of the test statistic for
                //! specific p-values and counts.
                static const double T_VALUES[13][16];

            public:
                CCramerVonMises(std::size_t size);

                //! Create by traversing a state document.
                CCramerVonMises(core::CStateRestoreTraverser &traverser);

                //! Persist state by passing information to the supplied inserter
                void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

                //! Add a value of the cumulative density function.
                void addF(double f);

                //! Get the mean test p-value for the observations
                //! to date.
                double pValue(void) const;

                //! Age out old p-values of the test.
                void age(double factor);

                //! Get a checksum for this object.
                uint64_t checksum(uint64_t seed = 0) const;

            private:
                //! Create by traversing a state document.
                bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

            private:
                //! The scale used to convert doubles in the interval
                //! [0,1] to 16 bit integers.
                static const double SCALE;

            private:
                typedef CBasicStatistics::SSampleMean<double>::TAccumulator TMeanAccumulator;

            private:
                //! The "count - 1" in the test statistic.
                std::size_t m_Size;
                //! The mean value of the test statistic.
                TMeanAccumulator m_T;
                //! The current values in the test statistic scaled
                //! and converted to 16 bit integers.
                TUInt16Vec m_F;
        };
};

}
}
#endif // INCLUDED_ml_maths_CStatisticalTests_h
