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

#ifndef INCLUDED_ml_maths_CSampling_h
#define INCLUDED_ml_maths_CSampling_h

#include <core/CFastMutex.h>
#include <core/CNonCopyable.h>
#include <core/CNonInstantiatable.h>
#include <core/CScopedFastLock.h>

#include <maths/CLinearAlgebraFwd.h>
#include <maths/CPRNG.h>
#include <maths/ImportExport.h>

#include <boost/optional.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include <vector>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {

//! \brief Sampling functionality.
//!
//! DEFINITION:\n
//! This is a place holder for random sampling utilities and algorithms.
class MATHS_EXPORT CSampling : private core::CNonInstantiatable {
public:
    using TDoubleVec = std::vector<double>;
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TSizeVec = std::vector<std::size_t>;
    using TPtrdiffVec = std::vector<std::ptrdiff_t>;

    //! \brief A mockable random number generator which uses boost::random::mt11213b.
    class MATHS_EXPORT CRandomNumberGenerator {
    public:
        using result_type = boost::random::mt11213b::result_type;

    public:
        //! Mock the random number generator to produce a constant.
        void mock(void);

        //! Unmock the random number generator.
        void unmock(void);

        //! Seed the random number generator.
        void seed(void);

        //! Returns the smallest value that the generator can produce.
        static result_type min(void) { return boost::random::mt11213b::min(); }

        //! Returns the largest value that the generator can produce.
        static result_type max(void) { return boost::random::mt11213b::max(); }

        //! Produces the next value of the generator.
        result_type operator()(void) {
            if (m_Mock) {
                return *m_Mock;
            }
            return m_Rng.operator()();
        }

        //! Fills a range with random values.
        template <typename ITR> void generate(ITR first, ITR last) {
            if (m_Mock) {
                for (/**/; first != last; ++first) {
                    *first = *m_Mock;
                }
            }
            m_Rng.generate(first, last);
        }

        //! Writes the mersenne_twister_engine to a std::ostream.
        template <class CHAR, class TRAITS>
        friend std::basic_ostream<CHAR, TRAITS> &operator<<(std::basic_ostream<CHAR, TRAITS> &o,
                                                            const CRandomNumberGenerator &g) {
            return o << g.m_Rng;
        }

        //! Reads a mersenne_twister_engine from a std::istream.
        template <class CHAR, class TRAITS>
        friend std::basic_istream<CHAR, TRAITS> &operator>>(std::basic_istream<CHAR, TRAITS> &i,
                                                            CRandomNumberGenerator &g) {
            return i >> g.m_Rng;
        }

    private:
        using TOptionalResultType = boost::optional<result_type>;

    private:
        TOptionalResultType m_Mock;
        boost::random::mt11213b m_Rng;
    };

    //! \brief Setup and tears down mock random numbers in the scope in which
    //! it is constructed.
    class MATHS_EXPORT CScopeMockRandomNumberGenerator {
    public:
        CScopeMockRandomNumberGenerator(void);
        ~CScopeMockRandomNumberGenerator(void);
    };

public:
    //! \name Persistence
    //@{
    //! Restore the static members of this class from persisted state
    static bool staticsAcceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

    //! Persist the static members of this class
    static void staticsAcceptPersistInserter(core::CStatePersistInserter &inserter);
    //@}

    //! Reinitialize the random number generator.
    static void seed(void);

    //! \name Uniform Sampling
    //!
    //! Sample uniformly from a specified range
    //@{
#define UNIFORM_SAMPLE(TYPE)                                                                       \
    static TYPE uniformSample(TYPE a, TYPE b);                                                     \
    static TYPE uniformSample(CPRNG::CXorOShiro128Plus &rng, TYPE a, TYPE b);                      \
    static TYPE uniformSample(CPRNG::CXorShift1024Mult &rng, TYPE a, TYPE b);                      \
    static void uniformSample(TYPE a, TYPE b, std::size_t n, std::vector<TYPE> &result);           \
    static void uniformSample(                                                                     \
        CPRNG::CXorOShiro128Plus &rng, TYPE a, TYPE b, std::size_t n, std::vector<TYPE> &result);  \
    static void uniformSample(                                                                     \
        CPRNG::CXorShift1024Mult &rng, TYPE a, TYPE b, std::size_t n, std::vector<TYPE> &result);
    UNIFORM_SAMPLE(std::size_t)
    UNIFORM_SAMPLE(std::ptrdiff_t)
    UNIFORM_SAMPLE(double)
#undef UNIFORM_SAMPLE
    //@}

    //! Get a normal sample with mean and variance \p mean and
    //! \p variance, respectively.
    static double normalSample(double mean, double variance);

    //! Get a normal sample with mean and variance \p mean and
    //! \p variance, respectively.
    static double normalSample(CPRNG::CXorOShiro128Plus &rng, double mean, double variance);

    //! Get a normal sample with mean and variance \p mean and
    //! \p variance, respectively.
    static double normalSample(CPRNG::CXorShift1024Mult &rng, double mean, double variance);

    //! Get \p n normal samples with mean and variance \p mean and
    //! \p variance, respectively.
    static void normalSample(double mean, double variance, std::size_t n, TDoubleVec &result);

    //! Get \p n normal samples with mean and variance \p mean and
    //! \p variance, respectively, using \p rng.
    static void normalSample(CPRNG::CXorOShiro128Plus &rng,
                             double mean,
                             double variance,
                             std::size_t n,
                             TDoubleVec &result);

    //! Get \p n normal samples with mean and variance \p mean and
    //! \p variance, respectively, using \p rng.
    static void normalSample(CPRNG::CXorShift1024Mult &rng,
                             double mean,
                             double variance,
                             std::size_t n,
                             TDoubleVec &result);

    //! Get \p n samples of a \f$\chi^2\f$ random variable with \p f
    //! degrees of freedom.
    static void chiSquaredSample(double f, std::size_t n, TDoubleVec &result);

    //! Get \p n samples of a \f$\chi^2\f$ random variable with \p f
    //! degrees of freedom using \p rng.
    static void
    chiSquaredSample(CPRNG::CXorOShiro128Plus &rng, double f, std::size_t n, TDoubleVec &result);

    //! Get \p n samples of a \f$\chi^2\f$ random variable with \p f
    //! degrees of freedom using \p rng.
    static void
    chiSquaredSample(CPRNG::CXorShift1024Mult &rng, double f, std::size_t n, TDoubleVec &result);

    //! \name Multivariate Normal Sampling
    //@{
    //! Sample from the normal distribution with mean \p mean and
    //! covariance matrix \p covariance.
    //!
    //! \param[in] mean The mean vector.
    //! \param[in] covariance The covariance matrix.
    //! \param[in] n The number of samples to generate.
    //! \param[out] samples Filled in with IID samples of the
    //! multivariate normal.
    static bool multivariateNormalSample(const TDoubleVec &mean,
                                         const TDoubleVecVec &covariance,
                                         std::size_t n,
                                         TDoubleVecVec &samples);

    //! Overload of multivariate normal sample using \p rng
    static bool multivariateNormalSample(CPRNG::CXorOShiro128Plus &rng,
                                         const TDoubleVec &mean,
                                         const TDoubleVecVec &covariance,
                                         std::size_t n,
                                         TDoubleVecVec &samples);

    //! Overload of multivariate normal sample using \p rng
    static bool multivariateNormalSample(CPRNG::CXorShift1024Mult &rng,
                                         const TDoubleVec &mean,
                                         const TDoubleVecVec &covariance,
                                         std::size_t n,
                                         TDoubleVecVec &samples);

#define MULTIVARIATE_NORMAL_SAMPLE(N)                                                              \
    static void multivariateNormalSample(const CVectorNx1<double, N> &mean,                        \
                                         const CSymmetricMatrixNxN<double, N> &covariance,         \
                                         std::size_t n,                                            \
                                         std::vector<CVectorNx1<double, N>> &samples);             \
    static void multivariateNormalSample(CPRNG::CXorOShiro128Plus &rng,                            \
                                         const CVectorNx1<double, N> &mean,                        \
                                         const CSymmetricMatrixNxN<double, N> &covariance,         \
                                         std::size_t n,                                            \
                                         std::vector<CVectorNx1<double, N>> &samples);             \
    static void multivariateNormalSample(CPRNG::CXorShift1024Mult &rng,                            \
                                         const CVectorNx1<double, N> &mean,                        \
                                         const CSymmetricMatrixNxN<double, N> &covariance,         \
                                         std::size_t n,                                            \
                                         std::vector<CVectorNx1<double, N>> &samples)
    MULTIVARIATE_NORMAL_SAMPLE(2);
    MULTIVARIATE_NORMAL_SAMPLE(3);
    MULTIVARIATE_NORMAL_SAMPLE(4);
    MULTIVARIATE_NORMAL_SAMPLE(5);
    MULTIVARIATE_NORMAL_SAMPLE(6);
    MULTIVARIATE_NORMAL_SAMPLE(7);
    MULTIVARIATE_NORMAL_SAMPLE(8);
    MULTIVARIATE_NORMAL_SAMPLE(9);
#undef MULTIVARIATE_NORMAL_SAMPLE
    //@}

    //! \name Categorical Sampling
    //@{
    //! Generate a sample from a categorical distribution with
    //! category probabilities \p probabilities.
    static std::size_t categoricalSample(TDoubleVec &probabilities);

    //! Generate a sample from a categorical distribution with
    //! category probabilities \p probabilities using \p rng.
    static std::size_t categoricalSample(CPRNG::CXorOShiro128Plus &rng, TDoubleVec &probabilities);

    //! Generate a sample from a categorical distribution with
    //! category probabilities \p probabilities using \p rng.
    static std::size_t categoricalSample(CPRNG::CXorShift1024Mult &rng, TDoubleVec &probabilities);

    //! Generate \p n samples from a categorical distribution
    //! with category probabilities \p probabilities assuming
    //! the values are replaced between draws.
    static void
    categoricalSampleWithReplacement(TDoubleVec &probabilities, std::size_t n, TSizeVec &result);

    //! Generate \p n samples from a categorical distribution
    //! with category probabilities \p probabilities using \p rng
    //! assuming the values are replaced between draws.
    static void categoricalSampleWithReplacement(CPRNG::CXorOShiro128Plus &rng,
                                                 TDoubleVec &probabilities,
                                                 std::size_t n,
                                                 TSizeVec &result);

    //! Generate \p n samples from a categorical distribution
    //! with category probabilities \p probabilities using \p rng
    //! assuming the values are replaced between draws.
    static void categoricalSampleWithReplacement(CPRNG::CXorShift1024Mult &rng,
                                                 TDoubleVec &probabilities,
                                                 std::size_t n,
                                                 TSizeVec &result);

    //! Generate \p n samples from a categorical distribution
    //! with category probabilities \p probabilities assuming
    //! the values are *not* replaced between draws.
    static void
    categoricalSampleWithoutReplacement(TDoubleVec &probabilities, std::size_t n, TSizeVec &result);

    //! Generate \p n samples from a categorical distribution
    //! with category probabilities \p probabilities using \p rng
    //! assuming the values are *not* replaced between draws.
    static void categoricalSampleWithoutReplacement(CPRNG::CXorOShiro128Plus &rng,
                                                    TDoubleVec &probabilities,
                                                    std::size_t n,
                                                    TSizeVec &result);

    //! Generate \p n samples from a categorical distribution
    //! with category probabilities \p probabilities using \p rng
    //! assuming the values are *not* replaced between draws.
    static void categoricalSampleWithoutReplacement(CPRNG::CXorShift1024Mult &rng,
                                                    TDoubleVec &probabilities,
                                                    std::size_t n,
                                                    TSizeVec &result);

    //! Generate samples from a multinomial distribution with number
    //! of trials \p n and category probabilities \p probabilities.
    //!
    //! This generates a sample from the distribution function
    //! <pre class="fragment">
    //!   \f$\(\displaystyle f(\{n_i\}) = \frac{n!}{\prod_i{n_i}}\prod_i{p_i^{n_i}}\)\f$
    //! </pre>
    //!
    //! The probabilities are sorted in decreasing order and
    //! \p sample is filled with the counts of each probability
    //! in this sorted order. This is the most efficient strategy
    //! since often \p sample will then be much smaller than
    //! \p probabilities and the sampling loop can end early.
    //!
    //! \param[in,out] probabilities The category probabilities,
    //! which should be normalized.
    //! \param[in] n The number of trials.
    //! \param[out] sample Filled in with the counts of the each
    //! category in the sorted probabilities order. Missing counts
    //! are zero.
    //! \param[in] sorted Set to true if the probabilities are
    //! already sorted in descending order.
    static void multinomialSampleFast(TDoubleVec &probabilities,
                                      std::size_t n,
                                      TSizeVec &sample,
                                      bool sorted = false);

    //! Generate samples according to the multinomial distribution
    //! with number of trials \p n and category probabilities
    //! \p probabilities.
    //!
    //! This generates a sample from the distribution function
    //! <pre class="fragment">
    //!   \f$\(\displaystyle f(\{n_i\}) = \frac{n!}{\prod_i{n_i}}\prod_i{p_i^{n_i}}\)\f$
    //! </pre>
    //!
    //! \param[in] probabilities The category probabilities, which
    //! should be normalized.
    //! \param[in] n The number of trials.
    //! \param[out] sample Filled in with the counts of each category
    //! in \p probabilities. This sample includes zeros explicitly,
    //! contrast with multinomialSampleFast.
    static void multinomialSampleStable(TDoubleVec probabilities, std::size_t n, TSizeVec &sample);
    //@}

    //! Sample a random permutation of the value [\p first, \p last).
    //!
    //! Reorders the elements in the range [\p first, \p last) using the
    //! supplied random number generator.
    //!
    //! \note We provide our own implementation of std::random_shuffle
    //! based on the libc++ implementation because this is different from
    //! the libstdc++ implementation which can cause platform specific
    //! differences.
    template <typename RNG, typename ITR>
    static void random_shuffle(RNG &rng, ITR first, ITR last) {
        auto d = last - first;
        if (d > 1) {
            CUniform0nGenerator<RNG> rand(rng);
            for (--last; first < last; ++first, --d) {
                auto i = rand(d);
                if (i > 0) {
                    std::iter_swap(first, first + i);
                }
            }
        }
    }

    //! Sample a random permutation of the value [\p first, \p last).
    //!
    //! Reorders the elements in the range [\p first, \p last) using the
    //! internal random number generator to provide a random distribution.
    template <typename ITR> static void random_shuffle(ITR first, ITR last) {
        core::CScopedFastLock scopedLock(ms_Lock);
        random_shuffle(ms_Rng, first, last);
    }

    //! Optimal (in a sense to be defined below) weighted sampling
    //! algorithm.
    //!
    //! Compute a set \p numberSamples (\f$\{n(i)\}\f$) such that
    //! the following constraints are satisfied.
    //! <pre class="fragment">
    //!   \f$\displaystyle n_i \in \{0,1,...,n\}\ \ \forall i\f$
    //!   \f$\displaystyle \sum_i{n_i} = [n \sum_i{w(i)}]\f$
    //!   \f$\displaystyle \sum_i{|n_i - n w(i)|}\ is\ minimized\f$
    //! </pre>
    //!
    //! Typically, the weights will sum to one.
    //!
    //! \param n The total number of samples required.
    //! \param weights The weights with which to sample.
    //! \param sampling Filled in with the weighted sampling.
    static void weightedSample(std::size_t n, const TDoubleVec &weights, TSizeVec &sampling);

    //! Sample the expectation of the normal distribution with \p mean
    //! and \p variance on the \p n quantile intervals.
    static void
    normalSampleQuantiles(double mean, double variance, std::size_t n, TDoubleVec &result);

    //! Sample the expectation of the gamma distribution with \p shape
    //! and \p rate on the \p n quantile intervals.
    static void gammaSampleQuantiles(double shape, double rate, std::size_t n, TDoubleVec &result);

private:
    //! \brief A uniform generator on the interval [0, n).
    template <typename RNG> class CUniform0nGenerator {
    public:
        CUniform0nGenerator(RNG &generator) : m_Generator(&generator) {}
        std::size_t operator()(std::size_t n) const {
            boost::random::uniform_int_distribution<std::size_t> uniform(0, n - 1);
            return uniform(*m_Generator);
        }

    private:
        RNG *m_Generator;
    };

private:
    //! The mutex for protecting access to the random number generator.
    static core::CFastMutex ms_Lock;

    //! The uniform random number generator.
    static CRandomNumberGenerator ms_Rng;
};
}
}

#endif// INCLUDED_ml_maths_CSampling_h
