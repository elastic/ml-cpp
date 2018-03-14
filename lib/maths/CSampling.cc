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

#include <maths/CSampling.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CScopedFastLock.h>

#include <maths/CLinearAlgebraEigen.h>
#include <maths/COrderings.h>
#include <maths/CTools.h>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include <algorithm>
#include <numeric>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

#include <math.h>

namespace ml {
namespace maths {

namespace {

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeVec = std::vector<std::size_t>;

//! Defines the appropriate integer random number generator.
template<typename INTEGER>
struct SRng {
    using Type = boost::random::uniform_int_distribution<INTEGER>;
    static INTEGER min(INTEGER a) {
        return a;
    }
    static INTEGER max(INTEGER b) {
        return b - 1;
    }
};
//! Specialization for a real uniform random number generator.
template<>
struct SRng<double> {
    using Type = boost::random::uniform_real_distribution<double>;
    static double min(double a) {
        return a;
    }
    static double max(double b) {
        return b;
    }
};

//! Implementation of uniform sampling.
template<typename RNG, typename TYPE>
TYPE doUniformSample(RNG &rng, TYPE a, TYPE b) {
    typename SRng<TYPE>::Type uniform(SRng<TYPE>::min(a), SRng<TYPE>::max(b));
    return uniform(rng);
}

//! Implementation of uniform sampling.
template<typename RNG, typename TYPE>
void doUniformSample(RNG &rng, TYPE a, TYPE b, std::size_t n, std::vector<TYPE> &result) {
    result.clear();
    result.reserve(n);
    typename SRng<TYPE>::Type uniform(SRng<TYPE>::min(a), SRng<TYPE>::max(b));
    for (std::size_t i = 0u; i < n; ++i) {
        result.push_back(uniform(rng));
    }
}

//! Implementation of normal sampling.
template<typename RNG>
double doNormalSample(RNG &rng, double mean, double variance) {
    if (variance < 0.0) {
        LOG_ERROR("Invalid variance " << variance);
        return mean;
    }
    boost::random::normal_distribution<double> normal(mean, ::sqrt(variance));
    return normal(rng);
}

//! Implementation of normal sampling.
template<typename RNG>
void doNormalSample(RNG &rng, double mean, double variance, std::size_t n, TDoubleVec &result) {
    result.clear();
    if (variance < 0.0) {
        LOG_ERROR("Invalid variance " << variance);
        return;
    } else if (variance == 0.0) {
        result.resize(n, mean);
    }

    result.reserve(n);
    boost::random::normal_distribution<double> normal(mean, ::sqrt(variance));
    for (std::size_t i = 0u; i < n; ++i) {
        result.push_back(normal(rng));
    }
}

//! Implementation of chi^2 sampling.
template<typename RNG>
void doChiSquaredSample(RNG &rng, double f, std::size_t n, TDoubleVec &result) {
    result.clear();
    result.reserve(n);
    boost::random::chi_squared_distribution<double> chi2(f);
    for (std::size_t i = 0u; i < n; ++i) {
        result.push_back(chi2(rng));
    }
}

//! Implementation of categorical sampling.
template<typename RNG>
std::size_t doCategoricalSample(RNG &rng, TDoubleVec &probabilities) {
    // We use inverse transform sampling to generate the categorical
    // samples from a random samples on [0,1].

    std::size_t p = probabilities.size();

    // Construct the transform function.
    for (std::size_t i = 1u; i < p; ++i) {
        probabilities[i] += probabilities[i - 1];
    }

    double uniform0X;
    if (probabilities[p - 1] == 0.0) {
        return doUniformSample(rng, std::size_t(0), p);
    } else {
        boost::random::uniform_real_distribution<> uniform(0.0, probabilities[p - 1]);
        uniform0X = uniform(rng);
    }

    return std::min(static_cast<std::size_t>(
                        std::lower_bound(probabilities.begin(),
                                         probabilities.end(),
                                         uniform0X) - probabilities.begin()),
                    probabilities.size() - 1);
}

//! Implementation of categorical sampling with replacement.
template<typename RNG>
void doCategoricalSampleWithReplacement(RNG &rng,
                                        TDoubleVec &probabilities,
                                        std::size_t n,
                                        TSizeVec &result) {
    // We use inverse transform sampling to generate the categorical
    // samples from random samples on [0,1].

    result.clear();
    if (n == 0) {
        return;
    }

    std::size_t p = probabilities.size();

    // Construct the transform function.
    for (std::size_t i = 1u; i < p; ++i) {
        probabilities[i] += probabilities[i - 1];
    }

    if (probabilities[p - 1] == 0.0) {
        doUniformSample(rng, std::size_t(0), p, n, result);
    } else {
        result.reserve(n);
        boost::random::uniform_real_distribution<> uniform(0.0, probabilities[p - 1]);
        for (std::size_t i = 0u; i < n; ++i) {
            double uniform0X = uniform(rng);
            result.push_back(std::min(static_cast<std::size_t>(
                                          std::lower_bound(probabilities.begin(),
                                                           probabilities.end(),
                                                           uniform0X) - probabilities.begin()),
                                      probabilities.size() - 1));
        }
    }
}

//! Implementation of categorical sampling without replacement.
template<typename RNG>
void doCategoricalSampleWithoutReplacement(RNG &rng,
                                           TDoubleVec &probabilities,
                                           std::size_t n,
                                           TSizeVec &result) {
    // We use inverse transform sampling to generate the categorical
    // samples from random samples on [0,1] and update the probabilities
    // throughout the sampling to exclude the values already taken.

    result.clear();
    if (n == 0) {
        return;
    }

    std::size_t p = probabilities.size();
    if (n >= p) {
        result.assign(boost::counting_iterator<std::size_t>(0),
                      boost::counting_iterator<std::size_t>(p));
    }

    // Construct the transform function.
    for (std::size_t i = 1u; i < p; ++i) {
        probabilities[i] += probabilities[i - 1];
    }

    result.reserve(n);
    TSizeVec indices(boost::counting_iterator<std::size_t>(0),
                     boost::counting_iterator<std::size_t>(p));
    TSizeVec s(1);

    for (std::size_t i = 0u; i < n; ++i, --p) {
        if (probabilities[p - 1] <= 0.0) {
            doUniformSample(rng, std::size_t(0), indices.size(), 1, s);
            result.push_back(indices[s[0]]);
        } else {
            boost::random::uniform_real_distribution<> uniform(0.0, probabilities[p - 1]);
            double                                     uniform0X = uniform(rng);
            s[0] = std::min(static_cast<std::size_t>(
                                std::lower_bound(probabilities.begin(),
                                                 probabilities.end(),
                                                 uniform0X) - probabilities.begin()),
                            probabilities.size() - 1);

            result.push_back(indices[s[0]]);

            double ps = probabilities[s[0]] - (s[0] == 0 ? 0.0 : probabilities[s[0] - 1]);
            for (std::size_t j = s[0] + 1; j < p; ++j) {
                probabilities[j - 1] = probabilities[j] - ps;
            }
            probabilities.pop_back();
        }
        indices.erase(indices.begin() + s[0]);
    }
}

//! Implementation of multivariate normal sampling.
template<typename RNG>
bool doMultivariateNormalSample(RNG &rng,
                                const TDoubleVec &mean,
                                const TDoubleVecVec &covariance,
                                std::size_t n,
                                TDoubleVecVec &samples) {
    using TJacobiSvd = Eigen::JacobiSVD<CDenseMatrix<double> >;

    if (mean.size() != covariance.size()) {
        LOG_ERROR("Incompatible mean and covariance: "
                  << core::CContainerPrinter::print(mean)
                  << ", "
                  << core::CContainerPrinter::print(covariance));
        return false;
    }

    samples.clear();
    if (n == 0) {
        return true;
    }

    // This computes the singular value decomposition of the covariance
    // matrix (note since the covariance matrix is symmetric only its
    // upper triangle is passed), i.e. C = U * S * V'.
    //
    // The singular values S should be rank full and U should equal V
    // for positive definite matrix. Therefore, using the fact that
    // linear combinations of normal random variables are normal we can
    // construct each sample using independent samples
    //   X = { X_i = N(0, [S]_ii) }
    //
    // and transforming as Y = m + U * X. Note that E[Y] is clearly m
    // and its covariance matrix is
    //   E[(Y - E[Y]) * (Y - E[Y])'] = U * E[X * X'] * U' = U * S * U'
    //
    // as required.

    std::size_t d = mean.size();
    LOG_TRACE("Dimension = " << d);
    LOG_TRACE("mean = " << core::CContainerPrinter::print(mean));

    CDenseMatrix<double> C(d,d);
    for (std::size_t i = 0u; i < d; ++i) {
        C(i, i) = covariance[i][i];
        if (covariance[i].size() < d - i) {
            LOG_ERROR("Bad covariance matrix: "
                      << core::CContainerPrinter::print(covariance));
            return false;
        }
        for (std::size_t j = 0; j < i; ++j) {
            C(i, j) = covariance[i][j];
            C(j, i) = covariance[i][j];
        }
    }
    LOG_TRACE("C =" << core_t::LINE_ENDING << ' ' << C);

    TJacobiSvd svd(C, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Get the singular values, these are the variances of the normals
    // to sample.
    const CDenseVector<double> &S = svd.singularValues();
    const CDenseMatrix<double> &U = svd.matrixU();
    TDoubleVec                  stddevs;
    stddevs.reserve(d);
    for (std::size_t i = 0u; i < d; ++i) {
        stddevs.push_back(::sqrt(std::max(S(i), 0.0)));
    }
    LOG_TRACE("Singular values of C = " << S.transpose());
    LOG_TRACE("stddevs = " << core::CContainerPrinter::print(stddevs));
    LOG_TRACE("U =" << core_t::LINE_ENDING << ' ' << U);
    LOG_TRACE("V =" << core_t::LINE_ENDING << ' ' << svd.matrixV());

    {
        samples.resize(n, mean);
        CDenseVector<double> sample(d);
        for (std::size_t i = 0u; i < n; ++i) {
            for (std::size_t j = 0u; j < d; ++j) {
                if (stddevs[j] == 0.0) {
                    sample(j) = 0.0;
                } else {
                    boost::random::normal_distribution<> normal(0.0, stddevs[j]);
                    sample(j) = normal(rng);
                }
            }
            sample = U * sample;
            for (std::size_t j = 0u; j < d; ++j) {
                samples[i][j] += sample(j);
            }
        }
    }

    return true;
}

//! Implementation of multivariate normal sampling.
template<typename RNG, typename T, std::size_t N>
void doMultivariateNormalSample(RNG &rng,
                                const CVectorNx1<T, N> &mean,
                                const CSymmetricMatrixNxN<T, N> &covariance,
                                std::size_t n,
                                std::vector<CVectorNx1<T, N> > &samples) {
    using TDenseVector = typename SDenseVector<CVectorNx1<T, N> >::Type;
    using TDenseMatrix = typename SDenseMatrix<CSymmetricMatrixNxN<T, N> >::Type;
    using TJacobiSvd = Eigen::JacobiSVD<TDenseMatrix>;

    samples.clear();
    if (n == 0) {
        return;
    }

    // See the other implementation for an explanation.

    TJacobiSvd svd(toDenseMatrix(covariance), Eigen::ComputeFullU | Eigen::ComputeFullV);

    // Get the singular values, these are the variances of the normals
    // to sample.
    const TDenseVector &S = svd.singularValues();
    const TDenseMatrix &U = svd.matrixU();
    T                   stddevs[N] = {};
    for (std::size_t i = 0u; i < N; ++i) {
        stddevs[i] = ::sqrt(std::max(S(i), 0.0));
    }

    {
        samples.resize(n, mean);
        TDenseVector sample(N);
        for (std::size_t i = 0u; i < n; ++i) {
            for (std::size_t j = 0u; j < N; ++j) {
                if (stddevs[j] == 0.0) {
                    sample(j) = 0.0;
                } else {
                    boost::random::normal_distribution<> normal(0.0, stddevs[j]);
                    sample(j) = normal(rng);
                }
            }
            sample = U * sample;
            samples[i] += fromDenseVector(sample);
        }
    }
}

//! Implementation of distribution quantile sampling.
template<typename DISTRIBUTION>
void sampleQuantiles(const DISTRIBUTION &distribution, std::size_t n, TDoubleVec &result) {
    CTools::SIntervalExpectation expectation;
    double                       dq = 1.0 / static_cast<double>(n);

    double a = boost::numeric::bounds<double>::lowest();
    for (std::size_t i = 1u; i < n; ++i) {
        double q = static_cast<double>(i) * dq;
        double b = boost::math::quantile(distribution, q);
        result.push_back(expectation(distribution, a, b));
        a = b;
    }
    double b = boost::numeric::bounds<double>::highest();
    result.push_back(expectation(distribution, a, b));
}

static const std::string RNG_TAG("a");

}

bool CSampling::staticsAcceptRestoreTraverser(core::CStateRestoreTraverser &traverser) {
    // Note we require that we only ever do one persistence per process.

    do {
        const std::string &name = traverser.name();

        if (name == RNG_TAG) {
            std::string value(traverser.value());
            // See acceptPersistInserter
            std::replace(value.begin(), value.end(), '_', ' ');
            std::istringstream    ss(value);
            core::CScopedFastLock scopedLock(ms_Lock);
            ss >> ms_Rng;
        }
    } while (traverser.next());

    return true;
}

void CSampling::staticsAcceptPersistInserter(core::CStatePersistInserter &inserter) {
    // Note we require that we only ever do one persistence per process.

    std::ostringstream ss;
    {
        core::CScopedFastLock scopedLock(ms_Lock);
        ss << ms_Rng;
    }
    std::string rng(ss.str());
    // These are space separated integers. We replace spaces or else
    // the JSON parser gets confused.
    std::replace(rng.begin(), rng.end(), ' ', '_');
    inserter.insertValue(RNG_TAG, rng);
}

void CSampling::seed(void) {
    core::CScopedFastLock scopedLock(ms_Lock);
    ms_Rng.seed();
}

#define UNIFORM_SAMPLE(TYPE)                                                 \
    TYPE CSampling::uniformSample(TYPE a, TYPE b)                                \
    {                                                                            \
        core::CScopedFastLock scopedLock(ms_Lock);                               \
        return doUniformSample(ms_Rng, a, b);                                    \
    }                                                                            \
    TYPE CSampling::uniformSample(CPRNG::CXorOShiro128Plus &rng, TYPE a, TYPE b) \
    {                                                                            \
        return doUniformSample(rng, a, b);                                       \
    }                                                                            \
    TYPE CSampling::uniformSample(CPRNG::CXorShift1024Mult &rng, TYPE a, TYPE b) \
    {                                                                            \
        return doUniformSample(rng, a, b);                                       \
    }                                                                            \
    void CSampling::uniformSample(TYPE a, TYPE b, std::size_t n,                 \
                                  std::vector<TYPE> &result)                     \
    {                                                                            \
        core::CScopedFastLock scopedLock(ms_Lock);                               \
        doUniformSample(ms_Rng, a, b, n, result);                                \
    }                                                                            \
    void CSampling::uniformSample(CPRNG::CXorOShiro128Plus &rng,                 \
                                  TYPE a, TYPE b, std::size_t n,                 \
                                  std::vector<TYPE> &result)                     \
    {                                                                            \
        doUniformSample(rng, a, b, n, result);                                   \
    }                                                                            \
    void CSampling::uniformSample(CPRNG::CXorShift1024Mult &rng,                 \
                                  TYPE a, TYPE b, std::size_t n,                 \
                                  std::vector<TYPE> &result)                     \
    {                                                                            \
        doUniformSample(rng, a, b, n, result);                                   \
    }
UNIFORM_SAMPLE(std::size_t)
UNIFORM_SAMPLE(std::ptrdiff_t)
UNIFORM_SAMPLE(double)
#undef UNIFORM_SAMPLE

double CSampling::normalSample(double mean, double variance) {
    core::CScopedFastLock scopedLock(ms_Lock);
    return doNormalSample(ms_Rng, mean, variance);
}

double CSampling::normalSample(CPRNG::CXorOShiro128Plus &rng, double mean, double variance) {
    return doNormalSample(rng, mean, variance);
}

double CSampling::normalSample(CPRNG::CXorShift1024Mult &rng, double mean, double variance) {
    return doNormalSample(rng, mean, variance);
}

void CSampling::normalSample(double mean, double variance, std::size_t n, TDoubleVec &result) {
    core::CScopedFastLock scopedLock(ms_Lock);
    doNormalSample(ms_Rng, mean, variance, n, result);
}

void CSampling::normalSample(CPRNG::CXorOShiro128Plus &rng,
                             double mean,
                             double variance,
                             std::size_t n,
                             TDoubleVec &result) {
    doNormalSample(rng, mean, variance, n, result);
}

void CSampling::normalSample(CPRNG::CXorShift1024Mult &rng,
                             double mean,
                             double variance,
                             std::size_t n,
                             TDoubleVec &result) {
    doNormalSample(rng, mean, variance, n, result);
}

void CSampling::chiSquaredSample(double f, std::size_t n, TDoubleVec &result) {
    core::CScopedFastLock scopedLock(ms_Lock);
    doChiSquaredSample(ms_Rng, f, n, result);
}

void CSampling::chiSquaredSample(CPRNG::CXorOShiro128Plus &rng,
                                 double f,
                                 std::size_t n,
                                 TDoubleVec &result) {
    doChiSquaredSample(rng, f, n, result);
}

void CSampling::chiSquaredSample(CPRNG::CXorShift1024Mult &rng,
                                 double f,
                                 std::size_t n,
                                 TDoubleVec &result) {
    doChiSquaredSample(rng, f, n, result);
}

bool CSampling::multivariateNormalSample(const TDoubleVec &mean,
                                         const TDoubleVecVec &covariance,
                                         std::size_t n,
                                         TDoubleVecVec &samples) {
    core::CScopedFastLock scopedLock(ms_Lock);
    return doMultivariateNormalSample(ms_Rng, mean, covariance, n, samples);
}

bool CSampling::multivariateNormalSample(CPRNG::CXorOShiro128Plus &rng,
                                         const TDoubleVec &mean,
                                         const TDoubleVecVec &covariance,
                                         std::size_t n,
                                         TDoubleVecVec &samples) {
    return doMultivariateNormalSample(rng, mean, covariance, n, samples);
}

bool CSampling::multivariateNormalSample(CPRNG::CXorShift1024Mult &rng,
                                         const TDoubleVec &mean,
                                         const TDoubleVecVec &covariance,
                                         std::size_t n,
                                         TDoubleVecVec &samples) {
    return doMultivariateNormalSample(rng, mean, covariance, n, samples);
}

#define MULTIVARIATE_NORMAL_SAMPLE(N)                                                      \
    void CSampling::multivariateNormalSample(const CVectorNx1<double, N> &mean,                \
                                             const CSymmetricMatrixNxN<double, N> &covariance, \
                                             std::size_t n,                                    \
                                             std::vector<CVectorNx1<double, N> > &samples)     \
    {                                                                                          \
        core::CScopedFastLock scopedLock(ms_Lock);                                             \
        doMultivariateNormalSample(ms_Rng, mean, covariance, n, samples);                      \
    }                                                                                          \
    void CSampling::multivariateNormalSample(CPRNG::CXorOShiro128Plus &rng,                    \
                                             const CVectorNx1<double, N> &mean,                \
                                             const CSymmetricMatrixNxN<double, N> &covariance, \
                                             std::size_t n,                                    \
                                             std::vector<CVectorNx1<double, N> > &samples)     \
    {                                                                                          \
        doMultivariateNormalSample(rng, mean, covariance, n, samples);                         \
    }                                                                                          \
    void CSampling::multivariateNormalSample(CPRNG::CXorShift1024Mult &rng,                    \
                                             const CVectorNx1<double, N> &mean,                \
                                             const CSymmetricMatrixNxN<double, N> &covariance, \
                                             std::size_t n,                                    \
                                             std::vector<CVectorNx1<double, N> > &samples)     \
    {                                                                                          \
        doMultivariateNormalSample(rng, mean, covariance, n, samples);                         \
    }
MULTIVARIATE_NORMAL_SAMPLE(2)
MULTIVARIATE_NORMAL_SAMPLE(3)
MULTIVARIATE_NORMAL_SAMPLE(4)
MULTIVARIATE_NORMAL_SAMPLE(5)
#undef MULTIVARIATE_NORMAL_SAMPLE

std::size_t CSampling::categoricalSample(TDoubleVec &probabilities) {
    core::CScopedFastLock scopedLock(ms_Lock);
    return doCategoricalSample(ms_Rng, probabilities);
}

std::size_t CSampling::categoricalSample(CPRNG::CXorOShiro128Plus &rng, TDoubleVec &probabilities) {
    return doCategoricalSample(rng, probabilities);
}

std::size_t CSampling::categoricalSample(CPRNG::CXorShift1024Mult &rng, TDoubleVec &probabilities) {
    return doCategoricalSample(rng, probabilities);
}

void CSampling::categoricalSampleWithReplacement(TDoubleVec &probabilities,
                                                 std::size_t n,
                                                 TSizeVec &result) {
    core::CScopedFastLock scopedLock(ms_Lock);
    doCategoricalSampleWithReplacement(ms_Rng, probabilities, n, result);
}

void CSampling::categoricalSampleWithReplacement(CPRNG::CXorOShiro128Plus &rng,
                                                 TDoubleVec &probabilities,
                                                 std::size_t n,
                                                 TSizeVec &result) {
    doCategoricalSampleWithReplacement(rng, probabilities, n, result);
}

void CSampling::categoricalSampleWithReplacement(CPRNG::CXorShift1024Mult &rng,
                                                 TDoubleVec &probabilities,
                                                 std::size_t n,
                                                 TSizeVec &result) {
    doCategoricalSampleWithReplacement(rng, probabilities, n, result);
}

void CSampling::categoricalSampleWithoutReplacement(TDoubleVec &probabilities,
                                                    std::size_t n,
                                                    TSizeVec &result) {
    core::CScopedFastLock scopedLock(ms_Lock);
    doCategoricalSampleWithoutReplacement(ms_Rng, probabilities, n, result);
}

void CSampling::categoricalSampleWithoutReplacement(CPRNG::CXorOShiro128Plus &rng,
                                                    TDoubleVec &probabilities,
                                                    std::size_t n,
                                                    TSizeVec &result) {
    doCategoricalSampleWithReplacement(rng, probabilities, n, result);
}

void CSampling::categoricalSampleWithoutReplacement(CPRNG::CXorShift1024Mult &rng,
                                                    TDoubleVec &probabilities,
                                                    std::size_t n,
                                                    TSizeVec &result) {
    doCategoricalSampleWithReplacement(rng, probabilities, n, result);
}

void CSampling::multinomialSampleFast(TDoubleVec &probabilities,
                                      std::size_t n,
                                      TSizeVec &sample,
                                      bool sorted) {
    sample.clear();

    if (n == 0 || probabilities.empty()) {
        return;
    }

    // This uses the fact that we can decompose the probability mass
    // function for the multinomial as follows:
    //   f({n_i}) = Prod_i{ f(n_i | { n_j : j < i }) }
    //
    // The conditional mass function of category i given categories
    // j < i is found by marginalizing over the categories j > i of
    // the full marginal distribution. The marginal distribution is
    // multinomial with number of trials n - Sum_{j < i}{ nj } and
    // probabilities p_j -> p_j / (1 - Sum_k{k < i}{ pk }). The marginal
    // distribution is just binomial with the same number of trials
    // and probability p_i (adjusted as above).
    //
    // In order to sample from the full distribution we sample each
    // marginal and then condition the next marginal on the values
    // we've sampled so far. It is straightforward to verify that
    // the probability distribution is then:
    //   P({n_i}) = Int{ Prod_i{ f(n_i | { n_j : j < i }) } }dn_i
    //            = Int{ f(n_i) }dn_i
    //
    // So the samples are distributed according to the joint distribution
    // function, i.e. multinomial in this case.
    //
    // We sort the probabilities in descending order to make sampling
    // as efficient as possible (since this means that the the loop
    // will often terminate as early as possible on average).

    if (!sorted) {
        std::sort(probabilities.begin(), probabilities.end(), std::greater<double>());
    }

    {
        std::size_t           r = n;
        double                p = 1.0;
        std::size_t           m = probabilities.size() - 1;
        core::CScopedFastLock scopedLock(ms_Lock);
        for (std::size_t i = 0u; r > 0 && i < m; ++i) {
            boost::random::binomial_distribution<> binomial(static_cast<int>(r),
                                                            probabilities[i] / p);
            std::size_t ni = static_cast<std::size_t>(binomial(ms_Rng));
            sample.push_back(ni);
            r -= ni;
            p -= probabilities[i];
        }
        if (r > 0) {
            sample.push_back(r);
        }
    }
}

void CSampling::multinomialSampleStable(TDoubleVec probabilities,
                                        std::size_t n,
                                        TSizeVec &sample) {
    TSizeVec indices;
    indices.reserve(probabilities.size());
    for (std::size_t i = 0u; i < probabilities.size(); ++i) {
        indices.push_back(i);
    }
    COrderings::simultaneousSort(probabilities, indices, std::greater<double>());

    sample.reserve(probabilities.size());

    multinomialSampleFast(probabilities, n, sample);

    sample.resize(probabilities.size(), 0);
    for (std::size_t i = 0u; i < sample.size(); /**/) {
        std::size_t j = indices[i];
        if (i != j) {
            std::swap(sample[i], sample[j]);
            std::swap(indices[i], indices[j]);
        } else {
            ++i;
        }
    }
}

void CSampling::weightedSample(std::size_t n,
                               const TDoubleVec &weights,
                               TSizeVec &sampling) {
    // We sample each category, corresponding to the index i of its,
    // weight according to its weight.
    //
    // Formally, we have a set of weights w(i) s.t. Sum_i{ w(i) } = 1.0.
    // We can only sample a model an integer number of times and we'd
    // like to sample each model as near as possible to w(i) * n times,
    // where n is the total number of samples. In addition we have the
    // constraint that the total number of samples must equal n. Defining
    // r(j, i) as follows:
    //   r(0, i) = w(i) * n - floor(w(i) * n)
    //   r(1, i) = w(i) * n - ceil(w(i) * n)
    //
    // The problem is equivalent to finding the rounding function c(i)
    // such that:
    //   Sum_i( r(c(i), i) ) = 0
    //   Sum_i( |r(c(i), i)| ) is minimized
    //
    // and then sampling the i'th model w(i) * n - r(i, c(i)) times.
    // This problem is solved optimally by a greedy algorithm. Note
    // that Sum_i{ w(i) } != 1.0 we round the number of samples to
    // the nearest integer to n * Sum_i{ p(i) }.

    typedef std::vector<unsigned int> TUIntVec;
    typedef std::pair<double, std::size_t> TDoubleSizePr;
    typedef std::vector<TDoubleSizePr> TDoubleSizePrVec;

    LOG_TRACE("Number samples = " << n);

    sampling.clear();

    double totalWeight = std::accumulate(weights.begin(), weights.end(), 0.0);

    n = std::max(static_cast<std::size_t>(totalWeight
                                          * static_cast<double>(n) + 0.5),
                 static_cast<std::size_t>(1u));

    LOG_TRACE("totalWeight = " << totalWeight << ", n = " << n);

    TUIntVec   choices;
    TDoubleVec remainders[] = { TDoubleVec(), TDoubleVec() };

    choices.reserve(weights.size());
    remainders[0].reserve(weights.size());
    remainders[1].reserve(weights.size());

    double totalRemainder = 0.0;
    for (std::size_t i = 0u; i < weights.size(); ++i) {
        // We need to re-normalize so that the probabilities sum to one.
        double number = weights[i] * static_cast<double>(n) / totalWeight;
        choices.push_back((number - ::floor(number) < 0.5) ? 0u : 1u);
        remainders[0].push_back(number - ::floor(number));
        remainders[1].push_back(number - ::ceil(number));
        totalRemainder += remainders[choices.back()].back();
    }

    // The remainder will be integral so checking against 0.5 avoids
    // floating point problems.

    if (::fabs(totalRemainder) > 0.5) {
        LOG_TRACE("ideal choice function = "
                  << core::CContainerPrinter::print(choices));

        TDoubleSizePrVec candidates;
        for (std::size_t i = 0u; i < choices.size(); ++i) {
            if (   (totalRemainder > 0.0 && choices[i] == 0u) ||
                   (totalRemainder < 0.0 && choices[i] == 1u)) {
                candidates.emplace_back(-::fabs(remainders[choices[i]][i]), i);
            }
        }
        std::sort(candidates.begin(), candidates.end());
        LOG_TRACE("candidates = "
                  << core::CContainerPrinter::print(candidates));

        for (std::size_t i = 0u;
             i < candidates.size() && ::fabs(totalRemainder) > 0.5;
             ++i) {
            std::size_t  j = candidates[i].second;
            unsigned int choice = choices[j];
            choices[j] = (choice + 1u) % 2u;
            totalRemainder += remainders[choices[j]][j] - remainders[choice][j];
        }
    }
    LOG_TRACE("choice function = "
              << core::CContainerPrinter::print(choices));

    sampling.reserve(weights.size());
    for (std::size_t i = 0u; i < weights.size(); ++i) {
        double number = weights[i] * static_cast<double>(n) / totalWeight;

        sampling.push_back(static_cast<std::size_t>(
                               choices[i] == 0u ? ::floor(number) : ::ceil(number)));
    }
}

void CSampling::normalSampleQuantiles(double mean,
                                      double variance,
                                      std::size_t n,
                                      TDoubleVec &result) {
    result.clear();
    if (n == 0) {
        return;
    }

    if (variance == 0.0) {
        result.resize(n, mean);
        return;
    }

    try {
        boost::math::normal_distribution<> normal(mean, ::sqrt(variance));
        sampleQuantiles(normal, n, result);
    } catch (const std::exception &e) {
        LOG_ERROR("Failed to sample normal quantiles: " << e.what()
                                                        << ", mean = " << mean
                                                        << ", variance = " << variance);
        result.clear();
    }
}

void CSampling::gammaSampleQuantiles(double shape,
                                     double rate,
                                     std::size_t n,
                                     TDoubleVec &result) {
    result.clear();
    if (n == 0) {
        return;
    }

    try {
        boost::math::gamma_distribution<> gamma(shape, 1.0 / rate);
        sampleQuantiles(gamma, n, result);
    } catch (const std::exception &e) {
        LOG_ERROR("Failed to sample normal quantiles: " << e.what()
                                                        << ", shape = " << shape
                                                        << ", rate = " << rate);
        result.clear();
    }
}

core::CFastMutex                  CSampling::ms_Lock;
CSampling::CRandomNumberGenerator CSampling::ms_Rng;


void CSampling::CRandomNumberGenerator::mock(void) {
    m_Mock.reset((min() + max()) / 2);
}

void CSampling::CRandomNumberGenerator::unmock(void) {
    m_Mock.reset();
}

void CSampling::CRandomNumberGenerator::seed(void) {
    m_Rng.seed();
}

CSampling::CScopeMockRandomNumberGenerator::CScopeMockRandomNumberGenerator(void) {
    CSampling::ms_Rng.mock();
}

CSampling::CScopeMockRandomNumberGenerator::~CScopeMockRandomNumberGenerator(void) {
    CSampling::ms_Rng.unmock();
}

}
}
