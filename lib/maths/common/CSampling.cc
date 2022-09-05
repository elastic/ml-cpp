/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <maths/common/CSampling.h>

#include <core/CLogger.h>
#include <core/CScopedFastLock.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

#include <maths/common/CLinearAlgebraEigen.h>
#include <maths/common/COrderings.h>
#include <maths/common/COrderingsSimultaneousSort.h>
#include <maths/common/CTools.h>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

namespace ml {
namespace maths {
namespace common {
namespace {
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeVec = std::vector<std::size_t>;

//! \brief A mockable random number generator which uses boost::random::mt11213b.
class CRandomNumberGenerator {
public:
    using result_type = boost::random::mt11213b::result_type;

public:
    //! Mock the random number generator to produce a constant.
    void mock() { m_Mock.emplace((min() + max()) / 2); }

    //! Unmock the random number generator.
    void unmock() { m_Mock.reset(); }

    //! Seed the random number generator.
    void seed() { m_Rng.seed(); }

    //! Returns the smallest value that the generator can produce.
    static result_type min() { return boost::random::mt11213b::min(); }

    //! Returns the largest value that the generator can produce.
    static result_type max() { return boost::random::mt11213b::max(); }

    //! Produces the next value of the generator.
    result_type operator()() {
        if (m_Mock) {
            return *m_Mock;
        }
        return m_Rng.operator()();
    }

    //! Writes the mersenne_twister_engine to a std::ostream.
    template<class CHAR, class TRAITS>
    friend std::basic_ostream<CHAR, TRAITS>&
    operator<<(std::basic_ostream<CHAR, TRAITS>& o, const CRandomNumberGenerator& g) {
        return o << g.m_Rng;
    }

    //! Reads a mersenne_twister_engine from a std::istream.
    template<class CHAR, class TRAITS>
    friend std::basic_istream<CHAR, TRAITS>&
    operator>>(std::basic_istream<CHAR, TRAITS>& i, CRandomNumberGenerator& g) {
        return i >> g.m_Rng;
    }

private:
    using TOptionalResultType = std::optional<result_type>;

private:
    TOptionalResultType m_Mock;
    boost::random::mt11213b m_Rng;
};

//! The mutex for protecting access to the default random number generator.
core::CFastMutex defaultRngMutex;

//! The default uniform random number generator.
CRandomNumberGenerator defaultRng;

//! Defines the appropriate integer random number generator.
template<typename INTEGER>
struct SRng {
    using Type = boost::random::uniform_int_distribution<INTEGER>;
    static INTEGER min(INTEGER a) { return a; }
    static INTEGER max(INTEGER b) { return b - 1; }
};
//! Specialization for a real uniform random number generator.
template<>
struct SRng<double> {
    using Type = boost::random::uniform_real_distribution<double>;
    static double min(double a) { return a; }
    static double max(double b) { return b; }
};

//! Implementation of uniform sampling.
template<typename RNG, typename TYPE>
TYPE doUniformSample(RNG& rng, TYPE a, TYPE b) {
    if (a >= b) {
        return a;
    }
    typename SRng<TYPE>::Type uniform(SRng<TYPE>::min(a), SRng<TYPE>::max(b));
    return uniform(rng);
}

//! Implementation of uniform sampling.
template<typename RNG, typename TYPE>
void doUniformSample(RNG& rng, TYPE a, TYPE b, std::size_t n, std::vector<TYPE>& result) {
    result.clear();
    if (a >= b) {
        result.resize(n, a);
        return;
    }
    result.reserve(n);
    typename SRng<TYPE>::Type uniform(SRng<TYPE>::min(a), SRng<TYPE>::max(b));
    for (std::size_t i = 0; i < n; ++i) {
        result.push_back(uniform(rng));
    }
}

//! Implementation of normal sampling.
template<typename RNG>
double doNormalSample(RNG& rng, double mean, double variance) {
    if (variance < 0.0) {
        LOG_ERROR(<< "Invalid variance " << variance);
        return mean;
    }
    boost::random::normal_distribution<double> normal(mean, std::sqrt(variance));
    return normal(rng);
}

//! Implementation of normal sampling.
template<typename RNG>
void doNormalSample(RNG& rng, double mean, double variance, std::size_t n, TDoubleVec& result) {
    result.clear();
    if (variance < 0.0) {
        LOG_ERROR(<< "Invalid variance " << variance);
        return;
    }
    if (variance == 0.0) {
        result.resize(n, mean);
        return;
    }

    result.reserve(n);
    boost::random::normal_distribution<double> normal(mean, std::sqrt(variance));
    for (std::size_t i = 0; i < n; ++i) {
        result.push_back(normal(rng));
    }
}

//! Implementation of Poisson sampling.
template<typename RNG>
void doPoissonSample(RNG& rng, double rate, std::size_t n, TSizeVec& result) {
    result.clear();
    if (rate <= 0.0) {
        LOG_ERROR(<< "Invalid rate " << rate);
        return;
    }

    result.reserve(n);
    boost::random::poisson_distribution<std::size_t> poisson{rate};
    for (std::size_t i = 0; i < n; ++i) {
        result.push_back(poisson(rng));
    }
}

//! Implementation of chi^2 sampling.
template<typename RNG>
void doChiSquaredSample(RNG& rng, double f, std::size_t n, TDoubleVec& result) {
    result.clear();
    if (f <= 0.0) {
        LOG_ERROR(<< "Invalid degrees freedom " << f);
        return;
    }

    result.reserve(n);
    boost::random::chi_squared_distribution<double> chi2(f);
    for (std::size_t i = 0; i < n; ++i) {
        result.push_back(chi2(rng));
    }
}

//! Implementation of categorical sampling.
template<typename RNG>
std::size_t doCategoricalSample(RNG& rng, TDoubleVec& probabilities) {
    // We use inverse transform sampling to generate the categorical
    // samples from a random samples on [0,1].

    std::size_t p = probabilities.size();

    // Construct the transform function.
    for (std::size_t i = 1; i < p; ++i) {
        probabilities[i] += probabilities[i - 1];
    }

    if (probabilities[p - 1] == 0.0) {
        return doUniformSample(rng, static_cast<std::size_t>(0), p);
    }

    double uniform0X{doUniformSample(rng, 0.0, probabilities[p - 1])};
    return std::min(static_cast<std::size_t>(std::lower_bound(probabilities.begin(),
                                                              probabilities.end(), uniform0X) -
                                             probabilities.begin()),
                    probabilities.size() - 1);
}

//! Implementation of categorical sampling with replacement.
template<typename RNG>
void doCategoricalSampleWithReplacement(RNG& rng,
                                        TDoubleVec& probabilities,
                                        std::size_t n,
                                        TSizeVec& result) {
    // We use inverse transform sampling to generate the categorical
    // samples from random samples on [0,1].

    result.clear();
    if (n == 0) {
        return;
    }

    std::size_t m{probabilities.size()};

    // Construct the transform function.
    for (std::size_t i = 1; i < m; ++i) {
        probabilities[i] += probabilities[i - 1];
    }

    if (probabilities[m - 1] == 0.0) {
        doUniformSample(rng, static_cast<std::size_t>(0), m, n, result);
    } else {
        result.reserve(n);
        boost::random::uniform_real_distribution<> uniform(0.0, probabilities[m - 1]);
        for (std::size_t i = 0; i < n; ++i) {
            double u0X{uniform(rng)};
            result.push_back(std::min(
                static_cast<std::size_t>(std::lower_bound(probabilities.begin(),
                                                          probabilities.end(), u0X) -
                                         probabilities.begin()),
                m - 1));
        }
    }
}

//! Implementation of categorical sampling without replacement.
template<typename RNG>
void doCategoricalSampleWithoutReplacement(RNG& rng, TDoubleVec& weights, std::size_t n, TSizeVec& result) {

    // We switch between two strategies:
    //   1. For n < eps |weights| we perform categorical sampling and discard
    //      duplicates. This approach has better constants but compexity
    //      O(|weights| + n log(|weights|))
    //   2. Otherwise, we notionally reservoir sample n from a stream of size
    //      |weights| and use the Efraimidis and Spirakis trick. This approach
    //      has complexity O(|weights|).

    using TBoolVec = std::vector<bool>;

    result.clear();
    if (n == 0) {
        return;
    }

    if (5 * n < weights.size()) {
        result.resize(n);
        std::partial_sum(weights.begin(), weights.end(), weights.begin());
        TBoolVec mask(weights.size(), false);
        for (std::size_t i = 0, m = 0; 2 * i < 3 * n; ++i) {
            auto j = std::min(std::lower_bound(weights.begin(), weights.end(),
                                               doUniformSample(rng, 0.0, weights.back())) -
                                  weights.begin(),
                              static_cast<std::ptrdiff_t>(weights.size()) - 1);

            if (mask[j] == false) {
                mask[j] = true;
                result[m++] = j;
                if (m == n) {
                    return;
                }
            }
        }
        // This should happen rarely so the amortised cost is negligible.
        std::adjacent_difference(weights.begin(), weights.end(), weights.begin());
    }

    result.resize(weights.size());
    std::iota(result.begin(), result.end(), 0);

    // We won't sample any values assigned zero weight.
    result.erase(std::remove_if(result.begin(), result.end(),
                                [&](auto i) { return weights[i] == 0.0; }),
                 result.end());

    if (n >= result.size()) {
        return;
    }

    // Use the inverse density transform to generate exponential samples.
    for (auto i : result) {
        weights[i] = -CTools::fastLog(doUniformSample(rng, 0.0, 1.0)) / weights[i];
    }

    std::nth_element(
        result.begin(), result.begin() + n, result.end(),
        [&](auto lhs, auto rhs) { return weights[lhs] < weights[rhs]; });
    result.resize(n);
}

//! Implementation of multivariate normal sampling.
template<typename RNG>
bool doMultivariateNormalSample(RNG& rng,
                                const TDoubleVec& mean,
                                const TDoubleVecVec& covariance,
                                std::size_t n,
                                TDoubleVecVec& samples) {
    if (mean.size() != covariance.size()) {
        LOG_ERROR(<< "Incompatible mean and covariance: " << mean << ", " << covariance);
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
    LOG_TRACE(<< "Dimension = " << d);
    LOG_TRACE(<< "mean = " << mean);

    CDenseMatrix<double> C(d, d);
    for (std::size_t i = 0; i < d; ++i) {
        C(i, i) = covariance[i][i];
        if (covariance[i].size() < d - i) {
            LOG_ERROR(<< "Bad covariance matrix: " << covariance);
            return false;
        }
        for (std::size_t j = 0; j < i; ++j) {
            C(i, j) = covariance[i][j];
            C(j, i) = covariance[i][j];
        }
    }
    LOG_TRACE(<< "C =" << core_t::LINE_ENDING << ' ' << C);

    auto svd = C.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Get the singular values, these are the variances of the normals
    // to sample.
    const auto& S = svd.singularValues();
    const auto& U = svd.matrixU();
    TDoubleVec stddevs;
    stddevs.reserve(d);
    for (std::size_t i = 0; i < d; ++i) {
        stddevs.push_back(std::sqrt(std::max(S(i), 0.0)));
    }
    LOG_TRACE(<< "Singular values of C = " << S.transpose());
    LOG_TRACE(<< "stddevs = " << stddevs);
    LOG_TRACE(<< "U =" << core_t::LINE_ENDING << ' ' << U);
    LOG_TRACE(<< "V =" << core_t::LINE_ENDING << ' ' << svd.matrixV());

    {
        samples.resize(n, mean);
        CDenseVector<double> sample(d);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < d; ++j) {
                if (stddevs[j] == 0.0) {
                    sample(j) = 0.0;
                } else {
                    boost::random::normal_distribution<> normal(0.0, stddevs[j]);
                    sample(j) = normal(rng);
                }
            }
            sample = U * sample;
            for (std::size_t j = 0; j < d; ++j) {
                samples[i][j] += sample(j);
            }
        }
    }

    return true;
}

//! Implementation of multivariate normal sampling.
template<typename RNG, typename T, std::size_t N>
void doMultivariateNormalSample(RNG& rng,
                                const CVectorNx1<T, N>& mean,
                                const CSymmetricMatrixNxN<T, N>& covariance,
                                std::size_t n,
                                std::vector<CVectorNx1<T, N>>& samples) {
    using TDenseVector = typename SDenseVector<CVectorNx1<T, N>>::Type;

    samples.clear();
    if (n == 0) {
        return;
    }

    // See the other implementation for an explanation.

    auto svd = toDenseMatrix(covariance).jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

    // Get the singular values, these are the variances of the normals
    // to sample.
    const auto& S = svd.singularValues();
    const auto& U = svd.matrixU();
    T stddevs[N] = {};
    for (std::size_t i = 0; i < N; ++i) {
        stddevs[i] = std::sqrt(std::max(S(i), 0.0));
    }

    samples.resize(n, mean);
    TDenseVector sample(N);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
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

//! Implementation of distribution quantile sampling.
template<typename DISTRIBUTION>
void sampleQuantiles(const DISTRIBUTION& distribution, std::size_t n, TDoubleVec& result) {
    CTools::SIntervalExpectation expectation;
    double dq = 1.0 / static_cast<double>(n);

    double a = std::numeric_limits<double>::lowest();
    for (std::size_t i = 1; i < n; ++i) {
        double q = static_cast<double>(i) * dq;
        double b = boost::math::quantile(distribution, q);
        result.push_back(expectation(distribution, a, b));
        a = b;
    }
    double b = std::numeric_limits<double>::max();
    result.push_back(expectation(distribution, a, b));
}

const std::string RNG_TAG("a");
}

bool CSampling::staticsAcceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    // Note we require that we only ever do one persistence per process.

    do {
        const std::string& name = traverser.name();

        if (name == RNG_TAG) {
            std::string value(traverser.value());
            // See acceptPersistInserter
            std::replace(value.begin(), value.end(), '_', ' ');
            std::istringstream ss(value);
            core::CScopedFastLock scopedLock{defaultRngMutex};
            ss >> defaultRng;
        }
    } while (traverser.next());

    return true;
}

void CSampling::staticsAcceptPersistInserter(core::CStatePersistInserter& inserter) {
    // Note we require that we only ever do one persistence per process.

    std::ostringstream ss;
    {
        core::CScopedFastLock scopedLock{defaultRngMutex};
        ss << defaultRng;
    }
    std::string rng(ss.str());
    // These are space separated integers. We replace spaces or else
    // the JSON parser gets confused.
    std::replace(rng.begin(), rng.end(), ' ', '_');
    inserter.insertValue(RNG_TAG, rng);
}

void CSampling::seed() {
    core::CScopedFastLock scopedLock{defaultRngMutex};
    defaultRng.seed();
}

#define UNIFORM_SAMPLE(TYPE)                                                                  \
    TYPE CSampling::uniformSample(TYPE a, TYPE b) {                                           \
        core::CScopedFastLock scopedLock{defaultRngMutex};                                    \
        return doUniformSample(defaultRng, a, b);                                             \
    }                                                                                         \
    TYPE CSampling::uniformSample(CPRNG::CXorOShiro128Plus& rng, TYPE a, TYPE b) {            \
        return doUniformSample(rng, a, b);                                                    \
    }                                                                                         \
    TYPE CSampling::uniformSample(CPRNG::CXorShift1024Mult& rng, TYPE a, TYPE b) {            \
        return doUniformSample(rng, a, b);                                                    \
    }                                                                                         \
    void CSampling::uniformSample(TYPE a, TYPE b, std::size_t n, std::vector<TYPE>& result) { \
        core::CScopedFastLock scopedLock{defaultRngMutex};                                    \
        doUniformSample(defaultRng, a, b, n, result);                                         \
    }                                                                                         \
    void CSampling::uniformSample(CPRNG::CXorOShiro128Plus& rng, TYPE a, TYPE b,              \
                                  std::size_t n, std::vector<TYPE>& result) {                 \
        doUniformSample(rng, a, b, n, result);                                                \
    }                                                                                         \
    void CSampling::uniformSample(CPRNG::CXorShift1024Mult& rng, TYPE a, TYPE b,              \
                                  std::size_t n, std::vector<TYPE>& result) {                 \
        doUniformSample(rng, a, b, n, result);                                                \
    }
UNIFORM_SAMPLE(std::size_t)
UNIFORM_SAMPLE(std::ptrdiff_t)
UNIFORM_SAMPLE(double)
#undef UNIFORM_SAMPLE

double CSampling::normalSample(double mean, double variance) {
    core::CScopedFastLock scopedLock{defaultRngMutex};
    return doNormalSample(defaultRng, mean, variance);
}

double CSampling::normalSample(CPRNG::CXorOShiro128Plus& rng, double mean, double variance) {
    return doNormalSample(rng, mean, variance);
}

double CSampling::normalSample(CPRNG::CXorShift1024Mult& rng, double mean, double variance) {
    return doNormalSample(rng, mean, variance);
}

void CSampling::normalSample(double mean, double variance, std::size_t n, TDoubleVec& result) {
    core::CScopedFastLock scopedLock{defaultRngMutex};
    doNormalSample(defaultRng, mean, variance, n, result);
}

void CSampling::normalSample(CPRNG::CXorOShiro128Plus& rng,
                             double mean,
                             double variance,
                             std::size_t n,
                             TDoubleVec& result) {
    doNormalSample(rng, mean, variance, n, result);
}

void CSampling::normalSample(CPRNG::CXorShift1024Mult& rng,
                             double mean,
                             double variance,
                             std::size_t n,
                             TDoubleVec& result) {
    doNormalSample(rng, mean, variance, n, result);
}

void CSampling::poissonSample(double rate, std::size_t n, TSizeVec& result) {
    core::CScopedFastLock scopedLock{defaultRngMutex};
    doPoissonSample(defaultRng, rate, n, result);
}

void CSampling::poissonSample(CPRNG::CXorOShiro128Plus& rng, double rate, std::size_t n, TSizeVec& result) {
    doPoissonSample(rng, rate, n, result);
}

void CSampling::poissonSample(CPRNG::CXorShift1024Mult& rng, double rate, std::size_t n, TSizeVec& result) {
    doPoissonSample(rng, rate, n, result);
}

void CSampling::chiSquaredSample(double f, std::size_t n, TDoubleVec& result) {
    core::CScopedFastLock scopedLock{defaultRngMutex};
    doChiSquaredSample(defaultRng, f, n, result);
}

void CSampling::chiSquaredSample(CPRNG::CXorOShiro128Plus& rng,
                                 double f,
                                 std::size_t n,
                                 TDoubleVec& result) {
    doChiSquaredSample(rng, f, n, result);
}

void CSampling::chiSquaredSample(CPRNG::CXorShift1024Mult& rng,
                                 double f,
                                 std::size_t n,
                                 TDoubleVec& result) {
    doChiSquaredSample(rng, f, n, result);
}

bool CSampling::multivariateNormalSample(const TDoubleVec& mean,
                                         const TDoubleVecVec& covariance,
                                         std::size_t n,
                                         TDoubleVecVec& samples) {
    core::CScopedFastLock scopedLock{defaultRngMutex};
    return doMultivariateNormalSample(defaultRng, mean, covariance, n, samples);
}

bool CSampling::multivariateNormalSample(CPRNG::CXorOShiro128Plus& rng,
                                         const TDoubleVec& mean,
                                         const TDoubleVecVec& covariance,
                                         std::size_t n,
                                         TDoubleVecVec& samples) {
    return doMultivariateNormalSample(rng, mean, covariance, n, samples);
}

bool CSampling::multivariateNormalSample(CPRNG::CXorShift1024Mult& rng,
                                         const TDoubleVec& mean,
                                         const TDoubleVecVec& covariance,
                                         std::size_t n,
                                         TDoubleVecVec& samples) {
    return doMultivariateNormalSample(rng, mean, covariance, n, samples);
}

#define MULTIVARIATE_NORMAL_SAMPLE(N)                                                        \
    void CSampling::multivariateNormalSample(                                                \
        const CVectorNx1<double, N>& mean, const CSymmetricMatrixNxN<double, N>& covariance, \
        std::size_t n, std::vector<CVectorNx1<double, N>>& samples) {                        \
        core::CScopedFastLock scopedLock{defaultRngMutex};                                   \
        doMultivariateNormalSample(defaultRng, mean, covariance, n, samples);                \
    }                                                                                        \
    void CSampling::multivariateNormalSample(                                                \
        CPRNG::CXorOShiro128Plus& rng, const CVectorNx1<double, N>& mean,                    \
        const CSymmetricMatrixNxN<double, N>& covariance, std::size_t n,                     \
        std::vector<CVectorNx1<double, N>>& samples) {                                       \
        doMultivariateNormalSample(rng, mean, covariance, n, samples);                       \
    }                                                                                        \
    void CSampling::multivariateNormalSample(                                                \
        CPRNG::CXorShift1024Mult& rng, const CVectorNx1<double, N>& mean,                    \
        const CSymmetricMatrixNxN<double, N>& covariance, std::size_t n,                     \
        std::vector<CVectorNx1<double, N>>& samples) {                                       \
        doMultivariateNormalSample(rng, mean, covariance, n, samples);                       \
    }
MULTIVARIATE_NORMAL_SAMPLE(2)
MULTIVARIATE_NORMAL_SAMPLE(3)
MULTIVARIATE_NORMAL_SAMPLE(4)
MULTIVARIATE_NORMAL_SAMPLE(5)
#undef MULTIVARIATE_NORMAL_SAMPLE

std::size_t CSampling::categoricalSample(TDoubleVec& probabilities) {
    core::CScopedFastLock scopedLock{defaultRngMutex};
    return doCategoricalSample(defaultRng, probabilities);
}

std::size_t CSampling::categoricalSample(CPRNG::CXorOShiro128Plus& rng,
                                         TDoubleVec& probabilities) {
    return doCategoricalSample(rng, probabilities);
}

std::size_t CSampling::categoricalSample(CPRNG::CXorShift1024Mult& rng,
                                         TDoubleVec& probabilities) {
    return doCategoricalSample(rng, probabilities);
}

void CSampling::categoricalSampleWithReplacement(TDoubleVec& probabilities,
                                                 std::size_t n,
                                                 TSizeVec& result) {
    core::CScopedFastLock scopedLock{defaultRngMutex};
    doCategoricalSampleWithReplacement(defaultRng, probabilities, n, result);
}

void CSampling::categoricalSampleWithReplacement(CPRNG::CXorOShiro128Plus& rng,
                                                 TDoubleVec& probabilities,
                                                 std::size_t n,
                                                 TSizeVec& result) {
    doCategoricalSampleWithReplacement(rng, probabilities, n, result);
}

void CSampling::categoricalSampleWithReplacement(CPRNG::CXorShift1024Mult& rng,
                                                 TDoubleVec& probabilities,
                                                 std::size_t n,
                                                 TSizeVec& result) {
    doCategoricalSampleWithReplacement(rng, probabilities, n, result);
}

void CSampling::categoricalSampleWithoutReplacement(TDoubleVec& probabilities,
                                                    std::size_t n,
                                                    TSizeVec& result) {
    core::CScopedFastLock scopedLock{defaultRngMutex};
    doCategoricalSampleWithoutReplacement(defaultRng, probabilities, n, result);
}

void CSampling::categoricalSampleWithoutReplacement(CPRNG::CXorOShiro128Plus& rng,
                                                    TDoubleVec& probabilities,
                                                    std::size_t n,
                                                    TSizeVec& result) {
    doCategoricalSampleWithoutReplacement(rng, probabilities, n, result);
}

void CSampling::categoricalSampleWithoutReplacement(CPRNG::CXorShift1024Mult& rng,
                                                    TDoubleVec& probabilities,
                                                    std::size_t n,
                                                    TSizeVec& result) {
    doCategoricalSampleWithoutReplacement(rng, probabilities, n, result);
}

void CSampling::multinomialSampleFast(TDoubleVec& probabilities,
                                      std::size_t n,
                                      TSizeVec& sample,
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
    // the full distribution. The marginal distribution is multinomial
    // with number of trials n - Sum_{j < i}{ nj } and probabilities
    // p_j -> p_j / (1 - Sum_k{k < i}{ pk }). The marginal distribution
    // is just binomial with the same number of trials and probability
    // p_i (adjusted as above).
    //
    // In order to sample from the full distribution we sample each
    // marginal and then condition the next marginal on the values
    // we've sampled so far. It is straightforward to verify that
    // the probability distribution is then:
    //   P({n_i}) = Int{ Prod_i{ f(n_i | { n_j : j < i }) } }dn_i
    //            = Int{ f(n_i) }dn_i
    //
    // So the samples are distributed according to the full distribution
    // function as required.
    //
    // We sort the probabilities in descending order to make sampling
    // as efficient as possible: this means that the loop will terminate
    // as early as possible on average.

    if (!sorted) {
        std::sort(probabilities.begin(), probabilities.end(), std::greater<>());
    }

    {
        std::size_t r = n;
        double p = 1.0;
        std::size_t m = probabilities.size() - 1;
        core::CScopedFastLock scopedLock{defaultRngMutex};
        for (std::size_t i = 0; r > 0 && i < m; ++i) {
            boost::random::binomial_distribution<> binomial(static_cast<int>(r),
                                                            probabilities[i] / p);
            auto ni = static_cast<std::size_t>(binomial(defaultRng));
            sample.push_back(ni);
            r -= ni;
            p -= probabilities[i];
        }
        if (r > 0) {
            sample.push_back(r);
        }
    }
}

void CSampling::multinomialSampleStable(TDoubleVec probabilities, std::size_t n, TSizeVec& sample) {
    TSizeVec indices;
    indices.reserve(probabilities.size());
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        indices.push_back(i);
    }
    COrderings::simultaneousSortWith(std::greater<>(), probabilities, indices);

    sample.reserve(probabilities.size());

    multinomialSampleFast(probabilities, n, sample);

    sample.resize(probabilities.size(), 0);
    for (std::size_t i = 0; i < sample.size(); /**/) {
        std::size_t j = indices[i];
        if (i != j) {
            std::swap(sample[i], sample[j]);
            std::swap(indices[i], indices[j]);
        } else {
            ++i;
        }
    }
}

void CSampling::weightedSample(std::size_t n, const TDoubleVec& weights, TSizeVec& sampling) {
    // We sample each category according to its weight.
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

    using TUIntVec = std::vector<unsigned int>;
    using TDoubleSizePr = std::pair<double, std::size_t>;
    using TDoubleSizePrVec = std::vector<TDoubleSizePr>;

    LOG_TRACE(<< "Number samples = " << n);

    sampling.clear();

    double totalWeight = std::accumulate(weights.begin(), weights.end(), 0.0);

    n = std::max(static_cast<std::size_t>(totalWeight * static_cast<double>(n) + 0.5),
                 static_cast<std::size_t>(1u));

    LOG_TRACE(<< "totalWeight = " << totalWeight << ", n = " << n);

    TUIntVec choices;
    TDoubleVec remainders[] = {TDoubleVec(), TDoubleVec()};

    choices.reserve(weights.size());
    remainders[0].reserve(weights.size());
    remainders[1].reserve(weights.size());

    double totalRemainder = 0.0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        // We need to re-normalize so that the probabilities sum to one.
        double number = weights[i] * static_cast<double>(n) / totalWeight;
        choices.push_back((number - std::floor(number) < 0.5) ? 0 : 1);
        remainders[0].push_back(number - std::floor(number));
        remainders[1].push_back(number - std::ceil(number));
        totalRemainder += remainders[choices.back()].back();
    }

    // The remainder will be integral so checking against 0.5 avoids
    // floating point problems.

    if (std::fabs(totalRemainder) > 0.5) {
        LOG_TRACE(<< "ideal choice function = " << choices);

        TDoubleSizePrVec candidates;
        for (std::size_t i = 0; i < choices.size(); ++i) {
            if ((totalRemainder > 0.0 && choices[i] == 0) ||
                (totalRemainder < 0.0 && choices[i] == 1)) {
                candidates.emplace_back(-std::fabs(remainders[choices[i]][i]), i);
            }
        }
        std::sort(candidates.begin(), candidates.end());
        LOG_TRACE(<< "candidates = " << candidates);

        for (std::size_t i = 0;
             i < candidates.size() && std::fabs(totalRemainder) > 0.5; ++i) {
            std::size_t j = candidates[i].second;
            unsigned int choice = choices[j];
            choices[j] = (choice + 1u) % 2;
            totalRemainder += remainders[choices[j]][j] - remainders[choice][j];
        }
    }
    LOG_TRACE(<< "choice function = " << choices);

    sampling.reserve(weights.size());
    for (std::size_t i = 0; i < weights.size(); ++i) {
        double number = weights[i] * static_cast<double>(n) / totalWeight;

        sampling.push_back(static_cast<std::size_t>(
            choices[i] == 0 ? std::floor(number) : std::ceil(number)));
    }
}

void CSampling::normalSampleQuantiles(double mean, double variance, std::size_t n, TDoubleVec& result) {
    result.clear();
    if (n == 0) {
        return;
    }
    if (variance <= 0.0) {
        result.resize(n, mean);
        return;
    }

    try {
        boost::math::normal normal(mean, std::sqrt(variance));
        sampleQuantiles(normal, n, result);
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to sample normal quantiles: " << e.what()
                  << ", mean = " << mean << ", variance = " << variance);
        result.clear();
    }
}

void CSampling::gammaSampleQuantiles(double shape, double rate, std::size_t n, TDoubleVec& result) {
    result.clear();
    if (n == 0) {
        return;
    }

    try {
        boost::math::gamma_distribution<> gamma(shape, 1.0 / rate);
        sampleQuantiles(gamma, n, result);
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to sample normal quantiles: " << e.what()
                  << ", shape = " << shape << ", rate = " << rate);
        result.clear();
    }
}

void CSampling::sobolSequenceSample(std::size_t dim, std::size_t n, TDoubleVecVec& samples) {
    samples.clear();
    if (n == 0 || dim == 0) {
        return;
    }

    try {
        using TSobolGenerator =
            boost::variate_generator<boost::random::sobol&, boost::uniform_01<double>>;
        boost::random::sobol engine(dim);
        TSobolGenerator gen(engine, boost::uniform_01<double>());
        samples.resize(n, TDoubleVec(dim));
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < dim; ++j) {
                samples[i][j] = gen();
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to sample Sobol sequence: " << e.what()
                  << ", dim = " << dim << ", n = " << n);
        samples.clear();
    }
}

CSampling::CScopeMockRandomNumberGenerator::CScopeMockRandomNumberGenerator() {
    defaultRng.mock();
}

CSampling::CScopeMockRandomNumberGenerator::~CScopeMockRandomNumberGenerator() {
    defaultRng.unmock();
}
}
}
}
