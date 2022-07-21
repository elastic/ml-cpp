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

#include <core/CLogger.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CCompositeFunctions.h>
#include <maths/common/CGammaRateConjugate.h>
#include <maths/common/CLogNormalMeanPrecConjugate.h>
#include <maths/common/CNormalMeanPrecConjugate.h>
#include <maths/common/COrderingsSimultaneousSort.h>
#include <maths/common/CPrior.h>
#include <maths/common/CPriorDetail.h>
#include <maths/common/CTools.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include "TestUtils.h"

#include <boost/math/distributions/normal.hpp>
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CPriorTest)

using namespace ml;
using namespace handy_typedefs;

namespace {

using TDoubleVec = std::vector<double>;

class CX {
public:
    bool operator()(const double& x, double& result) const {
        result = x;
        return true;
    }
};

class CVariance {
public:
    CVariance(const double mean) : m_Mean(mean) {}

    bool operator()(const double& x, double& result) const {
        result = (x - m_Mean) * (x - m_Mean);
        return true;
    }

private:
    double m_Mean;
};

class CMinusLogLikelihood {
public:
    using TDoubleVecVec = std::vector<TDoubleVec>;

public:
    CMinusLogLikelihood(const maths::common::CPrior& prior)
        : m_Prior(&prior), m_X(1, 0.0) {}

    bool operator()(const double& x, double& result) const {
        m_X[0] = x;
        maths_t::EFloatingPointErrorStatus status = m_Prior->jointLogMarginalLikelihood(
            m_X, maths_t::CUnitWeights::SINGLE_UNIT, result);
        result = -result;
        return !(status & maths_t::E_FpFailed);
    }

private:
    const maths::common::CPrior* m_Prior;
    mutable TDoubleVec m_X;
};
}

BOOST_AUTO_TEST_CASE(testExpectation) {
    using TMeanVarAccumulator =
        maths::common::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

    test::CRandomNumbers rng;

    maths::common::CNormalMeanPrecConjugate prior(
        maths::common::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData));

    TDoubleVec samples;
    rng.generateNormalSamples(1.0, 1.5, 10000u, samples);

    TMeanVarAccumulator moments;
    moments.add(samples);
    prior.addSamples(samples, maths_t::TDoubleWeightsAry1Vec(
                                  samples.size(), maths_t::CUnitWeights::UNIT));

    double trueMean = maths::common::CBasicStatistics::mean(moments);
    LOG_DEBUG(<< "true mean = " << trueMean);
    for (std::size_t n = 1; n < 10; ++n) {
        double mean;
        BOOST_TEST_REQUIRE(prior.expectation(CX(), n, mean));
        LOG_DEBUG(<< "n = " << n << ", mean = " << mean
                  << ", error = " << std::fabs(mean - trueMean));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(trueMean, mean, 1e-7);
    }

    double varianceErrors[] = {1.4,    0.1,    0.05,   0.01,  0.005,
                               0.0008, 0.0008, 0.0007, 0.0005};
    double trueVariance = maths::common::CBasicStatistics::variance(moments);
    LOG_DEBUG(<< "true variance = " << trueVariance);
    for (std::size_t n = 1; n < 10; ++n) {
        double variance;
        BOOST_TEST_REQUIRE(prior.expectation(CVariance(prior.mean()), n, variance));
        LOG_DEBUG(<< "n = " << n << ", variance = " << variance
                  << ", error = " << std::fabs(variance - trueVariance));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(trueVariance, variance, varianceErrors[n - 1]);
    }

    double entropyErrors[] = {0.5,    0.05,   0.01,   0.005, 0.001,
                              0.0003, 0.0003, 0.0002, 0.0002};
    boost::math::normal_distribution<> normal(trueMean, std::sqrt(trueVariance));
    double trueEntropy = maths::common::CTools::differentialEntropy(normal);
    LOG_DEBUG(<< "true differential entropy = " << trueEntropy);
    for (std::size_t n = 1; n < 10; ++n) {
        double entropy;
        BOOST_TEST_REQUIRE(prior.expectation(CMinusLogLikelihood(prior), n, entropy));
        LOG_DEBUG(<< "n = " << n << ", differential entropy = " << entropy
                  << ", error = " << std::fabs(entropy - trueEntropy));
        BOOST_REQUIRE_CLOSE_ABSOLUTE(trueEntropy, entropy, entropyErrors[n - 1]);
    }
}

BOOST_AUTO_TEST_SUITE_END()
