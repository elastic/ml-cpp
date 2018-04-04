/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CPriorTest.h"

#include <core/CLogger.h>

#include <maths/CBasicStatistics.h>
#include <maths/CCompositeFunctions.h>
#include <maths/CGammaRateConjugate.h>
#include <maths/CLogNormalMeanPrecConjugate.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/CPrior.h>
#include <maths/CPriorDetail.h>
#include <maths/CTools.h>

#include <test/CRandomNumbers.h>

#include "TestUtils.h"

#include <boost/math/distributions/normal.hpp>

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
    CMinusLogLikelihood(const maths::CPrior& prior)
        : m_Prior(&prior), m_WeightStyle(1, maths_t::E_SampleCountWeight), m_X(1, 0.0), m_Weight(1, TDoubleVec(1, 1.0)) {}

    bool operator()(const double& x, double& result) const {
        m_X[0] = x;
        maths_t::EFloatingPointErrorStatus status = m_Prior->jointLogMarginalLikelihood(m_WeightStyle, m_X, m_Weight, result);
        result = -result;
        return !(status & maths_t::E_FpFailed);
    }

private:
    const maths::CPrior* m_Prior;
    maths_t::TWeightStyleVec m_WeightStyle;
    mutable TDoubleVec m_X;
    TDoubleVecVec m_Weight;
};
}

void CPriorTest::testExpectation() {
    LOG_DEBUG("+-------------------------------+");
    LOG_DEBUG("|  CPriorTest::testExpectation  |");
    LOG_DEBUG("+-------------------------------+");

    using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using CNormalMeanPrecConjugate = CPriorTestInterfaceMixin<maths::CNormalMeanPrecConjugate>;

    test::CRandomNumbers rng;

    CNormalMeanPrecConjugate prior(maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData));

    TDoubleVec samples;
    rng.generateNormalSamples(1.0, 1.5, 10000u, samples);

    TMeanVarAccumulator moments;
    moments.add(samples);
    prior.addSamples(samples);

    double trueMean = maths::CBasicStatistics::mean(moments);
    LOG_DEBUG("true mean = " << trueMean);
    for (std::size_t n = 1; n < 10; ++n) {
        double mean;
        CPPUNIT_ASSERT(prior.expectation(CX(), n, mean));
        LOG_DEBUG("n = " << n << ", mean = " << mean << ", error = " << std::fabs(mean - trueMean));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(trueMean, mean, 1e-10);
    }

    double varianceErrors[] = {1.4, 0.1, 0.05, 0.01, 0.005, 0.0008, 0.0008, 0.0007, 0.0005};
    double trueVariance = maths::CBasicStatistics::variance(moments);
    LOG_DEBUG("true variance = " << trueVariance);
    for (std::size_t n = 1; n < 10; ++n) {
        double variance;
        CPPUNIT_ASSERT(prior.expectation(CVariance(prior.mean()), n, variance));
        LOG_DEBUG("n = " << n << ", variance = " << variance << ", error = " << std::fabs(variance - trueVariance));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(trueVariance, variance, varianceErrors[n - 1]);
    }

    double entropyErrors[] = {0.5, 0.05, 0.01, 0.005, 0.001, 0.0003, 0.0003, 0.0002, 0.0002};
    boost::math::normal_distribution<> normal(trueMean, std::sqrt(trueVariance));
    double trueEntropy = maths::CTools::differentialEntropy(normal);
    LOG_DEBUG("true differential entropy = " << trueEntropy);
    for (std::size_t n = 1; n < 10; ++n) {
        double entropy;
        CPPUNIT_ASSERT(prior.expectation(CMinusLogLikelihood(prior), n, entropy));
        LOG_DEBUG("n = " << n << ", differential entropy = " << entropy << ", error = " << std::fabs(entropy - trueEntropy));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(trueEntropy, entropy, entropyErrors[n - 1]);
    }
}

CppUnit::Test* CPriorTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CPriorTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CPriorTest>("CPriorTest::testExpectation", &CPriorTest::testExpectation));

    return suiteOfTests;
}
