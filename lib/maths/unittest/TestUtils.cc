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

#include "TestUtils.h"

#include <maths/CEqualWithTolerance.h>
#include <maths/CIntegration.h>
#include <maths/CPrior.h>
#include <maths/CSolvers.h>
#include <maths/CTools.h>

#include <cmath>

namespace ml {
using namespace maths;
using namespace handy_typedefs;

namespace {
const core_t::TTime HALF_HOUR{core::constants::HOUR / 2};
const core_t::TTime DAY{core::constants::DAY};
const core_t::TTime WEEK{core::constants::WEEK};

//! \brief Computes the c.d.f. of the prior minus the target supplied
//! to its constructor at specific locations.
class CCdf : public std::unary_function<double, double> {
public:
    enum EStyle { E_Lower, E_Upper, E_GeometricMean };

public:
    CCdf(EStyle style, const CPrior& prior, double target)
        : m_Style(style), m_Prior(&prior), m_Target(target), m_X(1u) {}

    double operator()(double x) const {
        double lowerBound, upperBound;

        m_X[0] = x;
        if (!m_Prior->minusLogJointCdf(m_X, maths_t::CUnitWeights::SINGLE_UNIT,
                                       lowerBound, upperBound)) {
            // We have no choice but to throw because this is
            // invoked inside a boost root finding function.

            LOG_ERROR(<< "Failed to evaluate c.d.f. at " << x);
            throw std::runtime_error("Failed to evaluate c.d.f.");
        }

        switch (m_Style) {
        case E_Lower:
            return std::exp(-lowerBound) - m_Target;
        case E_Upper:
            return std::exp(-upperBound) - m_Target;
        case E_GeometricMean:
            return std::exp(-(lowerBound + upperBound) / 2.0) - m_Target;
        }
        return std::exp(-(lowerBound + upperBound) / 2.0) - m_Target;
    }

private:
    EStyle m_Style;
    const CPrior* m_Prior;
    double m_Target;
    mutable TDouble1Vec m_X;
};

//! Set \p result to \p x.
bool identity(double x, double& result) {
    result = x;
    return true;
}

//! Computes the residual from a specified mean.
class CResidual {
public:
    using result_type = double;

public:
    CResidual(double mean) : m_Mean(mean) {}

    bool operator()(double x, double& result) const {
        result = (x - m_Mean) * (x - m_Mean);
        return true;
    }

private:
    double m_Mean;
};
}

CPriorTestInterface::CPriorTestInterface(CPrior& prior) : m_Prior(&prior) {
}

void CPriorTestInterface::addSamples(const TDouble1Vec& samples) {
    maths_t::TDoubleWeightsAry1Vec weights(samples.size(), TWeights::UNIT);
    m_Prior->addSamples(samples, weights);
}

maths_t::EFloatingPointErrorStatus
CPriorTestInterface::jointLogMarginalLikelihood(const TDouble1Vec& samples,
                                                double& result) const {
    maths_t::TDoubleWeightsAry1Vec weights(samples.size(), TWeights::UNIT);
    return m_Prior->jointLogMarginalLikelihood(samples, weights, result);
}

bool CPriorTestInterface::minusLogJointCdf(const TDouble1Vec& samples,
                                           double& lowerBound,
                                           double& upperBound) const {
    maths_t::TDoubleWeightsAry1Vec weights(samples.size(), TWeights::UNIT);
    return m_Prior->minusLogJointCdf(samples, weights, lowerBound, upperBound);
}

bool CPriorTestInterface::minusLogJointCdfComplement(const TDouble1Vec& samples,
                                                     double& lowerBound,
                                                     double& upperBound) const {
    maths_t::TDoubleWeightsAry1Vec weights(samples.size(), TWeights::UNIT);
    return m_Prior->minusLogJointCdfComplement(samples, weights, lowerBound, upperBound);
}

bool CPriorTestInterface::probabilityOfLessLikelySamples(maths_t::EProbabilityCalculation calculation,
                                                         const TDouble1Vec& samples,
                                                         double& lowerBound,
                                                         double& upperBound) const {
    maths_t::TDoubleWeightsAry1Vec weights(samples.size(), TWeights::UNIT);
    maths_t::ETail tail;
    return m_Prior->probabilityOfLessLikelySamples(calculation, samples, weights,
                                                   lowerBound, upperBound, tail);
}

bool CPriorTestInterface::anomalyScore(maths_t::EProbabilityCalculation calculation,
                                       const TDouble1Vec& samples,
                                       double& result) const {

    result = 0.0;

    double lowerBound, upperBound;
    maths_t::ETail tail;
    if (!m_Prior->probabilityOfLessLikelySamples(calculation, samples,
                                                 maths_t::CUnitWeights::SINGLE_UNIT,
                                                 lowerBound, upperBound, tail)) {
        LOG_ERROR(<< "Failed computing probability of less likely samples");
        return false;
    }

    result = CTools::deviation((lowerBound + upperBound) / 2.0);

    return true;
}

bool CPriorTestInterface::marginalLikelihoodQuantileForTest(double percentage,
                                                            double eps,
                                                            double& result) const {

    result = 0.0;

    percentage /= 100.0;
    double step = 1.0;
    TDoubleDoublePr bracket(0.0, step);

    try {
        CCdf cdf(percentage < 0.5 ? CCdf::E_Lower : CCdf::E_Upper, *m_Prior, percentage);
        TDoubleDoublePr fBracket(cdf(bracket.first), cdf(bracket.second));

        std::size_t maxIterations = 100u;
        for (/**/; fBracket.first * fBracket.second > 0.0 && maxIterations > 0; --maxIterations) {
            step *= 2.0;
            if (fBracket.first > 0.0) {
                bracket.first -= step;
                fBracket.first = cdf(bracket.first);
            } else if (fBracket.second < 0.0) {
                bracket.second += step;
                fBracket.second = cdf(bracket.second);
            }
        }

        CEqualWithTolerance<double> equal(CToleranceTypes::E_AbsoluteTolerance, 2.0 * eps);

        CSolvers::solve(bracket.first, bracket.second, fBracket.first,
                        fBracket.second, cdf, maxIterations, equal, result);
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to compute quantile: " << e.what()
                  << ", quantile = " << percentage);
        return false;
    }

    return true;
}

bool CPriorTestInterface::marginalLikelihoodMeanForTest(double& result) const {

    using TMarginalLikelihood =
        CCompositeFunctions::CExp<const CPrior::CLogMarginalLikelihood&>;
    using TFunctionTimesMarginalLikelihood =
        CCompositeFunctions::CProduct<bool (*)(double, double&), TMarginalLikelihood>;

    const double eps = 1e-3;
    unsigned int steps = 100u;

    result = 0.0;

    double a, b;
    if (!this->marginalLikelihoodQuantileForTest(0.001, eps, a) ||
        !this->marginalLikelihoodQuantileForTest(99.999, eps, b)) {
        LOG_ERROR(<< "Unable to compute mean likelihood");
        return false;
    }

    if (m_Prior->dataType() == maths_t::E_IntegerData) {
        b = std::ceil(b);
        a = std::floor(a);
        steps = static_cast<unsigned int>(b - a) + 1;
    }

    CPrior::CLogMarginalLikelihood logLikelihood(*m_Prior);
    TFunctionTimesMarginalLikelihood xTimesLikelihood(
        identity, TMarginalLikelihood(logLikelihood));

    double x = a;
    double step = (b - a) / static_cast<double>(steps);

    for (unsigned int i = 0; i < steps; ++i, x += step) {
        double integral;
        if (!CIntegration::gaussLegendre<CIntegration::OrderThree>(
                xTimesLikelihood, x, x + step, integral)) {
            return false;
        }
        result += integral;
    }

    return true;
}

bool CPriorTestInterface::marginalLikelihoodVarianceForTest(double& result) const {

    using TMarginalLikelihood =
        CCompositeFunctions::CExp<const CPrior::CLogMarginalLikelihood&>;
    using TResidualTimesMarginalLikelihood =
        CCompositeFunctions::CProduct<CResidual, TMarginalLikelihood>;

    const double eps = 1e-3;
    unsigned int steps = 100u;

    result = 0.0;

    double a, b;
    if (!this->marginalLikelihoodQuantileForTest(0.001, eps, a) ||
        !this->marginalLikelihoodQuantileForTest(99.999, eps, b)) {
        LOG_ERROR(<< "Unable to compute mean likelihood");
        return false;
    }

    if (m_Prior->dataType() == maths_t::E_IntegerData) {
        b = std::ceil(b);
        a = std::floor(a);
        steps = static_cast<unsigned int>(b - a) + 1;
    }

    CPrior::CLogMarginalLikelihood logLikelihood(*m_Prior);
    TResidualTimesMarginalLikelihood residualTimesLikelihood(
        CResidual(m_Prior->marginalLikelihoodMean()), TMarginalLikelihood(logLikelihood));

    double x = a;
    double step = (b - a) / static_cast<double>(steps);
    for (unsigned int i = 0; i < steps; ++i, x += step) {
        double integral;
        if (!CIntegration::gaussLegendre<CIntegration::OrderThree>(
                residualTimesLikelihood, x, x + step, integral)) {
            return false;
        }
        result += integral;
    }

    return true;
}

double constant(core_t::TTime /*time*/) {
    return 4.0;
}

double ramp(core_t::TTime time) {
    return 0.1 * static_cast<double>(time) / static_cast<double>(WEEK);
}

double markov(core_t::TTime time) {
    static double state{0.2};
    if (time % WEEK == 0) {
        core::CHashing::CMurmurHash2BT<core_t::TTime> hasher;
        state = 2.0 * static_cast<double>(hasher(time)) /
                static_cast<double>(std::numeric_limits<std::size_t>::max());
    }
    return state;
}

double smoothDaily(core_t::TTime time) {
    return std::sin(boost::math::double_constants::two_pi *
                    static_cast<double>(time) / static_cast<double>(DAY));
}

double smoothWeekly(core_t::TTime time) {
    return std::sin(boost::math::double_constants::two_pi *
                    static_cast<double>(time) / static_cast<double>(WEEK));
}

double spikeyDaily(core_t::TTime time) {
    double pattern[]{1.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1,
                     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1,
                     0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0,
                     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                     0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1};
    return pattern[(time % DAY) / HALF_HOUR];
}

double spikeyWeekly(core_t::TTime time) {
    double pattern[]{
        1.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1,
        0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2,
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1,
        0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        1.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2,
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1,
        0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        1.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1,
        0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.2,
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1,
        0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1,
        0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1};
    return pattern[(time % WEEK) / HALF_HOUR];
}

double weekends(core_t::TTime time) {
    double amplitude[] = {1.0, 0.9, 0.8, 0.9, 1.1, 0.2, 0.05};
    return amplitude[(time % WEEK) / DAY] *
           std::sin(boost::math::double_constants::two_pi *
                    static_cast<double>(time) / static_cast<double>(DAY));
}

double scale(double scale, core_t::TTime time, TGenerator generator) {
    return generator(static_cast<core_t::TTime>(scale * static_cast<double>(time)));
}
}
