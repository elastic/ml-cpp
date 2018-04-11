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

#include <maths/CProbabilityCalibrator.h>

#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

#include <maths/CQDigest.h>
#include <maths/CTools.h>

#include <boost/bind.hpp>
#include <boost/math/distributions/beta.hpp>

#include <cmath>
#include <string>

namespace ml {
namespace maths {

namespace {

const uint64_t QUANTILE_SIZE = 200u;
const double DISCRETIZATION_FACTOR = 100.0;

//! Convert a probability to a positive integer.
uint32_t discreteProbability(const double probability) {
    return static_cast<uint32_t>(DISCRETIZATION_FACTOR * -std::log(probability) + 0.5);
}

//! Convert a discrete probability integer into the
//! approximate probability which generated it.
double rawProbability(const uint32_t& discreteProbability) {
    return std::exp(-static_cast<double>(discreteProbability) / DISCRETIZATION_FACTOR);
}

const std::string STYLE_TAG("a");
const std::string CUTOFF_PROBABILITY_TAG("b");
const std::string DISCRETE_PROBABILITY_QUANTILE_TAG("c");
const std::string EMPTY_STRING;
}

CProbabilityCalibrator::CProbabilityCalibrator(EStyle style, double cutoffProbability)
    : m_Style(style), m_CutoffProbability(cutoffProbability), m_DiscreteProbabilityQuantiles(new CQDigest(QUANTILE_SIZE)) {
    if (!(m_CutoffProbability >= 0.0 && m_CutoffProbability <= 1.0)) {
        LOG_ERROR(<< "Invalid cutoff probability " << m_CutoffProbability);
        CTools::truncate(m_CutoffProbability, 0.0, 1.0);
    }
}

void CProbabilityCalibrator::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(STYLE_TAG, static_cast<int>(m_Style));
    inserter.insertValue(CUTOFF_PROBABILITY_TAG, m_CutoffProbability);
    inserter.insertLevel(DISCRETE_PROBABILITY_QUANTILE_TAG,
                         boost::bind(&CQDigest::acceptPersistInserter, m_DiscreteProbabilityQuantiles.get(), _1));
}

bool CProbabilityCalibrator::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        if (name == STYLE_TAG) {
            int style;
            if (core::CStringUtils::stringToType(traverser.value(), style) == false) {
                LOG_ERROR(<< "Invalid style in " << traverser.value());
                return false;
            }
            m_Style = static_cast<EStyle>(style);
        } else if (name == CUTOFF_PROBABILITY_TAG) {
            if (core::CStringUtils::stringToType(traverser.value(), m_CutoffProbability) == false) {
                LOG_ERROR(<< "Invalid cutoff in " << traverser.value());
                return false;
            }
        } else if (name == DISCRETE_PROBABILITY_QUANTILE_TAG) {
            if (traverser.traverseSubLevel(boost::bind(&CQDigest::acceptRestoreTraverser, m_DiscreteProbabilityQuantiles.get(), _1)) ==
                false) {
                LOG_ERROR(<< "Invalid quantiles in " << traverser.value());
                return false;
            }
        }
    } while (traverser.next());

    return true;
}

void CProbabilityCalibrator::add(double probability) {
    uint32_t pDiscrete = discreteProbability(probability);
    m_DiscreteProbabilityQuantiles->add(pDiscrete);
}

double CProbabilityCalibrator::calibrate(double probability) const {
    LOG_TRACE(<< "Calibrating " << probability);

    // The basic idea is to calibrate the probability to the historical
    // fractions by noting that the number of probabilities less than
    // p should tend to p * n in a sample set of n. We perform different
    // degrees of calibration depending on the assumptions we are prepared
    // to make about the probability set.

    uint32_t pDiscrete = discreteProbability(probability);
    m_DiscreteProbabilityQuantiles->superlevelSetInfimum(pDiscrete, pDiscrete);
    double Fl;
    double Fu;
    m_DiscreteProbabilityQuantiles->cdf(pDiscrete, 0.0, Fl, Fu);
    LOG_TRACE(<< "1 - F = [" << 1.0 - Fu << "," << 1.0 - Fl << "]");

    // We need to account for the fact that we have a finite sample
    // size when computing the fraction f. We use a Bayesian approach.
    // The count of events with fraction f is binomially distributed.
    // Therefore, if we assume a non-informative beta prior we can
    // get a posterior distribution for the true fraction simply by
    // setting alpha = alpha + f * n and beta = beta + n * (1 - f).
    // We will use a high confidence (ninetieth percentile) lower
    // bound for the probability.

    Fu = CTools::truncate(Fu, 0.0, 1.0);
    double n = static_cast<double>(m_DiscreteProbabilityQuantiles->n());
    double a = n * Fu + 1.0;
    double b = n * (1.0 - Fu) + 1.0;
    boost::math::beta_distribution<> beta(a, b);
    Fu = boost::math::quantile(beta, 0.75);
    LOG_TRACE(<< "(1 - F)(25) = " << 1.0 - Fu);

    // For partial calibration we use a cutoff probability (based
    // on the value of f at the cutoff) for, which we assume that
    // we can't accurately determine f for smaller values of f, by
    // observing historical probabilities. This would be the choice
    // when we expect there may systematic error in f, for small f,
    // and we don't and we don't want to calibrate this away since
    // it could  stop us detecting genuine anomalies. We choose the
    // scale to apply to small probabilities by ensuring that the
    // transformation is continuous across the cutoff probability.
    //
    // Note also that we choose never to decrease probability. This
    // would also be important when we do not expect probabilities
    // to match historic fractions as would be the case if the
    // quantities we are modeling aren't truly random. A good
    // counter example would be the series 1, 1, 1, 1, ..., 1.
    // Clearly it would be wrong to say that the probability of 1
    // is anything other than 1, but the historical fractions of
    // probabilities will be very far from what we'd expect, i.e.
    // we won't see any probabilities less than 1.

    switch (m_Style) {
    case E_PartialCalibration:
        if (Fu > m_CutoffProbability) {
            uint32_t pThreshold;
            m_DiscreteProbabilityQuantiles->quantileSublevelSetSupremum(m_CutoffProbability, pThreshold);
            m_DiscreteProbabilityQuantiles->cdf(pThreshold, 0.0, Fl, Fu);
            a = n * Fu + 1.0;
            b = n * (1.0 - Fu) + 1.0;
            beta = boost::math::beta_distribution<>(a, b);
            Fu = boost::math::quantile(beta, 0.75);
            double scale = std::max((1.0 - Fu) / rawProbability(pThreshold), 1.0);
            LOG_TRACE(<< "scale = " << scale << ", 1 - F = " << 1.0 - Fu << ", p = " << rawProbability(pThreshold));
            return probability * scale;
        }
        return std::max(probability, 1.0 - Fu);

    case E_FullCalibration:
        return 1.0 - Fu;
    }

    LOG_ABORT(<< "Unexpected style " << m_Style);
}
}
}
