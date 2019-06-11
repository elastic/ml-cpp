/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/COneOfNPrior.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStringUtils.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CMathsFuncs.h>
#include <maths/CMathsFuncsForMatrixAndVectorTypes.h>
#include <maths/CPriorStateSerialiser.h>
#include <maths/CRestoreParams.h>
#include <maths/CSampling.h>
#include <maths/CTools.h>
#include <maths/Constants.h>

#include <boost/numeric/conversion/bounds.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <utility>

namespace ml {
namespace maths {

namespace {

using TBool5Vec = core::CSmallVector<bool, 5>;
using TDouble5Vec = core::CSmallVector<double, 5>;
using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMaxAccumulator = CBasicStatistics::SMax<double>::TAccumulator;

//! Compute the log of \p n.
double logn(std::size_t n) {
    static const double LOG_N[] = {0.0, std::log(2.0), std::log(3.0),
                                   std::log(4.0), std::log(5.0)};
    return n < boost::size(LOG_N) ? LOG_N[n - 1] : std::log(static_cast<double>(n));
}

const double DERATE = 0.99999;
const double MINUS_INF = DERATE * boost::numeric::bounds<double>::lowest();
const double INF = DERATE * boost::numeric::bounds<double>::highest();
const double MAXIMUM_LOG_BAYES_FACTOR = std::log(1e8);
const double MAXIMUM_UNPENALISED_RELATIVE_VARIANCE_ERROR = 9.0;
const double MINIMUM_SIGNIFICANT_WEIGHT = 0.01;
const double MAXIMUM_RELATIVE_ERROR = 1e-3;
const double LOG_MAXIMUM_RELATIVE_ERROR = std::log(MAXIMUM_RELATIVE_ERROR);

const std::string VERSION_7_1_TAG("7.1");

// Version 7.1
const std::string MODEL_7_1_TAG("a");
const std::string SAMPLE_MOMENTS_7_1_TAG("b");
const std::string NUMBER_SAMPLES_7_1_TAG("c");
const std::string DECAY_RATE_7_1_TAG("d");

const std::string READABLE_MODEL_7_1_TAG("model");
const std::string READABLE_SAMPLE_MOMENTS_7_1_TAG("sample_moments");
const std::string READABLE_NUMBER_SAMPLES_7_1_TAG("number_samples");
const std::string READABLE_DECAY_RATE_7_1_TAG("decay_rate");

// Version < 7.1
const std::string MODEL_OLD_TAG("a");
const std::string NUMBER_SAMPLES_OLD_TAG("b");
const std::string DECAY_RATE_OLD_TAG("e");

// Nested tags
const std::string READABLE_WEIGHT_TAG("weight");
const std::string READABLE_PRIOR_TAG("prior");

const std::string WEIGHT_TAG("a");
const std::string PRIOR_TAG("b");

const std::string EMPTY_STRING;

//! Persist state for a models by passing information to \p inserter.
void modelAcceptPersistInserter(const CModelWeight& weight,
                                const CPrior& prior,
                                core::CStatePersistInserter& inserter) {
    const bool readableTags{inserter.readableTags()};
    inserter.insertLevel(
        readableTags ? READABLE_WEIGHT_TAG : WEIGHT_TAG, std::bind(&CModelWeight::acceptPersistInserter, &weight, std::placeholders::_1));
    inserter.insertLevel(readableTags ? READABLE_PRIOR_TAG : PRIOR_TAG, std::bind<void>(CPriorStateSerialiser(),
                                                      std::cref(prior), std::placeholders::_1));
}
}

//////// COneOfNPrior Implementation ////////

COneOfNPrior::COneOfNPrior(const TPriorPtrVec& models, maths_t::EDataType dataType, double decayRate)
    : CPrior(dataType, decayRate) {
    if (models.empty()) {
        LOG_ERROR(<< "Can't initialize one-of-n with no models!");
        return;
    }

    // Create a new model vector using uniform weights.
    m_Models.reserve(models.size());
    CModelWeight weight(1.0);
    for (const auto& model : models) {
        m_Models.emplace_back(weight, TPriorPtr(model->clone()));
    }
}

COneOfNPrior::COneOfNPrior(const TDoublePriorPtrPrVec& models,
                           maths_t::EDataType dataType,
                           double decayRate /*= 0.0*/)
    : CPrior(dataType, decayRate) {
    if (models.empty()) {
        LOG_ERROR(<< "Can't initialize mixed model with no models!");
        return;
    }

    CScopeCanonicalizeWeights<TPriorPtr> canonicalize(m_Models);

    // Create a new model vector using the specified models and their associated weights.
    m_Models.reserve(models.size());
    for (const auto& model : models) {
        m_Models.emplace_back(CModelWeight(model.first), TPriorPtr(model.second->clone()));
    }
}

COneOfNPrior::COneOfNPrior(const SDistributionRestoreParams& params,
                           core::CStateRestoreTraverser& traverser)
    : CPrior(params.s_DataType, params.s_DecayRate) {
    traverser.traverseSubLevel(std::bind(&COneOfNPrior::acceptRestoreTraverser, this,
                                         std::cref(params), std::placeholders::_1));
}

bool COneOfNPrior::acceptRestoreTraverser(const SDistributionRestoreParams& params,
                                          core::CStateRestoreTraverser& traverser) {
    if (traverser.name() == VERSION_7_1_TAG) {
        while (traverser.next()) {
            const std::string& name = traverser.name();
            RESTORE_SETUP_TEARDOWN(DECAY_RATE_7_1_TAG, double decayRate,
                                   core::CStringUtils::stringToType(traverser.value(), decayRate),
                                   this->decayRate(decayRate))
            RESTORE(MODEL_7_1_TAG, traverser.traverseSubLevel(std::bind(
                                       &COneOfNPrior::modelAcceptRestoreTraverser, this,
                                       std::cref(params), std::placeholders::_1)))
            RESTORE(SAMPLE_MOMENTS_7_1_TAG,
                    m_SampleMoments.fromDelimited(traverser.value()))
            RESTORE_SETUP_TEARDOWN(
                NUMBER_SAMPLES_7_1_TAG, double numberSamples,
                core::CStringUtils::stringToType(traverser.value(), numberSamples),
                this->numberSamples(numberSamples))
        }
    } else {
        do {
            const std::string& name = traverser.name();
            RESTORE_SETUP_TEARDOWN(DECAY_RATE_OLD_TAG, double decayRate,
                                   core::CStringUtils::stringToType(traverser.value(), decayRate),
                                   this->decayRate(decayRate))
            RESTORE(MODEL_OLD_TAG, traverser.traverseSubLevel(std::bind(
                                       &COneOfNPrior::modelAcceptRestoreTraverser, this,
                                       std::cref(params), std::placeholders::_1)))
            RESTORE_SETUP_TEARDOWN(
                NUMBER_SAMPLES_OLD_TAG, double numberSamples,
                core::CStringUtils::stringToType(traverser.value(), numberSamples),
                this->numberSamples(numberSamples))
        } while (traverser.next());

        if (!this->isNonInformative()) {
            double variance{INF};
            for (const auto& model : m_Models) {
                variance = std::min(variance, model.second->marginalLikelihoodVariance());
            }
            m_SampleMoments = CBasicStatistics::momentsAccumulator(
                this->numberSamples(), this->marginalLikelihoodMean(), variance);
        }
    }
    return true;
}

COneOfNPrior::COneOfNPrior(const COneOfNPrior& other)
    : CPrior(other.dataType(), other.decayRate()),
      m_SampleMoments(other.m_SampleMoments) {

    // Clone all the models up front so we can implement strong exception safety.
    m_Models.reserve(other.m_Models.size());
    for (const auto& model : other.m_Models) {
        m_Models.emplace_back(model.first, TPriorPtr(model.second->clone()));
    }

    this->CPrior::addSamples(other.numberSamples());
}

COneOfNPrior& COneOfNPrior::operator=(const COneOfNPrior& rhs) {
    if (this != &rhs) {
        COneOfNPrior tmp(rhs);
        this->swap(tmp);
    }
    return *this;
}

void COneOfNPrior::swap(COneOfNPrior& other) {
    this->CPrior::swap(other);
    m_Models.swap(other.m_Models);
    std::swap(m_SampleMoments, other.m_SampleMoments);
}

COneOfNPrior::EPrior COneOfNPrior::type() const {
    return E_OneOfN;
}

COneOfNPrior* COneOfNPrior::clone() const {
    return new COneOfNPrior(*this);
}

void COneOfNPrior::dataType(maths_t::EDataType value) {
    this->CPrior::dataType(value);
    for (auto& model : m_Models) {
        model.second->dataType(value);
    }
}

void COneOfNPrior::decayRate(double value) {
    this->CPrior::decayRate(value);
    for (auto& model : m_Models) {
        model.second->decayRate(this->decayRate());
    }
}

void COneOfNPrior::setToNonInformative(double offset, double decayRate) {
    for (auto& model : m_Models) {
        model.first.age(0.0);
        model.second->setToNonInformative(offset, decayRate);
    }
    m_SampleMoments = TMeanVarAccumulator();
    this->decayRate(decayRate);
    this->numberSamples(0.0);
}

void COneOfNPrior::removeModels(CModelFilter& filter) {
    CScopeCanonicalizeWeights<TPriorPtr> canonicalize(m_Models);

    std::size_t last = 0u;
    for (std::size_t i = 0u; i < m_Models.size(); ++i) {
        if (last != i) {
            std::swap(m_Models[last], m_Models[i]);
        }
        if (!filter(m_Models[last].second->type())) {
            ++last;
        }
    }
    m_Models.erase(m_Models.begin() + last, m_Models.end());
}

bool COneOfNPrior::needsOffset() const {
    for (const auto& model : m_Models) {
        if (model.second->needsOffset()) {
            return true;
        }
    }
    return false;
}

double COneOfNPrior::adjustOffset(const TDouble1Vec& samples,
                                  const TDoubleWeightsAry1Vec& weights) {
    TMeanAccumulator result;

    TDouble5Vec penalties;
    for (auto& model : m_Models) {
        double penalty = model.second->adjustOffset(samples, weights);
        penalties.push_back(penalty);
        result.add(penalty, model.first);
    }

    if (CBasicStatistics::mean(result) != 0.0) {
        CScopeCanonicalizeWeights<TPriorPtr> canonicalize(m_Models);
        for (std::size_t i = 0u; i < penalties.size(); ++i) {
            if (m_Models[i].second->participatesInModelSelection() &&
                CMathsFuncs::isFinite(penalties)) {
                CModelWeight& weight = m_Models[i].first;
                weight.logWeight(weight.logWeight() + penalties[i]);
            }
        }
    }

    return CBasicStatistics::mean(result);
}

double COneOfNPrior::offset() const {
    double offset = 0.0;
    for (const auto& model : m_Models) {
        offset = std::max(offset, model.second->offset());
    }
    return offset;
}

void COneOfNPrior::addSamples(const TDouble1Vec& samples,
                              const TDoubleWeightsAry1Vec& weights) {
    if (samples.empty()) {
        return;
    }
    if (samples.size() != weights.size()) {
        LOG_ERROR(<< "Mismatch in samples '"
                  << core::CContainerPrinter::print(samples) << "' and weights '"
                  << core::CContainerPrinter::print(weights) << "'");
        return;
    }

    this->adjustOffset(samples, weights);

    double n{this->numberSamples()};
    this->CPrior::addSamples(samples, weights);
    n = this->numberSamples() - n;

    for (std::size_t i = 0; i < samples.size(); ++i) {
        m_SampleMoments.add(samples[i], maths_t::countForUpdate(weights[i]));
    }

    // For this 1-of-n model we assume that all the data come from one
    // the distributions which comprise the model. The model can be
    // treated exactly like a discrete parameter and assigned a posterior
    // probability in a Bayesian sense. In particular, we have:
    //   f({p(m), m} | x) = L(x | p(m), m) * f(p(m)) * P(m) / N      (1)
    //
    // where,
    //   x = {x(1), x(2), ... , x(n)} is the sample vector,
    //   f({p(m), m} | x) is the posterior distribution for p(m) and m,
    //   p(m) are the parameters of the model,
    //   m is the model,
    //   L(x | p(m), m) is the likelihood of the data given the model m'th,
    //   f(p(m)) is the prior for the m'th model parameters,
    //   P(m) is the prior probability the data are from the m'th model,
    //   N is a normalization factor.
    //
    // There is one such relation for each model and N is computed over
    // all models:
    //   N = Sum_m( Integral_dp(m)( f({p(m), m}) ) )
    //
    // Note that we can write the r.h.s. of (1) as:
    //   ((L(x | p(m), m) / N'(m)) * f(p(m))) * (N'(m) / N * P(m))   (2)
    //
    // where,
    //   N'(m) = Integral_dp(m)( L(x | {p(m), m}) ),
    //   N = Sum_m( N'(m | x) ) by definition.
    //
    // This means that the joint posterior distribution factorises into the
    // posterior distribution for the model parameters given the data and
    // the posterior weights for each model, i.e.
    //   f({p(m), m} | x) = f'(p(m) | x) * P'(m | x)
    //
    // where f' and P' come from (2). Finally, note that N'(m) is really
    // a function of the data, say h_m(x), and satisfies the relation:
    //   h_m({x(1), x(2), ... , x(k+1)})
    //     = L(x(k+1) | {p(m), m, x(1), x(2), ... , x(k)})
    //       * h_m({x(1), x(2), ... , x(k)})
    //
    // Here, L(x(k+1) | {p(m), m, x(1), x(2), ... , x(k)}) is the likelihood
    // of the (k+1)'th datum given model m and all previous data. Really, the
    // data just enter into this via the updated model parameters p(m). This
    // is the form we use below.
    //
    // Note that the weight of the sample x(i) is interpreted as its count,
    // i.e. n(i), so for example updating with {(x, 2)} is equivalent to
    // updating with {x, x}.

    CScopeCanonicalizeWeights<TPriorPtr> canonicalize(m_Models);

    // We need to check *before* adding samples to the constituent models.
    bool isNonInformative{this->isNonInformative()};

    TDouble5Vec minusBics;
    TDouble5Vec varianceMismatchPenalties;
    TBool5Vec used;
    TBool5Vec uses;
    bool failed{false};

    // If none of the models are a good fit for the new data then restrict
    // the maximum Bayes Factor since the data likelihood given the model
    // is most useful if one of the models is (mostly) correct.
    double m{std::max(n, 1.0)};
    double maxLogBayesFactor{-m * MAXIMUM_LOG_BAYES_FACTOR};

    for (auto& model : m_Models) {

        double minusBic{0.0};
        maths_t::EFloatingPointErrorStatus status{maths_t::E_FpOverflowed};

        used.push_back(model.second->participatesInModelSelection());
        if (used.back()) {
            double logLikelihood;
            status = model.second->jointLogMarginalLikelihood(samples, weights, logLikelihood);

            minusBic = 2.0 * logLikelihood -
                       model.second->unmarginalizedParameters() * CTools::fastLog(n);

            double modeLogLikelihood;
            TDouble1Vec modes;
            modes.reserve(weights.size());
            for (const auto& weight : weights) {
                modes.push_back(model.second->marginalLikelihoodMode(weight));
            }
            model.second->jointLogMarginalLikelihood(modes, weights, modeLogLikelihood);
            maxLogBayesFactor = std::max(
                maxLogBayesFactor, logLikelihood - std::max(modeLogLikelihood, logLikelihood));
        }

        minusBics.push_back((status & maths_t::E_FpOverflowed) ? MINUS_INF : minusBic);
        failed |= (status & maths_t::E_FpFailed);

        // Update the component prior distribution.
        model.second->addSamples(samples, weights);

        varianceMismatchPenalties.push_back(
            -m * MAXIMUM_LOG_BAYES_FACTOR *
            std::max(1.0 - MAXIMUM_UNPENALISED_RELATIVE_VARIANCE_ERROR *
                               CBasicStatistics::variance(m_SampleMoments) /
                               model.second->marginalLikelihoodVariance(),
                     0.0));

        uses.push_back(model.second->participatesInModelSelection());
        if (!uses.back()) {
            model.first.logWeight(MINUS_INF);
        }
    }

    if (failed) {
        LOG_ERROR(<< "Failed to compute log-likelihood");
        LOG_ERROR(<< "samples = " << core::CContainerPrinter::print(samples));
        return;
    }
    if (isNonInformative) {
        return;
    }

    maxLogBayesFactor += m * MAXIMUM_LOG_BAYES_FACTOR;

    LOG_TRACE(<< "BICs = " << core::CContainerPrinter::print(minusBics));
    LOG_TRACE(<< "max Bayes Factor = " << maxLogBayesFactor);
    LOG_TRACE(<< "variance penalties = "
              << core::CContainerPrinter::print(varianceMismatchPenalties));

    double maxLogModelWeight{MINUS_INF + m * MAXIMUM_LOG_BAYES_FACTOR};
    double maxMinusBic{*std::max_element(minusBics.begin(), minusBics.end())};
    for (std::size_t i = 0; i < m_Models.size(); ++i) {
        if (used[i] && uses[i]) {
            m_Models[i].first.addLogFactor(
                std::max(minusBics[i] / 2.0, maxMinusBic / 2.0 - maxLogBayesFactor) +
                varianceMismatchPenalties[i]);
            maxLogModelWeight = std::max(maxLogModelWeight, m_Models[i].first.logWeight());
        }
    }

    for (std::size_t i = 0; i < m_Models.size(); ++i) {
        if (!used[i] && uses[i]) {
            m_Models[i].first.logWeight(maxLogModelWeight - m * MAXIMUM_LOG_BAYES_FACTOR);
        }
    }

    if (this->badWeights()) {
        LOG_ERROR(<< "Update failed (" << this->debugWeights() << ")");
        LOG_ERROR(<< "samples = " << core::CContainerPrinter::print(samples));
        LOG_ERROR(<< "weights = " << core::CContainerPrinter::print(weights));
        this->setToNonInformative(this->offsetMargin(), this->decayRate());
    }
}

void COneOfNPrior::propagateForwardsByTime(double time) {
    if (!CMathsFuncs::isFinite(time) || time < 0.0) {
        LOG_ERROR(<< "Bad propagation time " << time);
        return;
    }

    CScopeCanonicalizeWeights<TPriorPtr> canonicalize(m_Models);

    double alpha = std::exp(-this->decayRate() * time);

    for (auto& model : m_Models) {
        model.first.age(alpha);
        model.second->propagateForwardsByTime(time);
    }
    m_SampleMoments.age(alpha);

    this->numberSamples(this->numberSamples() * alpha);

    LOG_TRACE(<< "numberSamples = " << this->numberSamples());
}

COneOfNPrior::TDoubleDoublePr COneOfNPrior::marginalLikelihoodSupport() const {

    TDoubleDoublePr result(MINUS_INF, INF);

    // We define this is as the intersection of the component model supports.
    for (const auto& model : m_Models) {
        if (model.second->participatesInModelSelection()) {
            TDoubleDoublePr modelSupport = model.second->marginalLikelihoodSupport();
            result.first = std::max(result.first, modelSupport.first);
            result.second = std::min(result.second, modelSupport.second);
        }
    }

    return result;
}

double COneOfNPrior::marginalLikelihoodMean() const {

    if (this->isNonInformative()) {
        return this->medianModelMean();
    }

    // This is E_{P(i)}[ E[X | P(i)] ] and the conditional expectation
    // is just the individual model expectation. Note we exclude models
    // with low weight because typically the means are similar between
    // models and if they are very different we don't want to include
    // the model if there is strong evidence against it.

    double result = 0.0;
    double Z = 0.0;
    for (const auto& model : m_Models) {
        double wi = model.first;
        if (wi > MINIMUM_SIGNIFICANT_WEIGHT) {
            result += wi * model.second->marginalLikelihoodMean();
            Z += wi;
        }
    }
    return result / Z;
}

double COneOfNPrior::nearestMarginalLikelihoodMean(double value) const {

    if (this->isNonInformative()) {
        return this->medianModelMean();
    }

    // See marginalLikelihoodMean for discussion.

    double result = 0.0;
    double Z = 0.0;
    for (const auto& model : m_Models) {
        double wi = model.first;
        if (wi > MINIMUM_SIGNIFICANT_WEIGHT) {
            result += wi * model.second->nearestMarginalLikelihoodMean(value);
            Z += wi;
        }
    }
    return result / Z;
}

double COneOfNPrior::marginalLikelihoodMode(const TDoubleWeightsAry& weights) const {

    // We approximate this as the weighted average of the component
    // model modes.

    // Declared outside the loop to minimize the number of times
    // they are created.
    TDouble1Vec sample(1);
    TDoubleWeightsAry1Vec weight(1, weights);

    TMeanAccumulator mode;
    for (const auto& model : m_Models) {
        if (model.second->participatesInModelSelection()) {
            double wi = model.first;
            double mi = model.second->marginalLikelihoodMode(weights);
            double logLikelihood;
            sample[0] = mi;
            model.second->jointLogMarginalLikelihood(sample, weight, logLikelihood);
            mode.add(mi, wi * std::exp(logLikelihood));
        }
    }

    double result = CBasicStatistics::mean(mode);
    TDoubleDoublePr support = this->marginalLikelihoodSupport();
    return CTools::truncate(result, support.first, support.second);
}

double COneOfNPrior::marginalLikelihoodVariance(const TDoubleWeightsAry& weights) const {

    if (this->isNonInformative()) {
        return INF;
    }

    // This is E_{P(i)}[ Var[X | i] ] and the conditional expectation
    // is just the individual model expectation. Note we exclude models
    // with low weight because typically the means are similar between
    // models and if they are very different we don't want to include
    // the model if there is strong evidence against it.

    double result = 0.0;
    double Z = 0.0;
    for (const auto& model : m_Models) {
        double wi = model.first;
        if (wi > MINIMUM_SIGNIFICANT_WEIGHT) {
            result += wi * model.second->marginalLikelihoodVariance(weights);
            Z += wi;
        }
    }
    return result / Z;
}

COneOfNPrior::TDoubleDoublePr
COneOfNPrior::marginalLikelihoodConfidenceInterval(double percentage,
                                                   const TDoubleWeightsAry& weights) const {

    // We approximate this as the weighted sum of the component model
    // intervals. To compute the weights we expand all component model
    // marginal likelihoods about a reasonable estimate for the true
    // interval end points, i.e.
    //   [a_0, b_0] = [Sum_m( a(m) * w(m) ), Sum_m( b(m) * w(m) )]
    //
    // Here, m ranges over the component models, w(m) are the model
    // weights and P([a(m), b(m)]) = p where p is the desired percentage
    // expressed as a probability. Note that this will be accurate in
    // the limit that one model dominates.
    //
    // Note P([a, b]) = F(b) - F(a) where F(.) is the c.d.f. of the
    // marginal likelihood f(.) so it possible to compute a first order
    // correction to [a_0, b_0] as follows:
    //   a_1 = a_0 + ((1 - p) / 2 - F(a_0)) / f(a_0)
    //   b_1 = b_0 + ((1 - p) / 2 - F(b_0)) / f(b_0)             (1)
    //
    // For the time being we just compute a_0 and b_0. We can revisit
    // this calculation if the accuracy proves to be a problem.

    TMeanAccumulator x1, x2;

    for (const auto& model : m_Models) {
        double weight = model.first;
        if (weight >= MAXIMUM_RELATIVE_ERROR) {
            TDoubleDoublePr interval =
                model.second->marginalLikelihoodConfidenceInterval(percentage, weights);
            x1.add(interval.first, weight);
            x2.add(interval.second, weight);
        }
    }
    LOG_TRACE(<< "x1 = " << x1 << ", x2 = " << x2);

    return {CBasicStatistics::mean(x1), CBasicStatistics::mean(x2)};
}

maths_t::EFloatingPointErrorStatus
COneOfNPrior::jointLogMarginalLikelihood(const TDouble1Vec& samples,
                                         const TDoubleWeightsAry1Vec& weights,
                                         double& result) const {

    result = 0.0;

    if (samples.empty()) {
        LOG_ERROR(<< "Can't compute likelihood for empty sample set");
        return maths_t::E_FpFailed;
    }

    if (samples.size() != weights.size()) {
        LOG_ERROR(<< "Mismatch in samples '"
                  << core::CContainerPrinter::print(samples) << "' and weights '"
                  << core::CContainerPrinter::print(weights) << "'");
        return maths_t::E_FpFailed;
    }

    // We get that:
    //   marginal_likelihood(x) = Sum_m( L(x | m) * P(m) ).
    //
    // where,
    //   x = {x(1), x(2), ... , x(n)} the sample vector,
    //   L(x | m) = Integral_du(m)( L(x | {m, p(m)}) ),
    //   p(m) are the parameters of the component and
    //   P(m) is the prior probability the data are from the m'th model.

    // We re-normalize the data so that the maximum likelihood is one
    // to avoid underflow.
    TDouble5Vec logLikelihoods;
    double Z = 0.0;
    TMaxAccumulator maxLogLikelihood;

    for (const auto& model : m_Models) {
        if (model.second->participatesInModelSelection()) {
            double logLikelihood;
            maths_t::EFloatingPointErrorStatus status =
                model.second->jointLogMarginalLikelihood(samples, weights, logLikelihood);
            if (status & maths_t::E_FpFailed) {
                return status;
            }
            if (!(status & maths_t::E_FpOverflowed)) {
                logLikelihood += model.first.logWeight();
                logLikelihoods.push_back(logLikelihood);
                maxLogLikelihood.add(logLikelihood);
            }
            Z += std::exp(model.first.logWeight());
        }
    }

    if (maxLogLikelihood.count() == 0) {
        // Technically, the marginal likelihood is zero here so the
        // log would be infinite. We use minus max double because
        // log(0) = HUGE_VALUE, which causes problems for Windows.
        // Calling code is notified when the calculation overflows
        // and should avoid taking the exponential since this will
        // underflow and pollute the floating point environment.
        // This may cause issues for some library function
        // implementations (see fe*exceptflag for more details).
        result = MINUS_INF;
        return maths_t::E_FpOverflowed;
    }

    for (auto logLikelihood : logLikelihoods) {
        result += std::exp(logLikelihood - maxLogLikelihood[0]);
    }

    result = maxLogLikelihood[0] + CTools::fastLog(result / Z);

    maths_t::EFloatingPointErrorStatus status = CMathsFuncs::fpStatus(result);
    if (status & maths_t::E_FpFailed) {
        LOG_ERROR(<< "Failed to compute log likelihood (" << this->debugWeights() << ")");
        LOG_ERROR(<< "samples = " << core::CContainerPrinter::print(samples));
        LOG_ERROR(<< "weights = " << core::CContainerPrinter::print(weights));
        LOG_ERROR(<< "logLikelihoods = " << core::CContainerPrinter::print(logLikelihoods));
        LOG_ERROR(<< "maxLogLikelihood = " << maxLogLikelihood[0]);
    } else if (status & maths_t::E_FpOverflowed) {
        LOG_ERROR(<< "Log likelihood overflowed for (" << this->debugWeights() << ")");
        LOG_TRACE(<< "likelihoods = " << core::CContainerPrinter::print(logLikelihoods));
        LOG_TRACE(<< "samples = " << core::CContainerPrinter::print(samples));
        LOG_TRACE(<< "weights = " << core::CContainerPrinter::print(weights));
    }
    return status;
}

void COneOfNPrior::sampleMarginalLikelihood(std::size_t numberSamples,
                                            TDouble1Vec& samples) const {

    samples.clear();

    if (numberSamples == 0) {
        return;
    }

    TDouble5Vec weights;
    double Z = 0.0;
    for (const auto& model : m_Models) {
        weights.push_back(model.first);
        Z += model.first;
    }
    for (auto& weight : weights) {
        weight /= Z;
    }

    CSampling::TSizeVec sampling;
    CSampling::weightedSample(numberSamples, weights, sampling);
    LOG_TRACE(<< "weights = " << core::CContainerPrinter::print(weights)
              << ", sampling = " << core::CContainerPrinter::print(sampling));

    if (sampling.size() != m_Models.size()) {
        LOG_ERROR(<< "Failed to sample marginal likelihood");
        return;
    }

    TDoubleDoublePr support = this->marginalLikelihoodSupport();
    support.first = CTools::shiftRight(support.first);
    support.second = CTools::shiftLeft(support.second);

    samples.reserve(numberSamples);
    TDouble1Vec modelSamples;
    for (std::size_t i = 0u; i < m_Models.size(); ++i) {
        modelSamples.clear();
        m_Models[i].second->sampleMarginalLikelihood(sampling[i], modelSamples);
        for (auto sample : modelSamples) {
            samples.push_back(CTools::truncate(sample, support.first, support.second));
        }
    }
    LOG_TRACE(<< "samples = " << core::CContainerPrinter::print(samples));
}

bool COneOfNPrior::minusLogJointCdfImpl(bool complement,
                                        const TDouble1Vec& samples,
                                        const TDoubleWeightsAry1Vec& weights,
                                        double& lowerBound,
                                        double& upperBound) const {

    lowerBound = upperBound = 0.0;

    if (samples.empty()) {
        LOG_ERROR(<< "Can't compute c.d.f. " << (complement ? "complement " : "")
                  << "for empty sample set");
        return false;
    }

    if (this->isNonInformative()) {
        lowerBound = upperBound = -std::log(complement ? 1.0 - CTools::IMPROPER_CDF
                                                       : CTools::IMPROPER_CDF);
        return true;
    }

    // We get that:
    //   cdf(x) = Integral_dx( Sum_m( L(x | m) * P(m) )
    //
    // where,
    //   x = {x(1), x(2), ... , x(n)} the sample vector,
    //   L(x | m) = Integral_du(m)( L(x | {m, p(m)}) ),
    //   p(m) are the parameters of the component,
    //   P(m) is the prior probability the data are from the m'th model and
    //   Integral_dx is over [-inf, x(1)] o [-inf, x(2)] o ... o [-inf, x(n)]
    //   and o denotes the outer product.

    TDoubleSizePr5Vec logWeights = this->normalizedLogWeights();
    LOG_TRACE(<< "logWeights = " << core::CContainerPrinter::print(logWeights));

    TDouble5Vec logLowerBounds;
    TDouble5Vec logUpperBounds;
    TMaxAccumulator maxLogLowerBound;
    TMaxAccumulator maxLogUpperBound;
    double logMaximumRemainder = MINUS_INF;
    for (std::size_t i = 0u, n = logWeights.size(); i < n; ++i) {
        double wi = logWeights[i].first;
        const CPrior& model = *m_Models[logWeights[i].second].second;

        double li = 0.0;
        double ui = 0.0;
        if (complement && !model.minusLogJointCdfComplement(samples, weights, li, ui)) {
            LOG_ERROR(<< "Failed computing c.d.f. complement for "
                      << core::CContainerPrinter::print(samples));
            return false;
        } else if (!complement && !model.minusLogJointCdf(samples, weights, li, ui)) {
            LOG_ERROR(<< "Failed computing c.d.f. for "
                      << core::CContainerPrinter::print(samples));
            return false;
        }
        li = wi - li;
        ui = wi - ui;

        logLowerBounds.push_back(li);
        logUpperBounds.push_back(ui);
        maxLogLowerBound.add(li);
        maxLogUpperBound.add(ui);

        // Check if we can exit early with reasonable precision.
        if (i + 1 < n) {
            logMaximumRemainder = logn(n - i - 1) + logWeights[i + 1].first;
            if (logMaximumRemainder < maxLogLowerBound[0] + LOG_MAXIMUM_RELATIVE_ERROR &&
                logMaximumRemainder < maxLogUpperBound[0] + LOG_MAXIMUM_RELATIVE_ERROR) {
                break;
            }
        }
    }

    if (!CTools::logWillUnderflow<double>(maxLogLowerBound[0])) {
        maxLogLowerBound[0] = 0.0;
    }
    if (!CTools::logWillUnderflow<double>(maxLogUpperBound[0])) {
        maxLogUpperBound[0] = 0.0;
    }
    for (std::size_t i = 0u; i < logLowerBounds.size(); ++i) {
        lowerBound += std::exp(logLowerBounds[i] - maxLogLowerBound[0]);
        upperBound += std::exp(logUpperBounds[i] - maxLogUpperBound[0]);
    }
    lowerBound = -std::log(lowerBound) - maxLogLowerBound[0];
    upperBound = -std::log(upperBound) - maxLogUpperBound[0];
    if (logLowerBounds.size() < logWeights.size()) {
        upperBound += -std::log(1.0 + std::exp(logMaximumRemainder + upperBound));
    }
    lowerBound = std::max(lowerBound, 0.0);
    upperBound = std::max(upperBound, 0.0);

    LOG_TRACE(<< "Joint -log(c.d.f." << (complement ? " complement" : "")
              << ") = [" << lowerBound << "," << upperBound << "]");

    return true;
}

bool COneOfNPrior::minusLogJointCdf(const TDouble1Vec& samples,
                                    const TDoubleWeightsAry1Vec& weights,
                                    double& lowerBound,
                                    double& upperBound) const {
    return this->minusLogJointCdfImpl(false /*complement*/, samples, weights,
                                      lowerBound, upperBound);
}

bool COneOfNPrior::minusLogJointCdfComplement(const TDouble1Vec& samples,
                                              const TDoubleWeightsAry1Vec& weights,
                                              double& lowerBound,
                                              double& upperBound) const {
    return this->minusLogJointCdfImpl(true /*complement*/, samples, weights,
                                      lowerBound, upperBound);
}

bool COneOfNPrior::probabilityOfLessLikelySamples(maths_t::EProbabilityCalculation calculation,
                                                  const TDouble1Vec& samples,
                                                  const TDoubleWeightsAry1Vec& weights,
                                                  double& lowerBound,
                                                  double& upperBound,
                                                  maths_t::ETail& tail) const {

    lowerBound = upperBound = 0.0;
    tail = maths_t::E_UndeterminedTail;

    if (samples.empty()) {
        LOG_ERROR(<< "Can't compute distribution for empty sample set");
        return false;
    }
    if (this->isNonInformative()) {
        lowerBound = upperBound = 1.0;
        return true;
    }

    // The joint probability of less likely collection of samples can be
    // computed from the conditional probabilities of a less likely collection
    // of samples from each component model:
    //   P(R) = Sum_i( P(R | m) * P(m) )
    //
    // where,
    //   P(R | m) is the probability of a less likely collection of samples
    //   from the m'th model and
    //   P(m) is the prior probability the data are from the m'th model.

    using TDoubleTailPr = std::pair<double, maths_t::ETail>;
    using TDoubleTailPrMaxAccumulator = CBasicStatistics::SMax<TDoubleTailPr>::TAccumulator;

    TDoubleSizePr5Vec logWeights = this->normalizedLogWeights();

    TDoubleTailPrMaxAccumulator tail_;
    for (std::size_t i = 0u; i < logWeights.size(); ++i) {
        double weight = std::exp(logWeights[i].first);
        const CPrior& model = *m_Models[logWeights[i].second].second;

        if (lowerBound > static_cast<double>(m_Models.size() - i) * weight / MAXIMUM_RELATIVE_ERROR) {
            // The probability calculation is relatively expensive so don't
            // evaluate the probabilities that aren't needed to get good
            // accuracy.
            break;
        }

        double modelLowerBound, modelUpperBound;
        maths_t::ETail modelTail;
        if (!model.probabilityOfLessLikelySamples(calculation, samples, weights, modelLowerBound,
                                                  modelUpperBound, modelTail)) {
            // Logging handled at a lower level.
            return false;
        }

        LOG_TRACE(<< "weight = " << weight << ", modelLowerBound = " << modelLowerBound
                  << ", modelUpperBound = " << modelLowerBound);

        lowerBound += weight * modelLowerBound;
        upperBound += weight * modelUpperBound;
        tail_.add({weight * (modelLowerBound + modelUpperBound), modelTail});
    }

    if (!(lowerBound >= 0.0 && lowerBound <= 1.001) ||
        !(upperBound >= 0.0 && upperBound <= 1.001)) {
        LOG_ERROR(<< "Bad probability bounds = [" << lowerBound << ", " << upperBound << "]"
                  << ", " << core::CContainerPrinter::print(logWeights));
    }

    if (CMathsFuncs::isNan(lowerBound)) {
        lowerBound = 0.0;
    }
    if (CMathsFuncs::isNan(upperBound)) {
        upperBound = 1.0;
    }
    lowerBound = CTools::truncate(lowerBound, 0.0, 1.0);
    upperBound = CTools::truncate(upperBound, 0.0, 1.0);
    tail = tail_[0].second;

    LOG_TRACE(<< "Probability = [" << lowerBound << "," << upperBound << "]");

    return true;
}

bool COneOfNPrior::isNonInformative() const {
    for (const auto& model : m_Models) {
        if (model.second->participatesInModelSelection() &&
            model.second->isNonInformative()) {
            return true;
        }
    }
    return false;
}

void COneOfNPrior::print(const std::string& indent, std::string& result) const {

    result += core_t::LINE_ENDING + indent + "one-of-n";
    if (this->isNonInformative()) {
        result += " non-informative";
    }

    result += ':';
    result += core_t::LINE_ENDING + indent + " # samples " +
              core::CStringUtils::typeToStringPretty(this->numberSamples());
    for (const auto& model : m_Models) {
        double weight = model.first;
        if (weight >= MINIMUM_SIGNIFICANT_WEIGHT) {
            std::string indent_ = indent + " weight " +
                                  core::CStringUtils::typeToStringPretty(weight) + "  ";
            model.second->print(indent_, result);
        }
    }
}

std::string COneOfNPrior::printJointDensityFunction() const {
    return "Not supported";
}

uint64_t COneOfNPrior::checksum(uint64_t seed) const {
    seed = this->CPrior::checksum(seed);
    seed = CChecksum::calculate(seed, m_Models);
    seed = CChecksum::calculate(seed, m_SampleMoments);
    return seed;
}

void COneOfNPrior::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("COneOfNPrior");
    core::CMemoryDebug::dynamicSize("m_Models", m_Models, mem);
}

std::size_t COneOfNPrior::memoryUsage() const {
    return core::CMemory::dynamicSize(m_Models);
}

std::size_t COneOfNPrior::staticSize() const {
    return sizeof(*this);
}

void COneOfNPrior::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    const bool readableTags{inserter.readableTags()};
    inserter.insertValue(VERSION_7_1_TAG, "");
    for (const auto& model : m_Models) {
        inserter.insertLevel(readableTags ? READABLE_MODEL_7_1_TAG : MODEL_7_1_TAG,
                                                        std::bind(&modelAcceptPersistInserter,
                                                        std::cref(model.first),
                                                        std::cref(*model.second),
                                                        std::placeholders::_1));
    }
    inserter.insertValue(readableTags ? READABLE_SAMPLE_MOMENTS_7_1_TAG : SAMPLE_MOMENTS_7_1_TAG, m_SampleMoments.toDelimited());
    inserter.insertValue(readableTags ? READABLE_DECAY_RATE_7_1_TAG : DECAY_RATE_7_1_TAG, this->decayRate(), core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(readableTags ? READABLE_NUMBER_SAMPLES_7_1_TAG : NUMBER_SAMPLES_7_1_TAG, this->numberSamples(),
                         core::CIEEE754::E_SinglePrecision);
}

COneOfNPrior::TDoubleVec COneOfNPrior::weights() const {
    TDoubleVec result = this->logWeights();
    for (auto& weight : result) {
        weight = std::exp(weight);
    }
    return result;
}

COneOfNPrior::TDoubleVec COneOfNPrior::logWeights() const {

    TDoubleVec result;
    result.reserve(m_Models.size());

    double Z = 0.0;
    for (const auto& model : m_Models) {
        result.push_back(model.first.logWeight());
        Z += std::exp(result.back());
    }
    Z = std::log(Z);
    for (auto& weight : result) {
        weight -= Z;
    }

    return result;
}

COneOfNPrior::TPriorCPtrVec COneOfNPrior::models() const {
    TPriorCPtrVec result;
    result.reserve(m_Models.size());
    for (const auto& model : m_Models) {
        result.push_back(model.second.get());
    }
    return result;
}

bool COneOfNPrior::modelAcceptRestoreTraverser(const SDistributionRestoreParams& params,
                                               core::CStateRestoreTraverser& traverser) {
    CModelWeight weight(1.0);
    bool gotWeight = false;
    TPriorPtr model;

    do {
        const std::string& name = traverser.name();
        RESTORE_SETUP_TEARDOWN(
            WEIGHT_TAG,
            /*no-op*/,
            traverser.traverseSubLevel(std::bind(&CModelWeight::acceptRestoreTraverser,
                                                 &weight, std::placeholders::_1)),
            gotWeight = true)
        RESTORE(PRIOR_TAG, traverser.traverseSubLevel(std::bind<bool>(
                               CPriorStateSerialiser(), std::cref(params),
                               std::ref(model), std::placeholders::_1)))
    } while (traverser.next());

    if (!gotWeight) {
        LOG_ERROR(<< "No weight found");
        return false;
    }
    if (model == nullptr) {
        LOG_ERROR(<< "No model found");
        return false;
    }

    m_Models.emplace_back(weight, std::move(model));

    return true;
}

COneOfNPrior::TDoubleSizePr5Vec COneOfNPrior::normalizedLogWeights() const {

    TDoubleSizePr5Vec result;
    double Z = 0.0;
    for (std::size_t i = 0u; i < m_Models.size(); ++i) {
        if (m_Models[i].second->participatesInModelSelection()) {
            double logWeight = m_Models[i].first.logWeight();
            result.emplace_back(logWeight, i);
            Z += std::exp(logWeight);
        }
    }
    Z = std::log(Z);
    for (auto& logWeight : result) {
        logWeight.first -= Z;
    }
    std::sort(result.begin(), result.end(), std::greater<TDoubleSizePr>());
    return result;
}

double COneOfNPrior::medianModelMean() const {
    TDoubleVec means;
    means.reserve(m_Models.size());
    for (const auto& model : m_Models) {
        if (model.second->participatesInModelSelection()) {
            means.push_back(model.second->marginalLikelihoodMean());
        }
    }
    return CBasicStatistics::median(means);
}

bool COneOfNPrior::badWeights() const {
    for (const auto& model : m_Models) {
        if (!CMathsFuncs::isFinite(model.first.logWeight())) {
            return true;
        }
    }
    return false;
}

std::string COneOfNPrior::debugWeights() const {
    if (m_Models.empty()) {
        return std::string();
    }
    std::ostringstream result;
    result << std::scientific << std::setprecision(15);
    for (const auto& model : m_Models) {
        result << " " << model.first.logWeight();
    }
    result << " ";
    return result.str();
}
}
}
