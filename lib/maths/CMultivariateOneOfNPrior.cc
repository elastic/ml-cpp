/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CMultivariateOneOfNPrior.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStringUtils.h>
#include <core/RestoreMacros.h>

#include <maths/CChecksum.h>
#include <maths/CMathsFuncs.h>
#include <maths/COneOfNPrior.h>
#include <maths/CPriorStateSerialiser.h>
#include <maths/CRestoreParams.h>
#include <maths/CSampling.h>
#include <maths/CTools.h>
#include <maths/Constants.h>

#include <boost/bind.hpp>
#include <boost/make_unique.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/ref.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iterator>

namespace ml {
namespace maths {
namespace {

using TBool3Vec = core::CSmallVector<bool, 3>;
using TDouble3Vec = CMultivariateOneOfNPrior::TDouble3Vec;
using TDouble10Vec = CMultivariateOneOfNPrior::TDouble10Vec;
using TDouble10VecDouble10VecPr = CMultivariateOneOfNPrior::TDouble10VecDouble10VecPr;
using TDouble10Vec1Vec = CMultivariateOneOfNPrior::TDouble10Vec1Vec;
using TDouble10Vec10Vec = CMultivariateOneOfNPrior::TDouble10Vec10Vec;
using TDouble10VecWeightsAry1Vec = CMultivariateOneOfNPrior::TDouble10VecWeightsAry1Vec;
using TPriorPtr = CMultivariateOneOfNPrior::TPriorPtr;
using TWeightPriorPtrPr = CMultivariateOneOfNPrior::TWeightPriorPtrPr;
using TWeightPriorPtrPrVec = CMultivariateOneOfNPrior::TWeightPriorPtrPrVec;

// We use short field names to reduce the state size
const std::string MODEL_TAG("a");
const std::string NUMBER_SAMPLES_TAG("b");
const std::string WEIGHT_TAG("c");
const std::string PRIOR_TAG("d");
const std::string DECAY_RATE_TAG("e");

//! Add elements of \p x to \p y.
void add(const TDouble10Vec& x, TDouble10Vec& y) {
    for (std::size_t i = 0u; i < x.size(); ++i) {
        y[i] += x[i];
    }
}

//! Get the min of \p x and \p y.
TDouble10Vec min(const TDouble10Vec& x, const TDouble10Vec& y) {
    TDouble10Vec result(x);
    for (std::size_t i = 0u; i < x.size(); ++i) {
        result[i] = std::min(result[i], y[i]);
    }
    return result;
}

//! Get the max of \p x and \p y.
TDouble10Vec max(const TDouble10Vec& x, const TDouble10Vec& y) {
    TDouble10Vec result(x);
    for (std::size_t i = 0u; i < x.size(); ++i) {
        result[i] = std::max(result[i], y[i]);
    }
    return result;
}

//! Update the arithmetic mean \p mean with \p x and weight \p nx.
void updateMean(const TDouble10Vec& x, double nx, TDouble10Vec& mean, double& n) {
    if (nx <= 0.0) {
        return;
    }
    for (std::size_t i = 0u; i < x.size(); ++i) {
        mean[i] = (n * mean[i] + nx * x[i]) / (n + nx);
    }
    n += nx;
}

//! Update the arithmetic mean \p mean with \p x and weight \p nx.
void updateMean(const TDouble10Vec10Vec& x, double nx, TDouble10Vec10Vec& mean, double& n) {
    if (nx <= 0.0) {
        return;
    }
    for (std::size_t i = 0u; i < x.size(); ++i) {
        for (std::size_t j = 0u; j < x[i].size(); ++j) {
            mean[i][j] = (n * mean[i][j] + nx * x[i][j]) / (n + nx);
        }
    }
    n += nx;
}

//! Get the largest element of \p x.
double largest(const TDouble10Vec& x) {
    return *std::max_element(x.begin(), x.end());
}

//! Add a model vector entry reading parameters from \p traverser.
bool modelAcceptRestoreTraverser(const SDistributionRestoreParams& params,
                                 TWeightPriorPtrPrVec& models,
                                 core::CStateRestoreTraverser& traverser) {
    CModelWeight weight(1.0);
    bool gotWeight = false;
    TPriorPtr model;

    do {
        const std::string& name = traverser.name();
        RESTORE_SETUP_TEARDOWN(WEIGHT_TAG,
                               /**/,
                               traverser.traverseSubLevel(boost::bind(
                                   &CModelWeight::acceptRestoreTraverser, &weight, _1)),
                               gotWeight = true)
        RESTORE(PRIOR_TAG, traverser.traverseSubLevel(boost::bind<bool>(
                               CPriorStateSerialiser(), boost::cref(params),
                               boost::ref(model), _1)))
    } while (traverser.next());

    if (!gotWeight) {
        LOG_ERROR(<< "No weight found");
        return false;
    }
    if (model == nullptr) {
        LOG_ERROR(<< "No model found");
        return false;
    }

    models.emplace_back(weight, std::move(model));

    return true;
}

//! Read the models, decay rate and number of samples from the supplied traverser.
bool acceptRestoreTraverser(const SDistributionRestoreParams& params,
                            TWeightPriorPtrPrVec& models,
                            double& decayRate,
                            double& numberSamples,
                            core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE_BUILT_IN(DECAY_RATE_TAG, decayRate)
        RESTORE(MODEL_TAG, traverser.traverseSubLevel(boost::bind(
                               &modelAcceptRestoreTraverser,
                               boost::cref(params), boost::ref(models), _1)))
        RESTORE_BUILT_IN(NUMBER_SAMPLES_TAG, numberSamples)
    } while (traverser.next());

    return true;
}

//! Persist state for one of the models by passing information
//! to the supplied inserter.
void modelAcceptPersistInserter(const CModelWeight& weight,
                                const CMultivariatePrior& prior,
                                core::CStatePersistInserter& inserter) {
    inserter.insertLevel(
        WEIGHT_TAG, boost::bind(&CModelWeight::acceptPersistInserter, &weight, _1));
    inserter.insertLevel(PRIOR_TAG, boost::bind<void>(CPriorStateSerialiser(),
                                                      boost::cref(prior), _1));
}

const double DERATE = 0.99999;
const double MINUS_INF = DERATE * boost::numeric::bounds<double>::lowest();
const double INF = DERATE * boost::numeric::bounds<double>::highest();
const double LOG_INITIAL_WEIGHT = std::log(1e-6);
const double MINIMUM_SIGNIFICANT_WEIGHT = 0.01;
}

CMultivariateOneOfNPrior::CMultivariateOneOfNPrior(std::size_t dimension,
                                                   const TPriorPtrVec& models,
                                                   maths_t::EDataType dataType,
                                                   double decayRate)
    : CMultivariatePrior(dataType, decayRate), m_Dimension(dimension) {
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

CMultivariateOneOfNPrior::CMultivariateOneOfNPrior(std::size_t dimension,
                                                   const TDoublePriorPtrPrVec& models,
                                                   maths_t::EDataType dataType,
                                                   double decayRate)
    : CMultivariatePrior(dataType, decayRate), m_Dimension(dimension) {
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

CMultivariateOneOfNPrior::CMultivariateOneOfNPrior(std::size_t dimension,
                                                   const SDistributionRestoreParams& params,
                                                   core::CStateRestoreTraverser& traverser)
    : CMultivariatePrior(params.s_DataType, params.s_DecayRate),
      m_Dimension(dimension) {
    double decayRate{0.0};
    double numberSamples{0.0};
    if (traverser.traverseSubLevel(boost::bind(
            &acceptRestoreTraverser, boost::cref(params), boost::ref(m_Models),
            boost::ref(decayRate), boost::ref(numberSamples), _1)) == false) {
        return;
    }
    this->decayRate(decayRate);
    this->numberSamples(numberSamples);
}

CMultivariateOneOfNPrior::CMultivariateOneOfNPrior(const CMultivariateOneOfNPrior& other)
    : CMultivariatePrior(other.dataType(), other.decayRate()),
      m_Dimension(other.m_Dimension) {
    // Clone all the models up front so we can implement strong exception safety.
    m_Models.reserve(other.m_Models.size());
    for (const auto& model : other.m_Models) {
        m_Models.emplace_back(model.first, TPriorPtr(model.second->clone()));
    }

    this->CMultivariatePrior::addSamples(other.numberSamples());
}

CMultivariateOneOfNPrior& CMultivariateOneOfNPrior::
operator=(const CMultivariateOneOfNPrior& rhs) {
    if (this != &rhs) {
        CMultivariateOneOfNPrior tmp(rhs);
        this->swap(tmp);
    }
    return *this;
}

void CMultivariateOneOfNPrior::swap(CMultivariateOneOfNPrior& other) {
    this->CMultivariatePrior::swap(other);
    m_Models.swap(other.m_Models);
}

CMultivariateOneOfNPrior* CMultivariateOneOfNPrior::clone() const {
    return new CMultivariateOneOfNPrior(*this);
}

std::size_t CMultivariateOneOfNPrior::dimension() const {
    return m_Dimension;
}

void CMultivariateOneOfNPrior::dataType(maths_t::EDataType value) {
    this->CMultivariatePrior::dataType(value);
    for (auto& model : m_Models) {
        model.second->dataType(value);
    }
}

void CMultivariateOneOfNPrior::decayRate(double value) {
    this->CMultivariatePrior::decayRate(value);
    for (auto& model : m_Models) {
        model.second->decayRate(this->decayRate());
    }
}

void CMultivariateOneOfNPrior::setToNonInformative(double offset, double decayRate) {
    for (auto& model : m_Models) {
        model.first.age(0.0);
        model.second->setToNonInformative(offset, decayRate);
    }
    this->decayRate(decayRate);
    this->numberSamples(0.0);
}

void CMultivariateOneOfNPrior::adjustOffset(const TDouble10Vec1Vec& samples,
                                            const TDouble10VecWeightsAry1Vec& weights) {
    for (auto& model : m_Models) {
        model.second->adjustOffset(samples, weights);
    }
}

void CMultivariateOneOfNPrior::addSamples(const TDouble10Vec1Vec& samples,
                                          const TDouble10VecWeightsAry1Vec& weights) {
    if (samples.empty()) {
        return;
    }
    if (!this->check(samples, weights)) {
        return;
    }

    this->adjustOffset(samples, weights);

    double penalty = CTools::fastLog(this->numberSamples());
    this->CMultivariatePrior::addSamples(samples, weights);
    penalty = (penalty - CTools::fastLog(this->numberSamples())) / 2.0;

    // See COneOfNPrior::addSamples for a discussion.

    CScopeCanonicalizeWeights<TPriorPtr> canonicalize(m_Models);

    // We need to check *before* adding samples to the constituent models.
    bool isNonInformative = this->isNonInformative();

    // Compute the unnormalized posterior weights and update the component
    // priors. These weights are computed on the side since they are only
    // updated if all marginal likelihoods can be computed.
    TDouble3Vec logLikelihoods;
    TMaxAccumulator maxLogLikelihood;
    TBool3Vec used, uses;
    for (auto& model : m_Models) {
        bool use = model.second->participatesInModelSelection();

        // Update the weights with the marginal likelihoods.
        double logLikelihood = 0.0;
        maths_t::EFloatingPointErrorStatus status =
            use ? model.second->jointLogMarginalLikelihood(samples, weights, logLikelihood)
                : maths_t::E_FpOverflowed;
        if (status & maths_t::E_FpFailed) {
            LOG_ERROR(<< "Failed to compute log-likelihood");
            LOG_ERROR(<< "samples = " << core::CContainerPrinter::print(samples));
        } else {
            if (!(status & maths_t::E_FpOverflowed)) {
                logLikelihood += model.second->unmarginalizedParameters() * penalty;
                logLikelihoods.push_back(logLikelihood);
                maxLogLikelihood.add(logLikelihood);
            } else {
                logLikelihoods.push_back(MINUS_INF);
            }

            // Update the component prior distribution.
            model.second->addSamples(samples, weights);

            used.push_back(use);
            uses.push_back(model.second->participatesInModelSelection());
        }
    }

    TDouble10Vec n(m_Dimension, 0.0);
    for (const auto& weight : weights) {
        add(maths_t::count(weight), n);
    }

    if (!isNonInformative && maxLogLikelihood.count() > 0) {
        LOG_TRACE(<< "logLikelihoods = " << core::CContainerPrinter::print(logLikelihoods));

        // The idea here is to limit the amount which extreme samples
        // affect model selection, particularly early on in the model
        // life-cycle.
        double l = largest(n);
        double minLogLikelihood =
            maxLogLikelihood[0] -
            l * std::min(maxModelPenalty(this->numberSamples()), 100.0);

        TMaxAccumulator maxLogWeight;
        for (std::size_t i = 0; i < logLikelihoods.size(); ++i) {
            CModelWeight& weight = m_Models[i].first;
            if (!uses[i]) {
                weight.logWeight(MINUS_INF);
            } else if (used[i]) {
                weight.addLogFactor(std::max(logLikelihoods[i], minLogLikelihood));
                maxLogWeight.add(weight.logWeight());
            }
        }
        for (std::size_t i = 0u; i < m_Models.size(); ++i) {
            if (!used[i] && uses[i]) {
                m_Models[i].first.logWeight(maxLogWeight[0] + LOG_INITIAL_WEIGHT);
            }
        }
    }

    if (this->badWeights()) {
        LOG_ERROR(<< "Update failed (" << this->debugWeights() << ")");
        LOG_ERROR(<< "samples = " << core::CContainerPrinter::print(samples));
        LOG_ERROR(<< "weights = " << core::CContainerPrinter::print(weights));
        this->setToNonInformative(this->offsetMargin(), this->decayRate());
    }
}

void CMultivariateOneOfNPrior::propagateForwardsByTime(double time) {
    if (!CMathsFuncs::isFinite(time) || time < 0.0) {
        LOG_ERROR(<< "Bad propagation time " << time);
        return;
    }

    CScopeCanonicalizeWeights<TPriorPtr> canonicalize(m_Models);

    double alpha = std::exp(-this->scaledDecayRate() * time);

    for (auto& model : m_Models) {
        if (!this->isForForecasting()) {
            model.first.age(alpha);
        }
        model.second->propagateForwardsByTime(time);
    }

    this->numberSamples(this->numberSamples() * alpha);

    LOG_TRACE(<< "numberSamples = " << this->numberSamples());
}

CMultivariateOneOfNPrior::TUnivariatePriorPtrDoublePr
CMultivariateOneOfNPrior::univariate(const TSize10Vec& marginalize,
                                     const TSizeDoublePr10Vec& condition) const {
    COneOfNPrior::TDoublePriorPtrPrVec models;
    TDouble3Vec weights;
    TMaxAccumulator maxWeight;
    double Z = 0.0;

    for (const auto& model : m_Models) {
        if (model.second->participatesInModelSelection()) {
            TUnivariatePriorPtrDoublePr prior(model.second->univariate(marginalize, condition));
            if (prior.first == nullptr) {
                return {};
            }
            models.emplace_back(1.0, std::move(prior.first));
            weights.push_back(prior.second + model.first.logWeight());
            maxWeight.add(weights.back());
            Z += std::exp(model.first.logWeight());
        }
    }

    for (std::size_t i = 0u; i < weights.size(); ++i) {
        models[i].first *= std::exp(weights[i] - maxWeight[0]) / Z;
    }

    return {boost::make_unique<COneOfNPrior>(models, this->dataType(), this->decayRate()),
            maxWeight.count() > 0 ? maxWeight[0] : 0.0};
}

CMultivariateOneOfNPrior::TPriorPtrDoublePr
CMultivariateOneOfNPrior::bivariate(const TSize10Vec& marginalize,
                                    const TSizeDoublePr10Vec& condition) const {
    if (m_Dimension == 2) {
        return {TPriorPtr(this->clone()), 0.0};
    }

    TDoublePriorPtrPrVec models;
    TDouble3Vec weights;
    TMaxAccumulator maxWeight;
    double Z = 0.0;

    for (const auto& model : m_Models) {
        if (model.second->participatesInModelSelection()) {
            TPriorPtrDoublePr prior(model.second->bivariate(marginalize, condition));
            if (prior.first == nullptr) {
                return {};
            }
            models.emplace_back(1.0, std::move(prior.first));
            weights.push_back(prior.second + model.first.logWeight());
            maxWeight.add(weights.back());
            Z += std::exp(model.first.logWeight());
        }
    }

    for (std::size_t i = 0u; i < weights.size(); ++i) {
        models[i].first *= std::exp(weights[i] - maxWeight[0]) / Z;
    }

    return {boost::make_unique<CMultivariateOneOfNPrior>(2, models, this->dataType(),
                                                         this->decayRate()),
            maxWeight.count() > 0 ? maxWeight[0] : 0.0};
}

TDouble10VecDouble10VecPr CMultivariateOneOfNPrior::marginalLikelihoodSupport() const {

    // We define this is as the intersection of the component model
    // supports.

    TDouble10VecDouble10VecPr result(TDouble10Vec(m_Dimension, MINUS_INF),
                                     TDouble10Vec(m_Dimension, INF));
    TDouble10VecDouble10VecPr modelSupport;
    for (const auto& model : m_Models) {
        if (model.second->participatesInModelSelection()) {
            modelSupport = model.second->marginalLikelihoodSupport();
            result.first = max(result.first, modelSupport.first);
            result.second = min(result.second, modelSupport.second);
        }
    }

    return result;
}

TDouble10Vec CMultivariateOneOfNPrior::marginalLikelihoodMean() const {

    // This is E_{P(i)}[ E[X | P(i)] ] and the conditional expectation
    // is just the individual model expectation. Note we exclude models
    // with low weight because typically the means are similar between
    // models and if they are very different we don't want to include
    // the model if there is strong evidence against it.

    TDouble10Vec result(m_Dimension, 0.0);
    double w = 0.0;
    for (const auto& model : m_Models) {
        double wi = model.first;
        if (wi > MINIMUM_SIGNIFICANT_WEIGHT) {
            updateMean(model.second->marginalLikelihoodMean(), wi, result, w);
        }
    }
    return result;
}

TDouble10Vec
CMultivariateOneOfNPrior::nearestMarginalLikelihoodMean(const TDouble10Vec& value) const {

    // See marginalLikelihoodMean for discussion.

    TDouble10Vec result(m_Dimension, 0.0);
    double w = 0.0;
    for (const auto& model : m_Models) {
        double wi = model.first;
        if (wi > MINIMUM_SIGNIFICANT_WEIGHT) {
            updateMean(model.second->nearestMarginalLikelihoodMean(value), wi, result, w);
        }
    }
    return result;
}

TDouble10Vec10Vec CMultivariateOneOfNPrior::marginalLikelihoodCovariance() const {

    TDouble10Vec10Vec result(m_Dimension, TDouble10Vec(m_Dimension, 0.0));
    if (this->isNonInformative()) {
        for (std::size_t i = 0u; i < m_Dimension; ++i) {
            result[i][i] = INF;
        }
        return result;
    }

    // This is E_{P(i)}[ Cov[X | i] ] and the conditional expectation
    // is just the individual model expectation. Note we exclude models
    // with low weight because typically the variance are similar between
    // models and if they are very different we don't want to include
    // the model if there is strong evidence against it.

    double w = 0.0;
    for (const auto& model : m_Models) {
        double wi = model.first;
        if (wi > MINIMUM_SIGNIFICANT_WEIGHT) {
            updateMean(model.second->marginalLikelihoodCovariance(), wi, result, w);
        }
    }
    return result;
}

TDouble10Vec CMultivariateOneOfNPrior::marginalLikelihoodVariances() const {

    if (this->isNonInformative()) {
        return TDouble10Vec(m_Dimension, INF);
    }

    TDouble10Vec result(m_Dimension, 0.0);
    double w = 0.0;
    for (const auto& model : m_Models) {
        double wi = model.first;
        if (wi > MINIMUM_SIGNIFICANT_WEIGHT) {
            updateMean(model.second->marginalLikelihoodVariances(), wi, result, w);
        }
    }
    return result;
}

TDouble10Vec
CMultivariateOneOfNPrior::marginalLikelihoodMode(const TDouble10VecWeightsAry& weights) const {

    // We approximate this as the weighted average of the component
    // model modes.

    // Declared outside the loop to minimize the number of times
    // it is created.
    TDouble10Vec1Vec sample(1);
    TDouble10VecWeightsAry1Vec sampleWeights(1, weights);

    TDouble10Vec result(m_Dimension, 0.0);
    double w = 0.0;
    for (const auto& model : m_Models) {
        if (model.second->participatesInModelSelection()) {
            sample[0] = model.second->marginalLikelihoodMode(weights);
            double logLikelihood;
            model.second->jointLogMarginalLikelihood(sample, sampleWeights, logLikelihood);
            updateMean(sample[0], model.first * std::exp(logLikelihood), result, w);
        }
    }

    TDouble10VecDouble10VecPr support = this->marginalLikelihoodSupport();
    return CTools::truncate(result, support.first, support.second);
}

maths_t::EFloatingPointErrorStatus
CMultivariateOneOfNPrior::jointLogMarginalLikelihood(const TDouble10Vec1Vec& samples,
                                                     const TDouble10VecWeightsAry1Vec& weights,
                                                     double& result) const {
    result = 0.0;

    if (samples.empty()) {
        LOG_ERROR(<< "Can't compute likelihood for empty sample set");
        return maths_t::E_FpFailed;
    }
    if (!this->check(samples, weights)) {
        return maths_t::E_FpFailed;
    }

    // See COneOfNPrior::jointLogMarginalLikelihood for a discussion.

    // We re-normalize the data so that the maximum likelihood is one
    // to avoid underflow.
    TDouble3Vec logLikelihoods;
    TMaxAccumulator maxLogLikelihood;
    double Z = 0.0;

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

void CMultivariateOneOfNPrior::sampleMarginalLikelihood(std::size_t numberSamples,
                                                        TDouble10Vec1Vec& samples) const {
    samples.clear();

    if (numberSamples == 0 || this->isNonInformative()) {
        return;
    }

    TDouble3Vec weights;
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

    TDouble10VecDouble10VecPr support = this->marginalLikelihoodSupport();
    for (std::size_t i = 0u; i < m_Dimension; ++i) {
        support.first[i] = CTools::shiftRight(support.first[i]);
        support.second[i] = CTools::shiftLeft(support.second[i]);
    }

    samples.reserve(numberSamples);
    TDouble10Vec1Vec modelSamples;
    for (std::size_t i = 0u; i < m_Models.size(); ++i) {
        modelSamples.clear();
        m_Models[i].second->sampleMarginalLikelihood(sampling[i], modelSamples);
        for (const auto& sample : modelSamples) {
            samples.push_back(CTools::truncate(sample, support.first, support.second));
        }
    }
    LOG_TRACE(<< "samples = " << core::CContainerPrinter::print(samples));
}

bool CMultivariateOneOfNPrior::isNonInformative() const {
    for (const auto& model : m_Models) {
        if (model.second->isNonInformative()) {
            return true;
        }
    }
    return false;
}

void CMultivariateOneOfNPrior::print(const std::string& separator, std::string& result) const {
    result += core_t::LINE_ENDING + separator + " one-of-n";
    if (this->isNonInformative()) {
        result += " non-informative";
    }

    result += ':';
    result += core_t::LINE_ENDING + separator + " # samples " +
              core::CStringUtils::typeToStringPretty(this->numberSamples());

    std::string separator_ = separator + separator;

    for (const auto& model : m_Models) {
        double weight = model.first;
        if (weight >= MINIMUM_SIGNIFICANT_WEIGHT) {
            result += core_t::LINE_ENDING + separator_ + " weight " +
                      core::CStringUtils::typeToStringPretty(weight);
            model.second->print(separator_, result);
        }
    }
}

uint64_t CMultivariateOneOfNPrior::checksum(uint64_t seed) const {
    seed = this->CMultivariatePrior::checksum(seed);
    return CChecksum::calculate(seed, m_Models);
}

void CMultivariateOneOfNPrior::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CMultivariateOneOfNPrior");
    core::CMemoryDebug::dynamicSize("m_Models", m_Models, mem);
}

std::size_t CMultivariateOneOfNPrior::memoryUsage() const {
    return core::CMemory::dynamicSize(m_Models);
}

std::size_t CMultivariateOneOfNPrior::staticSize() const {
    return sizeof(*this);
}

std::string CMultivariateOneOfNPrior::persistenceTag() const {
    return ONE_OF_N_TAG + core::CStringUtils::typeToString(m_Dimension);
}

void CMultivariateOneOfNPrior::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    for (const auto& model : m_Models) {
        inserter.insertLevel(MODEL_TAG, boost::bind(&modelAcceptPersistInserter,
                                                    boost::cref(model.first),
                                                    boost::cref(*model.second), _1));
    }
    inserter.insertValue(DECAY_RATE_TAG, this->decayRate(), core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(NUMBER_SAMPLES_TAG, this->numberSamples(),
                         core::CIEEE754::E_SinglePrecision);
}

CMultivariateOneOfNPrior::TDouble3Vec CMultivariateOneOfNPrior::weights() const {
    TDouble3Vec result = this->logWeights();
    for (auto& weight : result) {
        weight = std::exp(weight);
    }
    return result;
}

CMultivariateOneOfNPrior::TDouble3Vec CMultivariateOneOfNPrior::logWeights() const {
    TDouble3Vec result;

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

CMultivariateOneOfNPrior::TPriorCPtr3Vec CMultivariateOneOfNPrior::models() const {
    TPriorCPtr3Vec result;
    for (const auto& model : m_Models) {
        result.push_back(model.second.get());
    }
    return result;
}

bool CMultivariateOneOfNPrior::badWeights() const {
    for (const auto& model : m_Models) {
        if (!CMathsFuncs::isFinite(model.first.logWeight())) {
            return true;
        }
    }
    return false;
}

std::string CMultivariateOneOfNPrior::debugWeights() const {
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

const double CMultivariateOneOfNPrior::MAXIMUM_RELATIVE_ERROR = 1e-3;
const double CMultivariateOneOfNPrior::LOG_MAXIMUM_RELATIVE_ERROR =
    std::log(MAXIMUM_RELATIVE_ERROR);
}
}
