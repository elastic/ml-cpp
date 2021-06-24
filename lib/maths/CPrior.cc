/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CPrior.h>

#include <core/CLogger.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CCompositeFunctions.h>
#include <maths/CEqualWithTolerance.h>
#include <maths/CIntegration.h>
#include <maths/CMathsFuncs.h>
#include <maths/COrderings.h>
#include <maths/CPriorDetail.h>
#include <maths/CSolvers.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace ml {
namespace maths {

namespace {

namespace detail {

//! Set the decay rate, validating the input.
void setDecayRate(double value, double fallback, CFloatStorage& result) {
    if (CMathsFuncs::isFinite(value)) {
        result = value;
    } else {
        LOG_ERROR(<< "Invalid decay rate " << value);
        result = fallback;
    }
}
}

const std::size_t ADJUST_OFFSET_TRIALS{20};
}

CPrior::CPrior()
    : m_DataType(maths_t::E_DiscreteData), m_DecayRate(0.0), m_NumberSamples(0) {
}

CPrior::CPrior(maths_t::EDataType dataType, double decayRate)
    : m_DataType(dataType), m_NumberSamples(0) {
    detail::setDecayRate(decayRate, FALLBACK_DECAY_RATE, m_DecayRate);
}

void CPrior::swap(CPrior& other) noexcept {
    std::swap(m_DataType, other.m_DataType);
    std::swap(m_DecayRate, other.m_DecayRate);
    std::swap(m_NumberSamples, other.m_NumberSamples);
}

bool CPrior::isDiscrete() const {
    return m_DataType == maths_t::E_DiscreteData || m_DataType == maths_t::E_IntegerData;
}

bool CPrior::isInteger() const {
    return m_DataType == maths_t::E_IntegerData;
}

maths_t::EDataType CPrior::dataType() const {
    return m_DataType;
}

void CPrior::dataType(maths_t::EDataType value) {
    m_DataType = value;
}

double CPrior::decayRate() const {
    return m_DecayRate;
}

void CPrior::decayRate(double value) {
    detail::setDecayRate(value, FALLBACK_DECAY_RATE, m_DecayRate);
}

void CPrior::removeModels(CModelFilter& /*filter*/) {
}

double CPrior::offsetMargin() const {
    return 0.0;
}

void CPrior::addSamples(const TDouble1Vec& samples, const TDoubleWeightsAry1Vec& weights) {
    double n = 0.0;
    try {
        for (std::size_t i = 0; i < weights.size(); ++i) {
            double xi = samples[i];
            double ni = maths_t::countForUpdate(weights[i]);
            if (CMathsFuncs::isFinite(xi) && CMathsFuncs::isFinite(ni)) {
                n += ni;
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to extract sample counts: " << e.what());
    }
    this->addSamples(n);
}

double CPrior::nearestMarginalLikelihoodMean(double /*value*/) const {
    return this->marginalLikelihoodMean();
}

CPrior::TDouble1Vec CPrior::marginalLikelihoodModes(const TDoubleWeightsAry& weights) const {
    return TDouble1Vec{this->marginalLikelihoodMode(weights)};
}

std::string CPrior::print() const {
    std::string result;
    this->print("", result);
    return result;
}

std::string CPrior::printMarginalLikelihoodFunction(double weight) const {
    // We'll plot the marginal likelihood function over the range
    // where most of the mass is.

    static const unsigned int POINTS = 501;

    SPlot plot = this->marginalLikelihoodPlot(POINTS, weight);

    std::ostringstream abscissa;
    std::ostringstream likelihood;

    abscissa << "x = [";
    likelihood << "likelihood = [";
    for (std::size_t i = 0; i < plot.s_Abscissa.size(); ++i) {
        abscissa << plot.s_Abscissa[i] << " ";
        likelihood << plot.s_Ordinates[i] << " ";
    }
    abscissa << "];" << core_t::LINE_ENDING;
    likelihood << "];" << core_t::LINE_ENDING << "plot(x, likelihood);";

    return abscissa.str() + likelihood.str();
}

CPrior::SPlot CPrior::marginalLikelihoodPlot(unsigned int numberPoints, double weight) const {
    if (this->isNonInformative()) {
        // The non-informative likelihood is improper 0 everywhere.
        return CPrior::SPlot();
    }

    CPrior::SPlot plot;
    if (numberPoints == 0) {
        return plot;
    }

    plot.s_Abscissa.reserve(numberPoints);
    plot.s_Ordinates.reserve(numberPoints);
    this->sampleMarginalLikelihood(numberPoints, plot.s_Abscissa);
    std::sort(plot.s_Abscissa.begin(), plot.s_Abscissa.end());

    for (auto x : plot.s_Abscissa) {
        double likelihood;
        maths_t::EFloatingPointErrorStatus status =
            this->jointLogMarginalLikelihood({x}, TWeights::SINGLE_UNIT, likelihood);
        if (status & maths_t::E_FpFailed) {
            // Ignore point.
        } else if (status & maths_t::E_FpOverflowed) {
            plot.s_Ordinates.push_back(0.0);
        } else {
            plot.s_Ordinates.push_back(weight * std::exp(likelihood));
        }
    }

    return plot;
}

uint64_t CPrior::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_DataType);
    seed = CChecksum::calculate(seed, m_DecayRate);
    return CChecksum::calculate(seed, m_NumberSamples);
}

double CPrior::numberSamples() const {
    return m_NumberSamples;
}

void CPrior::numberSamples(double numberSamples) {
    m_NumberSamples = numberSamples;
}

bool CPrior::participatesInModelSelection() const {
    return true;
}

double CPrior::unmarginalizedParameters() const {
    return 0.0;
}

void CPrior::adjustOffsetResamples(double minimumSample,
                                   TDouble1Vec& resamples,
                                   TDoubleWeightsAry1Vec& resamplesWeights) const {

    this->sampleMarginalLikelihood(ADJUST_OFFSET_SAMPLE_SIZE, resamples);
    std::size_t n = resamples.size();
    resamples.erase(std::remove_if(resamples.begin(), resamples.end(),
                                   [](double d) {
                                       return CMathsFuncs::SIsFinite()(d) == false;
                                   }),
                    resamples.end());
    if (resamples.size() != n) {
        LOG_ERROR(<< "Bad samples (" << this->debug() << ")");
        n = resamples.size();
    }
    for (std::size_t i = 0; i < n; ++i) {
        resamples[i] = std::max(resamples[i], minimumSample);
    }

    double resamplesWeight = 1.0;
    if (n > 0) {
        resamplesWeight = this->numberSamples() / static_cast<double>(n);
        resamplesWeights.resize(n, maths_t::countWeight(resamplesWeight));
    }
}

double CPrior::adjustOffsetWithCost(const TDouble1Vec& samples,
                                    const TDoubleWeightsAry1Vec& weights,
                                    COffsetCost& cost,
                                    CApplyOffset& apply) {
    if (samples.empty() ||
        CMathsFuncs::beginFinite(samples) == CMathsFuncs::endFinite(samples)) {
        return 0.0;
    }

    // Ideally we'd like to minimize a suitable measure of the difference
    // between the two marginal likelihoods w.r.t. to the new prior parameters.
    // However, even the Kullback-Leibler divergence can't be computed in
    // closed form, so there is no easy way of doing this. We would have
    // to use a non-linear maximization scheme, which computes the divergence
    // numerically to evaluate the objective function, but this will be
    // slow and it would be difficult to analyse its numerical stability.
    // Instead we simply sample marginal likelihood and update the shifted
    // prior with these samples. We search for a global maximum of the log
    // likelihood of these samples w.r.t. the offset.

    double margin = this->offsetMargin();
    double minimumSample = *std::min_element(CMathsFuncs::beginFinite(samples),
                                             CMathsFuncs::endFinite(samples));
    if (minimumSample + this->offset() >= margin) {
        return 0.0;
    }

    static const double EPS = 0.01;

    double offset = margin - minimumSample;
    offset *= (offset < 0.0 ? (1.0 - EPS) : (1.0 + EPS));

    cost.samples(samples, weights);
    cost.resample(minimumSample);
    apply.resample(minimumSample);

    if (this->isNonInformative()) {
        apply(offset);
        return 0.0;
    }

    TDouble1Vec resamples;
    TDoubleWeightsAry1Vec resamplesWeights;
    this->adjustOffsetResamples(minimumSample, resamples, resamplesWeights);

    double before;
    this->jointLogMarginalLikelihood(resamples, resamplesWeights, before);

    double maximumSample = *std::max_element(samples.begin(), samples.end());
    double range = resamples.empty()
                       ? maximumSample - minimumSample
                       : std::max(maximumSample - minimumSample,
                                  resamples[resamples.size() - 1] - resamples[0]);
    double increment = std::max(
        (range - margin) / static_cast<double>(ADJUST_OFFSET_TRIALS - 1), 0.0);

    if (increment > 0.0) {
        TDouble1Vec trialOffsets;
        trialOffsets.reserve(ADJUST_OFFSET_TRIALS);
        for (std::size_t i = 0; i < ADJUST_OFFSET_TRIALS; ++i) {
            offset += increment;
            trialOffsets.push_back(offset);
        }
        double likelihood;
        CSolvers::globalMinimize(trialOffsets, cost, offset, likelihood);
        LOG_TRACE(<< "samples = " << core::CContainerPrinter::print(samples)
                  << ", offset = " << offset << ", likelihood = " << likelihood);
    }

    apply(offset);

    double after;
    this->jointLogMarginalLikelihood(resamples, resamplesWeights, after);
    return std::min(after - before, 0.0);
}

void CPrior::addSamples(double n) {
    m_NumberSamples += n;
}

std::string CPrior::debug() const {
    return std::string();
}

CPrior::TStrStrPr CPrior::printMarginalLikelihoodStatistics() const {
    if (this->isNonInformative()) {
        return {UNKNOWN_VALUE_STRING, UNKNOWN_VALUE_STRING};
    }

    return this->doPrintMarginalLikelihoodStatistics();
}
const double CPrior::FALLBACK_DECAY_RATE = 0.001;
const std::size_t CPrior::ADJUST_OFFSET_SAMPLE_SIZE = 50;
const std::string CPrior::UNKNOWN_VALUE_STRING = "<unknown>";

////////// CPrior::CModelFilter Implementation //////////

CPrior::CModelFilter::CModelFilter() : m_Filter(0) {
}

CPrior::CModelFilter& CPrior::CModelFilter::remove(EPrior model) {
    m_Filter = m_Filter | model;
    return *this;
}

bool CPrior::CModelFilter::operator()(EPrior model) const {
    return (m_Filter & model) != 0;
}

////////// CPrior::CLogMarginalLikelihood Implementation //////////

CPrior::CLogMarginalLikelihood::CLogMarginalLikelihood(const CPrior& prior,
                                                       const TDoubleWeightsAry1Vec& weights)
    : m_Prior(&prior), m_Weights(&weights), m_X(1) {
}

double CPrior::CLogMarginalLikelihood::operator()(double x) const {
    double result;
    if (!this->operator()(x, result)) {
        throw std::runtime_error("Unable to compute likelihood at " +
                                 core::CStringUtils::typeToString(x));
    }
    return result;
}

bool CPrior::CLogMarginalLikelihood::operator()(double x, double& result) const {
    m_X[0] = x;
    return !(m_Prior->jointLogMarginalLikelihood(m_X, *m_Weights, result) & maths_t::E_FpFailed);
}

////////// CPrior::COffsetParameters Implementation //////////

CPrior::COffsetParameters::COffsetParameters(CPrior& prior)
    : m_Prior(&prior), m_Samples(nullptr), m_Weights(nullptr) {
}

void CPrior::COffsetParameters::samples(const TDouble1Vec& samples,
                                        const TDoubleWeightsAry1Vec& weights) {
    m_Samples = &samples;
    m_Weights = &weights;
}

void CPrior::COffsetParameters::resample(double minimumSample) {
    m_Prior->adjustOffsetResamples(minimumSample, m_Resamples, m_ResamplesWeights);
}

CPrior& CPrior::COffsetParameters::prior() const {
    return *m_Prior;
}

const CPrior::TDouble1Vec& CPrior::COffsetParameters::samples() const {
    return *m_Samples;
}

const CPrior::TDoubleWeightsAry1Vec& CPrior::COffsetParameters::weights() const {
    return *m_Weights;
}

const CPrior::TDouble1Vec& CPrior::COffsetParameters::resamples() const {
    return m_Resamples;
}

const CPrior::TDoubleWeightsAry1Vec& CPrior::COffsetParameters::resamplesWeights() const {
    return m_ResamplesWeights;
}

////////// CPrior::COffsetCost Implementation //////////

CPrior::COffsetCost::COffsetCost(CPrior& prior) : COffsetParameters(prior) {
}

double CPrior::COffsetCost::operator()(double offset) const {
    this->resetPriors(offset);
    return this->computeCost(offset);
}

void CPrior::COffsetCost::resetPriors(double offset) const {
    this->prior().setToNonInformative(offset, this->prior().decayRate());
    this->prior().addSamples(this->resamples(), this->resamplesWeights());
    this->prior().addSamples(this->samples(), this->weights());
}

double CPrior::COffsetCost::computeCost(double offset) const {
    double resamplesLogLikelihood = 0.0;
    maths_t::EFloatingPointErrorStatus status;
    if (this->resamples().size() > 0) {
        status = this->prior().jointLogMarginalLikelihood(
            this->resamples(), this->resamplesWeights(), resamplesLogLikelihood);
        if (status != maths_t::E_FpNoErrors) {
            LOG_ERROR(<< "Failed evaluating log-likelihood at " << offset << " for samples "
                      << core::CContainerPrinter::print(this->resamples()) << " and weights "
                      << core::CContainerPrinter::print(this->resamplesWeights()) << ", the prior is "
                      << this->prior().print() << ": status " << status);
        }
    }
    double samplesLogLikelihood;
    status = this->prior().jointLogMarginalLikelihood(
        this->samples(), this->weights(), samplesLogLikelihood);
    if (status != maths_t::E_FpNoErrors) {
        LOG_ERROR(<< "Failed evaluating log-likelihood at " << offset << " for "
                  << core::CContainerPrinter::print(this->samples()) << " and weights "
                  << core::CContainerPrinter::print(this->weights()) << ", the prior is "
                  << this->prior().print() << ": status " << status);
    }
    return -(resamplesLogLikelihood + samplesLogLikelihood);
}

////////// CPrior::CApplyOffset Implementation //////////

CPrior::CApplyOffset::CApplyOffset(CPrior& prior) : COffsetParameters(prior) {
}

void CPrior::CApplyOffset::operator()(double offset) const {
    this->prior().setToNonInformative(offset, this->prior().decayRate());
    this->prior().addSamples(this->resamples(), this->resamplesWeights());
}
}
}
