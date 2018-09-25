/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CModel.h>
#include <maths/CModelDetail.h>

#include <core/CLogger.h>
#include <core/Constants.h>

#include <maths/CTools.h>

#include <boost/range.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace ml {
namespace maths {
namespace {

using TDouble2Vec = core::CSmallVector<double, 2>;

//! Check if all the elements of \p lhs are less than or equal to the \p rhs.
bool lessThanEqual(const TDouble2Vec& lhs, double rhs) {
    return std::find_if(lhs.begin(), lhs.end(),
                        [rhs](double lhs_) { return lhs_ > rhs; }) == lhs.end();
}

//! Check if all the elements of \p lhs are less than or equal to the \p rhs.
bool greaterThanEqual(const TDouble2Vec& lhs, double rhs) {
    return std::find_if(lhs.begin(), lhs.end(),
                        [rhs](double lhs_) { return lhs_ < rhs; }) == lhs.end();
}

//! Get the correction to apply to the one-sided probability calculations.
//!
//! The value of a feature is always zero on an empty bucket therefore
//! if the calculation is one sided below (above) we need to add on twice
//! the probability of zero if the actual feature value is greater (less)
//! than zero.
double oneSidedEmptyBucketCorrection(maths_t::EProbabilityCalculation calculation,
                                     const TDouble2Vec& value,
                                     double probabilityEmptyBucket) {
    switch (calculation) {
    case maths_t::E_OneSidedBelow:
        return greaterThanEqual(value, 0.0) ? 2.0 * probabilityEmptyBucket : 0.0;
    case maths_t::E_OneSidedAbove:
        return lessThanEqual(value, 0.0) ? 2.0 * probabilityEmptyBucket : 0.0;
    case maths_t::E_TwoSided:
        break;
    }
    return 0.0;
}

const std::string EMPTY_STRING;
const double EFFECTIVE_COUNT[]{1.0,  0.8,  0.7,  0.65, 0.6,
                               0.57, 0.54, 0.52, 0.51};

//! Get the parameters for the stub model.
CModelParams stubParameters() {
    return CModelParams{
        0, 1.0, 0.0, 0.0, 6 * core::constants::HOUR, core::constants::DAY};
}
}

//////// CModelParams ////////

CModelParams::CModelParams(core_t::TTime bucketLength,
                           double learnRate,
                           double decayRate,
                           double minimumSeasonalVarianceScale,
                           core_t::TTime minimumTimeToDetectChange,
                           core_t::TTime maximumTimeToTestForChange)
    : m_BucketLength(bucketLength), m_LearnRate(learnRate), m_DecayRate(decayRate),
      m_MinimumSeasonalVarianceScale(minimumSeasonalVarianceScale),
      m_MinimumTimeToDetectChange(std::max(minimumTimeToDetectChange, 6 * bucketLength)),
      m_MaximumTimeToTestForChange(std::max(maximumTimeToTestForChange, 12 * bucketLength)),
      m_ProbabilityBucketEmpty(0.0) {
}

core_t::TTime CModelParams::bucketLength() const {
    return m_BucketLength;
}

double CModelParams::learnRate() const {
    return m_LearnRate;
}

double CModelParams::decayRate() const {
    return m_DecayRate;
}

double CModelParams::averagingDecayRate() const {
    return 5.0 * m_DecayRate;
}

double CModelParams::minimumSeasonalVarianceScale() const {
    return m_MinimumSeasonalVarianceScale;
}

bool CModelParams::testForChange(core_t::TTime changeInterval) const {
    return changeInterval >= std::max(3 * m_BucketLength, core::constants::HOUR);
}

core_t::TTime CModelParams::minimumTimeToDetectChange() const {
    return m_MinimumTimeToDetectChange;
}

core_t::TTime CModelParams::maximumTimeToTestForChange() const {
    return m_MaximumTimeToTestForChange;
}

void CModelParams::probabilityBucketEmpty(double probability) {
    m_ProbabilityBucketEmpty = probability;
}

double CModelParams::probabilityBucketEmpty() const {
    return m_ProbabilityBucketEmpty;
}

//////// CModelAddSamplesParams ////////

CModelAddSamplesParams& CModelAddSamplesParams::integer(bool integer) {
    m_Type = integer ? maths_t::E_IntegerData : maths_t::E_ContinuousData;
    return *this;
}

maths_t::EDataType CModelAddSamplesParams::type() const {
    return m_Type;
}

CModelAddSamplesParams& CModelAddSamplesParams::nonNegative(bool nonNegative) {
    m_IsNonNegative = nonNegative;
    return *this;
}

bool CModelAddSamplesParams::isNonNegative() const {
    return m_IsNonNegative;
}

CModelAddSamplesParams& CModelAddSamplesParams::propagationInterval(double interval) {
    m_PropagationInterval = interval;
    return *this;
}

double CModelAddSamplesParams::propagationInterval() const {
    return m_PropagationInterval;
}

CModelAddSamplesParams&
CModelAddSamplesParams::trendWeights(const TDouble2VecWeightsAryVec& weights) {
    m_TrendWeights = &weights;
    return *this;
}

const CModelAddSamplesParams::TDouble2VecWeightsAryVec&
CModelAddSamplesParams::trendWeights() const {
    return *m_TrendWeights;
}

CModelAddSamplesParams&
CModelAddSamplesParams::priorWeights(const TDouble2VecWeightsAryVec& weights) {
    m_PriorWeights = &weights;
    return *this;
}

const CModelAddSamplesParams::TDouble2VecWeightsAryVec&
CModelAddSamplesParams::priorWeights() const {
    return *m_PriorWeights;
}

//////// CModelProbabilityParams ////////

CModelProbabilityParams::CModelProbabilityParams()
    : m_SeasonalConfidenceInterval{DEFAULT_SEASONAL_CONFIDENCE_INTERVAL} {
}

CModelProbabilityParams&
CModelProbabilityParams::addCalculation(maths_t::EProbabilityCalculation calculation) {
    m_Calculations.push_back(calculation);
    return *this;
}

std::size_t CModelProbabilityParams::calculations() const {
    return m_Calculations.size();
}

maths_t::EProbabilityCalculation CModelProbabilityParams::calculation(std::size_t i) const {
    return m_Calculations.size() == 1 ? m_Calculations[0] : m_Calculations[i];
}

CModelProbabilityParams& CModelProbabilityParams::seasonalConfidenceInterval(double confidence) {
    m_SeasonalConfidenceInterval = confidence;
    return *this;
}

double CModelProbabilityParams::seasonalConfidenceInterval() const {
    return m_SeasonalConfidenceInterval;
}

CModelProbabilityParams& CModelProbabilityParams::addBucketEmpty(const TBool2Vec& empty) {
    m_BucketEmpty.push_back(empty);
    return *this;
}

CModelProbabilityParams& CModelProbabilityParams::bucketEmpty(const TBool2Vec1Vec& empty) {
    m_BucketEmpty = empty;
    return *this;
}

const CModelProbabilityParams::TBool2Vec1Vec& CModelProbabilityParams::bucketEmpty() const {
    return m_BucketEmpty;
}

CModelProbabilityParams&
CModelProbabilityParams::addWeights(const TDouble2VecWeightsAry& weights) {
    m_Weights.push_back(weights);
    return *this;
}

CModelProbabilityParams&
CModelProbabilityParams::weights(const TDouble2VecWeightsAry1Vec& weights) {
    m_Weights = weights;
    return *this;
}

const CModelProbabilityParams::TDouble2VecWeightsAry1Vec&
CModelProbabilityParams::weights() const {
    return m_Weights;
}

CModelProbabilityParams::TDouble2VecWeightsAry1Vec& CModelProbabilityParams::weights() {
    return m_Weights;
}

CModelProbabilityParams& CModelProbabilityParams::addCoordinate(std::size_t coordinate) {
    m_Coordinates.push_back(coordinate);
    return *this;
}

const CModelProbabilityParams::TSize2Vec& CModelProbabilityParams::coordinates() const {
    return m_Coordinates;
}

CModelProbabilityParams& CModelProbabilityParams::mostAnomalousCorrelate(std::size_t correlate) {
    m_MostAnomalousCorrelate.reset(correlate);
    return *this;
}

CModelProbabilityParams::TOptionalSize CModelProbabilityParams::mostAnomalousCorrelate() const {
    return m_MostAnomalousCorrelate;
}

CModelProbabilityParams& CModelProbabilityParams::useMultibucketFeatures(bool use) {
    m_UseMultibucketFeatures = use;
    return *this;
}

bool CModelProbabilityParams::useMultibucketFeatures() const {
    return m_UseMultibucketFeatures;
}

CModelProbabilityParams& CModelProbabilityParams::useAnomalyModel(bool use) {
    m_UseAnomalyModel = use;
    return *this;
}

bool CModelProbabilityParams::useAnomalyModel() const {
    return m_UseAnomalyModel;
}

//////// SModelProbabilityResult::SFeatureProbability ////////

SModelProbabilityResult::SFeatureProbability::SFeatureProbability()
    : s_Label{boost::cref(EMPTY_STRING)} {
}

SModelProbabilityResult::SFeatureProbability::SFeatureProbability(const std::string& label,
                                                                  double probability)
    : s_Label{boost::cref(label)}, s_Probability{probability} {
}

//////// CModel ////////

CModel::EUpdateResult CModel::combine(EUpdateResult lhs, EUpdateResult rhs) {
    switch (lhs) {
    case E_Success:
        return rhs;
    case E_Reset:
        return rhs == E_Failure ? E_Failure : E_Reset;
    case E_Failure:
        return E_Failure;
    }
    return E_Failure;
}

CModel::CModel(const CModelParams& params) : m_Params(params) {
}

double CModel::effectiveCount(std::size_t n) {
    return n <= boost::size(EFFECTIVE_COUNT) ? EFFECTIVE_COUNT[n - 1] : 0.5;
}

const CModelParams& CModel::params() const {
    return m_Params;
}

CModelParams& CModel::params() {
    return m_Params;
}

double CModel::correctForEmptyBucket(maths_t::EProbabilityCalculation calculation,
                                     const TDouble2Vec& value,
                                     bool bucketEmpty,
                                     double probabilityBucketEmpty,
                                     double probability) {
    double pCorrected{(1.0 - probabilityBucketEmpty) * probability};

    if (!bucketEmpty) {
        double pOneSided{oneSidedEmptyBucketCorrection(calculation, value, probabilityBucketEmpty)};
        return std::min(pOneSided + pCorrected, 1.0);
    }

    return probabilityBucketEmpty + pCorrected;
}

double CModel::correctForEmptyBucket(maths_t::EProbabilityCalculation calculation,
                                     double value,
                                     const TBool2Vec& bucketEmpty,
                                     const TDouble2Vec& probabilityEmptyBucket,
                                     double probability) {
    if (!bucketEmpty[0] && !bucketEmpty[1]) {
        double pState{(1.0 - probabilityEmptyBucket[0]) *
                      (1.0 - probabilityEmptyBucket[1])};
        double pOneSided{oneSidedEmptyBucketCorrection(calculation, {value}, 1.0 - pState)};
        return std::min(pOneSided + pState * probability, 1.0);
    }

    if (!bucketEmpty[0]) {
        double pState{(1.0 - probabilityEmptyBucket[0]) * probabilityEmptyBucket[1]};
        double pOneSided{oneSidedEmptyBucketCorrection(calculation, {value},
                                                       probabilityEmptyBucket[0])};
        return std::min(pOneSided + pState + (1.0 - pState) * probability, 1.0);
    }

    if (!bucketEmpty[1]) {
        double pState{probabilityEmptyBucket[0] * (1.0 - probabilityEmptyBucket[1])};
        double pOneSided{oneSidedEmptyBucketCorrection(calculation, {value},
                                                       probabilityEmptyBucket[1])};
        return std::min(pOneSided + pState + (1.0 - pState) * probability, 1.0);
    }

    double pState = probabilityEmptyBucket[0] * probabilityEmptyBucket[1];
    return pState + (1.0 - pState) * probability;
}

//////// CModelStub ////////

CModelStub::CModelStub() : CModel(stubParameters()) {
}

std::size_t CModelStub::identifier() const {
    return 0;
}

CModelStub* CModelStub::clone(std::size_t /*id*/) const {
    return new CModelStub(*this);
}

CModelStub* CModelStub::cloneForPersistence() const {
    return new CModelStub(*this);
}

CModelStub* CModelStub::cloneForForecast() const {
    return new CModelStub(*this);
}

bool CModelStub::isForecastPossible() const {
    return false;
}

void CModelStub::modelCorrelations(CTimeSeriesCorrelations& /*model*/) {
}

CModelStub::TSize2Vec1Vec CModelStub::correlates() const {
    return {};
}

CModelStub::TDouble2Vec CModelStub::mode(core_t::TTime /*time*/,
                                         const TDouble2VecWeightsAry& /*weights*/) const {
    return {};
}

CModelStub::TDouble2Vec1Vec
CModelStub::correlateModes(core_t::TTime /*time*/,
                           const TDouble2VecWeightsAry1Vec& /*weights*/) const {
    return {};
}

CModelStub::TDouble2Vec1Vec
CModelStub::residualModes(const TDouble2VecWeightsAry& /*weights*/) const {
    return {};
}

void CModelStub::addBucketValue(const TTimeDouble2VecSizeTrVec& /*value*/) {
}

CModelStub::EUpdateResult CModelStub::addSamples(const CModelAddSamplesParams& /*params*/,
                                                 TTimeDouble2VecSizeTrVec /*samples*/) {
    return E_Success;
}

void CModelStub::skipTime(core_t::TTime /*gap*/) {
}

void CModelStub::detrend(const TTime2Vec1Vec& /*time*/,
                         double /*confidenceInterval*/,
                         TDouble2Vec1Vec& /*value*/) const {
}

CModelStub::TDouble2Vec CModelStub::predict(core_t::TTime /*time*/,
                                            const TSizeDoublePr1Vec& /*correlated*/,
                                            TDouble2Vec /*hint*/) const {
    return {};
}

CModelStub::TDouble2Vec3Vec
CModelStub::confidenceInterval(core_t::TTime /*time*/,
                               double /*confidenceInterval*/,
                               const TDouble2VecWeightsAry& /*weights*/) const {
    return {};
}

bool CModelStub::forecast(core_t::TTime /*firstDataTime*/,
                          core_t::TTime /*lastDataTime*/,
                          core_t::TTime /*startTime*/,
                          core_t::TTime /*endTime*/,
                          double /*confidenceInterval*/,
                          const TDouble2Vec& /*minimum*/,
                          const TDouble2Vec& /*maximum*/,
                          const TForecastPushDatapointFunc& /*forecastPushDataPointFunc*/,
                          std::string& /*messageOut*/) {
    return true;
}

bool CModelStub::probability(const CModelProbabilityParams& /*params*/,
                             const TTime2Vec1Vec& /*time*/,
                             const TDouble2Vec1Vec& /*value*/,
                             SModelProbabilityResult& result) const {
    result = SModelProbabilityResult{};
    return true;
}

CModelStub::TDouble2Vec CModelStub::winsorisationWeight(double /*derate*/,
                                                        core_t::TTime /*time*/,
                                                        const TDouble2Vec& /*value*/) const {
    return {};
}

CModelStub::TDouble2Vec CModelStub::seasonalWeight(double /*confidence*/,
                                                   core_t::TTime /*time*/) const {
    return {};
}

std::uint64_t CModelStub::checksum(std::uint64_t seed) const {
    return seed;
}

void CModelStub::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr /*mem*/) const {
}

std::size_t CModelStub::memoryUsage() const {
    return 0;
}

void CModelStub::acceptPersistInserter(core::CStatePersistInserter& /*inserter*/) const {
}

maths_t::EDataType CModelStub::dataType() const {
    return maths_t::E_MixedData;
}
}
}
