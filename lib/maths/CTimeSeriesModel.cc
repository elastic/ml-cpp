/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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

#include <maths/CTimeSeriesModel.h>

#include <core/CAllocationStrategy.h>
#include <core/CFunctional.h>
#include <core/CPersistUtils.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatisticsPersist.h>
#include <maths/CDecayRateController.h>
#include <maths/CModelDetail.h>
#include <maths/CMultivariatePrior.h>
#include <maths/CMultivariateNormalConjugate.h>
#include <maths/Constants.h>
#include <maths/COrderings.h>
#include <maths/CPrior.h>
#include <maths/CPriorStateSerialiser.h>
#include <maths/CRestoreParams.h>
#include <maths/CTimeSeriesDecomposition.h>
#include <maths/CTimeSeriesDecompositionStateSerialiser.h>
#include <maths/CTools.h>

#include <boost/make_shared.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>

namespace ml {
namespace maths {
namespace {
using TDoubleDoublePr = std::pair<double, double>;
using TDouble2Vec = core::CSmallVector<double, 2>;
using TDouble10Vec = core::CSmallVector<double, 10>;
using TDouble10Vec1Vec = core::CSmallVector<TDouble10Vec, 1>;
using TDouble10Vec2Vec = core::CSmallVector<TDouble10Vec, 2>;
using TDouble10Vec4Vec = core::CSmallVector<TDouble10Vec, 4>;
using TDouble10Vec4Vec1Vec = core::CSmallVector<TDouble10Vec4Vec, 1>;
using TSizeVec = std::vector<std::size_t>;
using TSize10Vec = core::CSmallVector<std::size_t, 10>;
using TSizeDoublePr = std::pair<std::size_t, double>;
using TSizeDoublePr10Vec = core::CSmallVector<TSizeDoublePr, 10>;
using TTail10Vec = core::CSmallVector<maths_t::ETail, 10>;
using TTime1Vec = CTimeSeriesCorrelations::TTime1Vec;
using TDouble1Vec = CTimeSeriesCorrelations::TDouble1Vec;
using TDouble4Vec = CTimeSeriesCorrelations::TDouble4Vec;
using TDouble4Vec1Vec = CTimeSeriesCorrelations::TDouble4Vec1Vec;
using TSize1Vec = CTimeSeriesCorrelations::TSize1Vec;
using TSize2Vec1Vec = CTimeSeriesCorrelations::TSize2Vec1Vec;
using TMultivariatePriorCPtrSizePr1Vec = CTimeSeriesCorrelations::TMultivariatePriorCPtrSizePr1Vec;

//! Computes the Winsorisation weight for \p value.
double computeWinsorisationWeight(const CPrior &prior, double derate, double scale, double value) {
    static const double WINSORISED_FRACTION = 1e-4;
    static const double MINIMUM_WEIGHT_FRACTION = 1e-12;
    static const double MINIMUM_WEIGHT = 0.05;
    static const double MINUS_LOG_TOLERANCE = -std::log(1.0 - 100.0 * std::numeric_limits<double>::epsilon());

    double deratedMinimumWeight =  MINIMUM_WEIGHT
                                  + (0.5 - MINIMUM_WEIGHT)
                                  * CTools::truncate(derate, 0.0, 1.0);

    double lowerBound;
    double upperBound;
    if (!prior.minusLogJointCdf(CConstantWeights::SEASONAL_VARIANCE,
                                {value}, {{scale}}, lowerBound, upperBound)) {
        return 1.0;
    }
    if (   upperBound < MINUS_LOG_TOLERANCE &&
           !prior.minusLogJointCdfComplement(CConstantWeights::SEASONAL_VARIANCE,
                                             {value}, {{scale}}, lowerBound, upperBound)) {
        return 1.0;
    }

    double f = std::exp(-(lowerBound + upperBound) / 2.0);
    f = std::min(f, 1.0 - f);
    if (f >= WINSORISED_FRACTION) {
        return 1.0;
    }
    if (f <= MINIMUM_WEIGHT_FRACTION) {
        return deratedMinimumWeight;
    }

    // We interpolate between 1.0 and the minimum weight on the
    // interval [WINSORISED_FRACTION, MINIMUM_WEIGHT_FRACTION]
    // by fitting (f / WF)^(-c log(f)) where WF is the Winsorised
    // fraction and c is determined by solving:
    //   MW = (MWF / WF)^(-c log(MWF))

    static const double EXPONENT = -std::log(MINIMUM_WEIGHT)
                                   / std::log(MINIMUM_WEIGHT_FRACTION)
                                   / std::log(MINIMUM_WEIGHT_FRACTION / WINSORISED_FRACTION);
    static const double LOG_WINSORISED_FRACTION = std::log(WINSORISED_FRACTION);

    double deratedExponent = EXPONENT;
    if (deratedMinimumWeight != MINIMUM_WEIGHT) {
        deratedExponent =  -std::log(deratedMinimumWeight)
                          / std::log(MINIMUM_WEIGHT_FRACTION)
                          / std::log(MINIMUM_WEIGHT_FRACTION / WINSORISED_FRACTION);
    }

    double logf = std::log(f);
    double result = std::exp(-deratedExponent * logf * (logf - LOG_WINSORISED_FRACTION));

    if (CMathsFuncs::isNan(result)) {
        return 1.0;
    }

    LOG_TRACE("sample = " << value << " min(F, 1-F) = " << f << ", weight = " << result);

    return result;
}

//! Computes the Winsorisation weight for \p value.
double computeWinsorisationWeight(const CMultivariatePrior &prior,
                                  std::size_t dimension,
                                  double derate,
                                  double scale,
                                  const TDouble10Vec &value) {
    static const TSize10Vec MARGINALIZE;

    std::size_t d = prior.dimension();

    TSizeDoublePr10Vec condition(d - 1);
    for (std::size_t i = 0u, j = 0u; i < d; ++i) {
        if (i != dimension) {
            condition[j++] = std::make_pair(i, value[i]);
        }
    }

    boost::shared_ptr<CPrior> conditional(prior.univariate(MARGINALIZE, condition).first);
    return computeWinsorisationWeight(*conditional, derate, scale, value[dimension]);
}

//! The decay rate controllers we maintain.
enum EDecayRateController {
    E_TrendControl = 0,
    E_PriorControl,
    E_NumberControls
};

// Models

// Version 6.3
const std::string VERSION_6_3_TAG("6.3");
const std::string ID_6_3_TAG{"a"};
const std::string IS_NON_NEGATIVE_6_3_TAG{"b"};
const std::string IS_FORECASTABLE_6_3_TAG{"c"};
const std::string RNG_6_3_TAG{"d"};
const std::string CONTROLLER_6_3_TAG{"e"};
const std::string TREND_6_3_TAG{"f"};
const std::string PRIOR_6_3_TAG{"g"};
const std::string ANOMALY_MODEL_6_3_TAG{"h"};
const std::string SLIDING_WINDOW_6_3_TAG{"i"};
// Version < 6.3
const std::string ID_OLD_TAG{"a"};
const std::string CONTROLLER_OLD_TAG{"b"};
const std::string TREND_OLD_TAG{"c"};
const std::string PRIOR_OLD_TAG{"d"};
const std::string ANOMALY_MODEL_OLD_TAG{"e"};
const std::string IS_NON_NEGATIVE_OLD_TAG{"g"};
const std::string IS_FORECASTABLE_OLD_TAG{"h"};

// Anomaly model
const std::string MEAN_ERROR_TAG{"a"};
const std::string ANOMALIES_TAG{"b"};
const std::string PRIOR_TAG{"d"};
// Anomaly model nested
const std::string TAG_TAG{"a"};
const std::string OPEN_TIME_TAG{"b"};
const std::string SIGN_TAG{"c"};
const std::string MEAN_ERROR_NORM_TAG{"d"};

// Correlations
const std::string K_MOST_CORRELATED_TAG{"a"};
const std::string CORRELATED_LOOKUP_TAG{"b"};
const std::string CORRELATED_PRIORS_TAG{"c"};
// Correlations nested
const std::string FIRST_CORRELATE_ID_TAG{"a"};
const std::string SECOND_CORRELATE_ID_TAG{"b"};
const std::string CORRELATE_PRIOR_TAG{"c"};
const std::string CORRELATION_TAG{"d"};

const std::size_t MAXIMUM_CORRELATIONS{5000};
const double MINIMUM_CORRELATE_PRIOR_SAMPLE_COUNT{24.0};
const std::size_t SLIDING_WINDOW_SIZE{12};
const TSize10Vec NOTHING_TO_MARGINALIZE;
const TSizeDoublePr10Vec NOTHING_TO_CONDITION;

namespace forecast {
const std::string INFO_INSUFFICIENT_HISTORY("Insufficient history to forecast");
const std::string ERROR_MULTIVARIATE("Forecast not supported for multivariate features");
}
}

//! \brief A model of anomalous sections of a time series.
class CTimeSeriesAnomalyModel {
    public:
        CTimeSeriesAnomalyModel(void);
        CTimeSeriesAnomalyModel(core_t::TTime bucketLength, double decayRate);

        //! Update the anomaly with prediction error and probability.
        //!
        //! This extends the current anomaly if \p probability is small.
        //! Otherwise it closes it.
        void updateAnomaly(const CModelProbabilityParams &params,
                           core_t::TTime time, TDouble2Vec errors, double probability);

        //! If the time series is currently anomalous, update the model
        //! with the anomaly feature vector.
        void sampleAnomaly(const CModelProbabilityParams &params, core_t::TTime time);

        //! Reset the mean error norm.
        void reset(void);

        //! If the time series is currently anomalous, compute the anomalousness
        //! of the anomaly feature vector.
        void probability(const CModelProbabilityParams &params, core_t::TTime time, double &probability) const;

        //! Age the model to account for \p time elapsed time.
        void propagateForwardsByTime(double time);

        //! Compute a checksum for this object.
        uint64_t checksum(uint64_t seed) const;

        //! Debug the memory used by this object.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this object.
        std::size_t memoryUsage(void) const;

        //! Initialize reading state from \p traverser.
        bool acceptRestoreTraverser(const SModelRestoreParams &params,
                                    core::CStateRestoreTraverser &traverser);

        //! Persist by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

    private:
        using TDouble10Vec = core::CSmallVector<double, 10>;
        using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
        using TMaxAccumulator = CBasicStatistics::SMax<double>::TAccumulator;
        using TMultivariateNormalConjugate = CMultivariateNormalConjugate<2>;
        using TMultivariateNormalConjugateVec = std::vector<TMultivariateNormalConjugate>;

        //! \brief Extracts features related to anomalous time periods.
        class CAnomaly {
            public:
                //! See core::CMemory.
                static bool dynamicSizeAlwaysZero(void) { return true; }

            public:
                CAnomaly(void) : m_Tag(0), m_OpenTime(0), m_Sign(0.0) {}
                CAnomaly(std::size_t tag, core_t::TTime time) :
                    m_Tag(tag), m_OpenTime(time), m_Sign(0.0)
                {}

                //! Get the anomaly tag.
                std::size_t tag(void) const { return m_Tag; }

                //! Add a result to the anomaly.
                void update(const TDouble2Vec &errors) {
                    double norm{0.0};
                    for (const auto &error : errors) {
                        norm   += std::pow(error, 2.0);
                        m_Sign += error;
                    }
                    m_MeanErrorNorm.add(std::sqrt(norm));
                }

                //! Get the weight to apply to this anomaly on update.
                double weight(core_t::TTime time) const {
                    return 1.0 / (1.0 + std::max(static_cast<double>(time - m_OpenTime), 0.0));
                }

                //! Check if this anomaly is positive or negative.
                bool positive(void) const { return m_Sign > 0.0; }

                //! Get the feature vector for this anomaly.
                TDouble10Vec features(core_t::TTime time) const {
                    return {static_cast<double>(time - m_OpenTime),
                            CBasicStatistics::mean(m_MeanErrorNorm)};
                }

                //! Compute a checksum for this object.
                uint64_t checksum(uint64_t seed) const {
                    seed = CChecksum::calculate(seed, m_Tag);
                    seed = CChecksum::calculate(seed, m_OpenTime);
                    seed = CChecksum::calculate(seed, m_Sign);
                    return CChecksum::calculate(seed, m_MeanErrorNorm);
                }

                //! Initialize reading state from \p traverser.
                bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser) {
                    do {
                        const std::string &name{traverser.name()};
                        RESTORE_BUILT_IN(TAG_TAG, m_Tag)
                        RESTORE_BUILT_IN(OPEN_TIME_TAG, m_OpenTime)
                        RESTORE_BUILT_IN(SIGN_TAG, m_Sign)
                        RESTORE(MEAN_ERROR_NORM_TAG, m_MeanErrorNorm.fromDelimited(traverser.value()))
                    } while (traverser.next());
                    return true;
                }

                //! Persist by passing information to \p inserter.
                void acceptPersistInserter(core::CStatePersistInserter &inserter) const {
                    inserter.insertValue(TAG_TAG, m_Tag);
                    inserter.insertValue(OPEN_TIME_TAG, m_OpenTime);
                    inserter.insertValue(SIGN_TAG, m_Sign, core::CIEEE754::E_SinglePrecision);
                    inserter.insertValue(MEAN_ERROR_NORM_TAG, m_MeanErrorNorm.toDelimited());
                }

            private:
                //! The anomaly tag.
                std::size_t m_Tag;

                //! The time at which the anomaly started.
                core_t::TTime m_OpenTime;

                //! The anomaly sign, i.e. is the mean error positive or negative.
                double m_Sign;

                //! The mean deviation from predictions.
                TMeanAccumulator m_MeanErrorNorm;
        };
        using TAnomaly1Vec = core::CSmallVector<CAnomaly, 1>;

    private:
        //! The largest anomalous probability.
        static const double LARGEST_ANOMALOUS_PROBABILITY;
        //! The log of the largest anomalous probability.
        static const double LOG_LARGEST_ANOMALOUS_PROBABILITY;
        //! The log of the largest probability that it is deemed
        //! significantly anomalous.
        static const double LOG_SMALL_PROBABILITY;
        //! A unit weight.
        static const TDouble10Vec4Vec1Vec UNIT;

    private:
        //! Update the appropriate anomaly model with \p anomaly.
        void sample(core_t::TTime time, const CAnomaly &anomaly, double weight) {
            std::size_t index(anomaly.positive() ? 0 : 1);
            TDouble10Vec1Vec features{anomaly.features(this->scale(time))};
            m_Priors[index].addSamples(CConstantWeights::COUNT, features,
                                       {{TDouble10Vec(2, weight)}});
        }

        //! Get the scaled time.
        core_t::TTime scale(core_t::TTime time) const { return time / m_BucketLength; }

    private:
        //! The data bucketing interval.
        core_t::TTime m_BucketLength;

        //! The mean prediction error.
        TMeanAccumulator m_MeanError;

        //! The current anomalies (if there are any).
        TAnomaly1Vec m_Anomalies;

        //! The model describing features of anomalous time periods.
        TMultivariateNormalConjugateVec m_Priors;
};

CTimeSeriesAnomalyModel::CTimeSeriesAnomalyModel(void) : m_BucketLength(0) {
    m_Priors.reserve(2);
    m_Priors.push_back(TMultivariateNormalConjugate::nonInformativePrior(maths_t::E_ContinuousData));
    m_Priors.push_back(TMultivariateNormalConjugate::nonInformativePrior(maths_t::E_ContinuousData));
}

CTimeSeriesAnomalyModel::CTimeSeriesAnomalyModel(core_t::TTime bucketLength, double decayRate) :
    m_BucketLength(bucketLength) {
    m_Priors.reserve(2);
    m_Priors.push_back(TMultivariateNormalConjugate::nonInformativePrior(maths_t::E_ContinuousData,
                                                                         0.5 * LARGEST_ANOMALOUS_PROBABILITY * decayRate));
    m_Priors.push_back(TMultivariateNormalConjugate::nonInformativePrior(maths_t::E_ContinuousData,
                                                                         0.5 * LARGEST_ANOMALOUS_PROBABILITY * decayRate));
}

void CTimeSeriesAnomalyModel::updateAnomaly(const CModelProbabilityParams &params,
                                            core_t::TTime time,
                                            TDouble2Vec errors,
                                            double probability) {
    if (params.updateAnomalyModel()) {
        std::size_t tag{params.tag()};
        auto anomaly = std::find_if(m_Anomalies.begin(), m_Anomalies.end(),
                                    [tag] (const CAnomaly &anomaly_) { return anomaly_.tag() == tag; });

        if (probability < LARGEST_ANOMALOUS_PROBABILITY) {
            m_MeanError.add(std::sqrt(std::accumulate(errors.begin(), errors.end(), 0.0,
                                                      [] (double n, double x) { return n + x * x; })));

            double scale{CBasicStatistics::mean(m_MeanError)};
            for (auto &error : errors) {
                error = scale == 0.0 ? 1.0 : error / scale;
            }

            if (anomaly == m_Anomalies.end()) {
                m_Anomalies.emplace_back(tag, this->scale(time));
                anomaly = m_Anomalies.end() - 1;
            }
            anomaly->update(errors);
        } else if (anomaly != m_Anomalies.end())   {
            this->sample(time, *anomaly, 1.0 - anomaly->weight(this->scale(time)));
            m_Anomalies.erase(anomaly);
        }
    }
}

void CTimeSeriesAnomalyModel::sampleAnomaly(const CModelProbabilityParams &params, core_t::TTime time) {
    if (params.updateAnomalyModel()) {
        std::size_t tag{params.tag()};
        auto anomaly = std::find_if(m_Anomalies.begin(), m_Anomalies.end(),
                                    [tag] (const CAnomaly &anomaly_) { return anomaly_.tag() == tag; });
        if (anomaly != m_Anomalies.end()) {
            this->sample(time, *anomaly, anomaly->weight(this->scale(time)));
        }
    }
}

void CTimeSeriesAnomalyModel::reset(void) {
    m_MeanError = TMeanAccumulator();
    for (auto &prior : m_Priors) {
        prior = TMultivariateNormalConjugate::nonInformativePrior(maths_t::E_ContinuousData, prior.decayRate());
    }
}

void CTimeSeriesAnomalyModel::probability(const CModelProbabilityParams &params,
                                          core_t::TTime time, double &probability) const {
    std::size_t tag{params.tag()};
    auto anomaly = std::find_if(m_Anomalies.begin(), m_Anomalies.end(),
                                [tag] (const CAnomaly &anomaly_) { return anomaly_.tag() == tag; });
    if (anomaly != m_Anomalies.end()) {
        std::size_t index(anomaly->positive() ? 0 : 1);
        TDouble10Vec1Vec features{anomaly->features(this->scale(time))};
        double pl, pu;
        TTail10Vec tail;
        if (   probability < LARGEST_ANOMALOUS_PROBABILITY &&
               !m_Priors[index].isNonInformative() &&
               m_Priors[index].probabilityOfLessLikelySamples(maths_t::E_OneSidedAbove,
                                                              CConstantWeights::COUNT, features, UNIT,
                                                              pl, pu, tail)) {
            double logp{CTools::fastLog(probability)};
            double alpha{0.5 * std::min(  (logp - LOG_LARGEST_ANOMALOUS_PROBABILITY)
                                          / (LOG_SMALL_PROBABILITY - LOG_LARGEST_ANOMALOUS_PROBABILITY), 1.0)};
            double pGivenAnomalous{(pl + pu) / 2.0};
            double pScore{CTools::deviation(probability)};
            double pScoreGivenAnomalous{CTools::deviation(pGivenAnomalous)};
            LOG_TRACE("features = " << features
                      << " score(.) = " << pScore
                      << " score(.|anomalous) = " << pScoreGivenAnomalous
                      << " p = " << probability);
            probability = std::min(CTools::inverseDeviation( (1.0 - alpha) * pScore
                                                             + alpha * pScoreGivenAnomalous),
                                   LARGEST_ANOMALOUS_PROBABILITY);
        }
    }
}

void CTimeSeriesAnomalyModel::propagateForwardsByTime(double time) {
    m_Priors[0].propagateForwardsByTime(time);
    m_Priors[1].propagateForwardsByTime(time);
}

uint64_t CTimeSeriesAnomalyModel::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_BucketLength);
    seed = CChecksum::calculate(seed, m_MeanError);
    seed = CChecksum::calculate(seed, m_Anomalies);
    seed = CChecksum::calculate(seed, m_Priors[0]);
    return CChecksum::calculate(seed, m_Priors[1]);
}

void CTimeSeriesAnomalyModel::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CTimeSeriesAnomalyModel");
    core::CMemoryDebug::dynamicSize("m_Anomalies", m_Anomalies, mem);
    core::CMemoryDebug::dynamicSize("m_Priors", m_Priors, mem);
}

std::size_t CTimeSeriesAnomalyModel::memoryUsage(void) const {
    return core::CMemory::dynamicSize(m_Anomalies)
           + core::CMemory::dynamicSize(m_Priors);
}

bool CTimeSeriesAnomalyModel::acceptRestoreTraverser(const SModelRestoreParams &params,
                                                     core::CStateRestoreTraverser &traverser) {
    m_BucketLength = boost::unwrap_ref(params.s_Params).bucketLength();
    std::size_t index{0};
    do {
        const std::string &name{traverser.name()};
        RESTORE(MEAN_ERROR_TAG, m_MeanError.fromDelimited(traverser.value()));
        RESTORE(ANOMALIES_TAG, core::CPersistUtils::restore(ANOMALIES_TAG, m_Anomalies, traverser));
        RESTORE(PRIOR_TAG, traverser.traverseSubLevel(
                    boost::bind(&TMultivariateNormalConjugate::acceptRestoreTraverser,
                                &m_Priors[index++], _1)))
    } while (traverser.next());
    return true;
}

void CTimeSeriesAnomalyModel::acceptPersistInserter(core::CStatePersistInserter &inserter) const {
    inserter.insertValue(MEAN_ERROR_TAG, m_MeanError.toDelimited());
    core::CPersistUtils::persist(ANOMALIES_TAG, m_Anomalies, inserter);
    inserter.insertLevel(PRIOR_TAG, boost::bind(&TMultivariateNormalConjugate::acceptPersistInserter, &m_Priors[0], _1));
    inserter.insertLevel(PRIOR_TAG, boost::bind(&TMultivariateNormalConjugate::acceptPersistInserter, &m_Priors[1], _1));
}

const double CTimeSeriesAnomalyModel::LARGEST_ANOMALOUS_PROBABILITY{0.1};
const double CTimeSeriesAnomalyModel::LOG_LARGEST_ANOMALOUS_PROBABILITY{CTools::fastLog(LARGEST_ANOMALOUS_PROBABILITY)};
const double CTimeSeriesAnomalyModel::LOG_SMALL_PROBABILITY{CTools::fastLog(SMALL_PROBABILITY)};
const TDouble10Vec4Vec1Vec CTimeSeriesAnomalyModel::UNIT{CConstantWeights::unit<TDouble10Vec>(2)};


CUnivariateTimeSeriesModel::CUnivariateTimeSeriesModel(const CModelParams &params,
                                                       std::size_t id,
                                                       const CTimeSeriesDecompositionInterface &trend,
                                                       const CPrior &prior,
                                                       const TDecayRateController2Ary *controllers,
                                                       bool modelAnomalies) :
    CModel(params),
    m_Id(id),
    m_IsNonNegative(false),
    m_IsForecastable(true),
    m_Trend(trend.clone()),
    m_Prior(prior.clone()),
    m_AnomalyModel(modelAnomalies ?
                   boost::make_shared<CTimeSeriesAnomalyModel>(params.bucketLength(),
                                                               params.decayRate()) :
                   TAnomalyModelPtr()),
    m_SlidingWindow(SLIDING_WINDOW_SIZE),
    m_Correlations(0) {
    if (controllers) {
        m_Controllers = boost::make_shared<TDecayRateController2Ary>(*controllers);
    }
}

CUnivariateTimeSeriesModel::CUnivariateTimeSeriesModel(const SModelRestoreParams &params,
                                                       core::CStateRestoreTraverser &traverser) :
    CModel(params.s_Params),
    m_IsForecastable(false),
    m_SlidingWindow(SLIDING_WINDOW_SIZE),
    m_Correlations(0) {
    traverser.traverseSubLevel(boost::bind(&CUnivariateTimeSeriesModel::acceptRestoreTraverser,
                                           this, boost::cref(params), _1));
}

CUnivariateTimeSeriesModel::~CUnivariateTimeSeriesModel(void) {
    if (m_Correlations) {
        m_Correlations->removeTimeSeries(m_Id);
    }
}

std::size_t CUnivariateTimeSeriesModel::identifier(void) const {
    return m_Id;
}

CUnivariateTimeSeriesModel *CUnivariateTimeSeriesModel::clone(std::size_t id) const {
    CUnivariateTimeSeriesModel *result{new CUnivariateTimeSeriesModel{*this, id}};
    if (m_Correlations) {
        result->modelCorrelations(*m_Correlations);
    }
    return result;
}

CUnivariateTimeSeriesModel *CUnivariateTimeSeriesModel::cloneForPersistence(void) const {
    return new CUnivariateTimeSeriesModel{*this, m_Id};
}

CUnivariateTimeSeriesModel *CUnivariateTimeSeriesModel::cloneForForecast(void) const {
    return new CUnivariateTimeSeriesModel{*this, m_Id};
}

bool CUnivariateTimeSeriesModel::isForecastPossible(void) const {
    return m_IsForecastable && !m_Prior->isNonInformative();
}

void CUnivariateTimeSeriesModel::modelCorrelations(CTimeSeriesCorrelations &model) {
    m_Correlations = &model;
    m_Correlations->addTimeSeries(m_Id, *this);
}

TSize2Vec1Vec CUnivariateTimeSeriesModel::correlates(void) const {
    TSize2Vec1Vec result;
    TSize1Vec correlated;
    TSize2Vec1Vec variables;
    TMultivariatePriorCPtrSizePr1Vec correlationDistributionModels;
    TModelCPtr1Vec correlatedTimeSeriesModels;
    this->correlationModels(correlated, variables,
                            correlationDistributionModels,
                            correlatedTimeSeriesModels);
    result.resize(correlated.size(), TSize2Vec(2));
    for (std::size_t i = 0u; i < correlated.size(); ++i) {
        result[i][variables[i][0]] = m_Id;
        result[i][variables[i][1]] = correlated[i];
    }
    return result;
}

void CUnivariateTimeSeriesModel::addBucketValue(const TTimeDouble2VecSizeTrVec &values) {
    for (const auto &value : values) {
        m_Prior->adjustOffset(CConstantWeights::COUNT,
                              {m_Trend->detrend(value.first, value.second[0], 0.0)},
                              CConstantWeights::SINGLE_UNIT);
    }
}

CUnivariateTimeSeriesModel::EUpdateResult
CUnivariateTimeSeriesModel::addSamples(const CModelAddSamplesParams &params,
                                       TTimeDouble2VecSizeTrVec samples) {
    if (samples.empty()) {
        return E_Success;
    }

    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TOptionalTimeDoublePr = boost::optional<TTimeDoublePr>;

    TSizeVec valueorder(samples.size());
    std::iota(valueorder.begin(), valueorder.end(), 0);
    std::stable_sort(valueorder.begin(), valueorder.end(),
                     [&samples] (std::size_t lhs, std::size_t rhs) {
                return samples[lhs].second < samples[rhs].second;
            });

    TOptionalTimeDoublePr randomSample;

    double p{SLIDING_WINDOW_SIZE * static_cast<double>(this->params().bucketLength())
             / static_cast<double>(core::constants::DAY)};
    if (p >= 1.0 || CSampling::uniformSample(m_Rng, 0.0, 1.0) < p) {
        std::size_t i{CSampling::uniformSample(m_Rng, 0, samples.size())};
        randomSample.reset({samples[valueorder[i]].first, samples[valueorder[i]].second[0]});
    }

    m_IsNonNegative = params.isNonNegative();

    EUpdateResult result{this->updateTrend(params.weightStyles(), samples, params.trendWeights())};

    for (auto &sample : samples) {
        sample.second[0] = m_Trend->detrend(sample.first, sample.second[0], 0.0);
    }

    std::stable_sort(valueorder.begin(), valueorder.end(),
                     [&samples] (std::size_t lhs, std::size_t rhs) {
                return samples[lhs].second < samples[rhs].second;
            });

    maths_t::EDataType type{params.type()};
    m_Prior->dataType(type);

    TDouble1Vec samples_;
    TDouble4Vec1Vec weights;
    samples_.reserve(samples.size());
    weights.reserve(samples.size());
    TMeanAccumulator averageTime;

    for (auto i : valueorder) {
        samples_.push_back(samples[i].second[0]);
        TDouble4Vec1Vec wi(1);
        wi[0].reserve(params.priorWeights()[i].size());
        for (const auto &weight : params.priorWeights()[i]) {
            wi[0].push_back(weight[0]);
        }
        weights.push_back(wi[0]);
        averageTime.add(static_cast<double>(samples[i].first));
    }

    m_Prior->addSamples(params.weightStyles(), samples_, weights);
    m_Prior->propagateForwardsByTime(params.propagationInterval());
    if (m_AnomalyModel) {
        m_AnomalyModel->propagateForwardsByTime(params.propagationInterval());
    }

    double multiplier{1.0};
    if (m_Controllers) {
        TDouble1VecVec errors[2];
        errors[0].reserve(samples.size());
        errors[1].reserve(samples.size());
        for (auto i : valueorder) {
            this->appendPredictionErrors(params.propagationInterval(),
                                         samples[i].second[0], errors);
        }
        {
            CDecayRateController &controller{(*m_Controllers)[E_TrendControl]};
            core_t::TTime time{static_cast<core_t::TTime>(CBasicStatistics::mean(averageTime))};
            TDouble1Vec prediction{m_Trend->mean(time)};
            multiplier = controller.multiplier(prediction, errors[E_TrendControl],
                                               this->params().bucketLength(),
                                               this->params().learnRate(),
                                               this->params().decayRate());
            if (multiplier != 1.0) {
                m_Trend->decayRate(multiplier * m_Trend->decayRate());
                LOG_TRACE("trend decay rate = " << m_Trend->decayRate());
            }
        }
        {
            CDecayRateController &controller{(*m_Controllers)[E_PriorControl]};
            TDouble1Vec prediction{m_Prior->marginalLikelihoodMean()};
            multiplier = controller.multiplier(prediction, errors[E_PriorControl],
                                               this->params().bucketLength(),
                                               this->params().learnRate(),
                                               this->params().decayRate());
            if (multiplier != 1.0) {
                m_Prior->decayRate(multiplier * m_Prior->decayRate());
                LOG_TRACE("prior decay rate = " << m_Prior->decayRate());
            }
        }
    }

    if (m_Correlations) {
        m_Correlations->addSamples(m_Id, type, samples, weights, params.propagationInterval(), multiplier);
    }

    if (randomSample) {
        m_SlidingWindow.push_back({randomSample->first, randomSample->second});
    }

    return result;
}

void CUnivariateTimeSeriesModel::skipTime(core_t::TTime gap) {
    m_Trend->skipTime(gap);
}

CUnivariateTimeSeriesModel::TDouble2Vec
CUnivariateTimeSeriesModel::mode(core_t::TTime time,
                                 const maths_t::TWeightStyleVec &weightStyles,
                                 const TDouble2Vec4Vec &weights_) const {
    TDouble4Vec weights;
    weights.reserve(weights_.size());
    for (const auto &weight : weights_) {
        weights.push_back(weight[0]);
    }
    return {  m_Prior->marginalLikelihoodMode(weightStyles, weights)
              + CBasicStatistics::mean(m_Trend->baseline(time))};
}

CUnivariateTimeSeriesModel::TDouble2Vec1Vec
CUnivariateTimeSeriesModel::correlateModes(core_t::TTime time,
                                           const maths_t::TWeightStyleVec &weightStyles,
                                           const TDouble2Vec4Vec1Vec &weights_) const {
    TDouble2Vec1Vec result;

    TSize1Vec correlated;
    TSize2Vec1Vec variables;
    TMultivariatePriorCPtrSizePr1Vec correlationDistributionModels;
    TModelCPtr1Vec correlatedTimeSeriesModels;
    if (this->correlationModels(correlated, variables,
                                correlationDistributionModels,
                                correlatedTimeSeriesModels)) {
        result.resize(correlated.size(), TDouble10Vec(2));

        double baseline[2];
        baseline[0] = CBasicStatistics::mean(m_Trend->baseline(time));
        for (std::size_t i = 0u; i < correlated.size(); ++i) {
            baseline[1] = CBasicStatistics::mean(correlatedTimeSeriesModels[i]->m_Trend->baseline(time));
            TDouble10Vec4Vec weights;
            weights.resize(weights_[i].size(), TDouble10Vec(2));
            for (std::size_t j = 0u; j < weights_[i].size(); ++j) {
                for (std::size_t d = 0u; d < 2; ++d) {
                    weights[j][d] = weights_[i][j][d];
                }
            }
            TDouble10Vec mode(correlationDistributionModels[i].first->marginalLikelihoodMode(weightStyles, weights));
            result[i][variables[i][0]] = baseline[0] + mode[variables[i][0]];
            result[i][variables[i][1]] = baseline[1] + mode[variables[i][1]];
        }
    }

    return result;
}

CUnivariateTimeSeriesModel::TDouble2Vec1Vec
CUnivariateTimeSeriesModel::residualModes(const maths_t::TWeightStyleVec &weightStyles,
                                          const TDouble2Vec4Vec &weights_) const {
    TDouble2Vec1Vec result;

    TDouble4Vec weights;
    weights.reserve(weights_.size());
    for (const auto &weight : weights_) {
        weights.push_back(weight[0]);
    }

    TDouble1Vec modes(m_Prior->marginalLikelihoodModes(weightStyles, weights));
    result.reserve(modes.size());
    for (auto mode : modes) {
        result.push_back({mode});
    }

    return result;
}

void CUnivariateTimeSeriesModel::detrend(const TTime2Vec1Vec &time,
                                         double confidenceInterval,
                                         TDouble2Vec1Vec &value) const {
    if (value.empty()) {
        return;
    }

    if (value[0].size() == 1) {
        value[0][0] = m_Trend->detrend(time[0][0], value[0][0], confidenceInterval);
    } else   {
        TSize1Vec correlated;
        TSize2Vec1Vec variables;
        TMultivariatePriorCPtrSizePr1Vec correlationDistributionModels;
        TModelCPtr1Vec correlatedTimeSeriesModels;
        if (this->correlationModels(correlated, variables,
                                    correlationDistributionModels,
                                    correlatedTimeSeriesModels)) {
            for (std::size_t i = 0u; i < variables.size(); ++i) {
                if (!value[i].empty()) {
                    value[i][variables[i][0]] = m_Trend->detrend(time[i][variables[i][0]],
                                                                 value[i][variables[i][0]],
                                                                 confidenceInterval);
                    value[i][variables[i][1]] =
                        correlatedTimeSeriesModels[i]->m_Trend->detrend(time[i][variables[i][1]],
                                                                        value[i][variables[i][1]],
                                                                        confidenceInterval);
                }
            }
        }
    }
}

CUnivariateTimeSeriesModel::TDouble2Vec
CUnivariateTimeSeriesModel::predict(core_t::TTime time,
                                    const TSizeDoublePr1Vec &correlatedValue,
                                    TDouble2Vec hint) const {
    double correlateCorrection{0.0};
    if (!correlatedValue.empty()) {
        TSize1Vec correlated{correlatedValue[0].first};
        TSize2Vec1Vec variables;
        TMultivariatePriorCPtrSizePr1Vec correlationModel;
        TModelCPtr1Vec correlatedModel;
        if (m_Correlations->correlationModels(m_Id, correlated, variables,
                                              correlationModel, correlatedModel)) {
            double sample{correlatedModel[0]->m_Trend->detrend(time, correlatedValue[0].second, 0.0)};
            TSize10Vec marginalize{variables[0][1]};
            TSizeDoublePr10Vec condition{{variables[0][1], sample}};
            const CMultivariatePrior *joint{correlationModel[0].first};
            TPriorPtr margin{joint->univariate(marginalize, NOTHING_TO_CONDITION).first};
            TPriorPtr conditional{joint->univariate(NOTHING_TO_MARGINALIZE, condition).first};
            correlateCorrection = conditional->marginalLikelihoodMean() - margin->marginalLikelihoodMean();
        }
    }

    double scale{1.0 - this->params().probabilityBucketEmpty()};

    double seasonalOffset{0.0};
    if (m_Trend->initialized()) {
        seasonalOffset = CBasicStatistics::mean(m_Trend->baseline(time));
    }

    if (hint.size() == 1) {
        hint[0] = m_Trend->detrend(time, hint[0], 0.0);
    }

    double median{m_Prior->isNonInformative() ?
                  m_Prior->marginalLikelihoodMean() :
                  (hint.empty() ? CBasicStatistics::mean(m_Prior->marginalLikelihoodConfidenceInterval(0.0)) :
                   m_Prior->nearestMarginalLikelihoodMean(hint[0]))};
    double result{scale * (seasonalOffset + median + correlateCorrection)};

    return {m_IsNonNegative ? std::max(result, 0.0) : result};
}

CUnivariateTimeSeriesModel::TDouble2Vec3Vec
CUnivariateTimeSeriesModel::confidenceInterval(core_t::TTime time,
                                               double confidenceInterval,
                                               const maths_t::TWeightStyleVec &weightStyles,
                                               const TDouble2Vec4Vec &weights_) const {
    if (m_Prior->isNonInformative()) {
        return TDouble2Vec3Vec();
    }

    double scale{1.0 - this->params().probabilityBucketEmpty()};

    double seasonalOffset{m_Trend->initialized() ?
                          CBasicStatistics::mean(m_Trend->baseline(time, confidenceInterval)) : 0.0};

    TDouble4Vec weights;
    weights.reserve(weights_.size());
    for (const auto &weight : weights_) {
        weights.push_back(weight[0]);
    }

    double median{CBasicStatistics::mean(
                      m_Prior->marginalLikelihoodConfidenceInterval(0.0, weightStyles, weights))};
    TDoubleDoublePr interval{
        m_Prior->marginalLikelihoodConfidenceInterval(confidenceInterval, weightStyles, weights)};

    double result[]{scale * (seasonalOffset + interval.first),
                    scale * (seasonalOffset + median),
                    scale * (seasonalOffset + interval.second)};

    return {{m_IsNonNegative ? std::max(result[0], 0.0) : result[0]},
            {m_IsNonNegative ? std::max(result[1], 0.0) : result[1]},
            {m_IsNonNegative ? std::max(result[2], 0.0) : result[2]}};
}

bool CUnivariateTimeSeriesModel::forecast(core_t::TTime startTime,
                                          core_t::TTime endTime,
                                          double confidenceInterval,
                                          const TDouble2Vec &minimum_,
                                          const TDouble2Vec &maximum_,
                                          const TForecastPushDatapointFunc &forecastPushDataPointFunc,
                                          std::string &messageOut) {
    if (m_Prior->isNonInformative()) {
        messageOut = forecast::INFO_INSUFFICIENT_HISTORY;
        return true;
    }

    using TDouble3Vec = core::CSmallVector<double, 3>;
    using TDouble3VecVec = std::vector<TDouble3Vec>;

    core_t::TTime bucketLength{this->params().bucketLength()};
    double minimum{m_IsNonNegative ? std::max(minimum_[0], 0.0) : minimum_[0]};
    double maximum{m_IsNonNegative ? std::max(maximum_[0], 0.0) : maximum_[0]};

    TDouble3VecVec predictions;
    m_Trend->forecast(startTime, endTime, bucketLength, confidenceInterval,
                      this->params().minimumSeasonalVarianceScale(), predictions);

    core_t::TTime time{startTime};
    for (const auto &prediction : predictions) {
        SErrorBar errorBar;
        errorBar.s_Time         = time;
        errorBar.s_BucketLength = bucketLength;
        errorBar.s_LowerBound   = CTools::truncate(prediction[0],
                                                   minimum,
                                                   maximum + prediction[0] - prediction[1]);
        errorBar.s_Predicted    = CTools::truncate(prediction[1], minimum, maximum);
        errorBar.s_UpperBound   = CTools::truncate(prediction[2],
                                                   minimum + prediction[2] - prediction[1],
                                                   maximum);
        forecastPushDataPointFunc(errorBar);
        time += bucketLength;
    }

    return true;
}

bool CUnivariateTimeSeriesModel::probability(const CModelProbabilityParams &params,
                                             const TTime2Vec1Vec &time_,
                                             const TDouble2Vec1Vec &value,
                                             double &probability,
                                             TTail2Vec &tail,
                                             bool &conditional,
                                             TSize1Vec &mostAnomalousCorrelate) const {
    probability = 1.0;
    tail.resize(1, maths_t::E_UndeterminedTail);
    conditional = false;
    mostAnomalousCorrelate.clear();

    if (value.empty()) {
        return true;
    }

    if (value[0].size() == 1) {
        core_t::TTime time{time_[0][0]};
        TDouble1Vec sample{m_Trend->detrend(time, value[0][0], params.seasonalConfidenceInterval())};

        TDouble4Vec1Vec weights(1);
        weights[0].reserve(params.weights()[0].size());
        for (const auto &weight : params.weights()[0]) {
            weights[0].push_back(weight[0]);
        }

        double pl, pu;
        maths_t::ETail tail_;
        if (m_Prior->probabilityOfLessLikelySamples(params.calculation(0),
                                                    params.weightStyles(),
                                                    sample, weights, pl, pu, tail_)) {
            LOG_TRACE("P(" << sample << " | weight = " << weights
                      << ", time = " << time << ") = " << (pl + pu) / 2.0);
        } else   {
            LOG_ERROR("Failed to compute P(" << sample
                      << " | weight = " << weights << ", time = " << time << ")");
            return false;
        }
        probability = correctForEmptyBucket(params.calculation(0), value[0],
                                            params.bucketEmpty()[0][0],
                                            this->params().probabilityBucketEmpty(),
                                            (pl + pu) / 2.0);

        if (m_AnomalyModel) {
            TDouble2Vec residual{ (sample[0] - m_Prior->nearestMarginalLikelihoodMean(sample[0]))
                                  / std::max(std::sqrt(this->seasonalWeight(0.0, time)[0]), 1.0)};
            m_AnomalyModel->updateAnomaly(params, time, residual, probability);
            m_AnomalyModel->probability(params, time, probability);
            m_AnomalyModel->sampleAnomaly(params, time);
        }
        tail[0] = tail_;
    } else   {
        TSize1Vec correlated;
        TSize2Vec1Vec variables;
        TMultivariatePriorCPtrSizePr1Vec correlationDistributionModels;
        TModelCPtr1Vec correlatedTimeSeriesModels;
        if (!this->correlationModels(correlated, variables,
                                     correlationDistributionModels,
                                     correlatedTimeSeriesModels)) {
            return false;
        }

        double neff{effectiveCount(variables.size())};
        CProbabilityOfExtremeSample aggregator;
        CBasicStatistics::COrderStatisticsStack<double, 1> minProbability;

        // Declared outside the loop to minimize the number of times they are created.
        TSize10Vec variable(1);
        TDouble10Vec1Vec sample{TDouble10Vec(2)};
        TDouble10Vec4Vec1Vec weights{TDouble10Vec4Vec(params.weightStyles().size(), TDouble10Vec(2))};
        TDouble2Vec probabilityBucketEmpty(2);
        TDouble10Vec2Vec pli, pui;
        TTail10Vec ti;
        core_t::TTime mostAnomalousTime{0};
        double mostAnomalousSample{0.0};
        TPriorPtr mostAnomalousPrior;

        for (std::size_t i = 0u; i < variables.size(); ++i) {
            if (!value[i].empty() || (!params.mostAnomalousCorrelate() || i == *params.mostAnomalousCorrelate())) {
                variable[0] = variables[i][0];
                sample[0][variables[i][0]] = m_Trend->detrend(time_[i][variables[i][0]],
                                                              value[i][variables[i][0]],
                                                              params.seasonalConfidenceInterval());
                sample[0][variables[i][1]] =
                    correlatedTimeSeriesModels[i]->m_Trend->detrend(time_[i][variables[i][1]],
                                                                    value[i][variables[i][1]],
                                                                    params.seasonalConfidenceInterval());
                for (std::size_t j = 0u; j < params.weights()[i].size(); ++j) {
                    for (std::size_t d = 0u; d < 2; ++d) {
                        weights[0][j][d] = params.weights()[i][j][d];
                    }
                }

                if (correlationDistributionModels[i].first->probabilityOfLessLikelySamples(params.calculation(0),
                                                                                           params.weightStyles(),
                                                                                           sample, weights,
                                                                                           variable, pli, pui, ti)) {
                    LOG_TRACE("Marginal P(" << sample << " | weight = " << weights
                              << ", coordinate = " << variable
                              << ") = " << (pli[0][0] + pui[0][0]) / 2.0);
                    LOG_TRACE("Conditional P(" << sample << " | weight = " << weights
                              << ", coordinate = " << variable
                              << ") = " << (pli[1][0] + pui[1][0]) / 2.0);
                } else   {
                    LOG_ERROR("Failed to compute P(" << sample
                              << " | weight = " << weights
                              << ", coordinate = " << variable << ")");
                    continue;
                }

                probabilityBucketEmpty[variables[i][0]] = this->params().probabilityBucketEmpty();
                probabilityBucketEmpty[variables[i][1]] =
                    correlatedTimeSeriesModels[i]->params().probabilityBucketEmpty();
                double pl{std::sqrt(pli[0][0] * pli[1][0])};
                double pu{std::sqrt(pui[0][0] * pui[1][0])};
                double p{correctForEmptyBucket(params.calculation(0), value[0][variable[0]],
                                               params.bucketEmpty()[i], probabilityBucketEmpty,
                                               (pl + pu) / 2.0)};

                aggregator.add(p, neff);
                if (minProbability.add(p)) {
                    static TSizeDoublePr10Vec CONDITION;
                    static TSize10Vec MARGINALIZE;

                    tail[0] = ti[0];
                    mostAnomalousCorrelate.assign(1, i);
                    conditional = ((pli[1][0] + pui[1][0]) < (pli[0][0] + pui[0][0]));
                    mostAnomalousTime = time_[0][variables[i][0]];
                    mostAnomalousSample = sample[0][variables[i][0]];
                    mostAnomalousPrior =
                        conditional ?
                        correlationDistributionModels[i].first->univariate({variables[i][1]}, CONDITION).first :
                        correlationDistributionModels[i].first->univariate(MARGINALIZE,
                                                                           {{variables[i][1],
                                                                             sample[0][variables[i][1]]}}).first;
                }
            } else   {
                aggregator.add(1.0, neff);
            }
        }
        aggregator.calculate(probability);

        if (m_AnomalyModel) {
            TDouble2Vec residual{ (  mostAnomalousSample
                                     - mostAnomalousPrior->nearestMarginalLikelihoodMean(mostAnomalousSample))
                                  / std::max(std::sqrt(this->seasonalWeight(0.0, mostAnomalousTime)[0]), 1.0)};
            m_AnomalyModel->updateAnomaly(params, mostAnomalousTime, residual, probability);
            m_AnomalyModel->probability(params, mostAnomalousTime, probability);
            m_AnomalyModel->sampleAnomaly(params, mostAnomalousTime);
        }
    }

    return true;
}

CUnivariateTimeSeriesModel::TDouble2Vec
CUnivariateTimeSeriesModel::winsorisationWeight(double derate,
                                                core_t::TTime time,
                                                const TDouble2Vec &value) const {
    double scale{this->seasonalWeight(0.0, time)[0]};
    double sample{m_Trend->detrend(time, value[0], 0.0)};
    return {computeWinsorisationWeight(*m_Prior, derate, scale, sample)};
}

CUnivariateTimeSeriesModel::TDouble2Vec
CUnivariateTimeSeriesModel::seasonalWeight(double confidence, core_t::TTime time) const {
    double scale{m_Trend->scale(time, m_Prior->marginalLikelihoodVariance(), confidence).second};
    return {std::max(scale, this->params().minimumSeasonalVarianceScale())};
}

uint64_t CUnivariateTimeSeriesModel::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_IsNonNegative);
    seed = CChecksum::calculate(seed, m_Controllers);
    seed = CChecksum::calculate(seed, m_Trend);
    seed = CChecksum::calculate(seed, m_Prior);
    seed = CChecksum::calculate(seed, m_AnomalyModel);
    seed = CChecksum::calculate(seed, m_SlidingWindow);
    return CChecksum::calculate(seed, m_Correlations != 0);
}

void CUnivariateTimeSeriesModel::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CUnivariateTimeSeriesModel");
    core::CMemoryDebug::dynamicSize("m_Controllers", m_Controllers, mem);
    core::CMemoryDebug::dynamicSize("m_Trend", m_Trend, mem);
    core::CMemoryDebug::dynamicSize("m_Prior", m_Prior, mem);
    core::CMemoryDebug::dynamicSize("m_AnomalyModel", m_AnomalyModel, mem);
    core::CMemoryDebug::dynamicSize("m_SlidingWindow", m_SlidingWindow, mem);
}

std::size_t CUnivariateTimeSeriesModel::memoryUsage(void) const {
    return core::CMemory::dynamicSize(m_Controllers)
           + core::CMemory::dynamicSize(m_Trend)
           + core::CMemory::dynamicSize(m_Prior)
           + core::CMemory::dynamicSize(m_AnomalyModel)
           + core::CMemory::dynamicSize(m_SlidingWindow);
}

bool CUnivariateTimeSeriesModel::acceptRestoreTraverser(const SModelRestoreParams &params,
                                                        core::CStateRestoreTraverser &traverser) {
    if (traverser.name() == VERSION_6_3_TAG) {
        while (traverser.next()) {
            const std::string &name{traverser.name()};
            RESTORE_BUILT_IN(ID_6_3_TAG, m_Id)
            RESTORE_BOOL(IS_NON_NEGATIVE_6_3_TAG, m_IsNonNegative)
            RESTORE_BOOL(IS_FORECASTABLE_6_3_TAG, m_IsForecastable)
            RESTORE(RNG_6_3_TAG, m_Rng.fromString(traverser.value()))
            RESTORE_SETUP_TEARDOWN(CONTROLLER_6_3_TAG,
                                   m_Controllers = boost::make_shared<TDecayRateController2Ary>(),
                                   core::CPersistUtils::restore(CONTROLLER_6_3_TAG, *m_Controllers, traverser),
                                   /**/)
            RESTORE(TREND_6_3_TAG, traverser.traverseSubLevel(boost::bind<bool>(
                                                                  CTimeSeriesDecompositionStateSerialiser(),
                                                                  boost::cref(params.s_DecompositionParams),
                                                                  boost::ref(m_Trend), _1)))
            RESTORE(PRIOR_6_3_TAG, traverser.traverseSubLevel(boost::bind<bool>(
                                                                  CPriorStateSerialiser(),
                                                                  boost::cref(params.s_DistributionParams),
                                                                  boost::ref(m_Prior), _1)))
            RESTORE_SETUP_TEARDOWN(ANOMALY_MODEL_6_3_TAG,
                                   m_AnomalyModel = boost::make_shared<CTimeSeriesAnomalyModel>(),
                                   traverser.traverseSubLevel(boost::bind(&CTimeSeriesAnomalyModel::acceptRestoreTraverser,
                                                                          m_AnomalyModel.get(), boost::cref(params), _1)),
                                   /**/)
            RESTORE(SLIDING_WINDOW_6_3_TAG,
                    core::CPersistUtils::restore(SLIDING_WINDOW_6_3_TAG, m_SlidingWindow, traverser))
        }
    } else   {
        // There is no version string this is historic state.
        do {
            const std::string &name{traverser.name()};
            RESTORE_BUILT_IN(ID_OLD_TAG, m_Id)
            RESTORE_BOOL(IS_NON_NEGATIVE_OLD_TAG, m_IsNonNegative)
            RESTORE_BOOL(IS_FORECASTABLE_OLD_TAG, m_IsForecastable)
            RESTORE_SETUP_TEARDOWN(CONTROLLER_OLD_TAG,
                                   m_Controllers = boost::make_shared<TDecayRateController2Ary>(),
                                   core::CPersistUtils::restore(CONTROLLER_OLD_TAG, *m_Controllers, traverser),
                                   /**/)
            RESTORE(TREND_OLD_TAG, traverser.traverseSubLevel(boost::bind<bool>(
                                                                  CTimeSeriesDecompositionStateSerialiser(),
                                                                  boost::cref(params.s_DecompositionParams),
                                                                  boost::ref(m_Trend), _1)))
            RESTORE(PRIOR_OLD_TAG, traverser.traverseSubLevel(boost::bind<bool>(
                                                                  CPriorStateSerialiser(),
                                                                  boost::cref(params.s_DistributionParams),
                                                                  boost::ref(m_Prior), _1)))
            RESTORE_SETUP_TEARDOWN(ANOMALY_MODEL_OLD_TAG,
                                   m_AnomalyModel = boost::make_shared<CTimeSeriesAnomalyModel>(),
                                   traverser.traverseSubLevel(boost::bind(&CTimeSeriesAnomalyModel::acceptRestoreTraverser,
                                                                          m_AnomalyModel.get(), boost::cref(params), _1)),
                                   /**/)
        } while (traverser.next());
    }
    return true;
}

void CUnivariateTimeSeriesModel::acceptPersistInserter(core::CStatePersistInserter &inserter) const {
    // Note that we don't persist this->params() or the correlations
    // because that state is reinitialized.
    inserter.insertValue(VERSION_6_3_TAG, "");
    inserter.insertValue(ID_6_3_TAG, m_Id);
    inserter.insertValue(IS_NON_NEGATIVE_6_3_TAG, static_cast<int>(m_IsNonNegative));
    inserter.insertValue(IS_FORECASTABLE_6_3_TAG, static_cast<int>(m_IsForecastable));
    inserter.insertValue(RNG_6_3_TAG, m_Rng.toString());
    if (m_Controllers) {
        core::CPersistUtils::persist(CONTROLLER_6_3_TAG, *m_Controllers, inserter);
    }
    inserter.insertLevel(TREND_6_3_TAG, boost::bind<void>(CTimeSeriesDecompositionStateSerialiser(),
                                                          boost::cref(*m_Trend), _1));
    inserter.insertLevel(PRIOR_6_3_TAG, boost::bind<void>(CPriorStateSerialiser(),
                                                          boost::cref(*m_Prior), _1));
    if (m_AnomalyModel) {
        inserter.insertLevel(ANOMALY_MODEL_6_3_TAG,
                             boost::bind(&CTimeSeriesAnomalyModel::acceptPersistInserter,
                                         m_AnomalyModel.get(), _1));
    }
    core::CPersistUtils::persist(SLIDING_WINDOW_6_3_TAG, m_SlidingWindow, inserter);
}

maths_t::EDataType CUnivariateTimeSeriesModel::dataType(void) const {
    return m_Prior->dataType();
}

const CUnivariateTimeSeriesModel::TTimeDoublePrCBuf &CUnivariateTimeSeriesModel::slidingWindow(void) const {
    return m_SlidingWindow;
}

const CTimeSeriesDecompositionInterface &CUnivariateTimeSeriesModel::trend(void) const {
    return *m_Trend;
}

const CPrior &CUnivariateTimeSeriesModel::prior(void) const {
    return *m_Prior;
}

CUnivariateTimeSeriesModel::CUnivariateTimeSeriesModel(const CUnivariateTimeSeriesModel &other,
                                                       std::size_t id) :
    CModel(other.params()),
    m_Id(id),
    m_IsNonNegative(other.m_IsNonNegative),
    m_IsForecastable(other.m_IsForecastable),
    m_Rng(other.m_Rng),
    m_Trend(other.m_Trend->clone()),
    m_Prior(other.m_Prior->clone()),
    m_AnomalyModel(other.m_AnomalyModel ?
                   boost::make_shared<CTimeSeriesAnomalyModel>(*other.m_AnomalyModel) :
                   TAnomalyModelPtr()),
    m_SlidingWindow(other.m_SlidingWindow),
    m_Correlations(0) {
    if (other.m_Controllers) {
        m_Controllers = boost::make_shared<TDecayRateController2Ary>(*other.m_Controllers);
    }
}

CUnivariateTimeSeriesModel::EUpdateResult
CUnivariateTimeSeriesModel::updateTrend(const maths_t::TWeightStyleVec &weightStyles,
                                        const TTimeDouble2VecSizeTrVec &samples,
                                        const TDouble2Vec4VecVec &weights) {
    for (const auto &sample : samples) {
        if (sample.second.size() != 1) {
            LOG_ERROR("Dimension mismatch: '" << sample.second.size() << " != 1'");
            return E_Failure;
        }
    }

    // Time order is not reliable, for example if the data are polled
    // or for count feature, the times of all samples will be the same.
    TSizeVec timeorder(samples.size());
    std::iota(timeorder.begin(), timeorder.end(), 0);
    std::stable_sort(timeorder.begin(), timeorder.end(),
                     [&samples] (std::size_t lhs, std::size_t rhs) {
                return COrderings::lexicographical_compare(samples[lhs].first,
                                                           samples[lhs].second,
                                                           samples[rhs].first,
                                                           samples[rhs].second);
            });

    EUpdateResult result = E_Success;
    {
        TDouble4Vec weight(weightStyles.size());
        for (auto i : timeorder) {
            core_t::TTime time{samples[i].first};
            double value{samples[i].second[0]};
            for (std::size_t j = 0u; j < weights[i].size(); ++j) {
                weight[j] = weights[i][j][0];
            }
            if (m_Trend->addPoint(time, value, weightStyles, weight)) {
                result = E_Reset;
            }
        }
    }
    if (result == E_Reset) {
        m_Prior->setToNonInformative(0.0, m_Prior->decayRate());
        TDouble4Vec1Vec weight{{std::max(this->params().learnRate(),
                                         5.0 / static_cast<double>(SLIDING_WINDOW_SIZE))}};
        for (const auto &value : m_SlidingWindow) {
            TDouble1Vec sample{m_Trend->detrend(value.first, value.second, 0.0)};
            m_Prior->addSamples(CConstantWeights::COUNT, sample, weight);
        }
        if (m_Correlations) {
            m_Correlations->removeTimeSeries(m_Id);
        }
        if (m_Controllers) {
            m_Prior->decayRate(  m_Prior->decayRate()
                                 / (*m_Controllers)[E_PriorControl].multiplier());
            m_Trend->decayRate(  m_Trend->decayRate()
                                 / (*m_Controllers)[E_TrendControl].multiplier());
            for (auto &controller : *m_Controllers) {
                controller.reset();
            }
        }
        if (m_AnomalyModel) {
            m_AnomalyModel->reset();
        }
    }

    return result;
}

void CUnivariateTimeSeriesModel::appendPredictionErrors(double interval,
                                                        double sample_,
                                                        TDouble1VecVec (&result)[2]) {
    using TDecompositionPtr1Vec = core::CSmallVector<TDecompositionPtr, 1>;
    TDouble1Vec sample{sample_};
    TDecompositionPtr1Vec trend{m_Trend};
    if (auto error = predictionError(interval, m_Prior, sample)) {
        result[E_PriorControl].push_back(*error);
    }
    if (auto error = predictionError(trend, sample)) {
        result[E_TrendControl].push_back(*error);
    }
}

bool CUnivariateTimeSeriesModel::correlationModels(TSize1Vec &correlated,
                                                   TSize2Vec1Vec &variables,
                                                   TMultivariatePriorCPtrSizePr1Vec &correlationDistributionModels,
                                                   TModelCPtr1Vec &correlatedTimeSeriesModels) const {
    if (m_Correlations) {
        correlated = m_Correlations->correlated(m_Id);
        m_Correlations->correlationModels(m_Id, correlated, variables,
                                          correlationDistributionModels,
                                          correlatedTimeSeriesModels);
    }
    return correlated.size() > 0;
}


CTimeSeriesCorrelations::CTimeSeriesCorrelations(double minimumSignificantCorrelation,
                                                 double decayRate) :
    m_MinimumSignificantCorrelation(minimumSignificantCorrelation),
    m_Correlations(MAXIMUM_CORRELATIONS, decayRate)
{}

CTimeSeriesCorrelations::CTimeSeriesCorrelations(const CTimeSeriesCorrelations &other,
                                                 bool isForPersistence) :
    m_MinimumSignificantCorrelation(other.m_MinimumSignificantCorrelation),
    m_SampleData(other.m_SampleData),
    m_Correlations(other.m_Correlations),
    m_CorrelatedLookup(other.m_CorrelatedLookup),
    m_TimeSeriesModels(isForPersistence ? TModelCPtrVec() : other.m_TimeSeriesModels) {
    for (const auto &model : other.m_CorrelationDistributionModels) {
        m_CorrelationDistributionModels.emplace(
            model.first, std::make_pair(TMultivariatePriorPtr(model.second.first->clone()),
                                        model.second.second));
    }
}

CTimeSeriesCorrelations *CTimeSeriesCorrelations::clone(void) const {
    return new CTimeSeriesCorrelations(*this);
}

CTimeSeriesCorrelations *CTimeSeriesCorrelations::cloneForPersistence(void) const {
    return new CTimeSeriesCorrelations(*this, true);
}

void CTimeSeriesCorrelations::processSamples(const maths_t::TWeightStyleVec &weightStyles) {
    using TSizeSizePrMultivariatePriorPtrDoublePrUMapCItr = TSizeSizePrMultivariatePriorPtrDoublePrUMap::const_iterator;
    using TSizeSizePrMultivariatePriorPtrDoublePrUMapCItrVec = std::vector<TSizeSizePrMultivariatePriorPtrDoublePrUMapCItr>;

    // The priors use a shared pseudo random number generator which
    // generates a fixed sequence of random numbers. Since the order
    // of the correlated priors map can change across persist and
    // restore we can effectively get a different sequence of random
    // numbers depending on whether we persist and restore or not.
    // We'd like results to be as independent as possible from the
    // action of persisting and restoring so sort the collection to
    // preserve the random number sequence.
    TSizeSizePrMultivariatePriorPtrDoublePrUMapCItrVec iterators;
    iterators.reserve(m_CorrelationDistributionModels.size());
    for (auto i = m_CorrelationDistributionModels.begin(); i != m_CorrelationDistributionModels.end(); ++i) {
        iterators.push_back(i);
    }
    std::sort(iterators.begin(), iterators.end(),
              core::CFunctional::SDereference<COrderings::SFirstLess>());

    TDouble10Vec1Vec multivariateSamples;
    TDouble10Vec4Vec1Vec multivariateWeights;
    for (auto i : iterators) {
        std::size_t pid1{i->first.first};
        std::size_t pid2{i->first.second};
        auto i1 = m_SampleData.find(pid1);
        auto i2 = m_SampleData.find(pid2);
        if (i1 == m_SampleData.end() || i2 == m_SampleData.end()) {
            continue;
        }

        const TMultivariatePriorPtr &prior{i->second.first};
        SSampleData *samples1{&i1->second};
        SSampleData *samples2{&i2->second};
        std::size_t n1{samples1->s_Times.size()};
        std::size_t n2{samples2->s_Times.size()};
        std::size_t indices[] = { 0, 1 };
        if (n1 < n2) {
            std::swap(samples1, samples2);
            std::swap(n1, n2);
            std::swap(indices[0], indices[1]);
        }
        multivariateSamples.assign(n1, TDouble10Vec(2));
        multivariateWeights.assign(n1, TDouble10Vec4Vec(weightStyles.size(), TDouble10Vec(2)));

        TSize1Vec &tags2{samples2->s_Tags};
        TTime1Vec &times2{samples2->s_Times};

        COrderings::simultaneousSort(tags2, times2, samples2->s_Samples, samples2->s_Weights);
        for (auto j = tags2.begin(); j != tags2.end(); /**/) {
            auto k = std::upper_bound(j, tags2.end(), *j);
            std::size_t a = j - tags2.begin();
            std::size_t b = k - tags2.begin();
            COrderings::simultaneousSort(core::make_range(times2, a, b),
                                         core::make_range(samples2->s_Samples, a, b),
                                         core::make_range(samples2->s_Weights, a, b));
            j = k;
        }

        for (std::size_t j1 = 0u; j1 < n1; ++j1) {
            std::size_t j2{0u};
            if (n2 > 1) {
                std::size_t tag{samples1->s_Tags[j1]};
                core_t::TTime time{samples1->s_Times[j1]};
                std::size_t a_ = std::lower_bound(tags2.begin(), tags2.end(), tag) - tags2.begin();
                std::size_t b_ = std::upper_bound(tags2.begin(), tags2.end(), tag) - tags2.begin();
                std::size_t b{CTools::truncate(static_cast<std::size_t>(
                                                   std::lower_bound(times2.begin() + a_,
                                                                    times2.begin() + b_, time) - times2.begin()),
                                               std::size_t(1), n2 - 1)};
                std::size_t a{b - 1};
                j2 = std::abs(times2[a] - time) < std::abs(times2[b] - time) ? a : b;
            }
            multivariateSamples[j1][indices[0]] = samples1->s_Samples[j1];
            multivariateSamples[j1][indices[1]] = samples2->s_Samples[j2];
            for (std::size_t w = 0u; w < weightStyles.size(); ++w) {
                multivariateWeights[j1][w][indices[0]] = samples1->s_Weights[j1][w];
                multivariateWeights[j1][w][indices[1]] = samples2->s_Weights[j2][w];
            }
        }
        LOG_TRACE("correlate samples = " << core::CContainerPrinter::print(multivariateSamples)
                  << ", correlate weights = " << core::CContainerPrinter::print(multivariateWeights));

        prior->dataType(   samples1->s_Type == maths_t::E_IntegerData ||
                           samples2->s_Type == maths_t::E_IntegerData ?
                           maths_t::E_IntegerData : maths_t::E_ContinuousData);
        prior->addSamples(weightStyles, multivariateSamples, multivariateWeights);
        prior->propagateForwardsByTime(std::min(samples1->s_Interval, samples2->s_Interval));
        prior->decayRate(std::sqrt(samples1->s_Multiplier * samples2->s_Multiplier) * prior->decayRate());
        LOG_TRACE("correlation prior:" << core_t::LINE_ENDING << prior->print());
        LOG_TRACE("decayRate = " << prior->decayRate());
    }

    m_Correlations.capture();
    m_SampleData.clear();
}

void CTimeSeriesCorrelations::refresh(const CTimeSeriesCorrelateModelAllocator &allocator) {
    using TDoubleVec = std::vector<double>;
    using TSizeSizePrVec = std::vector<TSizeSizePr>;

    if (m_Correlations.changed()) {
        TSizeSizePrVec correlated;
        TDoubleVec correlationCoeffs;
        m_Correlations.mostCorrelated(static_cast<std::size_t>(
                                          1.2 * static_cast<double>(allocator.maxNumberCorrelations())),
                                      correlated,
                                      &correlationCoeffs);
        LOG_TRACE("correlated = " << core::CContainerPrinter::print(correlated));
        LOG_TRACE("correlationCoeffs = " << core::CContainerPrinter::print(correlationCoeffs));

        ptrdiff_t cutoff{std::upper_bound(correlationCoeffs.begin(), correlationCoeffs.end(),
                                          0.5 * m_MinimumSignificantCorrelation,
                                          [] (double lhs, double rhs) {
                        return std::fabs(lhs) > std::fabs(rhs);
                    }) - correlationCoeffs.begin()};
        LOG_TRACE("cutoff = " << cutoff);

        correlated.erase(correlated.begin() + cutoff, correlated.end());
        if (correlated.empty()) {
            m_CorrelationDistributionModels.clear();
            this->refreshLookup();
            return;
        }

        // Extract the correlated pairs which are and aren't already
        // being modeled.
        TSizeSizePrVec present;
        TSizeVec presentRank;
        TSizeSizePrVec missing;
        TSizeVec missingRank;
        std::size_t np{static_cast<std::size_t>(
                           std::max(0.9 * static_cast<double>(correlated.size()), 1.0))};
        std::size_t nm{static_cast<std::size_t>(
                           std::max(0.1 * static_cast<double>(correlated.size()), 1.0))};
        present.reserve(np);
        presentRank.reserve(np);
        missing.reserve(nm);
        missingRank.reserve(nm);
        for (std::size_t j = 0u; j < correlated.size(); ++j) {
            bool isPresent{m_CorrelationDistributionModels.count(correlated[j]) > 0};
            (isPresent ? present : missing).push_back(correlated[j]);
            (isPresent ? presentRank : missingRank).push_back(j);
        }

        // Remove any weakly correlated models.
        std::size_t initial{m_CorrelationDistributionModels.size()};
        COrderings::simultaneousSort(present, presentRank);
        for (auto i = m_CorrelationDistributionModels.begin(); i != m_CorrelationDistributionModels.end(); /**/) {
            std::size_t j = std::lower_bound(present.begin(),
                                             present.end(), i->first) - present.begin();
            if (j == present.size() || i->first != present[j]) {
                i = m_CorrelationDistributionModels.erase(i);
            } else   {
                i->second.second = correlationCoeffs[presentRank[j]];
                ++i;
            }
        }

        // Remove the remaining most weakly correlated models subject
        // to the capacity constraint.
        COrderings::simultaneousSort(presentRank, present, std::greater<std::size_t>());
        for (std::size_t i = 0u; m_CorrelationDistributionModels.size() > allocator.maxNumberCorrelations(); ++i) {
            m_CorrelationDistributionModels.erase(present[i]);
        }

        if (allocator.areAllocationsAllowed()) {
            for (std::size_t i = 0u, nextChunk = std::min(allocator.maxNumberCorrelations(),
                                                          initial + allocator.chunkSize());
                 m_CorrelationDistributionModels.size() < allocator.maxNumberCorrelations() &&
                 i < missing.size() &&
                 (    m_CorrelationDistributionModels.size() <= initial ||
                      !allocator.exceedsLimit(m_CorrelationDistributionModels.size()));
                 nextChunk = std::min(allocator.maxNumberCorrelations(),
                                      nextChunk + allocator.chunkSize())) {
                for (/**/; i < missing.size() && m_CorrelationDistributionModels.size() < nextChunk; ++i) {
                    m_CorrelationDistributionModels.insert({missing[i], {allocator.newPrior(),
                                                                         correlationCoeffs[missingRank[i]]}});
                }
            }
        }

        this->refreshLookup();
    }
}

const CTimeSeriesCorrelations::TSizeSizePrMultivariatePriorPtrDoublePrUMap &
CTimeSeriesCorrelations::correlatePriors(void) const {
    return m_CorrelationDistributionModels;
}

void CTimeSeriesCorrelations::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CTimeSeriesCorrelations");
    core::CMemoryDebug::dynamicSize("m_SampleData", m_SampleData, mem);
    core::CMemoryDebug::dynamicSize("m_Correlations", m_Correlations, mem);
    core::CMemoryDebug::dynamicSize("m_CorrelatedLookup", m_CorrelatedLookup, mem);
    core::CMemoryDebug::dynamicSize("m_CorrelationDistributionModels", m_CorrelationDistributionModels, mem);
}

std::size_t CTimeSeriesCorrelations::memoryUsage(void) const {
    return core::CMemory::dynamicSize(m_SampleData)
           + core::CMemory::dynamicSize(m_Correlations)
           + core::CMemory::dynamicSize(m_CorrelatedLookup)
           + core::CMemory::dynamicSize(m_CorrelationDistributionModels);
}

bool CTimeSeriesCorrelations::acceptRestoreTraverser(const SDistributionRestoreParams &params,
                                                     core::CStateRestoreTraverser &traverser) {
    do {
        const std::string &name{traverser.name()};
        RESTORE(K_MOST_CORRELATED_TAG,
                traverser.traverseSubLevel(boost::bind(&CKMostCorrelated::acceptRestoreTraverser,
                                                       &m_Correlations, _1)))
        RESTORE(CORRELATED_LOOKUP_TAG,
                core::CPersistUtils::restore(CORRELATED_LOOKUP_TAG, m_CorrelatedLookup, traverser))
        RESTORE(CORRELATED_PRIORS_TAG,
                traverser.traverseSubLevel(boost::bind(&CTimeSeriesCorrelations::restoreCorrelatePriors,
                                                       this, boost::cref(params), _1)))
    } while (traverser.next());
    return true;
}

void CTimeSeriesCorrelations::acceptPersistInserter(core::CStatePersistInserter &inserter) const {
    // Note we don't persist the minimum significant correlation or the
    // models because that state is reinitialized. The sample is only
    // maintained transitively during an update at the end of a bucket
    // and so always empty at the point persistence occurs.

    inserter.insertLevel(K_MOST_CORRELATED_TAG,
                         boost::bind(&CKMostCorrelated::acceptPersistInserter, &m_Correlations, _1));
    core::CPersistUtils::persist(CORRELATED_LOOKUP_TAG, m_CorrelatedLookup, inserter);
    inserter.insertLevel(CORRELATED_PRIORS_TAG,
                         boost::bind(&CTimeSeriesCorrelations::persistCorrelatePriors, this, _1));
}

bool CTimeSeriesCorrelations::restoreCorrelatePriors(const SDistributionRestoreParams &params,
                                                     core::CStateRestoreTraverser &traverser) {
    do {
        const std::string &name{traverser.name()};
        RESTORE_SETUP_TEARDOWN(CORRELATE_PRIOR_TAG,
                               TSizeSizePrMultivariatePriorPtrDoublePrPr prior,
                               traverser.traverseSubLevel(
                                   boost::bind(&restore, boost::cref(params), boost::ref(prior), _1)),
                               m_CorrelationDistributionModels.insert(prior))
    } while (traverser.next());
    return true;
}

void CTimeSeriesCorrelations::persistCorrelatePriors(core::CStatePersistInserter &inserter) const {
    using TSizeSizePrMultivariatePriorPtrDoublePrUMapCItrVec =
        std::vector<TSizeSizePrMultivariatePriorPtrDoublePrUMap::const_iterator>;
    TSizeSizePrMultivariatePriorPtrDoublePrUMapCItrVec ordered;
    ordered.reserve(m_CorrelationDistributionModels.size());
    for (auto prior = m_CorrelationDistributionModels.begin(); prior != m_CorrelationDistributionModels.end(); ++prior) {
        ordered.push_back(prior);
    }
    std::sort(ordered.begin(), ordered.end(),
              core::CFunctional::SDereference<COrderings::SFirstLess>());
    for (auto prior : ordered) {
        inserter.insertLevel(CORRELATE_PRIOR_TAG, boost::bind(&persist, boost::cref(*prior), _1));
    }
}

bool CTimeSeriesCorrelations::restore(const SDistributionRestoreParams &params,
                                      TSizeSizePrMultivariatePriorPtrDoublePrPr &prior,
                                      core::CStateRestoreTraverser &traverser) {
    do {
        const std::string &name{traverser.name()};
        RESTORE_BUILT_IN(FIRST_CORRELATE_ID_TAG, prior.first.first)
        RESTORE_BUILT_IN(SECOND_CORRELATE_ID_TAG, prior.first.second)
        RESTORE(CORRELATE_PRIOR_TAG,
                traverser.traverseSubLevel(boost::bind<bool>(CPriorStateSerialiser(),
                                                             boost::cref(params),
                                                             boost::ref(prior.second.first), _1)))
        RESTORE_BUILT_IN(CORRELATION_TAG, prior.second.second)

    } while (traverser.next());
    return true;
}

void CTimeSeriesCorrelations::persist(const TSizeSizePrMultivariatePriorPtrDoublePrPr &prior,
                                      core::CStatePersistInserter &inserter) {
    inserter.insertValue(FIRST_CORRELATE_ID_TAG, prior.first.first);
    inserter.insertValue(SECOND_CORRELATE_ID_TAG, prior.first.second);
    inserter.insertLevel(CORRELATE_PRIOR_TAG, boost::bind<void>(CPriorStateSerialiser(),
                                                                boost::cref(*prior.second.first), _1));
    inserter.insertValue(CORRELATION_TAG,
                         prior.second.second,
                         core::CIEEE754::E_SinglePrecision);
}

void CTimeSeriesCorrelations::addTimeSeries(std::size_t id, const CUnivariateTimeSeriesModel &model) {
    m_Correlations.addVariables(id + 1);
    core::CAllocationStrategy::resize(m_TimeSeriesModels, std::max(id + 1, m_TimeSeriesModels.size()));
    m_TimeSeriesModels[id] = &model;
}

void CTimeSeriesCorrelations::removeTimeSeries(std::size_t id) {
    auto correlated_ = m_CorrelatedLookup.find(id);
    if (correlated_ != m_CorrelatedLookup.end()) {
        TSize1Vec &correlated{correlated_->second};
        for (const auto &correlate : correlated) {
            m_CorrelationDistributionModels.erase({id, correlate});
            m_CorrelationDistributionModels.erase({correlate, id});
        }
        this->refreshLookup();
    }
    m_Correlations.removeVariables({id});
    m_TimeSeriesModels[id] = 0;
}

void CTimeSeriesCorrelations::addSamples(std::size_t id,
                                         maths_t::EDataType type,
                                         const TTimeDouble2VecSizeTrVec &samples,
                                         const TDouble4Vec1Vec &weights,
                                         double interval,
                                         double multiplier) {
    SSampleData &data{m_SampleData[id]};
    data.s_Type = type;
    data.s_Times.reserve(samples.size());
    data.s_Samples.reserve(samples.size());
    data.s_Tags.reserve(samples.size());
    for (const auto &sample : samples) {
        data.s_Times.push_back(sample.first);
        data.s_Samples.push_back(sample.second[0]);
        data.s_Tags.push_back(sample.third);
    }
    data.s_Weights = weights;
    data.s_Interval = interval;
    data.s_Multiplier = multiplier;
    m_Correlations.add(id, CBasicStatistics::median(data.s_Samples));
}

TSize1Vec CTimeSeriesCorrelations::correlated(std::size_t id) const {
    auto correlated = m_CorrelatedLookup.find(id);
    return correlated != m_CorrelatedLookup.end() ? correlated->second : TSize1Vec();
}

bool CTimeSeriesCorrelations::correlationModels(std::size_t id,
                                                TSize1Vec &correlated,
                                                TSize2Vec1Vec &variables,
                                                TMultivariatePriorCPtrSizePr1Vec &correlationDistributionModels,
                                                TModelCPtr1Vec &correlatedTimeSeriesModels) const {
    variables.clear();
    correlationDistributionModels.clear();
    correlatedTimeSeriesModels.clear();

    if (correlated.empty()) {
        return false;
    }

    variables.reserve(correlated.size());
    correlationDistributionModels.reserve(correlated.size());
    correlatedTimeSeriesModels.reserve(correlated.size());
    std::size_t end{0u};
    for (auto correlate : correlated) {
        auto i = m_CorrelationDistributionModels.find({id, correlate});
        TSize2Vec variable{0, 1};
        if (i == m_CorrelationDistributionModels.end()) {
            i = m_CorrelationDistributionModels.find({correlate, id});
            std::swap(variable[0], variable[1]);
        }
        if (i == m_CorrelationDistributionModels.end()) {
            LOG_ERROR("Unexpectedly missing prior for correlation (" << id
                      << "," << correlate << ")");
            continue;
        }
        if (std::fabs(i->second.second) < m_MinimumSignificantCorrelation) {
            LOG_TRACE("Correlation " << i->second.second << " is too small to model");
            continue;
        }
        if (i->second.first->numberSamples() < MINIMUM_CORRELATE_PRIOR_SAMPLE_COUNT) {
            LOG_TRACE("Too few samples in correlate model");
            continue;
        }
        correlated[end] = correlate;
        variables.push_back(std::move(variable));
        correlationDistributionModels.push_back({i->second.first.get(), variable[0]});
        ++end;
    }

    correlated.resize(variables.size());
    for (auto correlate : correlated) {
        correlatedTimeSeriesModels.push_back(m_TimeSeriesModels[correlate]);
    }

    return correlationDistributionModels.size() > 0;
}

void CTimeSeriesCorrelations::refreshLookup(void) {
    m_CorrelatedLookup.clear();
    for (const auto &prior : m_CorrelationDistributionModels) {
        std::size_t x0{prior.first.first};
        std::size_t x1{prior.first.second};
        m_CorrelatedLookup[x0].push_back(x1);
        m_CorrelatedLookup[x1].push_back(x0);
    }
    for (auto &prior : m_CorrelatedLookup) {
        std::sort(prior.second.begin(), prior.second.end());
    }
}


CMultivariateTimeSeriesModel::CMultivariateTimeSeriesModel(const CModelParams &params,
                                                           const CTimeSeriesDecompositionInterface &trend,
                                                           const CMultivariatePrior &prior,
                                                           const TDecayRateController2Ary *controllers,
                                                           bool modelAnomalies) :
    CModel(params),
    m_IsNonNegative(false),
    m_Prior(prior.clone()),
    m_AnomalyModel(modelAnomalies ?
                   boost::make_shared<CTimeSeriesAnomalyModel>(params.bucketLength(),
                                                               params.decayRate()) :
                   TAnomalyModelPtr()),
    m_SlidingWindow(SLIDING_WINDOW_SIZE) {
    if (controllers) {
        m_Controllers = boost::make_shared<TDecayRateController2Ary>(*controllers);
    }
    for (std::size_t d = 0u; d < this->dimension(); ++d) {
        m_Trend.emplace_back(trend.clone());
    }
}

CMultivariateTimeSeriesModel::CMultivariateTimeSeriesModel(const CMultivariateTimeSeriesModel &other) :
    CModel(other.params()),
    m_IsNonNegative(other.m_IsNonNegative),
    m_Prior(other.m_Prior->clone()),
    m_AnomalyModel(other.m_AnomalyModel ?
                   boost::make_shared<CTimeSeriesAnomalyModel>(*other.m_AnomalyModel) :
                   TAnomalyModelPtr()),
    m_SlidingWindow(other.m_SlidingWindow) {
    if (other.m_Controllers) {
        m_Controllers = boost::make_shared<TDecayRateController2Ary>(*other.m_Controllers);
    }
    m_Trend.reserve(other.m_Trend.size());
    for (const auto &trend : other.m_Trend) {
        m_Trend.emplace_back(trend->clone());
    }
}

CMultivariateTimeSeriesModel::CMultivariateTimeSeriesModel(const SModelRestoreParams &params,
                                                           core::CStateRestoreTraverser &traverser) :
    CModel(params.s_Params),
    m_SlidingWindow(SLIDING_WINDOW_SIZE) {
    traverser.traverseSubLevel(boost::bind(&CMultivariateTimeSeriesModel::acceptRestoreTraverser,
                                           this, boost::cref(params), _1));
}

std::size_t CMultivariateTimeSeriesModel::identifier(void) const {
    return 0;
}

CMultivariateTimeSeriesModel *CMultivariateTimeSeriesModel::clone(std::size_t /*id*/) const {
    return new CMultivariateTimeSeriesModel{*this};
}

CMultivariateTimeSeriesModel *CMultivariateTimeSeriesModel::cloneForPersistence() const {
    return new CMultivariateTimeSeriesModel{*this};
}

CMultivariateTimeSeriesModel *CMultivariateTimeSeriesModel::cloneForForecast() const {
    // Note: placeholder as there is no forecast support for multivariate time series for now
    return new CMultivariateTimeSeriesModel{*this};
}

bool CMultivariateTimeSeriesModel::isForecastPossible(void) const {
    return false;
}

void CMultivariateTimeSeriesModel::modelCorrelations(CTimeSeriesCorrelations & /*model*/) {
    // no-op
}

TSize2Vec1Vec CMultivariateTimeSeriesModel::correlates(void) const {
    return TSize2Vec1Vec();
}

void CMultivariateTimeSeriesModel::addBucketValue(const TTimeDouble2VecSizeTrVec & /*value*/) {
    // no-op
}

CMultivariateTimeSeriesModel::EUpdateResult
CMultivariateTimeSeriesModel::addSamples(const CModelAddSamplesParams &params,
                                         TTimeDouble2VecSizeTrVec samples) {
    if (samples.empty()) {
        return E_Success;
    }

    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TOptionalTimeDouble2VecPr = boost::optional<TTimeDouble2VecPr>;

    TSizeVec valueorder(samples.size());
    std::iota(valueorder.begin(), valueorder.end(), 0);
    std::stable_sort(valueorder.begin(), valueorder.end(),
                     [&samples] (std::size_t lhs, std::size_t rhs) {
                return samples[lhs].second < samples[rhs].second;
            });

    TOptionalTimeDouble2VecPr randomSample;

    double p{SLIDING_WINDOW_SIZE * static_cast<double>(this->params().bucketLength())
             / static_cast<double>(core::constants::DAY)};
    if (p >= 1.0 || CSampling::uniformSample(m_Rng, 0.0, 1.0) < p) {
        std::size_t i{CSampling::uniformSample(m_Rng, 0, samples.size())};
        randomSample.reset({samples[valueorder[i]].first, samples[valueorder[i]].second});
    }

    m_IsNonNegative = params.isNonNegative();

    std::size_t dimension{this->dimension()};

    EUpdateResult result{this->updateTrend(params.weightStyles(), samples, params.trendWeights())};

    for (auto &sample : samples) {
        if (sample.second.size() != dimension) {
            LOG_ERROR("Unexpected sample dimension: '"
                      << sample.second.size() << " != " << this->dimension() << "' discarding");
            continue;
        }
        core_t::TTime time{sample.first};
        for (std::size_t d = 0u; d < sample.second.size(); ++d) {
            sample.second[d] = m_Trend[d]->detrend(time, sample.second[d], 0.0);
        }
    }

    std::stable_sort(valueorder.begin(), valueorder.end(),
                     [&samples] (std::size_t lhs, std::size_t rhs) {
                return samples[lhs].second < samples[rhs].second;
            });

    maths_t::EDataType type{params.type()};
    m_Prior->dataType(type);

    TDouble10Vec1Vec samples_;
    TDouble10Vec4Vec1Vec weights;
    samples_.reserve(samples.size());
    weights.reserve(samples.size());
    TMeanAccumulator averageTime;

    for (auto i : valueorder) {
        samples_.push_back(samples[i].second);
        TDouble10Vec4Vec wi(params.weightStyles().size(), TDouble10Vec(dimension));
        for (std::size_t j = 0u; j < params.priorWeights()[i].size(); ++j) {
            const TDouble2Vec &weight{params.priorWeights()[i][j]};
            for (std::size_t d = 0u; d < dimension; ++d) {
                wi[j][d] = weight[d];
            }
        }
        weights.push_back(wi);
        averageTime.add(static_cast<double>(samples[i].first));
    }

    m_Prior->addSamples(params.weightStyles(), samples_, weights);
    m_Prior->propagateForwardsByTime(params.propagationInterval());
    if (m_AnomalyModel) {
        m_AnomalyModel->propagateForwardsByTime(params.propagationInterval());
    }

    if (m_Controllers) {
        TDouble1VecVec errors[2];
        errors[0].reserve(samples.size());
        errors[1].reserve(samples.size());
        for (auto i : valueorder) {
            this->appendPredictionErrors(params.propagationInterval(), samples[i].second, errors);
        }
        {
            CDecayRateController &controller{(*m_Controllers)[E_TrendControl]};
            TDouble1Vec prediction(dimension);
            core_t::TTime time{static_cast<core_t::TTime>(CBasicStatistics::mean(averageTime))};
            for (std::size_t d = 0u; d < dimension; ++d) {
                prediction[d] = m_Trend[d]->mean(time);
            }
            double multiplier{controller.multiplier(prediction, errors[E_TrendControl],
                                                    this->params().bucketLength(),
                                                    this->params().learnRate(),
                                                    this->params().decayRate())};
            if (multiplier != 1.0) {
                for (const auto &trend : m_Trend) {
                    trend->decayRate(multiplier * trend->decayRate());
                }
                LOG_TRACE("trend decay rate = " << m_Trend[0]->decayRate());
            }
        }
        {
            CDecayRateController &controller{(*m_Controllers)[E_PriorControl]};
            TDouble1Vec prediction(m_Prior->marginalLikelihoodMean());
            double multiplier{controller.multiplier(prediction, errors[E_PriorControl],
                                                    this->params().bucketLength(),
                                                    this->params().learnRate(),
                                                    this->params().decayRate())};
            if (multiplier != 1.0) {
                m_Prior->decayRate(multiplier * m_Prior->decayRate());
                LOG_TRACE("prior decay rate = " << m_Prior->decayRate());
            }
        }
    }

    if (randomSample) {
        m_SlidingWindow.push_back({randomSample->first, randomSample->second});
    }

    return result;
}

void CMultivariateTimeSeriesModel::skipTime(core_t::TTime gap) {
    for (const auto &trend : m_Trend) {
        trend->skipTime(gap);
    }
}

CMultivariateTimeSeriesModel::TDouble2Vec
CMultivariateTimeSeriesModel::mode(core_t::TTime time,
                                   const maths_t::TWeightStyleVec &weightStyles,
                                   const TDouble2Vec4Vec &weights_) const {
    std::size_t dimension = this->dimension();

    TDouble2Vec result(dimension);

    TDouble10Vec4Vec weights(weights_.size());
    for (std::size_t i = 0u; i < weights_.size(); ++i) {
        for (std::size_t d = 0u; d < dimension; ++d) {
            weights[i].push_back(weights_[i][d]);
        }
    }

    TDouble10Vec mode(m_Prior->marginalLikelihoodMode(weightStyles, weights));

    for (std::size_t d = 0u; d < dimension; ++d) {
        result[d] = mode[d] + CBasicStatistics::mean(m_Trend[d]->baseline(time));
    }

    return result;
}

CMultivariateTimeSeriesModel::TDouble2Vec1Vec
CMultivariateTimeSeriesModel::correlateModes(core_t::TTime /*time*/,
                                             const maths_t::TWeightStyleVec & /*weightStyles*/,
                                             const TDouble2Vec4Vec1Vec & /*weights*/) const {
    return TDouble2Vec1Vec();
}

CMultivariateTimeSeriesModel::TDouble2Vec1Vec
CMultivariateTimeSeriesModel::residualModes(const maths_t::TWeightStyleVec &weightStyles,
                                            const TDouble2Vec4Vec &weights_) const {
    TDouble10Vec4Vec weights;
    weights.reserve(weights_.size());
    for (const auto &weight : weights_) {
        weights.emplace_back(weight[0]);
    }
    TDouble10Vec1Vec modes(m_Prior->marginalLikelihoodModes(weightStyles, weights));
    TDouble2Vec1Vec result;
    result.reserve(modes.size());
    for (const auto &mode : modes) {
        result.push_back(TDouble2Vec(mode));
    }
    return result;
}

void CMultivariateTimeSeriesModel::detrend(const TTime2Vec1Vec &time_,
                                           double confidenceInterval,
                                           TDouble2Vec1Vec &value) const {
    std::size_t dimension{this->dimension()};
    core_t::TTime time{time_[0][0]};
    for (std::size_t d = 0u; d < dimension; ++d) {
        value[0][d] = m_Trend[d]->detrend(time, value[0][d], confidenceInterval);
    }
}

CMultivariateTimeSeriesModel::TDouble2Vec
CMultivariateTimeSeriesModel::predict(core_t::TTime time,
                                      const TSizeDoublePr1Vec & /*correlated*/,
                                      TDouble2Vec hint) const {
    using TUnivariatePriorPtr = boost::shared_ptr<CPrior>;

    static const TSizeDoublePr10Vec CONDITION;

    std::size_t dimension{this->dimension()};
    double scale{1.0 - this->params().probabilityBucketEmpty()};

    if (hint.size() == dimension) {
        for (std::size_t d = 0u; d < dimension; ++d) {
            hint[d] = m_Trend[d]->detrend(time, hint[d], 0.0);
        }
    }

    TSize10Vec marginalize(dimension - 1);
    std::iota(marginalize.begin(), marginalize.end(), 1);

    TDouble2Vec result(dimension);
    TDouble10Vec mean(m_Prior->marginalLikelihoodMean());
    for (std::size_t d = 0u; d < dimension; --marginalize[std::min(d, dimension - 2)], ++d) {
        double seasonalOffset{0.0};
        if (m_Trend[d]->initialized()) {
            seasonalOffset = CBasicStatistics::mean(m_Trend[d]->baseline(time));
        }
        double median{mean[d]};
        if (!m_Prior->isNonInformative()) {
            TUnivariatePriorPtr marginal{m_Prior->univariate(marginalize, CONDITION).first};
            median = hint.empty() ? CBasicStatistics::mean(marginal->marginalLikelihoodConfidenceInterval(0.0)) :
                     marginal->nearestMarginalLikelihoodMean(hint[d]);
        }
        result[d] = scale * (seasonalOffset + median);
        if (m_IsNonNegative) {
            result[d] = std::max(result[d], 0.0);
        }
    }

    return result;
}

CMultivariateTimeSeriesModel::TDouble2Vec3Vec
CMultivariateTimeSeriesModel::confidenceInterval(core_t::TTime time,
                                                 double confidenceInterval,
                                                 const maths_t::TWeightStyleVec &weightStyles,
                                                 const TDouble2Vec4Vec &weights_) const {
    if (m_Prior->isNonInformative()) {
        return TDouble2Vec3Vec();
    }

    using TUnivariatePriorPtr = boost::shared_ptr<CPrior>;

    static const TSizeDoublePr10Vec CONDITION;

    std::size_t dimension{this->dimension()};
    double scale{1.0 - this->params().probabilityBucketEmpty()};

    TSize10Vec marginalize(dimension - 1);
    std::iota(marginalize.begin(), marginalize.end(), 1);

    TDouble2Vec3Vec result(3, TDouble2Vec(dimension));

    TDouble4Vec weights;
    for (std::size_t d = 0u; d < dimension; --marginalize[std::min(d, dimension - 2)], ++d) {
        double seasonalOffset{m_Trend[d]->initialized() ?
                              CBasicStatistics::mean(
                                  m_Trend[d]->baseline(time, confidenceInterval)) : 0.0};

        weights.clear();
        weights.reserve(weights_.size());
        for (const auto &weight : weights_) {
            weights.push_back(weight[d]);
        }

        TUnivariatePriorPtr marginal{m_Prior->univariate(marginalize, CONDITION).first};
        double median{CBasicStatistics::mean(marginal->marginalLikelihoodConfidenceInterval(0.0))};
        TDoubleDoublePr interval{
            marginal->marginalLikelihoodConfidenceInterval(confidenceInterval, weightStyles, weights)};

        result[0][d] = scale * (seasonalOffset + interval.first);
        result[1][d] = scale * (seasonalOffset + median);
        result[2][d] = scale * (seasonalOffset + interval.second);
        if (m_IsNonNegative) {
            result[0][d] = std::max(result[0][d], 0.0);
            result[1][d] = std::max(result[1][d], 0.0);
            result[2][d] = std::max(result[2][d], 0.0);
        }
    }

    return result;
}

bool CMultivariateTimeSeriesModel::forecast(core_t::TTime /*startTime*/,
                                            core_t::TTime /*endTime*/,
                                            double /*confidenceInterval*/,
                                            const TDouble2Vec & /*minimum*/,
                                            const TDouble2Vec & /*maximum*/,
                                            const TForecastPushDatapointFunc & /*forecastPushDataPointFunc*/,
                                            std::string &messageOut) {
    LOG_DEBUG(forecast::ERROR_MULTIVARIATE);
    messageOut = forecast::ERROR_MULTIVARIATE;
    return false;
}

bool CMultivariateTimeSeriesModel::probability(const CModelProbabilityParams &params,
                                               const TTime2Vec1Vec &time_,
                                               const TDouble2Vec1Vec &value,
                                               double &probability,
                                               TTail2Vec &tail,
                                               bool &conditional,
                                               TSize1Vec &mostAnomalousCorrelate) const {
    TSize2Vec coordinates(params.coordinates());
    if (coordinates.empty()) {
        coordinates.resize(this->dimension());
        std::iota(coordinates.begin(), coordinates.end(), 0);
    }

    probability = 1.0;
    tail.resize(coordinates.size(), maths_t::E_UndeterminedTail);
    conditional = false;
    mostAnomalousCorrelate.clear();

    std::size_t dimension{this->dimension()};
    core_t::TTime time{time_[0][0]};
    TDouble10Vec1Vec sample{TDouble10Vec(dimension)};
    TDouble10Vec4Vec1Vec weights{TDouble10Vec4Vec(params.weightStyles().size(),
                                                  TDouble10Vec(dimension))};
    for (std::size_t d = 0u; d < dimension; ++d) {
        sample[0][d] = m_Trend[d]->detrend(time, value[0][d],
                                           params.seasonalConfidenceInterval());
    }
    for (std::size_t i = 0u; i < params.weightStyles().size(); ++i) {
        for (std::size_t d = 0u; d < dimension; ++d) {
            weights[0][i][d] = params.weights()[0][i][d];
        }
    }
    bool bucketEmpty{params.bucketEmpty()[0][0]};
    double probabilityBucketEmpty{this->params().probabilityBucketEmpty()};

    CJointProbabilityOfLessLikelySamples pl_[2];
    CJointProbabilityOfLessLikelySamples pu_[2];

    TSize10Vec coordinate(1);
    TDouble10Vec2Vec pls;
    TDouble10Vec2Vec pus;
    TTail10Vec tail_;
    for (std::size_t i = 0u; i < coordinates.size(); ++i) {
        maths_t::EProbabilityCalculation calculation = params.calculation(i);
        coordinate[0] = coordinates[i];
        if (!m_Prior->probabilityOfLessLikelySamples(calculation,
                                                     params.weightStyles(),
                                                     sample, weights, coordinate,
                                                     pls, pus, tail_)) {
            LOG_ERROR("Failed to compute P(" << sample << " | weight = " << weights << ")");
            return false;
        }
        pl_[0].add(correctForEmptyBucket(calculation, value[0],
                                         bucketEmpty, probabilityBucketEmpty,
                                         pls[0][0]));
        pu_[0].add(correctForEmptyBucket(calculation, value[0],
                                         bucketEmpty, probabilityBucketEmpty,
                                         pus[0][0]));
        pl_[1].add(correctForEmptyBucket(calculation, value[0],
                                         bucketEmpty, probabilityBucketEmpty,
                                         pls[1][0]));
        pu_[1].add(correctForEmptyBucket(calculation, value[0],
                                         bucketEmpty, probabilityBucketEmpty,
                                         pus[1][0]));
        tail[i] = tail_[0];
    }
    double pl[2], pu[2];
    if (   !pl_[0].calculate(pl[0]) || !pu_[0].calculate(pu[0]) ||
           !pl_[1].calculate(pl[1]) || !pu_[1].calculate(pu[1])) {
        return false;
    }

    probability = (std::sqrt(pl[0] * pl[1]) + std::sqrt(pu[0] * pu[1])) / 2.0;

    if (m_AnomalyModel) {
        TDouble2Vec residual(dimension);
        TDouble10Vec nearest(m_Prior->nearestMarginalLikelihoodMean(sample[0]));
        TDouble2Vec scale(this->seasonalWeight(0.0, time));
        for (std::size_t i = 0u; i < dimension; ++i) {
            residual[i] = (sample[0][i] - nearest[i]) / std::max(std::sqrt(scale[i]), 1.0);
        }
        m_AnomalyModel->updateAnomaly(params, time, residual, probability);
        m_AnomalyModel->probability(params, time, probability);
        m_AnomalyModel->sampleAnomaly(params, time);
    }

    return true;
}

CMultivariateTimeSeriesModel::TDouble2Vec
CMultivariateTimeSeriesModel::winsorisationWeight(double derate,
                                                  core_t::TTime time,
                                                  const TDouble2Vec &value) const {
    TDouble2Vec result(this->dimension());

    std::size_t dimension{this->dimension()};
    TDouble2Vec scale(this->seasonalWeight(0.0, time));
    TDouble10Vec sample(dimension);
    for (std::size_t d = 0u; d < dimension; ++d) {
        sample[d] = m_Trend[d]->detrend(time, value[d], 0.0);
    }

    for (std::size_t d = 0u; d < dimension; ++d) {
        result[d] = computeWinsorisationWeight(*m_Prior, d, derate, scale[d], sample);
    }

    return result;
}

CMultivariateTimeSeriesModel::TDouble2Vec
CMultivariateTimeSeriesModel::seasonalWeight(double confidence, core_t::TTime time) const {
    TDouble2Vec result(this->dimension());
    TDouble10Vec variances(m_Prior->marginalLikelihoodVariances());
    for (std::size_t d = 0u, dimension = this->dimension(); d < dimension; ++d) {
        double scale{m_Trend[d]->scale(time, variances[d], confidence).second};
        result[d] = std::max(scale, this->params().minimumSeasonalVarianceScale());
    }
    return result;
}

uint64_t CMultivariateTimeSeriesModel::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_IsNonNegative);
    seed = CChecksum::calculate(seed, m_Controllers);
    seed = CChecksum::calculate(seed, m_Trend);
    seed = CChecksum::calculate(seed, m_Prior);
    seed = CChecksum::calculate(seed, m_AnomalyModel);
    return CChecksum::calculate(seed, m_SlidingWindow);
}

void CMultivariateTimeSeriesModel::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CUnivariateTimeSeriesModel");
    core::CMemoryDebug::dynamicSize("m_Controllers", m_Controllers, mem);
    core::CMemoryDebug::dynamicSize("m_Trend", m_Trend, mem);
    core::CMemoryDebug::dynamicSize("m_Prior", m_Prior, mem);
    core::CMemoryDebug::dynamicSize("m_AnomalyModel", m_AnomalyModel, mem);
    core::CMemoryDebug::dynamicSize("m_SlidingWindow", m_SlidingWindow, mem);
}

std::size_t CMultivariateTimeSeriesModel::memoryUsage(void) const {
    return core::CMemory::dynamicSize(m_Controllers)
           + core::CMemory::dynamicSize(m_Trend)
           + core::CMemory::dynamicSize(m_Prior)
           + core::CMemory::dynamicSize(m_AnomalyModel)
           + core::CMemory::dynamicSize(m_SlidingWindow);
}

bool CMultivariateTimeSeriesModel::acceptRestoreTraverser(const SModelRestoreParams &params,
                                                          core::CStateRestoreTraverser &traverser) {
    if (traverser.name() == VERSION_6_3_TAG) {
        while (traverser.next()) {
            const std::string &name{traverser.name()};
            RESTORE_BOOL(IS_NON_NEGATIVE_6_3_TAG, m_IsNonNegative)
            RESTORE(RNG_6_3_TAG, m_Rng.fromString(traverser.value()))
            RESTORE_SETUP_TEARDOWN(CONTROLLER_6_3_TAG,
                                   m_Controllers = boost::make_shared<TDecayRateController2Ary>(),
                                   core::CPersistUtils::restore(CONTROLLER_6_3_TAG, *m_Controllers, traverser),
                                   /**/)
            RESTORE_SETUP_TEARDOWN(TREND_6_3_TAG,
                                   m_Trend.push_back(TDecompositionPtr()),
                                   traverser.traverseSubLevel(boost::bind<bool>(
                                                                  CTimeSeriesDecompositionStateSerialiser(),
                                                                  boost::cref(params.s_DecompositionParams),
                                                                  boost::ref(m_Trend.back()), _1)),
                                   /**/)
            RESTORE(PRIOR_6_3_TAG, traverser.traverseSubLevel(boost::bind<bool>(
                                                                  CPriorStateSerialiser(),
                                                                  boost::cref(params.s_DistributionParams),
                                                                  boost::ref(m_Prior), _1)))
            RESTORE_SETUP_TEARDOWN(ANOMALY_MODEL_6_3_TAG,
                                   m_AnomalyModel = boost::make_shared<CTimeSeriesAnomalyModel>(),
                                   traverser.traverseSubLevel(boost::bind(&CTimeSeriesAnomalyModel::acceptRestoreTraverser,
                                                                          m_AnomalyModel.get(), boost::cref(params), _1)),
                                   /**/)
            RESTORE(SLIDING_WINDOW_6_3_TAG,
                    core::CPersistUtils::restore(SLIDING_WINDOW_6_3_TAG, m_SlidingWindow, traverser))
        }
    } else   {
        do {
            const std::string &name{traverser.name()};
            RESTORE_BOOL(IS_NON_NEGATIVE_OLD_TAG, m_IsNonNegative)
            RESTORE_SETUP_TEARDOWN(CONTROLLER_OLD_TAG,
                                   m_Controllers = boost::make_shared<TDecayRateController2Ary>(),
                                   core::CPersistUtils::restore(CONTROLLER_6_3_TAG, *m_Controllers, traverser),
                                   /**/)
            RESTORE_SETUP_TEARDOWN(TREND_OLD_TAG,
                                   m_Trend.push_back(TDecompositionPtr()),
                                   traverser.traverseSubLevel(boost::bind<bool>(
                                                                  CTimeSeriesDecompositionStateSerialiser(),
                                                                  boost::cref(params.s_DecompositionParams),
                                                                  boost::ref(m_Trend.back()), _1)),
                                   /**/)
            RESTORE(PRIOR_OLD_TAG, traverser.traverseSubLevel(boost::bind<bool>(
                                                                  CPriorStateSerialiser(),
                                                                  boost::cref(params.s_DistributionParams),
                                                                  boost::ref(m_Prior), _1)))
            RESTORE_SETUP_TEARDOWN(ANOMALY_MODEL_OLD_TAG,
                                   m_AnomalyModel = boost::make_shared<CTimeSeriesAnomalyModel>(),
                                   traverser.traverseSubLevel(boost::bind(&CTimeSeriesAnomalyModel::acceptRestoreTraverser,
                                                                          m_AnomalyModel.get(), boost::cref(params), _1)),
                                   /**/)
        } while (traverser.next());
    }
    return true;
}

void CMultivariateTimeSeriesModel::acceptPersistInserter(core::CStatePersistInserter &inserter) const {
    // Note that we don't persist this->params() because that state
    // is reinitialized.
    inserter.insertValue(VERSION_6_3_TAG, "");
    inserter.insertValue(IS_NON_NEGATIVE_6_3_TAG, static_cast<int>(m_IsNonNegative));
    if (m_Controllers) {
        core::CPersistUtils::persist(CONTROLLER_6_3_TAG, *m_Controllers, inserter);
    }
    for (const auto &trend : m_Trend) {
        inserter.insertLevel(TREND_6_3_TAG, boost::bind<void>(CTimeSeriesDecompositionStateSerialiser(),
                                                              boost::cref(*trend), _1));
    }
    inserter.insertLevel(PRIOR_6_3_TAG, boost::bind<void>(CPriorStateSerialiser(),
                                                          boost::cref(*m_Prior), _1));
    if (m_AnomalyModel) {
        inserter.insertLevel(ANOMALY_MODEL_6_3_TAG,
                             boost::bind(&CTimeSeriesAnomalyModel::acceptPersistInserter,
                                         m_AnomalyModel.get(), _1));
    }
    core::CPersistUtils::persist(SLIDING_WINDOW_6_3_TAG, m_SlidingWindow, inserter);
}

maths_t::EDataType CMultivariateTimeSeriesModel::dataType(void) const {
    return m_Prior->dataType();
}

const CMultivariateTimeSeriesModel::TTimeDouble2VecPrCBuf &CMultivariateTimeSeriesModel::slidingWindow(void) const {
    return m_SlidingWindow;
}

const CMultivariateTimeSeriesModel::TDecompositionPtr10Vec &CMultivariateTimeSeriesModel::trend(void) const {
    return m_Trend;
}

const CMultivariatePrior &CMultivariateTimeSeriesModel::prior(void) const {
    return *m_Prior;
}

CMultivariateTimeSeriesModel::EUpdateResult
CMultivariateTimeSeriesModel::updateTrend(const maths_t::TWeightStyleVec &weightStyles,
                                          const TTimeDouble2VecSizeTrVec &samples,
                                          const TDouble2Vec4VecVec &weights) {
    std::size_t dimension{this->dimension()};

    for (const auto &sample : samples) {
        if (sample.second.size() != dimension) {
            LOG_ERROR("Dimension mismatch: '"
                      << sample.second.size() << " != " << m_Trend.size() << "'");
            return E_Failure;
        }
    }

    // Time order is not reliable, for example if the data are polled
    // or for count feature, the times of all samples will be the same.
    TSizeVec timeorder(samples.size());
    std::iota(timeorder.begin(), timeorder.end(), 0);
    std::stable_sort(timeorder.begin(), timeorder.end(),
                     [&samples] (std::size_t lhs, std::size_t rhs) {
                return COrderings::lexicographical_compare(samples[lhs].first,
                                                           samples[lhs].second,
                                                           samples[rhs].first,
                                                           samples[rhs].second);
            });

    EUpdateResult result{E_Success};
    {
        TDouble4Vec weight(weightStyles.size());
        for (auto i : timeorder) {
            core_t::TTime time{samples[i].first};
            TDouble10Vec value(samples[i].second);
            for (std::size_t d = 0u; d < dimension; ++d) {
                for (std::size_t j = 0u; j < weights[i].size(); ++j) {
                    weight[j] = weights[i][j][d];
                }
                if (m_Trend[d]->addPoint(time, value[d], weightStyles, weight)) {
                    result = E_Reset;
                }
            }
        }
    }
    if (result == E_Reset) {
        m_Prior->setToNonInformative(0.0, m_Prior->decayRate());
        TDouble10Vec4Vec1Vec weight{{TDouble10Vec(
                                         dimension, std::max(this->params().learnRate(),
                                                             5.0 / static_cast<double>(SLIDING_WINDOW_SIZE)))}};
        for (const auto &value : m_SlidingWindow) {
            TDouble10Vec1Vec sample{TDouble10Vec(dimension)};
            for (std::size_t i = 0u; i < dimension; ++i) {
                sample[0][i] = m_Trend[i]->detrend(value.first, value.second[i], 0.0);
            }
            m_Prior->addSamples(CConstantWeights::COUNT, sample, weight);
        }
        if (m_Controllers) {
            m_Prior->decayRate(  m_Prior->decayRate()
                                 / (*m_Controllers)[E_PriorControl].multiplier());
            for (auto &trend : m_Trend) {
                trend->decayRate(  trend->decayRate()
                                   / (*m_Controllers)[E_TrendControl].multiplier());
            }
            for (auto &controller : *m_Controllers) {
                controller.reset();
            }
        }
        if (m_AnomalyModel) {
            m_AnomalyModel->reset();
        }
    }

    return result;
}

void CMultivariateTimeSeriesModel::appendPredictionErrors(double interval,
                                                          const TDouble2Vec &sample,
                                                          TDouble1VecVec (&result)[2]) {
    if (auto error = predictionError(interval, m_Prior, sample)) {
        result[E_PriorControl].push_back(*error);
    }
    if (auto error = predictionError(m_Trend, sample)) {
        result[E_TrendControl].push_back(*error);
    }
}

std::size_t CMultivariateTimeSeriesModel::dimension(void) const {
    return m_Prior->dimension();
}

}
}
