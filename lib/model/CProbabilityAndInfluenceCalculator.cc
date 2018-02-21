/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CProbabilityAndInfluenceCalculator.h>

#include <core/CLogger.h>

#include <maths/CBasicStatistics.h>
#include <maths/CModel.h>
#include <maths/CMultivariatePrior.h>
#include <maths/COrderings.h>
#include <maths/CTimeSeriesDecomposition.h>
#include <maths/CTimeSeriesModel.h>
#include <maths/CTools.h>

#include <model/CAnnotatedProbabilityBuilder.h>
#include <model/CStringStore.h>

namespace ml
{
namespace model
{
namespace
{

typedef CProbabilityAndInfluenceCalculator::TSize1Vec TSize1Vec;
typedef CProbabilityAndInfluenceCalculator::TSize2Vec TSize2Vec;
typedef CProbabilityAndInfluenceCalculator::TDouble1Vec TDouble1Vec;
typedef CProbabilityAndInfluenceCalculator::TDouble2Vec TDouble2Vec;
typedef CProbabilityAndInfluenceCalculator::TDouble4Vec TDouble4Vec;
typedef CProbabilityAndInfluenceCalculator::TDouble2Vec1Vec TDouble2Vec1Vec;
typedef CProbabilityAndInfluenceCalculator::TDouble4Vec1Vec TDouble4Vec1Vec;
typedef CProbabilityAndInfluenceCalculator::TDouble1VecDoublePr TDouble1VecDoublePr;
typedef CProbabilityAndInfluenceCalculator::TBool2Vec TBool2Vec;
typedef CProbabilityAndInfluenceCalculator::TTime2Vec TTime2Vec;
typedef CProbabilityAndInfluenceCalculator::TTime2Vec1Vec TTime2Vec1Vec;
typedef CProbabilityAndInfluenceCalculator::TStrCRefDouble1VecDoublePrPr TStrCRefDouble1VecDoublePrPr;
typedef CProbabilityAndInfluenceCalculator::TStrCRefDouble1VecDoublePrPrVec TStrCRefDouble1VecDoublePrPrVec;
typedef CProbabilityAndInfluenceCalculator::TStrCRefDouble1VecDouble1VecPrPr TStrCRefDouble1VecDouble1VecPrPr;
typedef CProbabilityAndInfluenceCalculator::TStrCRefDouble1VecDouble1VecPrPrVec TStrCRefDouble1VecDouble1VecPrPrVec;
typedef CProbabilityAndInfluenceCalculator::TStoredStringPtrStoredStringPtrPr TStoredStringPtrStoredStringPtrPr;
typedef CProbabilityAndInfluenceCalculator::TStoredStringPtrStoredStringPtrPrDoublePr TStoredStringPtrStoredStringPtrPrDoublePr;
typedef CProbabilityAndInfluenceCalculator::TStoredStringPtrStoredStringPtrPrDoublePrVec TStoredStringPtrStoredStringPtrPrDoublePrVec;
typedef core::CSmallVector<maths_t::ETail, 2> TTail2Vec;
typedef core::CSmallVector<maths_t::EProbabilityCalculation, 2> TProbabilityCalculation2Vec;
typedef core::CSmallVector<TDouble2Vec, 4> TDouble2Vec4Vec;
typedef core::CSmallVector<TDouble2Vec4Vec, 1> TDouble2Vec4Vec1Vec;
typedef std::pair<std::size_t, double> TSizeDoublePr;
typedef core::CSmallVector<TSizeDoublePr, 1> TSizeDoublePr1Vec;

//! Get the canonical influence string pointer.
core::CStoredStringPtr canonical(const std::string &influence)
{
    return CStringStore::influencers().get(influence);
}

//! \brief Orders two value influences by decreasing influence.
class CDecreasingValueInfluence
{
    public:
        CDecreasingValueInfluence(maths_t::ETail tail) : m_Tail(tail) {}

        bool operator()(const TStrCRefDouble1VecDoublePrPr &lhs,
                        const TStrCRefDouble1VecDoublePrPr &rhs) const
        {
            return m_Tail == maths_t::E_LeftTail ?
                   lhs.second.first < rhs.second.first :
                   lhs.second.first > rhs.second.first;
        }

    private:
        maths_t::ETail m_Tail;
};

//! \brief Orders two mean influences by decreasing influence.
class CDecreasingMeanInfluence
{
    public:
        typedef maths::CBasicStatistics::SSampleMean<double>::TAccumulator TMeanAccumulator;

    public:
        CDecreasingMeanInfluence(maths_t::ETail tail, const TDouble2Vec &value, double count) :
                m_Tail(tail),
                m_Mean(maths::CBasicStatistics::accumulator(count, value[0]))
        {}

        bool operator()(const TStrCRefDouble1VecDoublePrPr &lhs,
                        const TStrCRefDouble1VecDoublePrPr &rhs) const
        {
            TMeanAccumulator l = m_Mean - maths::CBasicStatistics::accumulator(lhs.second.second,
                                                                              lhs.second.first[0]);
            TMeanAccumulator r = m_Mean - maths::CBasicStatistics::accumulator(rhs.second.second,
                                                                               rhs.second.first[0]);
            double ml = maths::CBasicStatistics::mean(l);
            double nl = maths::CBasicStatistics::count(l);
            double mr = maths::CBasicStatistics::mean(r);
            double nr = maths::CBasicStatistics::count(r);
            return m_Tail == maths_t::E_LeftTail ?
                   maths::COrderings::lexicographical_compare(mr, nl, ml, nr) :
                   maths::COrderings::lexicographical_compare(ml, nl, mr, nr);
        }

    private:
        maths_t::ETail m_Tail;
        TMeanAccumulator m_Mean;
};

//! \brief Orders two variance influences by decreasing influence.
class CDecreasingVarianceInfluence
{
    public:
         typedef maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator TMeanVarAccumulator;

    public:
        CDecreasingVarianceInfluence(maths_t::ETail tail, const TDouble2Vec &value, double count) :
                m_Tail(tail),
                m_Variance(maths::CBasicStatistics::accumulator(count, value[1], value[0]))
        {}

        bool operator()(const TStrCRefDouble1VecDoublePrPr &lhs,
                        const TStrCRefDouble1VecDoublePrPr &rhs) const
        {
            TMeanVarAccumulator l = m_Variance - maths::CBasicStatistics::accumulator(lhs.second.second,
                                                                                      lhs.second.first[1],
                                                                                      lhs.second.first[0]);
            TMeanVarAccumulator r = m_Variance - maths::CBasicStatistics::accumulator(rhs.second.second,
                                                                                      rhs.second.first[1],
                                                                                      rhs.second.first[0]);
            double vl = maths::CBasicStatistics::maximumLikelihoodVariance(l);
            double nl = maths::CBasicStatistics::count(l);
            double vr = maths::CBasicStatistics::maximumLikelihoodVariance(r);
            double nr = maths::CBasicStatistics::count(r);
            return m_Tail == maths_t::E_LeftTail ?
                   maths::COrderings::lexicographical_compare(vr, nl, vl, nr) :
                   maths::COrderings::lexicographical_compare(vl, nl, vr, nr);
        }

    private:
        maths_t::ETail m_Tail;
        TMeanVarAccumulator m_Variance;
};

//! A safe ratio function \p numerator / \p denominator dealing
//! with the case \p n and/or \p d are zero.
double ratio(double numerator, double denominator, double zeroDividedByZero)
{
    if (denominator == 0.0)
    {
        if (numerator == 0.0)
        {
            return zeroDividedByZero;
        }
        return numerator < 0.0 ? -std::numeric_limits<double>::max() :
                                  std::numeric_limits<double>::max();
    }
    return numerator / denominator;
}

//! \brief Computes the value of summed statistics on the set difference.
class CValueDifference
{
    public:
        //! Features.
        void operator()(const TDouble2Vec &v, double n,
                        const TDouble1Vec &vi, double ni,
                        maths::CModelProbabilityParams &params,
                        TDouble2Vec &difference) const
        {
            params.addBucketEmpty(TBool2Vec{n == ni});
            for (std::size_t i = 0u; i < v.size(); ++i)
            {
                difference[i] = v[i] - vi[i];
            }
        }

        //! Correlates.
        void operator()(const TDouble2Vec &v, const TDouble2Vec &n,
                        const TDouble1Vec &vi, const TDouble1Vec &ni,
                        maths::CModelProbabilityParams &params,
                        TDouble2Vec &difference) const
        {
            TBool2Vec bucketEmpty(2);
            for (std::size_t i = 0u; i < v.size(); ++i)
            {
                bucketEmpty[i] = ((n[i] - ni[i]) == 0);
                difference[i]  = v[i] - vi[i];

            }
            params.addBucketEmpty(bucketEmpty);
        }
};

//! \brief Computes the value of min, max, dc, etc on the set intersection.
class CValueIntersection
{
    public:
        //! Features.
        void operator()(const TDouble2Vec &/*v*/, double /*n*/,
                        const TDouble1Vec &vi, double ni,
                        maths::CModelProbabilityParams &params,
                        TDouble2Vec &intersection) const
        {
            params.addBucketEmpty(TBool2Vec{ni == 0});
            for (std::size_t i = 0u; i < vi.size(); ++i)
            {
                intersection[i] = vi[i];
            }
        }

        //! Correlates.
        void operator()(const TDouble2Vec &/*v*/, const TDouble2Vec &/*n*/,
                        const TDouble1Vec &vi, const TDouble1Vec &ni,
                        maths::CModelProbabilityParams &params,
                        TDouble2Vec &intersection) const
        {
            TBool2Vec bucketEmpty(2);
            for (std::size_t i = 0u; i < vi.size(); ++i)
            {
                bucketEmpty[i]  = (ni[i] == 0);
                intersection[i] = vi[i];
            }
            params.addBucketEmpty(bucketEmpty);
        }
};

//! \brief Computes the value of the mean statistic on a set difference.
class CMeanDifference
{
    public:
        //! Features.
        void operator()(const TDouble2Vec &v, double n,
                        const TDouble1Vec &vi, double ni,
                        maths::CModelProbabilityParams &params,
                        TDouble2Vec &difference) const
        {
            params.addBucketEmpty(TBool2Vec{n == ni});
            for (std::size_t d = 0u; d < v.size(); ++d)
            {
                for (std::size_t i = 0u; i < params.weightStyles().size(); ++i)
                {
                    if (params.weightStyles()[i] == maths_t::E_SampleCountVarianceScaleWeight)
                    {
                        params.weights()[0][i][d] *= n / (n - ni);
                        break;
                    }
                }
                difference[d] = maths::CBasicStatistics::mean(
                                    maths::CBasicStatistics::accumulator( n,  v[d])
                                  - maths::CBasicStatistics::accumulator(ni, vi[d]));
            }
        }

        //! Correlates.
        void operator()(const TDouble2Vec &v, const TDouble2Vec &n,
                        const TDouble1Vec &vi, const TDouble1Vec &ni,
                        maths::CModelProbabilityParams &params,
                        TDouble2Vec &difference) const
        {
            TBool2Vec bucketEmpty(2);
            for (std::size_t d = 0u; d < 2; ++d)
            {
                bucketEmpty[d] = ((n[d] - ni[d]) == 0);
                for (std::size_t i = 0u; i < params.weightStyles().size(); ++i)
                {
                    if (params.weightStyles()[i] == maths_t::E_SampleCountVarianceScaleWeight)
                    {
                        params.weights()[0][i][d] *= n[d] / (n[d] - ni[d]);
                        break;
                    }
                }
                difference[d] = maths::CBasicStatistics::mean(
                                    maths::CBasicStatistics::accumulator( n[d],  v[d])
                                  - maths::CBasicStatistics::accumulator(ni[d], vi[d]));
            }
            params.addBucketEmpty(bucketEmpty);
        }
};

//! \brief Computes the value of the variance statistic on a set difference.
class CVarianceDifference
{
    public:
        //! Features.
        void operator()(const TDouble1Vec &v, double n,
                        const TDouble1Vec &vi, double ni,
                        maths::CModelProbabilityParams &params,
                        TDouble2Vec &difference) const
        {
            std::size_t dimension = v.size() / 2;
            params.addBucketEmpty(TBool2Vec{n == ni});
            for (std::size_t d = 0u; d < dimension; ++d)
            {
                for (std::size_t i = 0u; i < params.weightStyles().size(); ++i)
                {
                    if (params.weightStyles()[i] == maths_t::E_SampleCountVarianceScaleWeight)
                    {
                        params.weights()[0][i][d] *= n / (n - ni);
                        break;
                    }
                }
                difference[d] = maths::CBasicStatistics::maximumLikelihoodVariance(
                                    maths::CBasicStatistics::accumulator( n,  v[dimension + d],  v[d])
                                  - maths::CBasicStatistics::accumulator(ni, vi[dimension + d], vi[d]));
            }
        }

        //! Correlates.
        void operator()(const TDouble2Vec &v, const TDouble2Vec &n,
                        const TDouble1Vec &vi, const TDouble1Vec &ni,
                        maths::CModelProbabilityParams &params,
                        TDouble2Vec &difference) const
        {
            TBool2Vec bucketEmpty(2);
            for (std::size_t d = 0u; d < 2; ++d)
            {
                bucketEmpty[d] = ((n[d] - ni[d]) == 0);
                for (std::size_t i = 0u; i < params.weightStyles().size(); ++i)
                {
                    if (params.weightStyles()[i] == maths_t::E_SampleCountVarianceScaleWeight)
                    {
                        params.weights()[0][i][d] *= n[d] / (n[d] - ni[d]);
                        break;
                    }
                }
                difference[d] =  maths::CBasicStatistics::maximumLikelihoodVariance(
                                     maths::CBasicStatistics::accumulator( n[d],  v[2 + d],  v[d])
                                   - maths::CBasicStatistics::accumulator(ni[d], vi[2 + d], vi[d]));
            }
            params.addBucketEmpty(bucketEmpty);
        }
};

//! Sets all influences to one.
//!
//! \param[in] influencerName The name of the influencer field.
//! \param[in] influencerValues The feature values for the intersection
//! of the records in \p value with distinct values of \p influenceName.
//! \param[out] result Filled in with the influences of \p value.
template<typename INFLUENCER_VALUES>
void doComputeIndicatorInfluences(const core::CStoredStringPtr &influencerName,
                                  const INFLUENCER_VALUES &influencerValues,
                                  TStoredStringPtrStoredStringPtrPrDoublePrVec &result)
{
    result.reserve(influencerValues.size());
    for (const auto &influencerValue : influencerValues)
    {
        result.emplace_back(TStoredStringPtrStoredStringPtrPr(influencerName,
                                            canonical(influencerValue.first)), 1.0);
    }
}

//! The influence calculation for features using \p computeSample to
//! get the statistics and \p computeInfluence to compute the influences
//! from the corresponding probabilities.
//!
//! \param[in] computeInfluencedValue The function to compute the
//! influenced feature value for which to compute the probability.
//! \param[in] computeInfluence The function to compute influence.
//! \param[in] model The model to use to compute the probability.
//! \param[in] elapsedTime The time elapsed since the model was created.
//! \param[in] params The parameters need to compute the probability.
//! \param[in] time The time of \p value.
//! \param[in] value The influenced feature value.
//! \param[in] count The measurement count in \p value.
//! \param[in] probability The probability of \p value.
//! \param[in] influencerName The name of the influencer field.
//! \param[in] influencerValues The feature values for the intersection
//! of the records in \p value with distinct values of \p influenceName.
//! \param[in] cutoff The value at which there is no influence.
//! \param[in] includeCutoff If true then add in values for influences
//! less than the cutoff with estimated influence.
//! \param[out] result Filled in with the influences of \p value.
template<typename COMPUTE_INFLUENCED_VALUE, typename COMPUTE_INFLUENCE>
void doComputeInfluences(model_t::EFeature feature,
                         COMPUTE_INFLUENCED_VALUE computeInfluencedValue,
                         COMPUTE_INFLUENCE computeInfluence,
                         const maths::CModel &model,
                         core_t::TTime elapsedTime,
                         maths::CModelProbabilityParams &params,
                         const TTime2Vec1Vec &time,
                         const TDouble2Vec &value,
                         double count,
                         double probability,
                         const core::CStoredStringPtr &influencerName,
                         const TStrCRefDouble1VecDoublePrPrVec &influencerValues,
                         double cutoff,
                         bool includeCutoff,
                         TStoredStringPtrStoredStringPtrPrDoublePrVec &result)
{
    if (influencerValues.size() == 1)
    {
        result.emplace_back(TStoredStringPtrStoredStringPtrPr(influencerName,
                                            canonical(influencerValues[0].first)), 1.0);
        return;
    }
    if (probability == 1.0)
    {
        doComputeIndicatorInfluences(influencerName, influencerValues, result);
        return;
    }

    std::size_t dimension = model_t::dimension(feature);
    result.reserve(influencerValues.size());

    // Declared outside the loop to minimize the number of times they are created.
    TDouble2Vec1Vec influencedValue{TDouble2Vec(dimension)};
    TTail2Vec tail;
    TSize1Vec mostAnomalousCorrelate;

    double logp = maths::CTools::fastLog(probability);
    TDouble2Vec4Vec1Vec weights(params.weights());
    for (auto i = influencerValues.begin(); i != influencerValues.end(); ++i)
    {
        params.weights(weights)
              .updateAnomalyModel(false);

        computeInfluencedValue(value, count,
                               i->second.first,
                               i->second.second,
                               params, influencedValue[0]);

        double pi;
        bool conditional;
        if (!model.probability(params, time, influencedValue,
                               pi, tail, conditional, mostAnomalousCorrelate))
        {
            LOG_ERROR("Failed to compute P(" << influencedValue[0]
                      << " | influencer = " << core::CContainerPrinter::print(*i) << ")");
            continue;
        }
        pi = maths::CTools::truncate(pi, maths::CTools::smallestProbability(), 1.0);
        pi = model_t::adjustProbability(feature, elapsedTime, pi);

        double influence = computeInfluence(logp, maths::CTools::fastLog(pi));

        LOG_TRACE("log(p) = " << logp
                  << ", tail = " << core::CContainerPrinter::print(tail)
                  << ", v(i) = " << core::CContainerPrinter::print(influencedValue)
                  << ", log(p(i)) = " << ::log(pi)
                  << ", weight = " << core::CContainerPrinter::print(params.weights())
                  << ", influence = " << influence
                  << ", influencer field value = " << i->first.get());

        if (dimension == 1 && influence >= cutoff)
        {
            result.emplace_back(TStoredStringPtrStoredStringPtrPr(influencerName, canonical(i->first)),
                                influence);
        }
        else if (dimension == 1)
        {
            if (includeCutoff)
            {
                result.emplace_back(TStoredStringPtrStoredStringPtrPr(influencerName, canonical(i->first)),
                                    influence);
                for (++i; i != influencerValues.end(); ++i)
                {
                    result.emplace_back(TStoredStringPtrStoredStringPtrPr(influencerName, canonical(i->first)),
                                        0.5 * influence);
                }
            }
            break;
        }
        else if (influence >= cutoff)
        {
            result.emplace_back(TStoredStringPtrStoredStringPtrPr(influencerName, canonical(i->first)),
                                influence);
        }
        else if (includeCutoff)
        {
            result.emplace_back(TStoredStringPtrStoredStringPtrPr(influencerName, canonical(i->first)),
                                0.5 * influence);
        }
    }
}

//! Implement the influence calculation for correlates of univariate
//! features using \p computeSample to get the statistics and
//! \p computeInfluence to compute the influences from the corresponding
//! probabilities.
template<typename COMPUTE_INFLUENCED_VALUE, typename COMPUTE_INFLUENCE>
void doComputeCorrelateInfluences(model_t::EFeature feature,
                                  COMPUTE_INFLUENCED_VALUE computeInfluencedValue,
                                  COMPUTE_INFLUENCE computeInfluence,
                                  const maths::CModel &model,
                                  core_t::TTime elapsedTime,
                                  maths::CModelProbabilityParams &params,
                                  const TTime2Vec &time,
                                  const TDouble2Vec &value,
                                  const TDouble2Vec &count,
                                  double probability,
                                  const core::CStoredStringPtr &influencerName,
                                  const TStrCRefDouble1VecDouble1VecPrPrVec &influencerValues,
                                  double cutoff,
                                  bool includeCutoff,
                                  TStoredStringPtrStoredStringPtrPrDoublePrVec &result)
{
    if (influencerValues.size() == 1)
    {
        result.emplace_back(TStoredStringPtrStoredStringPtrPr(influencerName,
                                            canonical(influencerValues[0].first)), 1.0);
        return;
    }
    if (probability == 1.0)
    {
        doComputeIndicatorInfluences(influencerName, influencerValues, result);
        return;
    }

    result.reserve(influencerValues.size());

    // Declared outside the loop to minimize the number of times they are created.
    TDouble2Vec1Vec influencedValue{TDouble2Vec(2)};
    TTail2Vec tail;
    TSize1Vec mostAnomalousCorrelate;

    double logp = ::log(probability);
    TDouble2Vec4Vec1Vec weights(params.weights());
    for (const auto &influence_ : influencerValues)
    {
        params.weights(weights)
              .updateAnomalyModel(false);

        computeInfluencedValue(value, count,
                               influence_.second.first,
                               influence_.second.second,
                               params, influencedValue[0]);

        double pi;
        bool conditional;
        if (!model.probability(params, TTime2Vec1Vec{time}, influencedValue,
                               pi, tail, conditional, mostAnomalousCorrelate))
        {
            LOG_ERROR("Failed to compute P(" << core::CContainerPrinter::print(influencedValue)
                      << " | influencer = " << core::CContainerPrinter::print(influence_) << ")");
            continue;
        }
        pi = maths::CTools::truncate(pi, maths::CTools::smallestProbability(), 1.0);
        pi = model_t::adjustProbability(feature, elapsedTime, pi);

        double influence = computeInfluence(logp, ::log(pi));

        LOG_TRACE("log(p) = " << logp
                  << ", v(i) = " << core::CContainerPrinter::print(influencedValue)
                  << ", log(p(i)) = " << ::log(pi)
                  << ", weight(i) = " << core::CContainerPrinter::print(params.weights())
                  << ", influence = " << influence
                  << ", influencer field value = " << influence_.first.get());

        if (includeCutoff || influence >= cutoff)
        {
            result.emplace_back(TStoredStringPtrStoredStringPtrPr(influencerName,
                                                canonical(influence_.first)), influence);
        }
    }
}

}

CProbabilityAndInfluenceCalculator::CProbabilityAndInfluenceCalculator(double cutoff) :
        m_Cutoff(cutoff),
        m_InfluenceCalculator(0),
        m_ProbabilityTemplate(CModelTools::CProbabilityAggregator::E_Min),
        m_Probability(CModelTools::CProbabilityAggregator::E_Min),
        m_ProbabilityCache(0)
{}

bool CProbabilityAndInfluenceCalculator::empty(void) const
{
    return m_Probability.empty();
}

double CProbabilityAndInfluenceCalculator::cutoff(void) const
{
    return m_Cutoff;
}

void CProbabilityAndInfluenceCalculator::plugin(const CInfluenceCalculator &influenceCalculator)
{
    m_InfluenceCalculator = &influenceCalculator;
}

void CProbabilityAndInfluenceCalculator::addAggregator(const maths::CJointProbabilityOfLessLikelySamples &aggregator)
{
    m_ProbabilityTemplate.add(aggregator);
    m_Probability.add(aggregator);
}

void CProbabilityAndInfluenceCalculator::addAggregator(const maths::CProbabilityOfExtremeSample &aggregator)
{
    m_ProbabilityTemplate.add(aggregator);
    m_Probability.add(aggregator);
}

void CProbabilityAndInfluenceCalculator::addCache(CModelTools::CProbabilityCache &cache)
{
    m_ProbabilityCache = &cache;
}

void CProbabilityAndInfluenceCalculator::add(const CProbabilityAndInfluenceCalculator &other, double weight)
{
    double p = 0.0;
    if (!other.m_Probability.calculate(p))
    {
        return;
    }
    if (!other.m_Probability.empty())
    {
        m_Probability.add(p, weight);
    }
    for (const auto &aggregator : other.m_InfluencerProbabilities)
    {
        if (aggregator.second.calculate(p))
        {
            auto &aggregator_ = m_InfluencerProbabilities.emplace(aggregator.first,
                                                                  other.m_ProbabilityTemplate).first->second;
            if (!aggregator.second.empty())
            {
                aggregator_.add(p, weight);
            }
        }
    }
}

bool CProbabilityAndInfluenceCalculator::addAttributeProbability(const core::CStoredStringPtr &attribute,
                                                                 std::size_t cid,
                                                                 double pAttribute,
                                                                 SParams &params,
                                                                 CAnnotatedProbabilityBuilder &builder,
                                                                 double weight)
{
    model_t::CResultType type;
    TSize1Vec mostAnomalousCorrelate;
    if (this->addProbability(params.s_Feature, cid, *params.s_Model,
                             params.s_ElapsedTime,
                             params.s_ComputeProbabilityParams,
                             params.s_Time, params.s_Value,
                             params.s_Probability, params.s_Tail,
                             type, mostAnomalousCorrelate, weight))
    {
        static const TStoredStringPtr1Vec NO_CORRELATED_ATTRIBUTES;
        static const TSizeDoublePr1Vec NO_CORRELATES;
        builder.addAttributeProbability(cid, attribute,
                                        pAttribute, params.s_Probability,
                                        model_t::CResultType::E_Unconditional,
                                        params.s_Feature,
                                        NO_CORRELATED_ATTRIBUTES, NO_CORRELATES);
        return true;
    }
    return false;
}

bool CProbabilityAndInfluenceCalculator::addAttributeProbability(const core::CStoredStringPtr &attribute,
                                                                 std::size_t cid,
                                                                 double pAttribute,
                                                                 SCorrelateParams &params,
                                                                 CAnnotatedProbabilityBuilder &builder,
                                                                 double weight)
{
    model_t::CResultType type;
    params.s_MostAnomalousCorrelate.clear();
    if (this->addProbability(params.s_Feature, cid, *params.s_Model,
                             params.s_ElapsedTime,
                             params.s_ComputeProbabilityParams,
                             params.s_Times, params.s_Values,
                             params.s_Probability, params.s_Tail,
                             type, params.s_MostAnomalousCorrelate, weight))
    {
        TStoredStringPtr1Vec correlatedLabels_;
        TSizeDoublePr1Vec correlated_;
        if (!params.s_MostAnomalousCorrelate.empty())
        {
            std::size_t i = params.s_MostAnomalousCorrelate[0];
            correlatedLabels_.push_back(params.s_CorrelatedLabels[i]);
            correlated_.emplace_back(params.s_Correlated[i],
                                     params.s_Values[i][params.s_Variables[i][1]]);
        }
        builder.addAttributeProbability(cid, attribute,
                                        pAttribute, params.s_Probability,
                                        type, params.s_Feature,
                                        correlatedLabels_, correlated_);
        return true;
    }
    return false;
}

bool CProbabilityAndInfluenceCalculator::addProbability(model_t::EFeature feature,
                                                        std::size_t id,
                                                        const maths::CModel &model,
                                                        core_t::TTime elapsedTime,
                                                        const maths::CModelProbabilityParams &params,
                                                        const TTime2Vec1Vec &time,
                                                        const TDouble2Vec1Vec &values_,
                                                        double &probability,
                                                        TTail2Vec &tail,
                                                        model_t::CResultType &type,
                                                        TSize1Vec &mostAnomalousCorrelate,
                                                        double weight)
{
    if (values_.empty())
    {
        return false;
    }

    // Maybe check the cache.
    if (!model_t::isConstant(feature) && m_ProbabilityCache)
    {
        TDouble2Vec1Vec values(model_t::stripExtraStatistics(feature, values_));
        model.detrend(time, params.seasonalConfidenceInterval(), values);
        bool conditional;
        if (m_ProbabilityCache->lookup(feature, id, values, probability, tail,
                                       conditional, mostAnomalousCorrelate))
        {
            m_Probability.add(probability, weight);
            type.set(conditional ? model_t::CResultType::E_Conditional :
                                   model_t::CResultType::E_Unconditional);
            return true;
        }
    }

    // The accuracy from the cache isn't good enough so fall back
    // to calculating.
    TDouble2Vec1Vec values(model_t::stripExtraStatistics(feature, values_));
    bool conditional;
    if (model.probability(params, time, values,
                          probability, tail,
                          conditional, mostAnomalousCorrelate))
    {
        if (!model_t::isConstant(feature))
        {
            probability = model_t::adjustProbability(feature, elapsedTime, probability);
            m_Probability.add(probability, weight);
            type.set(conditional ? model_t::CResultType::E_Conditional :
                                   model_t::CResultType::E_Unconditional);
            if (m_ProbabilityCache)
            {
                m_ProbabilityCache->addModes(feature, id, model);
                m_ProbabilityCache->addProbability(feature, id, values, probability, tail,
                                                   conditional, mostAnomalousCorrelate);
            }
        }
        else
        {
            type.set(model_t::CResultType::E_Unconditional);
            mostAnomalousCorrelate.clear();
        }
        return true;
    }

    return false;
}

void CProbabilityAndInfluenceCalculator::addProbability(double probability, double weight)
{
    m_Probability.add(probability, weight);
    for (auto &&aggregator : m_InfluencerProbabilities)
    {
        aggregator.second.add(probability, weight);
    }
}

void CProbabilityAndInfluenceCalculator::addInfluences(const std::string &influencerName,
                                                       const TStrCRefDouble1VecDoublePrPrVec &influencerValues,
                                                       SParams &params,
                                                       double weight)
{
    if (!m_InfluenceCalculator)
    {
        LOG_ERROR("No influence calculator plug-in: can't compute influence");
        return;
    }

    const std::string *influencerValue = 0;
    if (influencerValues.empty())
    {
        for (std::size_t i = 0u; i < params.s_PartitioningFields.size(); ++i)
        {
            if (params.s_PartitioningFields[i].first.get() == influencerName)
            {
                influencerValue = params.s_PartitioningFields[i].second.get_pointer();
                break;
            }
        }
        if (!influencerValue)
        {
            return;
        }
    }

    double logp = ::log(std::max(params.s_Probability, maths::CTools::smallestProbability()));

    params.s_InfluencerName   = canonical(influencerName);
    params.s_InfluencerValues = influencerValues;
    params.s_Cutoff = 0.5 / std::max(-logp, 1.0);
    params.s_IncludeCutoff = true;

    m_InfluenceCalculator->computeInfluences(params);
    m_Influences.swap(params.s_Influences);
    if (m_Influences.empty() && influencerValue)
    {
        m_Influences.emplace_back(TStoredStringPtrStoredStringPtrPr(params.s_InfluencerName,
                                                  canonical(*influencerValue)), 1.0);
    }
    this->commitInfluences(params.s_Feature, logp, weight);
}

void CProbabilityAndInfluenceCalculator::addInfluences(const std::string &influencerName,
                                                       const TStrCRefDouble1VecDouble1VecPrPrVecVec &influencerValues,
                                                       SCorrelateParams &params,
                                                       double weight)
{
    if (!m_InfluenceCalculator)
    {
        LOG_ERROR("No influence calculator plug-in: can't compute influence");
        return;
    }

    const std::string *influencerValue = 0;
    if (influencerValues.empty())
    {
        for (std::size_t i = 0u; i < params.s_PartitioningFields.size(); ++i)
        {
            if (params.s_PartitioningFields[i].first.get() == influencerName)
            {
                influencerValue = params.s_PartitioningFields[i].second.get_pointer();
                break;
            }
        }
        if (!influencerValue)
        {
            return;
        }
    }

    double logp = ::log(std::max(params.s_Probability, maths::CTools::smallestProbability()));

    params.s_InfluencerName   = canonical(influencerName);
    params.s_InfluencerValues = influencerValues[params.s_MostAnomalousCorrelate[0]];
    params.s_Cutoff = 0.5 / std::max(-logp, 1.0);
    params.s_IncludeCutoff = true;

    m_InfluenceCalculator->computeInfluences(params);
    m_Influences.swap(params.s_Influences);
    if (m_Influences.empty() && influencerValue)
    {
        m_Influences.emplace_back(TStoredStringPtrStoredStringPtrPr(params.s_InfluencerName,
                                                  canonical(*influencerValue)), 1.0);
    }
    this->commitInfluences(params.s_Feature, logp, weight);
}

bool CProbabilityAndInfluenceCalculator::calculate(double &probability) const
{
    return m_Probability.calculate(probability);
}

bool CProbabilityAndInfluenceCalculator::calculate(double &probability,
                                                   TStoredStringPtrStoredStringPtrPrDoublePrVec &influences) const
{
    if (!m_Probability.calculate(probability))
    {
        return false;
    }

    LOG_TRACE("probability = " << probability);

    double logp = ::log(probability);

    influences.reserve(m_InfluencerProbabilities.size());
    for (const auto &aggregator : m_InfluencerProbabilities)
    {
        double probability_;
        if (!aggregator.second.calculate(probability_))
        {
            LOG_ERROR("Couldn't calculate probability for influencer "
                      << core::CContainerPrinter::print(aggregator.first));
        }
        LOG_TRACE("influence probability = " << probability_);
        double influence = CInfluenceCalculator::intersectionInfluence(logp, ::log(probability_));
        if (influence >= m_Cutoff)
        {
            influences.emplace_back(aggregator.first, influence);
        }
    }
    std::sort(influences.begin(), influences.end(), maths::COrderings::SSecondGreater());

    return true;
}

void CProbabilityAndInfluenceCalculator::commitInfluences(model_t::EFeature feature,
                                                          double logp,
                                                          double weight)
{
    LOG_TRACE("influences = " << core::CContainerPrinter::print(m_Influences));

    for (const auto &influence : m_Influences)
    {
        CModelTools::CProbabilityAggregator &aggregator =
                m_InfluencerProbabilities.emplace(influence.first,
                                                  m_ProbabilityTemplate).first->second;
        if (!model_t::isConstant(feature))
        {
            double probability = ::exp(influence.second * logp);
            LOG_TRACE("Adding = " << influence.first.second.get() << " " << probability);
            aggregator.add(probability, weight);
        }
    }
}

CProbabilityAndInfluenceCalculator::SParams::SParams(const CPartitioningFields &partitioningFields) :
        s_Feature(),
        s_Model(0),
        s_ElapsedTime(0),
        s_Count(0.0),
        s_Probability(1.0),
        s_PartitioningFields(partitioningFields),
        s_Cutoff(1.0),
        s_IncludeCutoff(false)
{}

std::string CProbabilityAndInfluenceCalculator::SParams::describe(void) const
{
    return core::CContainerPrinter::print(s_Value)
           + " | feature = " + model_t::print(s_Feature)
           + ", @ " + core::CContainerPrinter::print(s_Time)
           + ", elapsedTime = " + core::CStringUtils::typeToString(s_ElapsedTime);
}

CProbabilityAndInfluenceCalculator::SCorrelateParams::SCorrelateParams(const CPartitioningFields &partitioningFields) :
        s_Feature(),
        s_Model(0),
        s_ElapsedTime(0),
        s_Probability(1.0),
        s_PartitioningFields(partitioningFields),
        s_Cutoff(1.0),
        s_IncludeCutoff(false)
{}

std::string CProbabilityAndInfluenceCalculator::SCorrelateParams::describe(void) const
{
    return core::CContainerPrinter::print(s_Values)
           + " | feature = " + model_t::print(s_Feature)
           + ", @ " + core::CContainerPrinter::print(s_Times)
           + ", elapsedTime = " + core::CStringUtils::typeToString(s_ElapsedTime);
}


////// CInfluenceCalculator //////

CInfluenceCalculator::~CInfluenceCalculator(void) {}

double CInfluenceCalculator::intersectionInfluence(double logp, double logpi)
{
    return maths::CTools::truncate(ratio(logpi, logp, 1.0), 0.0, 1.0);
}

double CInfluenceCalculator::complementInfluence(double logp, double logpi)
{
    return maths::CTools::truncate(1.0 - ratio(logpi, logp, 0.0), 0.0, 1.0);
}

////// CInfluenceUnavailableCalculator //////

void CInfluenceUnavailableCalculator::computeInfluences(TParams &params) const
{
    params.s_Influences.clear();
}

void CInfluenceUnavailableCalculator::computeInfluences(TCorrelateParams &params) const
{
    params.s_Influences.clear();
}

////// CIndicatorInfluenceCalculator //////

void CIndicatorInfluenceCalculator::computeInfluences(TParams &params) const
{
    params.s_Influences.clear();
    doComputeIndicatorInfluences(params.s_InfluencerName,
                                 params.s_InfluencerValues,
                                 params.s_Influences);
}

void CIndicatorInfluenceCalculator::computeInfluences(TCorrelateParams &params) const
{
    params.s_Influences.clear();
    doComputeIndicatorInfluences(params.s_InfluencerName,
                                 params.s_InfluencerValues,
                                 params.s_Influences);
}

////// CLogProbabilityComplementInfluenceCalculator //////

void CLogProbabilityComplementInfluenceCalculator::computeInfluences(TParams &params) const
{
    params.s_Influences.clear();

    maths::CModelProbabilityParams params_;
    for (std::size_t i = 0u; i < params.s_Tail.size(); ++i)
    {
        if (params.s_Tail[i] == maths_t::E_RightTail)
        {
            params_.addCalculation(maths_t::E_OneSidedAbove)
                   .addCoordinate(i);
        }
    }

    if (params_.calculations() > 0)
    {
        params_.seasonalConfidenceInterval(params.s_ComputeProbabilityParams.seasonalConfidenceInterval())
               .weightStyles(params.s_ComputeProbabilityParams.weightStyles())
               .addWeights(params.s_ComputeProbabilityParams.weights()[0]);

        TStrCRefDouble1VecDoublePrPrVec &influencerValues = params.s_InfluencerValues;
        if (model_t::dimension(params.s_Feature) == 1)
        {
            std::sort(influencerValues.begin(), influencerValues.end(),
                      CDecreasingValueInfluence(params.s_Tail[0]));
        }
        LOG_TRACE("influencerValues = " << core::CContainerPrinter::print(influencerValues));

        doComputeInfluences(params.s_Feature, CValueDifference(), complementInfluence,
                            *params.s_Model, params.s_ElapsedTime, params_,
                            params.s_Time, params.s_Value[0], params.s_Count,
                            params.s_Probability, params.s_InfluencerName, params.s_InfluencerValues,
                            params.s_Cutoff, params.s_IncludeCutoff, params.s_Influences);
    }
}

void CLogProbabilityComplementInfluenceCalculator::computeInfluences(TCorrelateParams &params) const
{
    params.s_Influences.clear();

    if (params.s_Tail[0] == maths_t::E_RightTail)
    {
        std::size_t correlate = params.s_MostAnomalousCorrelate[0];
        maths::CModelProbabilityParams params_;
        params_.addCalculation(maths_t::E_OneSidedAbove)
               .seasonalConfidenceInterval(params.s_ComputeProbabilityParams.seasonalConfidenceInterval())
               .weightStyles(params.s_ComputeProbabilityParams.weightStyles())
               .addWeights(params.s_ComputeProbabilityParams.weights()[correlate])
               .mostAnomalousCorrelate(correlate);
        LOG_TRACE("influencerValues = " << core::CContainerPrinter::print(params.s_InfluencerValues));
        doComputeCorrelateInfluences(params.s_Feature, CValueDifference(), complementInfluence,
                                     *params.s_Model, params.s_ElapsedTime, params_,
                                     params.s_Times[correlate], params.s_Values[correlate], params.s_Counts[correlate],
                                     params.s_Probability, params.s_InfluencerName, params.s_InfluencerValues,
                                     params.s_Cutoff, params.s_IncludeCutoff, params.s_Influences);
    }
}

////// CLogProbabilityInfluenceCalculator //////

namespace
{

//! Maybe add \p coordinate and the appropriate calculation to \p params.
void addCoordinate(maths_t::ETail tail,
                   std::size_t coordinate,
                   maths::CModelProbabilityParams &params)
{
    switch (tail)
    {
    case maths_t::E_LeftTail:
    {
        params.addCalculation(maths_t::E_OneSidedBelow)
              .addCoordinate(coordinate);
        break;
    }
    case maths_t::E_RightTail:
    {
        params.addCalculation(maths_t::E_OneSidedAbove)
              .addCoordinate(coordinate);
        break;
    }
    case maths_t::E_MixedOrNeitherTail:
    case maths_t::E_UndeterminedTail:
        break;
    }
}

}

void CLogProbabilityInfluenceCalculator::computeInfluences(TParams &params) const
{
    params.s_Influences.clear();

    maths::CModelProbabilityParams params_;
    for (std::size_t i = 0u; i < params.s_Tail.size(); ++i)
    {
        addCoordinate(params.s_Tail[i], i, params_);
    }

    if (params_.calculations() > 0)
    {
        params_.seasonalConfidenceInterval(params.s_ComputeProbabilityParams.seasonalConfidenceInterval())
               .weightStyles(params.s_ComputeProbabilityParams.weightStyles())
               .addWeights(params.s_ComputeProbabilityParams.weights()[0]);

        TStrCRefDouble1VecDoublePrPrVec &influencerValues = params.s_InfluencerValues;
        if (model_t::dimension(params.s_Feature) == 1)
        {
            std::sort(influencerValues.begin(), influencerValues.end(),
                      CDecreasingValueInfluence(params.s_Tail[0]));
        }
        LOG_TRACE("influencerValues = " << core::CContainerPrinter::print(influencerValues));

        doComputeInfluences(params.s_Feature, CValueIntersection(), intersectionInfluence,
                            *params.s_Model, params.s_ElapsedTime, params_,
                            params.s_Time, params.s_Value[0], params.s_Count,
                            params.s_Probability, params.s_InfluencerName, params.s_InfluencerValues,
                            params.s_Cutoff, params.s_IncludeCutoff, params.s_Influences);
    }
}

void CLogProbabilityInfluenceCalculator::computeInfluences(TCorrelateParams &params) const
{
    params.s_Influences.clear();

    maths::CModelProbabilityParams params_;
    addCoordinate(params.s_Tail[0], 0, params_);

    if (params_.calculations() > 0)
    {
        std::size_t correlate = params.s_MostAnomalousCorrelate[0];
        params_.seasonalConfidenceInterval(params.s_ComputeProbabilityParams.seasonalConfidenceInterval())
               .weightStyles(params.s_ComputeProbabilityParams.weightStyles())
               .addWeights(params.s_ComputeProbabilityParams.weights()[correlate])
               .mostAnomalousCorrelate(correlate);
        LOG_TRACE("influencerValues = " << core::CContainerPrinter::print(params.s_InfluencerValues));
        doComputeCorrelateInfluences(params.s_Feature, CValueDifference(), intersectionInfluence,
                                     *params.s_Model, params.s_ElapsedTime, params_,
                                     params.s_Times[correlate], params.s_Values[correlate], params.s_Counts[correlate],
                                     params.s_Probability, params.s_InfluencerName, params.s_InfluencerValues,
                                     params.s_Cutoff, params.s_IncludeCutoff, params.s_Influences);
    }
}

////// CMeanInfluenceCalculator //////

void CMeanInfluenceCalculator::computeInfluences(TParams &params) const
{
    params.s_Influences.clear();

    maths::CModelProbabilityParams params_;
    for (std::size_t i = 0u; i < params.s_Tail.size(); ++i)
    {
        addCoordinate(params.s_Tail[i], i, params_);
    }

    if (params_.calculations() > 0)
    {
        params_.seasonalConfidenceInterval(params.s_ComputeProbabilityParams.seasonalConfidenceInterval())
               .weightStyles(params.s_ComputeProbabilityParams.weightStyles())
               .addWeights(params.s_ComputeProbabilityParams.weights()[0]);

        TStrCRefDouble1VecDoublePrPrVec &influencerValues = params.s_InfluencerValues;
        if (model_t::dimension(params.s_Feature) == 1)
        {
            std::sort(influencerValues.begin(), influencerValues.end(),
                      CDecreasingMeanInfluence(params.s_Tail[0], params.s_Value[0], params.s_Count));
        }
        LOG_TRACE("influencerValues = " << core::CContainerPrinter::print(params.s_InfluencerValues));
        doComputeInfluences(params.s_Feature, CMeanDifference(), complementInfluence,
                            *params.s_Model, params.s_ElapsedTime, params_,
                            params.s_Time, params.s_Value[0], params.s_Count,
                            params.s_Probability, params.s_InfluencerName, params.s_InfluencerValues,
                            params.s_Cutoff, params.s_IncludeCutoff, params.s_Influences);
    }
}

void CMeanInfluenceCalculator::computeInfluences(TCorrelateParams &params) const
{
    params.s_Influences.clear();

    maths::CModelProbabilityParams params_;
    addCoordinate(params.s_Tail[0], 0, params_);

    if (params_.calculations() > 0)
    {
        std::size_t correlate = params.s_MostAnomalousCorrelate[0];
        params_.seasonalConfidenceInterval(params.s_ComputeProbabilityParams.seasonalConfidenceInterval())
               .weightStyles(params.s_ComputeProbabilityParams.weightStyles())
               .addWeights(params.s_ComputeProbabilityParams.weights()[correlate])
               .mostAnomalousCorrelate(correlate);
        LOG_TRACE("influencerValues = " << core::CContainerPrinter::print(params.s_InfluencerValues));
        doComputeCorrelateInfluences(params.s_Feature, CMeanDifference(), complementInfluence,
                                     *params.s_Model, params.s_ElapsedTime, params_,
                                     params.s_Times[correlate], params.s_Values[correlate], params.s_Counts[correlate],
                                     params.s_Probability, params.s_InfluencerName, params.s_InfluencerValues,
                                     params.s_Cutoff, params.s_IncludeCutoff, params.s_Influences);
    }
}

////// CVarianceInfluenceCalculator //////

void CVarianceInfluenceCalculator::computeInfluences(TParams &params) const
{
    params.s_Influences.clear();

    maths::CModelProbabilityParams params_;
    for (std::size_t i = 0u; i < params.s_Tail.size(); ++i)
    {
        addCoordinate(params.s_Tail[i], i, params_);
    }

    if (params_.calculations() > 0)
    {
        params_.seasonalConfidenceInterval(params.s_ComputeProbabilityParams.seasonalConfidenceInterval())
               .weightStyles(params.s_ComputeProbabilityParams.weightStyles())
               .addWeights(params.s_ComputeProbabilityParams.weights()[0]);

        TStrCRefDouble1VecDoublePrPrVec &influencerValues = params.s_InfluencerValues;
        if (model_t::dimension(params.s_Feature) == 1)
        {
            std::sort(influencerValues.begin(), influencerValues.end(),
                      CDecreasingVarianceInfluence(params.s_Tail[0], params.s_Value[0], params.s_Count));
        }
        LOG_TRACE("influencerValues = " << core::CContainerPrinter::print(influencerValues));

        doComputeInfluences(params.s_Feature, CVarianceDifference(), complementInfluence,
                            *params.s_Model, params.s_ElapsedTime, params_,
                            params.s_Time, params.s_Value[0], params.s_Count,
                            params.s_Probability, params.s_InfluencerName, params.s_InfluencerValues,
                            params.s_Cutoff, params.s_IncludeCutoff, params.s_Influences);
    }
}

void CVarianceInfluenceCalculator::computeInfluences(TCorrelateParams &params) const
{
    params.s_Influences.clear();

    maths::CModelProbabilityParams params_;
    addCoordinate(params.s_Tail[0], 0, params_);

    if (params_.calculations() > 0)
    {
        std::size_t correlate = params.s_MostAnomalousCorrelate[0];
        params_.seasonalConfidenceInterval(params.s_ComputeProbabilityParams.seasonalConfidenceInterval())
               .weightStyles(params.s_ComputeProbabilityParams.weightStyles())
               .addWeights(params.s_ComputeProbabilityParams.weights()[correlate])
               .mostAnomalousCorrelate(correlate);
        LOG_TRACE("influencerValues = " << core::CContainerPrinter::print(params.s_InfluencerValues));
        doComputeCorrelateInfluences(params.s_Feature, CVarianceDifference(), complementInfluence,
                                     *params.s_Model, params.s_ElapsedTime, params_,
                                     params.s_Times[correlate], params.s_Values[correlate], params.s_Counts[correlate],
                                     params.s_Probability, params.s_InfluencerName, params.s_InfluencerValues,
                                     params.s_Cutoff, params.s_IncludeCutoff, params.s_Influences);
    }
}

}
}
