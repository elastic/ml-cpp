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

#ifndef INCLUDED_ml_model_CAnnotatedProbabilityBuilder_h
#define INCLUDED_ml_model_CAnnotatedProbabilityBuilder_h

#include <core/CStoredStringPtr.h>

#include <maths/CBasicStatistics.h>

#include <model/CAnnotatedProbability.h>
#include <model/FunctionTypes.h>
#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <boost/optional.hpp>

#include <utility>
#include <vector>


namespace ml
{
namespace maths
{
class CMultinomialConjugate;
}
namespace model
{
class CModel;

//! \brief Manages the creation of annotated probabilities using the
//! builder pattern.
class MODEL_EXPORT CAnnotatedProbabilityBuilder : private core::CNonCopyable
{
    public:
        using TSizeDoublePr = std::pair<std::size_t, double>;
        using TDouble1Vec = core::CSmallVector<double, 1>;
        using TSize1Vec = core::CSmallVector<std::size_t, 1>;
        using TSizeDoublePr1Vec = core::CSmallVector<TSizeDoublePr, 1>;
        using TStoredStringPtr1Vec = core::CSmallVector<core::CStoredStringPtr, 1>;

    public:
        CAnnotatedProbabilityBuilder(SAnnotatedProbability &annotatedProbability);

        CAnnotatedProbabilityBuilder(SAnnotatedProbability &annotatedProbability,
                                     std::size_t numberAttributeProbabilities,
                                     function_t::EFunction function,
                                     std::size_t numberOfPeople);

        void attributeProbabilityPrior(const maths::CMultinomialConjugate *prior);
        void personAttributeProbabilityPrior(const maths::CMultinomialConjugate *prior);
        void personFrequency(double frequency, bool everSeenBefore);
        void probability(double p);
        void addAttributeProbability(std::size_t cid,
                                     const core::CStoredStringPtr &attribute,
                                     double pAttribute,
                                     double pGivenAttribute,
                                     model_t::CResultType type,
                                     model_t::EFeature feature,
                                     const TStoredStringPtr1Vec &correlatedAttributes,
                                     const TSizeDoublePr1Vec &correlated);
        void build();

    private:
        void addAttributeDescriptiveData(std::size_t cid,
                                         double pAttribute,
                                         SAttributeProbability &attributeProbability);

        void addDescriptiveData();

    private:
        using TMinAccumulator = maths::CBasicStatistics::COrderStatisticsHeap<SAttributeProbability>;

    private:
        SAnnotatedProbability              &m_Result;
        std::size_t                         m_NumberAttributeProbabilities;
        std::size_t                         m_NumberOfPeople;
        const maths::CMultinomialConjugate *m_AttributeProbabilityPrior;
        const maths::CMultinomialConjugate *m_PersonAttributeProbabilityPrior;
        TMinAccumulator                     m_MinAttributeProbabilities;
        std::size_t                         m_DistinctTotalAttributes;
        std::size_t                         m_DistinctRareAttributes;
        double                              m_RareAttributes;
        bool                                m_IsPopulation;
        bool                                m_IsRare;
        bool                                m_IsFreqRare;
};

}
}

#endif // INCLUDED_ml_model_CAnnotatedProbabilityBuilder_h
