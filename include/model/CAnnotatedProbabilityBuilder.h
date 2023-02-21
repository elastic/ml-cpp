/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#ifndef INCLUDED_ml_model_CAnnotatedProbabilityBuilder_h
#define INCLUDED_ml_model_CAnnotatedProbabilityBuilder_h

#include <core/CNonCopyable.h>
#include <core/CStoredStringPtr.h>

#include <maths/common/CBasicStatistics.h>

#include <model/CAnnotatedProbability.h>
#include <model/FunctionTypes.h>
#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <utility>

namespace ml {
namespace maths {
namespace common {
class CMultinomialConjugate;
}
}
namespace model {

//! \brief Manages the creation of annotated probabilities using the
//! builder pattern.
class MODEL_EXPORT CAnnotatedProbabilityBuilder : private core::CNonCopyable {
public:
    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TDouble1Vec = core::CSmallVector<double, 1>;
    using TSize1Vec = core::CSmallVector<std::size_t, 1>;
    using TSizeDoublePr1Vec = core::CSmallVector<TSizeDoublePr, 1>;
    using TStoredStringPtr1Vec = core::CSmallVector<core::CStoredStringPtr, 1>;

public:
    explicit CAnnotatedProbabilityBuilder(SAnnotatedProbability& annotatedProbability);

    CAnnotatedProbabilityBuilder(SAnnotatedProbability& annotatedProbability,
                                 std::size_t numberAttributeProbabilities,
                                 function_t::EFunction function);

    void attributeProbabilityPrior(const maths::common::CMultinomialConjugate* prior);
    void personAttributeProbabilityPrior(const maths::common::CMultinomialConjugate* prior);
    void probability(double p);
    SAnnotatedProbability::TAnomalyScoreExplanation& anomalyScoreExplanation();
    void multiBucketImpact(double multiBucketImpact);
    void addAttributeProbability(std::size_t cid,
                                 const core::CStoredStringPtr& attribute,
                                 double pGivenAttribute,
                                 model_t::CResultType type,
                                 model_t::EFeature feature,
                                 const TStoredStringPtr1Vec& correlatedAttributes,
                                 const TSizeDoublePr1Vec& correlated);
    void build();

private:
    using TMinAccumulator =
        maths::common::CBasicStatistics::COrderStatisticsHeap<SAttributeProbability>;

private:
    SAnnotatedProbability& m_Result;
    std::size_t m_NumberAttributeProbabilities;
    const maths::common::CMultinomialConjugate* m_AttributeProbabilityPrior;
    const maths::common::CMultinomialConjugate* m_PersonAttributeProbabilityPrior;
    TMinAccumulator m_MinAttributeProbabilities;
    std::size_t m_DistinctTotalAttributes;
    bool m_IsPopulation;
    bool m_IsRare;
    bool m_IsFreqRare;
};
}
}

#endif // INCLUDED_ml_model_CAnnotatedProbabilityBuilder_h
