/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CAnnotatedProbability_h
#define INCLUDED_ml_model_CAnnotatedProbability_h

#include <core/CSmallVector.h>
#include <core/CStoredStringPtr.h>

#include <model/FunctionTypes.h>
#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <boost/optional.hpp>
#include <boost/ref.hpp>

#include <utility>
#include <vector>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}

namespace model {

namespace annotated_probability {
enum EDescriptiveData {
    E_PERSON_PERIOD = 0,
    E_PERSON_NEVER_SEEN_BEFORE = 1,
    E_PERSON_COUNT = 2,
    E_DISTINCT_RARE_ATTRIBUTES_COUNT = 3,
    E_DISTINCT_TOTAL_ATTRIBUTES_COUNT = 4,
    E_RARE_ATTRIBUTES_COUNT = 5,
    E_TOTAL_ATTRIBUTES_COUNT = 6,
    E_ATTRIBUTE_CONCENTRATION = 7,
    E_ACTIVITY_CONCENTRATION = 8
};
}

//! \brief A collection of data describing an attribute's probability.
struct MODEL_EXPORT SAttributeProbability {
    using TDouble1Vec = core::CSmallVector<double, 1>;
    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TSizeDoublePr1Vec = core::CSmallVector<TSizeDoublePr, 1>;
    using TDescriptiveDataDoublePr =
        std::pair<annotated_probability::EDescriptiveData, double>;
    using TDescriptiveDataDoublePr2Vec = core::CSmallVector<TDescriptiveDataDoublePr, 2>;
    using TStoredStringPtr1Vec = core::CSmallVector<core::CStoredStringPtr, 1>;

    SAttributeProbability();
    SAttributeProbability(std::size_t cid,
                          const core::CStoredStringPtr& attribute,
                          double probability,
                          model_t::CResultType type,
                          model_t::EFeature feature,
                          const TStoredStringPtr1Vec& correlatedAttributes,
                          const TSizeDoublePr1Vec& correlated);

    //! Total ordering of attribute probabilities by probability
    //! breaking ties using the attribute and finally the feature.
    bool operator<(const SAttributeProbability& other) const;

    //! Persist the probability passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Restore the probability reading state from \p traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Add the descriptive data \p value for \p key.
    void addDescriptiveData(annotated_probability::EDescriptiveData key, double value);

    //! The attribute identifier.
    std::size_t s_Cid;
    //! The attribute.
    core::CStoredStringPtr s_Attribute;
    //! The attribute probability.
    double s_Probability;
    //! The type of result (see CResultType for details).
    model_t::CResultType s_Type;
    //! The most unusual feature of the attribute.
    model_t::EFeature s_Feature;
    //! The correlated attributes.
    TStoredStringPtr1Vec s_CorrelatedAttributes;
    //! The correlated attribute identifiers (if any).
    TSizeDoublePr1Vec s_Correlated;
    //! A list of key-value pairs with extra descriptive data (if any).
    TDescriptiveDataDoublePr2Vec s_DescriptiveData;
    //! The current bucket value of the attribute (cached from the model).
    mutable TDouble1Vec s_CurrentBucketValue;
    //! The population mean (cached from the model).
    mutable TDouble1Vec s_BaselineBucketMean;
};

//! \brief A collection of data describing the result of a model
//! probability calculation.
//!
//! DESCRIPTION:\n
//! This includes all associated data such as a set of the smallest
//! attribute probabilities, the influences, extra descriptive data
//! and so on.
struct MODEL_EXPORT SAnnotatedProbability {
    using TAttributeProbability1Vec = core::CSmallVector<SAttributeProbability, 1>;
    using TStoredStringPtrStoredStringPtrPr =
        std::pair<core::CStoredStringPtr, core::CStoredStringPtr>;
    using TStoredStringPtrStoredStringPtrPrDoublePr =
        std::pair<TStoredStringPtrStoredStringPtrPr, double>;
    using TStoredStringPtrStoredStringPtrPrDoublePrVec =
        std::vector<TStoredStringPtrStoredStringPtrPrDoublePr>;
    using TDescriptiveDataDoublePr = SAttributeProbability::TDescriptiveDataDoublePr;
    using TDescriptiveDataDoublePr2Vec = SAttributeProbability::TDescriptiveDataDoublePr2Vec;
    using TOptionalDouble = boost::optional<double>;
    using TOptionalUInt64 = boost::optional<uint64_t>;

    SAnnotatedProbability();
    SAnnotatedProbability(double p);

    //! Add the descriptive data \p value for \p key.
    void addDescriptiveData(annotated_probability::EDescriptiveData key, double value);

    //! Efficiently swap the contents of this and \p other.
    void swap(SAnnotatedProbability& other);

    //! Is the result type interim?
    bool isInterim() const;

    //! Persist the probability passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Restore the probability reading state from \p traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! The probability of seeing the series' sample in a time interval.
    double s_Probability;

    //! The impact of multi/single bucket analysis on the probability
    double s_MultiBucketImpact;

    //! The smallest attribute probabilities and associated data describing
    //! the calculation.
    TAttributeProbability1Vec s_AttributeProbabilities;

    //! The field values which influence this probability.
    TStoredStringPtrStoredStringPtrPrDoublePrVec s_Influences;

    //! A list of (key, value) pairs with extra descriptive data (if any).
    TDescriptiveDataDoublePr2Vec s_DescriptiveData;

    //! The result type (interim or final)
    //!
    //! This field is transient - does not get persisted because interim results
    //! never get persisted.
    model_t::CResultType s_ResultType;

    //! The current bucket count for this probability (cached from the model).
    TOptionalUInt64 s_CurrentBucketCount;

    //! The baseline bucket count for this probability (cached from the model).
    TOptionalDouble s_BaselineBucketCount;
};
}
}

#endif // INCLUDED_ml_model_CAnnotatedProbability_h
