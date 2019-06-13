/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_model_CRuleScope_h
#define INCLUDED_ml_model_CRuleScope_h

#include <model/ImportExport.h>

#include <core/CPatternSet.h>
#include <core/CTriple.h>

#include <boost/ref.hpp>

#include <string>
#include <vector>

namespace ml {
namespace model {

class CAnomalyDetectorModel;

//! \brief The scope of the rule. It dictates the series where the rule applies.
//!
//! DESCRIPTION:\n
//! The rule scope allows to specify when a rule applies based on the series
//! split fields (partition, over, by). When the scope is empty, the rule
//! applies to all series. Fields can be specified to either be included in
//! the scope or excluded from it depending on whether they are contained
//! in a filter (i.e. a list of string patterns). Multiple fields are combined
//! with a logical AND.
class MODEL_EXPORT CRuleScope {
public:
    enum ERuleScopeFilterType { E_Include, E_Exclude };

    using TPatternSetCRef = std::reference_wrapper<const core::CPatternSet>;
    using TStrPatternSetCRefFilterTypeTriple =
        core::CTriple<std::string, TPatternSetCRef, ERuleScopeFilterType>;
    using TStrPatternSetCRefFilterTypeTripleVec = std::vector<TStrPatternSetCRefFilterTypeTriple>;

public:
    //! Default constructor.
    CRuleScope() = default;

    //! Adds a requirement for \p field to be in \p filter for the rule to apply
    void include(const std::string& field, const core::CPatternSet& filter);

    //! Adds a requirement for \p field not to be in \p filter for the rule to apply
    void exclude(const std::string& field, const core::CPatternSet& filter);

    //! Check whether the given series is in the rule scope.
    bool check(const CAnomalyDetectorModel& model, std::size_t pid, std::size_t cid) const;

    //! Pretty-print the scope.
    std::string print() const;

private:
    //! A vector that holds the triple of the field, filter and its type.
    TStrPatternSetCRefFilterTypeTripleVec m_Scope;
};
}
}

#endif // INCLUDED_ml_model_CRuleScope_h
