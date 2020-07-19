/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CFuzzyLogic_h
#define INCLUDED_ml_maths_CFuzzyLogic_h

#include <maths/ImportExport.h>

#include <maths/CTools.h>

namespace ml {
namespace maths {

//! \brief Fuzzy logical expression with multiplicative AND and OR.
//!
//! DESCRIPTION:\n
//! For more information see https://en.wikipedia.org/wiki/Fuzzy_logic#Fuzzy_logic_operators.
class MATHS_EXPORT CFuzzyTruthValue {
public:
    static const CFuzzyTruthValue TRUE;
    static const CFuzzyTruthValue FALSE;

public:
    CFuzzyTruthValue(double value = 0.0, double isTrueThreshold = 0.5)
        : m_Value{value}, m_IsTrueThreshold{isTrueThreshold} {}

    //! Check if this is less true than \p rhs.
    bool operator<(const CFuzzyTruthValue& rhs) const {
        return rhs.m_IsTrueThreshold * m_Value < m_IsTrueThreshold * rhs.m_Value;
    }

    //! Get the decision.
    bool boolean() const { return m_Value > m_IsTrueThreshold; }

    //! Get the continuous truth value.
    double value() const { return m_Value; }

    //! Get the fuzzy AND of two expressions.
    friend inline CFuzzyTruthValue operator&&(const CFuzzyTruthValue& lhs,
                                              const CFuzzyTruthValue& rhs) {
        return {lhs.m_Value * rhs.m_Value, lhs.m_IsTrueThreshold * rhs.m_IsTrueThreshold};
    }

    //! Get the fuzzy OR of two expressions.
    friend inline CFuzzyTruthValue operator||(const CFuzzyTruthValue& lhs,
                                              const CFuzzyTruthValue& rhs) {
        return {1.0 - (1.0 - lhs.m_Value) * (1.0 - rhs.m_Value),
                1.0 - (1.0 - lhs.m_IsTrueThreshold) * (1.0 - rhs.m_IsTrueThreshold)};
    }

private:
    double m_Value;
    double m_IsTrueThreshold;
};

//! Fuzzy check if \p value is greater than \p threshold.
//!
//! \param[in] value The value to compare to \p threshold.
//! \param[in] threshold The smallest true \p value.
//! \param[in] margin The fuzzyness expressed as the width of the logistic
//! function which defines the truth value of this expression.
inline CFuzzyTruthValue fuzzyGreaterThan(double value, double threshold, double margin) {
    return {CTools::logisticFunction(value, margin, threshold, +1.0)};
}

//! Fuzzy check if \p value is less than \p threshold.
//!
//! \param[in] value The value to compare to \p threshold.
//! \param[in] threshold The smallest true \p value.
//! \param[in] margin The fuzzyness expressed as the width of the logistic
//! function which defines the truth value of this expression.
inline CFuzzyTruthValue fuzzyLessThan(double value, double threshold, double margin) {
    return {CTools::logisticFunction(value, margin, threshold, -1.0)};
}
}
}

#endif // INCLUDED_ml_maths_CFuzzyLogic_h
