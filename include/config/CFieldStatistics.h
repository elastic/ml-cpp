/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_config_CFieldStatistics_h
#define INCLUDED_ml_config_CFieldStatistics_h

#include <config/CDataSemantics.h>
#include <config/CDataSummaryStatistics.h>
#include <config/ConfigTypes.h>
#include <config/ImportExport.h>

#include <boost/variant.hpp>

#include <functional>
#include <string>
#include <vector>

namespace ml {
namespace config {
class CAutoconfigurerParams;
class CPenalty;

//! \brief Gathers useful statistics relating to a single field.
//!
//! DESCRIPTION:\n
//! This wraps up the functionality to discover data semantics
//! and gather the appropriate summary statistics.
class CONFIG_EXPORT CFieldStatistics {
public:
    using TAutoconfigurerParamsCRef = std::reference_wrapper<const CAutoconfigurerParams>;

public:
    CFieldStatistics(const std::string& fieldName, const CAutoconfigurerParams& params);

    //! Get the name of the field.
    const std::string& name() const;

    //! If we have been able to determine the data type start
    //! capturing the appropriate statistics.
    void maybeStartCapturingTypeStatistics();

    //! Add an example value for the field.
    void add(core_t::TTime time, const std::string& example);

    //! Get the type of data we think we have.
    config_t::EDataType type() const;

    //! Get the data summary statistics if no more specific
    //! ones are available.
    const CDataSummaryStatistics* summary() const;

    //! Get the categorical summary statistics if we think
    //! the data are categorical.
    const CCategoricalDataSummaryStatistics* categoricalSummary() const;

    //! Get the numeric summary statistics if we think the
    //! data are numeric.
    const CNumericDataSummaryStatistics* numericSummary() const;

    //! Get the score for this field based on \p penalty.
    double score(const CPenalty& penalty) const;

private:
    using TTimeStrPr = std::pair<core_t::TTime, std::string>;
    using TTimeStrPrVec = std::vector<TTimeStrPr>;
    using TDataSummaryStatistics =
        boost::variant<CDataSummaryStatistics, CCategoricalDataSummaryStatistics, CNumericDataSummaryStatistics>;

private:
    //! The auto-configuration parameters.
    const CAutoconfigurerParams& params() const;

    //! Add the records in the buffer to the statistics.
    void replayBuffer();

private:
    //! A reference to the auto-configuration parameters.
    TAutoconfigurerParamsCRef m_Params;

    //! The field name.
    std::string m_FieldName;

    //! The number of examples added.
    uint64_t m_NumberExamples;

    //! A buffer of the records before the field has been classified.
    TTimeStrPrVec m_Buffer;

    //! Deduces the data semantics.
    CDataSemantics m_Semantics;

    //! Computes the summary statistics.
    TDataSummaryStatistics m_SummaryStatistics;
};
}
}

#endif // INCLUDED_ml_config_CFieldStatistics_h
