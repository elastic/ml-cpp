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

#ifndef INCLUDED_ml_config_CFieldStatistics_h
#define INCLUDED_ml_config_CFieldStatistics_h

#include <config/CDataSemantics.h>
#include <config/CDataSummaryStatistics.h>
#include <config/ConfigTypes.h>
#include <config/ImportExport.h>

#include <string>
#include <vector>

#include <boost/ref.hpp>
#include <boost/variant.hpp>

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
        typedef boost::reference_wrapper<const CAutoconfigurerParams> TAutoconfigurerParamsCRef;

    public:
        CFieldStatistics(const std::string &fieldName, const CAutoconfigurerParams &params);

        //! Get the name of the field.
        const std::string &name(void) const;

        //! If we have been able to determine the data type start
        //! capturing the appropriate statistics.
        void maybeStartCapturingTypeStatistics(void);

        //! Add an example value for the field.
        void add(core_t::TTime time, const std::string &example);

        //! Get the type of data we think we have.
        config_t::EDataType type(void) const;

        //! Get the data summary statistics if no more specific
        //! ones are available.
        const CDataSummaryStatistics *summary(void) const;

        //! Get the categorical summary statistics if we think
        //! the data are categorical.
        const CCategoricalDataSummaryStatistics *categoricalSummary(void) const;

        //! Get the numeric summary statistics if we think the
        //! data are numeric.
        const CNumericDataSummaryStatistics *numericSummary(void) const;

        //! Get the score for this field based on \p penalty.
        double score(const CPenalty &penalty) const;

    private:
        typedef std::pair<core_t::TTime, std::string> TTimeStrPr;
        typedef std::vector<TTimeStrPr> TTimeStrPrVec;
        typedef boost::variant<CDataSummaryStatistics,
                CCategoricalDataSummaryStatistics,
                CNumericDataSummaryStatistics> TDataSummaryStatistics;

    private:
        //! The auto-configuration parameters.
        const CAutoconfigurerParams &params(void) const;

        //! Add the records in the buffer to the statistics.
        void replayBuffer(void);

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
