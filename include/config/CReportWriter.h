/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_config_CReportWriter_h
#define INCLUDED_ml_config_CReportWriter_h

#include <api/COutputHandler.h>

#include <config/ConfigTypes.h>
#include <config/ImportExport.h>

#include <string>
#include <vector>

namespace ml
{
namespace config
{
class CCategoricalDataSummaryStatistics;
class CDataSummaryStatistics;
class CDetectorSpecification;
class CNumericDataSummaryStatistics;

//! \brief Generates summary reports about a data set comprising
//! one or more distinct fields.
//!
//! DESCRIPTION:\n
//! Responsible for generating the report which is the output of
//! automatic configuration. This includes data semantics, summary
//! statistics and suggested configurations.
//!
//! IMPLEMENTATION:\n
//! This uses the builder pattern accepting different objects from
//! which to create the report.
class CONFIG_EXPORT CReportWriter : public api::COutputHandler
{
    public:
        typedef std::vector<std::string> TStrVec;
        typedef std::vector<TStrVec> TStrVecVec;
        typedef std::vector<TStrVecVec> TStrVecVecVec;
        typedef std::vector<TStrVecVecVec> TStrVecVecVecVec;

        //! \name Summary Statistics.
        //@{
        static const std::size_t FIELD_NAME                 = 0;
        static const std::size_t DATA_TYPE                  = 1;
        static const std::size_t EARLIEST_TIME              = 2;
        static const std::size_t LATEST_TIME                = 3;
        static const std::size_t MEAN_RATE                  = 4;
        static const std::size_t CATEGORICAL_DISTINCT_COUNT = 5;
        static const std::size_t CATEGORICAL_TOP_N_COUNTS   = 6;
        static const std::size_t NUMERIC_MINIMUM            = 7;
        static const std::size_t NUMERIC_MEDIAN             = 8;
        static const std::size_t NUMERIC_MAXIMUM            = 9;
        static const std::size_t NUMERIC_DENSITY_CHART      = 10;
        static const std::size_t NUMBER_STATISTICS          = 11;
        //@}

        //! \name Detector Attributes.
        //@{
        static const std::size_t DESCRIPTION       = 0;
        static const std::size_t OVERALL_SCORE     = 1;
        static const std::size_t PARAMETER_SCORES  = 2;
        static const std::size_t DETECTOR_CONFIG   = 3;
        static const std::size_t NUMBER_ATTRIBUTES = 4;
        //@}

        //! \name Detector Parameter Labels
        //@{
        static const std::size_t BUCKET_LENGTH_PARAMETER = 0;
        static const std::size_t IGNORE_EMPTY_PARAMETER  = 1;
        static const std::size_t SCORE_PARAMETER         = 2;
        static const std::size_t DESCRIPTION_PARAMETER   = 3;
        static const std::size_t NUMBER_PARAMETERS       = 4;
        //@}

        //! The summary statistic labels.
        static const std::string STATISTIC_LABELS[NUMBER_STATISTICS];

        //! The detector parameter labels.
        static const std::string PARAMETER_LABELS[NUMBER_PARAMETERS];

    public:
        explicit CReportWriter(std::ostream &writeStream);

        // Bring the other overload of fieldNames() into scope.
        using api::COutputHandler::fieldNames;

        //! No-op.
        virtual bool fieldNames(const TStrVec &fieldNames,
                                const TStrVec &extraFieldNames);

        // Bring the other overload of writeRow() into scope.
        using api::COutputHandler::writeRow;

        //! No-op.
        virtual bool writeRow(const TStrStrUMap &dataRowFields,
                              const TStrStrUMap &overrideDataRowFields);

        //! Add the total number of records processed.
        void addTotalRecords(uint64_t n);

        //! Add the total number of invalid records processed.
        void addInvalidRecords(uint64_t n);

        //! Add the summary for \p field.
        void addFieldStatistics(const std::string &field,
                                config_t::EDataType type,
                                const CDataSummaryStatistics &summary);

        //! Add the summary for the categorical field \p field.
        void addFieldStatistics(const std::string &field,
                                config_t::EDataType type,
                                const CCategoricalDataSummaryStatistics &summary);

        //! Add the summary for the numeric field \p field.
        void addFieldStatistics(const std::string &field,
                                config_t::EDataType type,
                                const CNumericDataSummaryStatistics &summary);

        //! Add a summary of the detector \p detector.
        void addDetector(const CDetectorSpecification &spec);

        //! Write the report.
        //virtual void write(void) const = 0;
        void write(void) const;

    protected:
        //! The statistics in the summary table for unclassified fields.
        static const std::size_t UNCLASSIFIED_STATISTICS[5];

        //! The statistics in the summary table for categorical fields.
        static const std::size_t CATEGORICAL_STATISTICS[6];

        //! The statistics in the summary table for numeric fields.
        static const std::size_t NUMERIC_STATISTICS[8];

        //! The detector parameters in score table.
        static const std::size_t DETECTOR_PARAMETERS[4];

    private:
        //! The stream to which to write the report.
        std::ostream &m_WriteStream;

        //! The total number of records processed.
        std::string m_TotalRecords;

        //! The total number of invalid records processed.
        std::string m_InvalidRecords;

        //! The summary statistics.
        TStrVecVec m_UnclassifiedFields;

        //! The summary statistics for categorical fields.
        TStrVecVec m_CategoricalFields;

        //! The summary statistics for numeric fields.
        TStrVecVec m_NumericFields;

        //! The summary of a candidate detector.
        TStrVecVecVecVec m_Detectors;
};

}
}

#endif // INCLUDED_ml_config_CReportWriter_h
