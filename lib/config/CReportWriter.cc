/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <config/CReportWriter.h>

#include <core/CStringUtils.h>
#include <core/CTimeUtils.h>

#include <config/CDataSummaryStatistics.h>
#include <config/CDetectorSpecification.h>
#include <config/CTools.h>

#include <boost/range.hpp>

#include <cstdio>

namespace ml {
namespace config {
namespace {

using TSizeVec = std::vector<std::size_t>;
using TStrVec = std::vector<std::string>;
using TStrVecVec = std::vector<TStrVec>;

//! Pad \p value.
inline std::string pad(std::size_t padTo, const std::string& value) {
    return std::string((padTo - value.length()) / 2, ' ') + value +
           std::string((padTo - value.length() + 1) / 2, ' ');
}

//! Pass the string back.
const std::string& print(const std::string& s) {
    return s;
}

//! Convert to a string.
template<typename T>
inline std::string print(const T& t) {
    return core::CStringUtils::typeToString(t);
}

//! Convert to a string.
inline std::string print(double t) {
    return core::CStringUtils::typeToStringPrecise(t, core::CIEEE754::E_SinglePrecision);
}

//! Write out a pair space delimited.
template<typename U, typename V>
inline std::string print(const std::pair<U, V>& p, std::size_t padTo = 0) {
    std::string first = print(p.first);
    std::string second = print(p.second);
    return (padTo > 0 ? pad(padTo, first) : first) + " " +
           (padTo > 0 ? pad(padTo, second) : second);
}

//! Write out a vector of pairs new line delimited.
template<typename U, typename V>
inline std::string print(const std::vector<std::pair<U, V>>& v, std::size_t padTo = 0) {
    std::string result;
    for (std::size_t i = 0u; i < v.size(); ++i) {
        result += print(v[i], padTo) + "\n";
    }
    return result;
}

//!
TStrVec splitMultifields(const std::string& field) {
    TStrVec fields;
    std::string remainder;
    core::CStringUtils::tokenise("\n", field, fields, remainder);
    fields.push_back(remainder);
    return fields;
}

//! Compute the length of the longest field for \p statistic.
std::size_t longest(const TStrVecVec& fields, std::size_t statistic) {
    std::size_t longest = 0u;
    for (std::size_t i = 0u; i < fields.size(); ++i) {
        TStrVec fi = splitMultifields(fields[i][statistic]);
        for (std::size_t j = 0u; j < fi.size(); ++j) {
            longest = std::max(longest, fi[j].length());
        }
    }
    return longest;
}

//! Write a row of the summary statistic table.
template<std::size_t N, typename ROW>
void writeTableRow(std::ostream& o,
                   const TSizeVec& padTo,
                   const std::size_t (&stats)[N],
                   const ROW& row) {
    TStrVecVec columnFields;
    columnFields.reserve(N);
    std::size_t height = 1u;
    for (std::size_t i = 0u; i < N; ++i) {
        columnFields.push_back(splitMultifields(row[stats[i]]));
        height = std::max(height, columnFields[i].size());
    }
    for (std::size_t i = 0u; i < height; ++i) {
        for (std::size_t j = 0u; j < N; ++j) {
            o << (i < columnFields[j].size() ? pad(padTo[j], columnFields[j][i])
                                             : std::string(padTo[j], ' '));
        }
        o << "\n";
    }
}

//! Write the summary statistic table.
template<std::size_t M, std::size_t N>
void writeTable(std::ostream& o,
                const std::string (&labels)[M],
                const std::size_t (&stats)[N],
                const TStrVecVec& values) {
    // Compute the table pads.
    TSizeVec padTo(N, 0);
    std::size_t tableWidth = 0;
    for (std::size_t i = 0u; i < N; ++i) {
        const std::string& label = labels[stats[i]];
        padTo[i] = std::max(longest(values, stats[i]), label.length()) + 4;
        tableWidth += padTo[i];
    }

    // Write the table.
    writeTableRow(o, padTo, stats, labels);
    o << std::string(tableWidth, '-') << "\n";
    for (std::size_t i = 0u; i < values.size(); ++i) {
        writeTableRow(o, padTo, stats, values[i]);
    }
}

const TStrVec NO_STRINGS;
}

CReportWriter::CReportWriter(std::ostream& writeStream)
    : m_WriteStream(writeStream) {
}

bool CReportWriter::fieldNames(const TStrVec& /*fieldNames*/, const TStrVec& /*extraFieldNames*/) {
    return true;
}

const CReportWriter::TStrVec& CReportWriter::fieldNames() const {
    return NO_STRINGS;
}

bool CReportWriter::writeRow(const TStrStrUMap& /*dataRowFields*/,
                             const TStrStrUMap& /*overrideDataRowFields*/) {
    return true;
}

void CReportWriter::addTotalRecords(uint64_t n) {
    m_TotalRecords = print(n);
}

void CReportWriter::addInvalidRecords(uint64_t n) {
    m_InvalidRecords = print(n);
}

void CReportWriter::addFieldStatistics(const std::string& field,
                                       config_t::EDataType type,
                                       const CDataSummaryStatistics& summary) {
    std::size_t n = m_UnclassifiedFields.size();
    m_UnclassifiedFields.push_back(TStrVec(NUMBER_STATISTICS));
    m_UnclassifiedFields[n][FIELD_NAME] = field;
    m_UnclassifiedFields[n][DATA_TYPE] = config_t::print(type);
    m_UnclassifiedFields[n][EARLIEST_TIME] =
        core::CTimeUtils::toLocalString(summary.earliest());
    m_UnclassifiedFields[n][LATEST_TIME] =
        core::CTimeUtils::toLocalString(summary.latest());
    m_UnclassifiedFields[n][MEAN_RATE] = CTools::prettyPrint(summary.meanRate());
}

void CReportWriter::addFieldStatistics(const std::string& field,
                                       config_t::EDataType type,
                                       const CCategoricalDataSummaryStatistics& summary) {
    std::size_t n = m_CategoricalFields.size();
    m_CategoricalFields.push_back(TStrVec(NUMBER_STATISTICS));
    m_CategoricalFields[n][FIELD_NAME] = field;
    m_CategoricalFields[n][DATA_TYPE] = config_t::print(type);
    m_CategoricalFields[n][EARLIEST_TIME] =
        core::CTimeUtils::toLocalString(summary.earliest());
    m_CategoricalFields[n][LATEST_TIME] =
        core::CTimeUtils::toLocalString(summary.latest());
    m_CategoricalFields[n][MEAN_RATE] = CTools::prettyPrint(summary.meanRate());
    m_CategoricalFields[n][CATEGORICAL_DISTINCT_COUNT] = print(summary.distinctCount());
    CCategoricalDataSummaryStatistics::TStrSizePrVec topn;
    summary.topN(topn);
    m_CategoricalFields[n][CATEGORICAL_TOP_N_COUNTS] = print(topn);
}

void CReportWriter::addFieldStatistics(const std::string& field,
                                       config_t::EDataType type,
                                       const CNumericDataSummaryStatistics& summary) {
    std::size_t n = m_NumericFields.size();
    m_NumericFields.push_back(TStrVec(NUMBER_STATISTICS));
    m_NumericFields[n][FIELD_NAME] = field;
    m_NumericFields[n][DATA_TYPE] = config_t::print(type);
    m_NumericFields[n][EARLIEST_TIME] = core::CTimeUtils::toLocalString(summary.earliest());
    m_NumericFields[n][LATEST_TIME] = core::CTimeUtils::toLocalString(summary.latest());
    m_NumericFields[n][MEAN_RATE] = CTools::prettyPrint(summary.meanRate());
    m_NumericFields[n][NUMERIC_MINIMUM] = CTools::prettyPrint(summary.minimum());
    m_NumericFields[n][NUMERIC_MEDIAN] = CTools::prettyPrint(summary.median());
    m_NumericFields[n][NUMERIC_MAXIMUM] = CTools::prettyPrint(summary.maximum());
    CNumericDataSummaryStatistics::TDoubleDoublePrVec densitychart;
    summary.densityChart(densitychart);
    m_NumericFields[n][NUMERIC_DENSITY_CHART] = print(densitychart, 15);
}

void CReportWriter::addDetector(const CDetectorSpecification& spec) {
    std::size_t n = m_Detectors.size();
    m_Detectors.push_back(TStrVecVecVec(NUMBER_ATTRIBUTES));
    m_Detectors[n][DESCRIPTION].push_back(TStrVec(1, spec.description()));
    m_Detectors[n][OVERALL_SCORE].push_back(TStrVec(1, CTools::prettyPrint(spec.score())));
    CDetectorSpecification::TParamScoresVec scores;
    spec.scores(scores);
    m_Detectors[n][PARAMETER_SCORES].resize(scores.size(), TStrVec(NUMBER_PARAMETERS));
    for (std::size_t i = 0u; i < scores.size(); ++i) {
        m_Detectors[n][PARAMETER_SCORES][i][BUCKET_LENGTH_PARAMETER] =
            CTools::prettyPrint(scores[i].s_BucketLength);
        m_Detectors[n][PARAMETER_SCORES][i][IGNORE_EMPTY_PARAMETER] = scores[i].s_IgnoreEmpty;
        m_Detectors[n][PARAMETER_SCORES][i][SCORE_PARAMETER] =
            CTools::prettyPrint(scores[i].s_Score);
        m_Detectors[n][PARAMETER_SCORES][i][DESCRIPTION_PARAMETER] =
            scores[i].s_Descriptions.empty() ? std::string("-")
                                             : scores[i].s_Descriptions[0];
        for (std::size_t j = 1u; j < scores[i].s_Descriptions.size(); ++j) {
            m_Detectors[n][PARAMETER_SCORES][i][DESCRIPTION_PARAMETER] +=
                "\n" + scores[i].s_Descriptions[j];
        }
    }
    m_Detectors[n][DETECTOR_CONFIG].push_back(TStrVec(1, spec.detectorConfig()));
}

void CReportWriter::write() const {
    m_WriteStream << "============\n";
    m_WriteStream << "DATA SUMMARY\n";
    m_WriteStream << "============\n\n";

    m_WriteStream << "Found "
                  << (m_UnclassifiedFields.size() + m_CategoricalFields.size() +
                      m_NumericFields.size())
                  << " fields\n";
    m_WriteStream << "Processed " << m_TotalRecords << " records\n";
    m_WriteStream << "There were " << m_InvalidRecords << " invalid records\n";

    if (m_UnclassifiedFields.size() > 0) {
        m_WriteStream << "\nUnclassified Fields\n";
        m_WriteStream << "===================\n\n";
        writeTable(m_WriteStream, STATISTIC_LABELS, UNCLASSIFIED_STATISTICS, m_UnclassifiedFields);
    }
    if (m_CategoricalFields.size() > 0) {
        m_WriteStream << "\nCategorical Fields\n";
        m_WriteStream << "==================\n\n";
        writeTable(m_WriteStream, STATISTIC_LABELS, CATEGORICAL_STATISTICS, m_CategoricalFields);

        for (std::size_t i = 0u; i < m_CategoricalFields.size(); ++i) {
            m_WriteStream << "\nMost frequent for '"
                          << m_CategoricalFields[i][FIELD_NAME] << "':\n";
            m_WriteStream << m_CategoricalFields[i][CATEGORICAL_TOP_N_COUNTS];
        }
    }
    if (m_NumericFields.size() > 0) {
        m_WriteStream << "\nNumeric Fields\n";
        m_WriteStream << "==============\n\n";
        writeTable(m_WriteStream, STATISTIC_LABELS, NUMERIC_STATISTICS, m_NumericFields);
        for (std::size_t i = 0u; i < m_NumericFields.size(); ++i) {
            m_WriteStream << "\nProbability density for '"
                          << m_NumericFields[i][FIELD_NAME] << "':\n";
            m_WriteStream << pad(15, "x") << pad(15, "f(x)") << "\n";
            m_WriteStream << m_NumericFields[i][NUMERIC_DENSITY_CHART];
        }
    }
    if (m_Detectors.size() > 0) {
        m_WriteStream << "\n\n\n===================\n";
        m_WriteStream << "CANDIDATE DETECTORS\n";
        m_WriteStream << "===================";
        for (std::size_t i = 0u; i < m_Detectors.size(); ++i) {
            m_WriteStream << "\n\n\n"
                          << m_Detectors[i][DESCRIPTION][0][0] << "\n";
            m_WriteStream
                << std::string(m_Detectors[i][DESCRIPTION][0][0].length(), '=') << "\n";
            m_WriteStream << "\n  Best parameters score: "
                          << m_Detectors[i][OVERALL_SCORE][0][0] << "\n\n";
            writeTable(m_WriteStream, PARAMETER_LABELS, DETECTOR_PARAMETERS,
                       m_Detectors[i][PARAMETER_SCORES]);
            if (!m_Detectors[i][DETECTOR_CONFIG][0][0].empty()) {
                m_WriteStream << "\n"
                              << m_Detectors[i][DETECTOR_CONFIG][0][0] << "\n";
            }
        }
    }
}

const std::string CReportWriter::STATISTIC_LABELS[NUMBER_STATISTICS] = {
    std::string("Field Name"),
    std::string("Data Type"),
    std::string("Earliest Time"),
    std::string("Latest Time"),
    std::string("Mean Rate"),
    std::string("Distinct Categories"),
    std::string("Most Frequent Categories"),
    std::string("Minimum"),
    std::string("Median"),
    std::string("Maximum"),
    std::string("Probability Density Chart")};

const std::string CReportWriter::PARAMETER_LABELS[NUMBER_PARAMETERS] = {
    std::string("Bucket Length"), std::string("Ignore Empty"),
    std::string("Score"), std::string("Explanation")};

const std::size_t CReportWriter::UNCLASSIFIED_STATISTICS[] = {
    FIELD_NAME, DATA_TYPE, EARLIEST_TIME, LATEST_TIME, MEAN_RATE};

const std::size_t CReportWriter::CATEGORICAL_STATISTICS[] = {
    FIELD_NAME,  DATA_TYPE, EARLIEST_TIME,
    LATEST_TIME, MEAN_RATE, CATEGORICAL_DISTINCT_COUNT};

const std::size_t CReportWriter::NUMERIC_STATISTICS[] = {
    FIELD_NAME, DATA_TYPE,       EARLIEST_TIME,  LATEST_TIME,
    MEAN_RATE,  NUMERIC_MINIMUM, NUMERIC_MEDIAN, NUMERIC_MAXIMUM};

const std::size_t CReportWriter::DETECTOR_PARAMETERS[] = {
    BUCKET_LENGTH_PARAMETER, IGNORE_EMPTY_PARAMETER, SCORE_PARAMETER, DESCRIPTION_PARAMETER};
}
}
