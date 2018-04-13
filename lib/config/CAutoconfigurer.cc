/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <config/CAutoconfigurer.h>

#include <core/Constants.h>
#include <core/CStringUtils.h>
#include <core/CTimeUtils.h>

#include <maths/CTools.h>

#include <config/CAutoconfigurerDetectorPenalties.h>
#include <config/CAutoconfigurerFieldRolePenalties.h>
#include <config/CAutoconfigurerParams.h>
#include <config/CDataCountStatistics.h>
#include <config/CDetectorEnumerator.h>
#include <config/CDetectorRecord.h>
#include <config/CDetectorSpecification.h>
#include <config/CFieldRolePenalty.h>
#include <config/CFieldStatistics.h>
#include <config/ConfigTypes.h>
#include <config/Constants.h>
#include <config/CReportWriter.h>

#include <boost/numeric/conversion/bounds.hpp>
#include <boost/optional.hpp>
#include <boost/range.hpp>
#include <boost/unordered_map.hpp>

#include <cmath>
#include <string>
#include <vector>

namespace ml
{
namespace config
{
namespace
{

//! Check if we should report progress.
bool reportProgress(uint64_t records)
{
    static const double LOG_10 = maths::CTools::fastLog(10.0);
    double log10 = maths::CTools::fastLog(static_cast<double>(records) / 100.0) / LOG_10;
    uint64_t nextPow10 = static_cast<uint64_t>(std::pow(10, std::ceil(log10)));
    return records % std::max(nextPow10, uint64_t(100)) == 0;
}

const std::size_t UPDATE_SCORE_RECORD_COUNT_INTERVAL = 50000;
const core_t::TTime UPDATE_SCORE_TIME_INTERVAL       = 172800;

}

//! \brief The implementation of automatic configuration.
class CONFIG_EXPORT CAutoconfigurerImpl : public core::CNonCopyable
{
    public:
        using TStrVec = std::vector<std::string>;
        using TStrStrUMap = boost::unordered_map<std::string, std::string>;
        using TStrStrUMapCItr = TStrStrUMap::const_iterator;

    public:
        CAutoconfigurerImpl(const CAutoconfigurerParams &params, CReportWriter &reportWriter);

        //! Receive a single record to be processed.
        bool handleRecord(const TStrStrUMap &fieldValues);

        //! Generate the report.
        void finalise();

        //! Get the report writer.
        CReportWriter &reportWriter();

        //! How many records did we handle?
        uint64_t numRecordsHandled() const;

    private:
        using TTimeStrStrUMapPr = std::pair<core_t::TTime, TStrStrUMap>;
        using TTimeStrStrUMapPrVec = std::vector<TTimeStrStrUMapPr>;
        using TOptionalUserDataType = boost::optional<config_t::EUserDataType>;
        using TDetectorSpecificationVec = std::vector<CDetectorSpecification>;
        using TFieldStatisticsVec = std::vector<CFieldStatistics>;

    private:
        //! Extract the time from \p fieldValues.
        bool extractTime(const TStrStrUMap &fieldValues,
                         core_t::TTime &time) const;

        //! Initialize the field statistics.
        void initializeFieldStatisticsOnce(const TStrStrUMap &fieldValues);

        //! Actually process the content of the record.
        void processRecord(core_t::TTime time, const TStrStrUMap &dataRowFields);

        //! Update the statistics with \p time and \p fieldValues and maybe
        //! recompute detector scores and prune.
        void updateStatisticsAndMaybeComputeScores(core_t::TTime time,
                                                   const TStrStrUMap &fieldValues);

        //! Compute the detector scores.
        void computeScores(bool final);

        //! Generate the candidate detectors to evaluate.
        void generateCandidateDetectorsOnce();

        //! Run the records in the buffer through the detector scorers.
        void replayBuffer();

    private:
        //! The parameters.
        CAutoconfigurerParams m_Params;

        //! Set to true the first time initializeOnce is called.
        bool m_Initialized;

        //! The number of records supplied to handleRecord.
        uint64_t m_NumberRecords;

        //! The number of records with no time field.
        uint64_t m_NumberRecordsWithNoOrInvalidTime;

        //! The last time the detector scores were refreshed.
        core_t::TTime m_LastTimeScoresWereRefreshed;

        //! A buffer of the records before the configuration has begun.
        TTimeStrStrUMapPrVec m_Buffer;

        //! The field semantics and summary statistics.
        TFieldStatisticsVec m_FieldStatistics;

        //! The detector count data statistics.
        CDataCountStatisticsDirectAddressTable m_DetectorCountStatistics;

        //! The field role penalties.
        CAutoconfigurerFieldRolePenalties m_FieldRolePenalties;

        //! The detector penalties.
        CAutoconfigurerDetectorPenalties m_DetectorPenalties;

        //! Set to true the first time generateCandidateDetectorsOnce is called.
        bool m_GeneratedCandidateFieldNames;

        //! The candidate detectors.
        TDetectorSpecificationVec m_CandidateDetectors;

        //! Efficiently extracts the detector's records.
        CDetectorRecordDirectAddressTable m_DetectorRecordFactory;

        //! Writes out a report on the data and recommended configurations.
        CReportWriter &m_ReportWriter;
};


//////// CAutoconfigurer ////////

CAutoconfigurer::CAutoconfigurer(const CAutoconfigurerParams &params,
                                 CReportWriter &reportWriter) :
        m_Impl(new CAutoconfigurerImpl(params, reportWriter))
{
}

void CAutoconfigurer::newOutputStream()
{
    m_Impl->reportWriter().newOutputStream();
}

bool CAutoconfigurer::handleRecord(const TStrStrUMap &fieldValues)
{
    return m_Impl->handleRecord(fieldValues);
}

void CAutoconfigurer::finalise()
{
    m_Impl->finalise();
}

bool CAutoconfigurer::restoreState(core::CDataSearcher &/*restoreSearcher*/,
                                   core_t::TTime &/*completeToTime*/)
{
    return true;
}

bool CAutoconfigurer::persistState(core::CDataAdder &/*persister*/)
{
    return true;
}

uint64_t CAutoconfigurer::numRecordsHandled() const
{
    return m_Impl->numRecordsHandled();
}

api::COutputHandler &CAutoconfigurer::outputHandler()
{
    return m_Impl->reportWriter();
}


//////// CAutoconfigurerImpl ////////

CAutoconfigurerImpl::CAutoconfigurerImpl(const CAutoconfigurerParams &params,
                                         CReportWriter &reportWriter) :
        m_Params(params),
        m_Initialized(false),
        m_NumberRecords(0),
        m_NumberRecordsWithNoOrInvalidTime(0),
        m_LastTimeScoresWereRefreshed(boost::numeric::bounds<core_t::TTime>::lowest()),
        m_DetectorCountStatistics(m_Params),
        m_FieldRolePenalties(m_Params),
        m_DetectorPenalties(m_Params, m_FieldRolePenalties),
        m_GeneratedCandidateFieldNames(false),
        m_ReportWriter(reportWriter)
{
}

bool CAutoconfigurerImpl::handleRecord(const TStrStrUMap &fieldValues)
{
    ++m_NumberRecords;

    if (reportProgress(m_NumberRecords))
    {
        LOG_DEBUG("Processed " << m_NumberRecords << " records");
    }

    core_t::TTime time = 0;
    if (!this->extractTime(fieldValues, time))
    {
        ++m_NumberRecordsWithNoOrInvalidTime;
        return true;
    }
    this->initializeFieldStatisticsOnce(fieldValues);
    this->processRecord(time, fieldValues);

    return true;
}

void CAutoconfigurerImpl::finalise()
{
    LOG_TRACE("CAutoconfigurerImpl::finalise...");

    this->computeScores(true);

    m_ReportWriter.addTotalRecords(m_NumberRecords);
    m_ReportWriter.addInvalidRecords(m_NumberRecordsWithNoOrInvalidTime);

    for (std::size_t i = 0u; i < m_FieldStatistics.size(); ++i)
    {
        const std::string &name  = m_FieldStatistics[i].name();
        config_t::EDataType type = m_FieldStatistics[i].type();
        if (const CDataSummaryStatistics *summary = m_FieldStatistics[i].summary())
        {
            m_ReportWriter.addFieldStatistics(name, type, *summary);
        }
        if (const CCategoricalDataSummaryStatistics *summary = m_FieldStatistics[i].categoricalSummary())
        {
            m_ReportWriter.addFieldStatistics(name, type, *summary);
        }
        if (const CNumericDataSummaryStatistics *summary = m_FieldStatistics[i].numericSummary())
        {
            m_ReportWriter.addFieldStatistics(name, type, *summary);
        }
    }

    for (std::size_t i = 0u; i < m_CandidateDetectors.size(); ++i)
    {
        m_ReportWriter.addDetector(m_CandidateDetectors[i]);
    }

    m_ReportWriter.write();

    LOG_TRACE("CAutoconfigurerImpl::finalise done");
}

CReportWriter &CAutoconfigurerImpl::reportWriter()
{
    return m_ReportWriter;
}

uint64_t CAutoconfigurerImpl::numRecordsHandled() const
{
    return m_NumberRecords;
}

bool CAutoconfigurerImpl::extractTime(const TStrStrUMap &fieldValues,
                                      core_t::TTime &time) const
{
    TStrStrUMapCItr i = fieldValues.find(m_Params.timeFieldName());

    if (i == fieldValues.end())
    {
        LOG_ERROR("No time field '" << m_Params.timeFieldName()
                  << "' in record:" << core_t::LINE_ENDING
                  << CAutoconfigurer::debugPrintRecord(fieldValues));
        return false;
    }

    if (m_Params.timeFieldFormat().empty())
    {
        if (!core::CStringUtils::stringToType(i->second, time))
        {
            LOG_ERROR("Cannot interpret time field '" << m_Params.timeFieldName()
                      << "' in record:" << core_t::LINE_ENDING
                      << CAutoconfigurer::debugPrintRecord(fieldValues));
            return false;
        }
    }
    else if (!core::CTimeUtils::strptime(m_Params.timeFieldFormat(), i->second, time))
    {
        LOG_ERROR("Cannot interpret time field '" << m_Params.timeFieldName()
                  << "' using format '" << m_Params.timeFieldFormat()
                  << "' in record:" << core_t::LINE_ENDING
                  << CAutoconfigurer::debugPrintRecord(fieldValues));
        return false;
    }

    return true;
}

void CAutoconfigurerImpl::initializeFieldStatisticsOnce(const TStrStrUMap &fieldValues)
{
    if (m_Initialized)
    {
        return;
    }

    m_FieldStatistics.reserve(fieldValues.size());
    for (const auto &entry : fieldValues)
    {
        const std::string &fieldName = entry.first;
        if (fieldName != m_Params.timeFieldName() && m_Params.fieldOfInterest(fieldName))
        {
            LOG_DEBUG("Adding field '" << fieldName << "'");
            m_FieldStatistics.push_back(CFieldStatistics(fieldName, m_Params));
        }
    }

    m_Initialized = true;
}

void CAutoconfigurerImpl::processRecord(core_t::TTime time, const TStrStrUMap &fieldValues)
{
    for (std::size_t i = 0u; i < m_FieldStatistics.size(); ++i)
    {
        TStrStrUMapCItr j = fieldValues.find(m_FieldStatistics[i].name());
        if (j != fieldValues.end())
        {
            m_FieldStatistics[i].add(time, j->second);
        }
    }

    if (m_NumberRecords < m_Params.minimumRecordsToAttemptConfig())
    {
        m_Buffer.push_back(std::make_pair(time, fieldValues));
    }
    else
    {
        this->generateCandidateDetectorsOnce();
        this->replayBuffer();
        this->updateStatisticsAndMaybeComputeScores(time, fieldValues);
    }
}

void CAutoconfigurerImpl::updateStatisticsAndMaybeComputeScores(core_t::TTime time,
                                                                const TStrStrUMap &fieldValues)
{
    CDetectorRecordDirectAddressTable::TDetectorRecordVec records;
    m_DetectorRecordFactory.detectorRecords(time, fieldValues, m_CandidateDetectors, records);
    m_DetectorCountStatistics.add(records);
    if (   m_NumberRecords % UPDATE_SCORE_RECORD_COUNT_INTERVAL == 0
        && time >= m_LastTimeScoresWereRefreshed + UPDATE_SCORE_TIME_INTERVAL)
    {
        this->computeScores(false);
        m_LastTimeScoresWereRefreshed = time;
    }
}

void CAutoconfigurerImpl::computeScores(bool final)
{
    LOG_TRACE("CAutoconfigurerImpl::computeScores...");

    std::size_t last = 0u;

    for (std::size_t i = 0u; i < m_CandidateDetectors.size(); ++i)
    {
        LOG_TRACE("Refreshing scores for " << m_CandidateDetectors[i].description());
        m_CandidateDetectors[i].refreshScores();
        LOG_TRACE("score = " << m_CandidateDetectors[i].score());
        if (m_CandidateDetectors[i].score() > (final ? m_Params.minimumDetectorScore() : 0.0))
        {
            if (i > last)
            {
                m_CandidateDetectors[i].swap(m_CandidateDetectors[last]);
            }
            ++last;
        }
    }

    if (last < m_CandidateDetectors.size())
    {
        LOG_DEBUG("Removing " << m_CandidateDetectors.size() - last << " detectors");
        m_CandidateDetectors.erase(m_CandidateDetectors.begin() + last, m_CandidateDetectors.end());
        m_DetectorRecordFactory.build(m_CandidateDetectors);
        m_DetectorCountStatistics.pruneUnsed(m_CandidateDetectors);
    }

    LOG_TRACE("CAutoconfigurerImpl::computeScores done");
}

void CAutoconfigurerImpl::generateCandidateDetectorsOnce()
{
    if (m_GeneratedCandidateFieldNames)
    {
        return;
    }

    LOG_DEBUG("Generate Candidate Detectors:");

    using TAddField = void (CDetectorEnumerator::*)(const std::string &);
    using TCanUse = bool (CAutoconfigurerParams::*)(const std::string &) const;

    CDetectorEnumerator enumerator(m_Params);
    for (std::size_t i = 0u; i < m_Params.functionsCategoriesToConfigure().size(); ++i)
    {
        enumerator.addFunction(m_Params.functionsCategoriesToConfigure()[i]);
    }
    for (std::size_t i = 0u; i < m_FieldStatistics.size(); ++i)
    {
        static std::string FIELD_NAMES[] =
            {
                std::string("categorical argument"),
                std::string("metric argument"),
                std::string("by field"),
                std::string("rare function by field"),
                std::string("over field"),
                std::string("partition field")
            };
        static TAddField ADD_FIELD[] =
            {
                &CDetectorEnumerator::addCategoricalFunctionArgument,
                &CDetectorEnumerator::addMetricFunctionArgument,
                &CDetectorEnumerator::addByField,
                &CDetectorEnumerator::addRareByField,
                &CDetectorEnumerator::addOverField,
                &CDetectorEnumerator::addPartitionField
            };
        static TCanUse CAN_USE[] =
            {
                &CAutoconfigurerParams::canUseForFunctionArgument,
                &CAutoconfigurerParams::canUseForFunctionArgument,
                &CAutoconfigurerParams::canUseForByField,
                &CAutoconfigurerParams::canUseForByField,
                &CAutoconfigurerParams::canUseForOverField,
                &CAutoconfigurerParams::canUseForPartitionField
            };
        double scores[] =
            {
                m_FieldStatistics[i].score(m_FieldRolePenalties.categoricalFunctionArgumentPenalty()),
                m_FieldStatistics[i].score(m_FieldRolePenalties.metricFunctionArgumentPenalty()),
                m_FieldStatistics[i].score(m_FieldRolePenalties.byPenalty()),
                m_FieldStatistics[i].score(m_FieldRolePenalties.rareByPenalty()),
                m_FieldStatistics[i].score(m_FieldRolePenalties.overPenalty()),
                m_FieldStatistics[i].score(m_FieldRolePenalties.partitionPenalty())
            };

        const std::string &fieldName = m_FieldStatistics[i].name();
        for (std::size_t j = 0u; j < boost::size(FIELD_NAMES); ++j)
        {
            if ((m_Params.*CAN_USE[j])(fieldName) && scores[j] > 0.0)
            {
                LOG_DEBUG(FIELD_NAMES[j] << " '" << fieldName << "' with score " << scores[j]);
                (enumerator.*ADD_FIELD[j])(fieldName);
            }
        }
    }

    LOG_DEBUG("Generating...");
    enumerator.generate(m_CandidateDetectors);
    LOG_DEBUG("Got " << m_CandidateDetectors.size() << " detectors");

    m_DetectorCountStatistics.build(m_CandidateDetectors);
    m_DetectorRecordFactory.build(m_CandidateDetectors);

    for (std::size_t i = 0u; i < m_CandidateDetectors.size(); ++i)
    {
        CDetectorSpecification &spec = m_CandidateDetectors[i];
        spec.addFieldStatistics(m_FieldStatistics);
        spec.setPenalty(m_DetectorPenalties.penaltyFor(spec));
        spec.setCountStatistics(m_DetectorCountStatistics.statistics(spec));
    }

    m_GeneratedCandidateFieldNames = true;
}

void CAutoconfigurerImpl::replayBuffer()
{
    for (std::size_t i = 0u; i < m_Buffer.size(); ++i)
    {
        if (reportProgress(i))
        {
            LOG_DEBUG("Replayed " << i << " records");
        }
        this->updateStatisticsAndMaybeComputeScores(m_Buffer[i].first, m_Buffer[i].second);
    }
    TTimeStrStrUMapPrVec empty;
    m_Buffer.swap(empty);
}

}
}
