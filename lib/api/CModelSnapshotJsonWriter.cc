/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2018 Elasticsearch BV. All Rights Reserved.
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

#include <api/CModelSnapshotJsonWriter.h>

#include <api/CModelSizeStatsJsonWriter.h>

namespace ml
{
namespace api
{

// JSON field names
const std::string CModelSnapshotJsonWriter::JOB_ID("job_id");
const std::string CModelSnapshotJsonWriter::TIMESTAMP("timestamp");
const std::string CModelSnapshotJsonWriter::MODEL_SNAPSHOT("model_snapshot");
const std::string CModelSnapshotJsonWriter::SNAPSHOT_ID("snapshot_id");
const std::string CModelSnapshotJsonWriter::SNAPSHOT_DOC_COUNT("snapshot_doc_count");
const std::string CModelSnapshotJsonWriter::DESCRIPTION("description");
const std::string CModelSnapshotJsonWriter::LATEST_RECORD_TIME("latest_record_time_stamp");
const std::string CModelSnapshotJsonWriter::LATEST_RESULT_TIME("latest_result_time_stamp");
const std::string CModelSnapshotJsonWriter::QUANTILE_STATE("quantile_state");
const std::string CModelSnapshotJsonWriter::QUANTILES("quantiles");

CModelSnapshotJsonWriter::CModelSnapshotJsonWriter(const std::string &jobId, core::CJsonOutputStreamWrapper &strmOut)
    : m_JobId(jobId),
      m_Writer(strmOut)
{
    // Don't write any output in the constructor because, the way things work at
    // the moment, the output stream might be redirected after construction
}

void CModelSnapshotJsonWriter::write(const SModelSnapshotReport &report)
{
    m_Writer.StartObject();
    m_Writer.String(MODEL_SNAPSHOT);
    m_Writer.StartObject();

    m_Writer.String(JOB_ID);
    m_Writer.String(m_JobId);
    m_Writer.String(SNAPSHOT_ID);
    m_Writer.String(report.s_SnapshotId);

    m_Writer.String(SNAPSHOT_DOC_COUNT);
    m_Writer.Uint64(report.s_NumDocs);

    // Write as a Java timestamp - ms since the epoch rather than seconds
    int64_t javaTimestamp = int64_t(report.s_SnapshotTimestamp) * 1000;

    m_Writer.String(TIMESTAMP);
    m_Writer.Int64(javaTimestamp);

    m_Writer.String(DESCRIPTION);
    m_Writer.String(report.s_Description);

    CModelSizeStatsJsonWriter::write(m_JobId, report.s_ModelSizeStats, m_Writer);

    if (report.s_LatestRecordTime > 0)
    {
        javaTimestamp = int64_t(report.s_LatestRecordTime) * 1000;

        m_Writer.String(LATEST_RECORD_TIME);
        m_Writer.Int64(javaTimestamp);
    }
    if (report.s_LatestFinalResultTime > 0)
    {
        javaTimestamp = int64_t(report.s_LatestFinalResultTime) * 1000;

        m_Writer.String(LATEST_RESULT_TIME);
        m_Writer.Int64(javaTimestamp);
    }

    // write normalizerState here
    m_Writer.String(QUANTILES);

    writeQuantileState(m_JobId, report.s_NormalizerState, report.s_LatestFinalResultTime, m_Writer);

    m_Writer.EndObject();
    m_Writer.EndObject();

    m_Writer.flush();

    LOG_DEBUG("Wrote model snapshot report with ID " << report.s_SnapshotId <<
              " for: " << report.s_Description << ", latest final results at " << report.s_LatestFinalResultTime);
}

void CModelSnapshotJsonWriter::writeQuantileState(const std::string &jobId,
                                                  const std::string &state,
                                                  core_t::TTime time,
                                                  core::CRapidJsonConcurrentLineWriter &writer)
{
    writer.StartObject();
    writer.String(JOB_ID);
    writer.String(jobId);
    writer.String(QUANTILE_STATE);
    writer.String(state);
    writer.String(TIMESTAMP);
    writer.Int64(time * 1000);
    writer.EndObject();
}

}
}
