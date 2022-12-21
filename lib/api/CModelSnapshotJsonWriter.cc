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

#include <api/CModelSnapshotJsonWriter.h>

#include <api/CModelSizeStatsJsonWriter.h>

namespace ml {
namespace api {
namespace {

// JSON field names
const std::string JOB_ID("job_id");
const std::string MIN_VERSION("min_version");
const std::string TIMESTAMP("timestamp");
const std::string MODEL_SNAPSHOT("model_snapshot");
const std::string SNAPSHOT_ID("snapshot_id");
const std::string SNAPSHOT_DOC_COUNT("snapshot_doc_count");
const std::string DESCRIPTION("description");
const std::string LATEST_RECORD_TIME("latest_record_time_stamp");
const std::string LATEST_RESULT_TIME("latest_result_time_stamp");
const std::string QUANTILES("quantiles");
const std::string QUANTILE_STATE("quantile_state");
}

CModelSnapshotJsonWriter::CModelSnapshotJsonWriter(const std::string& jobId,
                                                   core::CJsonOutputStreamWrapper& strmOut)
    : m_JobId(jobId), m_Writer(strmOut) {
    // Don't write any output in the constructor because, the way things work at
    // the moment, the output stream might be redirected after construction
}

void CModelSnapshotJsonWriter::write(const SModelSnapshotReport& report) {
    m_Writer.StartObject();
    m_Writer.Key(MODEL_SNAPSHOT);
    m_Writer.StartObject();

    m_Writer.Key(JOB_ID);
    m_Writer.String(m_JobId);
    m_Writer.Key(MIN_VERSION);
    m_Writer.String(report.s_MinVersion);
    m_Writer.Key(SNAPSHOT_ID);
    m_Writer.String(report.s_SnapshotId);

    m_Writer.Key(SNAPSHOT_DOC_COUNT);
    m_Writer.Uint64(report.s_NumDocs);

    m_Writer.Key(TIMESTAMP);
    m_Writer.Time(report.s_SnapshotTimestamp);

    m_Writer.Key(DESCRIPTION);
    m_Writer.String(report.s_Description);

    CModelSizeStatsJsonWriter::write(m_JobId, report.s_ModelSizeStats, m_Writer);

    if (report.s_LatestRecordTime > 0) {
        m_Writer.Key(LATEST_RECORD_TIME);
        m_Writer.Time(report.s_LatestRecordTime);
    }
    if (report.s_LatestFinalResultTime > 0) {
        m_Writer.Key(LATEST_RESULT_TIME);
        m_Writer.Time(report.s_LatestFinalResultTime);
    }

    // write normalizerState here
    m_Writer.Key(QUANTILES);

    writeQuantileState(m_JobId, report.s_NormalizerState,
                       report.s_LatestFinalResultTime, m_Writer);

    m_Writer.EndObject();
    m_Writer.EndObject();

    m_Writer.flush();

    LOG_DEBUG(<< "Wrote model snapshot report with ID " << report.s_SnapshotId
              << " for: " << report.s_Description
              << ", latest final results at " << report.s_LatestFinalResultTime);
}

void CModelSnapshotJsonWriter::writeQuantileState(const std::string& jobId,
                                                  const std::string& state,
                                                  core_t::TTime time,
                                                  core::CRapidJsonConcurrentLineWriter& writer) {
    writer.StartObject();
    writer.Key(JOB_ID);
    writer.String(jobId);
    writer.Key(QUANTILE_STATE);
    writer.String(state);
    writer.Key(TIMESTAMP);
    writer.Time(time);
    writer.EndObject();
}
}
}
