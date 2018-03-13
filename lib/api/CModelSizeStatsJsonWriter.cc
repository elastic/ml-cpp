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

#include <api/CModelSizeStatsJsonWriter.h>

#include <core/CTimeUtils.h>

namespace ml
{
namespace api
{

// JSON field names
const std::string CModelSizeStatsJsonWriter::JOB_ID("job_id");
const std::string CModelSizeStatsJsonWriter::MODEL_SIZE_STATS("model_size_stats");
const std::string CModelSizeStatsJsonWriter::MODEL_BYTES("model_bytes");
const std::string CModelSizeStatsJsonWriter::TOTAL_BY_FIELD_COUNT("total_by_field_count");
const std::string CModelSizeStatsJsonWriter::TOTAL_OVER_FIELD_COUNT("total_over_field_count");
const std::string CModelSizeStatsJsonWriter::TOTAL_PARTITION_FIELD_COUNT("total_partition_field_count");
const std::string CModelSizeStatsJsonWriter::BUCKET_ALLOCATION_FAILURES_COUNT("bucket_allocation_failures_count");
const std::string CModelSizeStatsJsonWriter::MEMORY_STATUS("memory_status");
const std::string CModelSizeStatsJsonWriter::TIMESTAMP("timestamp");
const std::string CModelSizeStatsJsonWriter::LOG_TIME("log_time");

void CModelSizeStatsJsonWriter::write(const std::string &jobId,
                                      const model::CResourceMonitor::SResults &results,
                                      core::CRapidJsonConcurrentLineWriter &writer)
{
    writer.String(MODEL_SIZE_STATS);
    writer.StartObject();

    writer.String(JOB_ID);
    writer.String(jobId);
    writer.String(MODEL_BYTES);
    // Background persist causes the memory size to double due to copying
    // the models. On top of that, after the persist is done we may not
    // be able to retrieve that memory back. Thus, we report twice the
    // memory usage in order to allow for that.
    // See https://github.com/elastic/x-pack-elasticsearch/issues/1020.
    // Issue https://github.com/elastic/x-pack-elasticsearch/issues/857
    // discusses adding an option to perform only foreground persist.
    // If that gets implemented, we should only double when background
    // persist is configured.
    writer.Uint64(results.s_Usage * 2);

    writer.String(TOTAL_BY_FIELD_COUNT);
    writer.Uint64(results.s_ByFields);

    writer.String(TOTAL_OVER_FIELD_COUNT);
    writer.Uint64(results.s_OverFields);

    writer.String(TOTAL_PARTITION_FIELD_COUNT);
    writer.Uint64(results.s_PartitionFields);

    writer.String(BUCKET_ALLOCATION_FAILURES_COUNT);
    writer.Uint64(results.s_AllocationFailures);

    writer.String(MEMORY_STATUS);
    writer.String(print(results.s_MemoryStatus));

    writer.String(TIMESTAMP);
    writer.Int64(results.s_BucketStartTime * 1000);

    writer.String(LOG_TIME);
    writer.Int64(core::CTimeUtils::now() * 1000);

    writer.EndObject();
}


}
}

