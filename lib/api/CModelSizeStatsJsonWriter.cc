/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CModelSizeStatsJsonWriter.h>

#include <core/CTimeUtils.h>

#include <model/SCategorizerStats.h>

namespace ml {
namespace api {
namespace {

// JSON field names
const std::string JOB_ID{"job_id"};
const std::string MODEL_SIZE_STATS{"model_size_stats"};
const std::string MODEL_BYTES{"model_bytes"};
const std::string PEAK_MODEL_BYTES{"peak_model_bytes"};
const std::string MODEL_BYTES_EXCEEDED{"model_bytes_exceeded"};
const std::string MODEL_BYTES_MEMORY_LIMIT{"model_bytes_memory_limit"};
const std::string TOTAL_BY_FIELD_COUNT{"total_by_field_count"};
const std::string TOTAL_OVER_FIELD_COUNT{"total_over_field_count"};
const std::string TOTAL_PARTITION_FIELD_COUNT{"total_partition_field_count"};
const std::string BUCKET_ALLOCATION_FAILURES_COUNT{"bucket_allocation_failures_count"};
const std::string MEMORY_STATUS{"memory_status"};
const std::string CATEGORIZED_DOC_COUNT{"categorized_doc_count"};
const std::string TOTAL_CATEGORY_COUNT{"total_category_count"};
const std::string FREQUENT_CATEGORY_COUNT{"frequent_category_count"};
const std::string RARE_CATEGORY_COUNT{"rare_category_count"};
const std::string DEAD_CATEGORY_COUNT{"dead_category_count"};
const std::string FAILED_CATEGORY_COUNT{"failed_category_count"};
const std::string CATEGORIZATION_STATUS{"categorization_status"};
const std::string TIMESTAMP{"timestamp"};
const std::string LOG_TIME{"log_time"};
const std::string CATEGORIZER_STATS{"categorizer_stats"};
const std::string PARTITION_FIELD_NAME{"partition_field_name"};
const std::string PARTITION_FIELD_VALUE{"partition_field_value"};
}

void CModelSizeStatsJsonWriter::write(const std::string& jobId,
                                      const model::CResourceMonitor::SModelSizeStats& results,
                                      core::CRapidJsonConcurrentLineWriter& writer) {
    writer.Key(MODEL_SIZE_STATS);
    writer.StartObject();

    writer.Key(MODEL_BYTES);
    writer.Uint64(results.s_AdjustedUsage);

    writer.Key(PEAK_MODEL_BYTES);
    writer.Uint64(results.s_AdjustedPeakUsage);

    writer.Key(MODEL_BYTES_EXCEEDED);
    writer.Uint64(results.s_BytesExceeded);

    writer.Key(MODEL_BYTES_MEMORY_LIMIT);
    writer.Uint64(results.s_BytesMemoryLimit);

    writer.Key(TOTAL_BY_FIELD_COUNT);
    writer.Uint64(results.s_ByFields);

    writer.Key(TOTAL_OVER_FIELD_COUNT);
    writer.Uint64(results.s_OverFields);

    writer.Key(TOTAL_PARTITION_FIELD_COUNT);
    writer.Uint64(results.s_PartitionFields);

    writer.Key(BUCKET_ALLOCATION_FAILURES_COUNT);
    writer.Uint64(results.s_AllocationFailures);

    writer.Key(MEMORY_STATUS);
    writer.String(model_t::print(results.s_MemoryStatus));

    CModelSizeStatsJsonWriter::writeCommonFields(
        jobId, results.s_OverallCategorizerStats, results.s_BucketStartTime, writer);

    writer.EndObject();
}

void CModelSizeStatsJsonWriter::writeCategorizerStats(
    const std::string& jobId,
    const std::string& partitionFieldName,
    const std::string& partitionFieldValue,
    const model::SCategorizerStats& categorizerStats,
    const TOptionalTime& timestamp,
    core::CRapidJsonConcurrentLineWriter& writer) {

    writer.Key(CATEGORIZER_STATS);
    writer.StartObject();

    CModelSizeStatsJsonWriter::writeCommonFields(jobId, categorizerStats, timestamp, writer);

    if (partitionFieldName.empty() == false) {
        writer.Key(PARTITION_FIELD_NAME);
        writer.String(partitionFieldName);

        writer.Key(PARTITION_FIELD_VALUE);
        writer.String(partitionFieldValue);
    }

    writer.EndObject();
}

void CModelSizeStatsJsonWriter::writeCommonFields(const std::string& jobId,
                                                  const model::SCategorizerStats& categorizerStats,
                                                  const TOptionalTime& timestamp,
                                                  core::CRapidJsonConcurrentLineWriter& writer) {

    writer.Key(JOB_ID);
    writer.String(jobId);

    writer.Key(CATEGORIZED_DOC_COUNT);
    writer.Uint64(categorizerStats.s_CategorizedMessages);

    writer.Key(TOTAL_CATEGORY_COUNT);
    writer.Uint64(categorizerStats.s_TotalCategories);

    writer.Key(FREQUENT_CATEGORY_COUNT);
    writer.Uint64(categorizerStats.s_FrequentCategories);

    writer.Key(RARE_CATEGORY_COUNT);
    writer.Uint64(categorizerStats.s_RareCategories);

    writer.Key(DEAD_CATEGORY_COUNT);
    writer.Uint64(categorizerStats.s_DeadCategories);

    writer.Key(FAILED_CATEGORY_COUNT);
    writer.Uint64(categorizerStats.s_MemoryCategorizationFailures);

    writer.Key(CATEGORIZATION_STATUS);
    writer.String(model_t::print(categorizerStats.s_CategorizationStatus));

    std::int64_t nowMs{core::CTimeUtils::nowMs()};
    writer.Key(TIMESTAMP);
    if (timestamp.has_value()) {
        writer.Time(*timestamp);
    } else {
        writer.Int64(nowMs);
    }

    writer.Key(LOG_TIME);
    writer.Int64(nowMs);
}
}
}
