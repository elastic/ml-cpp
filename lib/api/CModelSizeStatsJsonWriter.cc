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

#include <api/CModelSizeStatsJsonWriter.h>

#include <core/CBoostJsonConcurrentLineWriter.h>
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
const std::string SYSTEM_MEMORY_BYTES{"system_memory_bytes"};
const std::string MAX_SYSTEM_MEMORY_BYTES{"max_system_memory_bytes"};
const std::string MODEL_BYTES_EXCEEDED{"model_bytes_exceeded"};
const std::string MODEL_BYTES_MEMORY_LIMIT{"model_bytes_memory_limit"};
const std::string TOTAL_BY_FIELD_COUNT{"total_by_field_count"};
const std::string TOTAL_OVER_FIELD_COUNT{"total_over_field_count"};
const std::string TOTAL_PARTITION_FIELD_COUNT{"total_partition_field_count"};
const std::string BUCKET_ALLOCATION_FAILURES_COUNT{"bucket_allocation_failures_count"};
const std::string MEMORY_STATUS{"memory_status"};
const std::string ASSIGNMENT_MEMORY_BASIS{"assignment_memory_basis"};
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
const std::string OUTPUT_MEMORY_ALLOCATOR_BYTES("output_memory_allocator_bytes");
}

void CModelSizeStatsJsonWriter::write(const std::string& jobId,
                                      const model::CResourceMonitor::SModelSizeStats& results,
                                      core::CBoostJsonConcurrentLineWriter& writer) {
    writer.onKey(MODEL_SIZE_STATS);
    writer.onObjectBegin();

    writer.onKey(MODEL_BYTES);
    writer.onUint64(results.s_AdjustedUsage);

    writer.onKey(PEAK_MODEL_BYTES);
    writer.onUint64(results.s_AdjustedPeakUsage);

    writer.onKey(MODEL_BYTES_EXCEEDED);
    writer.onUint64(results.s_BytesExceeded);

    writer.onKey(MODEL_BYTES_MEMORY_LIMIT);
    writer.onUint64(results.s_BytesMemoryLimit);

    writer.onKey(TOTAL_BY_FIELD_COUNT);
    writer.onUint64(results.s_ByFields);

    writer.onKey(TOTAL_OVER_FIELD_COUNT);
    writer.onUint64(results.s_OverFields);

    writer.onKey(TOTAL_PARTITION_FIELD_COUNT);
    writer.onUint64(results.s_PartitionFields);

    writer.onKey(BUCKET_ALLOCATION_FAILURES_COUNT);
    writer.onUint64(results.s_AllocationFailures);

    writer.onKey(MEMORY_STATUS);
    writer.onString(model_t::print(results.s_MemoryStatus));

    if (results.s_AssignmentMemoryBasis != model_t::E_AssignmentBasisUnknown) {
        writer.onKey(ASSIGNMENT_MEMORY_BASIS);
        writer.onString(model_t::print(results.s_AssignmentMemoryBasis));
    }

    writer.onKey(OUTPUT_MEMORY_ALLOCATOR_BYTES);
    writer.onUint64(results.s_OutputMemoryAllocatorUsage);

    CModelSizeStatsJsonWriter::writeCommonFields(
        jobId, results.s_OverallCategorizerStats, results.s_BucketStartTime, writer);

    writer.onObjectEnd();
}

void CModelSizeStatsJsonWriter::writeCategorizerStats(
    const std::string& jobId,
    const std::string& partitionFieldName,
    const std::string& partitionFieldValue,
    const model::SCategorizerStats& categorizerStats,
    const TOptionalTime& timestamp,
    core::CBoostJsonConcurrentLineWriter& writer) {

    writer.onKey(CATEGORIZER_STATS);
    writer.onObjectBegin();

    CModelSizeStatsJsonWriter::writeCommonFields(jobId, categorizerStats, timestamp, writer);

    if (partitionFieldName.empty() == false) {
        writer.onKey(PARTITION_FIELD_NAME);
        writer.onString(partitionFieldName);

        writer.onKey(PARTITION_FIELD_VALUE);
        writer.onString(partitionFieldValue);
    }

    writer.onObjectEnd();
}

void CModelSizeStatsJsonWriter::writeCommonFields(const std::string& jobId,
                                                  const model::SCategorizerStats& categorizerStats,
                                                  const TOptionalTime& timestamp,
                                                  core::CBoostJsonConcurrentLineWriter& writer) {

    writer.onKey(JOB_ID);
    writer.onString(jobId);

    writer.onKey(CATEGORIZED_DOC_COUNT);
    writer.onUint64(categorizerStats.s_CategorizedMessages);

    writer.onKey(TOTAL_CATEGORY_COUNT);
    writer.onUint64(categorizerStats.s_TotalCategories);

    writer.onKey(FREQUENT_CATEGORY_COUNT);
    writer.onUint64(categorizerStats.s_FrequentCategories);

    writer.onKey(RARE_CATEGORY_COUNT);
    writer.onUint64(categorizerStats.s_RareCategories);

    writer.onKey(DEAD_CATEGORY_COUNT);
    writer.onUint64(categorizerStats.s_DeadCategories);

    writer.onKey(FAILED_CATEGORY_COUNT);
    writer.onUint64(categorizerStats.s_MemoryCategorizationFailures);

    writer.onKey(CATEGORIZATION_STATUS);
    writer.onString(model_t::print(categorizerStats.s_CategorizationStatus));

    std::int64_t nowMs{core::CTimeUtils::nowMs()};
    writer.onKey(TIMESTAMP);
    if (timestamp.has_value()) {
        writer.onTime(*timestamp);
    } else {
        writer.onInt64(nowMs);
    }

    writer.onKey(LOG_TIME);
    writer.onInt64(nowMs);
}
}
}
