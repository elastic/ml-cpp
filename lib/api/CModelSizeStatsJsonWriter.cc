/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CModelSizeStatsJsonWriter.h>

#include <core/CTimeUtils.h>

namespace ml {
namespace api {
namespace {

// JSON field names
const std::string JOB_ID{"job_id"};
const std::string MODEL_SIZE_STATS{"model_size_stats"};
const std::string MODEL_BYTES{"model_bytes"};
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
const std::string CATEGORIZATION_STATUS{"categorization_status"};
const std::string TIMESTAMP{"timestamp"};
const std::string LOG_TIME{"log_time"};
}

void CModelSizeStatsJsonWriter::write(const std::string& jobId,
                                      const model::CResourceMonitor::SModelSizeStats& results,
                                      core::CRapidJsonConcurrentLineWriter& writer) {
    writer.String(MODEL_SIZE_STATS);
    writer.StartObject();

    writer.String(JOB_ID);
    writer.String(jobId);

    writer.String(MODEL_BYTES);
    writer.Uint64(results.s_AdjustedUsage);

    writer.String(MODEL_BYTES_EXCEEDED);
    writer.Uint64(results.s_BytesExceeded);

    writer.String(MODEL_BYTES_MEMORY_LIMIT);
    writer.Uint64(results.s_BytesMemoryLimit);

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

    writer.String(CATEGORIZED_DOC_COUNT);
    writer.Uint64(results.s_CategorizedMessages);

    writer.String(TOTAL_CATEGORY_COUNT);
    writer.Uint64(results.s_TotalCategories);

    writer.String(FREQUENT_CATEGORY_COUNT);
    writer.Uint64(results.s_FrequentCategories);

    writer.String(RARE_CATEGORY_COUNT);
    writer.Uint64(results.s_RareCategories);

    writer.String(DEAD_CATEGORY_COUNT);
    writer.Uint64(results.s_DeadCategories);

    writer.String(CATEGORIZATION_STATUS);
    writer.String(print(results.s_CategorizationStatus));

    writer.String(TIMESTAMP);
    writer.Time(results.s_BucketStartTime);

    writer.String(LOG_TIME);
    writer.Time(core::CTimeUtils::now());

    writer.EndObject();
}
}
}
