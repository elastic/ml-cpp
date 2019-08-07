/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CMemoryUsageEstimationResultJsonWriter.h>

namespace ml {
namespace api {
namespace {

// JSON field names
const std::string EXPECTED_MEMORY_USAGE_WITH_ONE_PARTITION("expected_memory_usage_with_one_partition");
const std::string EXPECTED_MEMORY_USAGE_WITH_MAX_PARTITIONS("expected_memory_usage_with_max_partitions");
}

CMemoryUsageEstimationResultJsonWriter::CMemoryUsageEstimationResultJsonWriter(core::CJsonOutputStreamWrapper& strmOut)
    : m_Writer(strmOut) {
    // Don't write any output in the constructor because, the way things work at
    // the moment, the output stream might be redirected after construction
}

void CMemoryUsageEstimationResultJsonWriter::write(const std::string& expectedMemoryUsageWithOnePartition,
                                                   const std::string& expectedMemoryUsageWithMaxPartitions) {
    m_Writer.StartObject();
    m_Writer.Key(EXPECTED_MEMORY_USAGE_WITH_ONE_PARTITION);
    m_Writer.String(expectedMemoryUsageWithOnePartition);
    m_Writer.Key(EXPECTED_MEMORY_USAGE_WITH_MAX_PARTITIONS);
    m_Writer.String(expectedMemoryUsageWithMaxPartitions);
    m_Writer.EndObject();
    m_Writer.flush();
}
}
}
