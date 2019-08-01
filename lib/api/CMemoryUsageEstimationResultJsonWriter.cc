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
const std::string MEMORY_USAGE_WITH_ONE_PARTITION("memory_usage_with_one_partition");
const std::string MEMORY_USAGE_WITH_MAX_PARTITIONS("memory_usage_with_max_partitions");
}

SMemoryUsageEstimationResult::SMemoryUsageEstimationResult(
    std::size_t memoryUsageWithOnePartition, std::size_t memoryUsageWithMaxPartitions):
s_MemoryUsageWithOnePartition(memoryUsageWithOnePartition), s_MemoryUsageWithMaxPartitions(memoryUsageWithMaxPartitions) {}

SMemoryUsageEstimationResult::SMemoryUsageEstimationResult(
    const SMemoryUsageEstimationResult& result):
SMemoryUsageEstimationResult(result.s_MemoryUsageWithOnePartition, result.s_MemoryUsageWithMaxPartitions) {}

CMemoryUsageEstimationResultJsonWriter::CMemoryUsageEstimationResultJsonWriter(core::CJsonOutputStreamWrapper& strmOut) : m_Writer(strmOut) {
    // Don't write any output in the constructor because, the way things work at
    // the moment, the output stream might be redirected after construction
}

void CMemoryUsageEstimationResultJsonWriter::write(const SMemoryUsageEstimationResult& result) {
    m_Writer.StartObject();
    m_Writer.Key(MEMORY_USAGE_WITH_ONE_PARTITION);
    m_Writer.Uint64(result.s_MemoryUsageWithOnePartition);
    m_Writer.Key(MEMORY_USAGE_WITH_MAX_PARTITIONS);
    m_Writer.Uint64(result.s_MemoryUsageWithMaxPartitions);
    m_Writer.EndObject();
    m_Writer.flush();
}
}
}
