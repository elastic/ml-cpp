/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CMemoryUsageEstimationResultJsonWriter_h
#define INCLUDED_ml_api_CMemoryUsageEstimationResultJsonWriter_h

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CNonCopyable.h>
#include <core/CRapidJsonConcurrentLineWriter.h>

#include <api/ImportExport.h>

#include <string>

namespace ml {
namespace api {

//! Structure to store the memory usage estimation result
struct API_EXPORT SMemoryUsageEstimationResult {
    SMemoryUsageEstimationResult(size_t memoryUsageWithOnePartition, size_t memoryUsageWithMaxPartitions);
    SMemoryUsageEstimationResult(const SMemoryUsageEstimationResult& result);

    const size_t s_MemoryUsageWithOnePartition;
    const size_t s_MemoryUsageWithMaxPartitions;
};

//! \brief
//! Write memory usage estimation result in JSON format
//!
//! DESCRIPTION:\n
//! Outputs the memory usage estimation result.
//!
class API_EXPORT CMemoryUsageEstimationResultJsonWriter : private core::CNonCopyable {
public:
    //! Constructor that causes output to be written to the specified wrapped stream
    CMemoryUsageEstimationResultJsonWriter(core::CJsonOutputStreamWrapper& strmOut);

    //! Writes the given memory usage estimation result in JSON format.
    void write(const SMemoryUsageEstimationResult& result);

private:
    //! JSON line writer
    core::CRapidJsonConcurrentLineWriter m_Writer;
};
}
}

#endif // INCLUDED_ml_api_CMemoryUsageEstimationResultJsonWriter_h
