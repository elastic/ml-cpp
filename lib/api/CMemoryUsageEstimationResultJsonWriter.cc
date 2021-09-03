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

#include <api/CMemoryUsageEstimationResultJsonWriter.h>

namespace ml {
namespace api {
namespace {

// JSON field names
const std::string EXPECTED_MEMORY_WITHOUT_DISK("expected_memory_without_disk");
const std::string EXPECTED_MEMORY_WITH_DISK("expected_memory_with_disk");
}

CMemoryUsageEstimationResultJsonWriter::CMemoryUsageEstimationResultJsonWriter(core::CJsonOutputStreamWrapper& strmOut)
    : m_Writer(strmOut) {
    // Don't write any output in the constructor because, the way things work at
    // the moment, the output stream might be redirected after construction
}

void CMemoryUsageEstimationResultJsonWriter::write(const std::string& expectedMemoryWithoutDisk,
                                                   const std::string& expectedMemoryWithDisk) {
    m_Writer.StartObject();
    m_Writer.Key(EXPECTED_MEMORY_WITHOUT_DISK);
    m_Writer.String(expectedMemoryWithoutDisk);
    m_Writer.Key(EXPECTED_MEMORY_WITH_DISK);
    m_Writer.String(expectedMemoryWithDisk);
    m_Writer.EndObject();
    m_Writer.flush();
}
}
}
