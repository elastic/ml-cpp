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
#ifndef INCLUDED_ml_api_CMemoryUsageEstimationResultJsonWriter_h
#define INCLUDED_ml_api_CMemoryUsageEstimationResultJsonWriter_h

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CNonCopyable.h>
#include <core/CRapidJsonConcurrentLineWriter.h>

#include <api/ImportExport.h>

#include <string>

namespace ml {
namespace api {

//! \brief
//! Write memory usage estimation result in JSON format
//!
//! DESCRIPTION:\n
//! Outputs the memory usage estimation result.
//!
class API_EXPORT CMemoryUsageEstimationResultJsonWriter : private core::CNonCopyable {
public:
    //! \param[in] strmOut The wrapped stream to which to write output.
    CMemoryUsageEstimationResultJsonWriter(core::CJsonOutputStreamWrapper& strmOut);

    //! Writes the given memory usage estimation result in JSON format.
    void write(const std::string& expectedMemoryUsageWithOnePartition,
               const std::string& expectedMemoryUsageWithMaxPartitions);

private:
    //! JSON line writer
    core::CRapidJsonConcurrentLineWriter m_Writer;
};
}
}

#endif // INCLUDED_ml_api_CMemoryUsageEstimationResultJsonWriter_h
