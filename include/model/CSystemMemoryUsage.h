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

#ifndef INCLUDED_ml_model_CSystemMemoryUsage_h
#define INCLUDED_ml_model_CSystemMemoryUsage_h

#include <model/ImportExport.h>

#include <cstddef>

namespace ml {
namespace model {

//! \brief Determines the memory used by the current process.
//!
//! DESCRIPTION:\n
//! Determines the memory used by the current process based on the operating system.
class MODEL_EXPORT CSystemMemoryUsage {
public:
    CSystemMemoryUsage() = default;
    ~CSystemMemoryUsage() = default;

    CSystemMemoryUsage(const CSystemMemoryUsage&) = delete;
    CSystemMemoryUsage(CSystemMemoryUsage&&) = delete;
    CSystemMemoryUsage& operator=(const CSystemMemoryUsage&) = delete;
    CSystemMemoryUsage& operator=(CSystemMemoryUsage&&) = delete;

    //! Return the system memory used by the current process.
    //! \param memSize An estimate of the process memory
    //! \return The system memory usage based on the current OS.
    std::size_t operator()(std::size_t memSize);
};
}
}

#endif //INCLUDED_ml_model_CSystemMemoryUsage_h
