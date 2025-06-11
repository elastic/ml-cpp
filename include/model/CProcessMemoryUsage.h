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

//! \brief Determines how to calculate the memory used by the current process.
//!
//! DESCRIPTION:\n
//! Determines how to calculate the memory used by the current process based on the operating system.
//! On some OS's (Mac, Windows) we use the estimated memory usage of the models,
//! while on others (Linux) we use the actual memory of the process as provided by system calls.
class MODEL_EXPORT CProcessMemoryUsage {
public:
    enum class EMemoryStrategy { E_Estimated, E_System };

public:
    CProcessMemoryUsage() = default;
    ~CProcessMemoryUsage() = default;

    CProcessMemoryUsage(const CProcessMemoryUsage&) = delete;
    CProcessMemoryUsage(CProcessMemoryUsage&&) = delete;
    CProcessMemoryUsage& operator=(const CProcessMemoryUsage&) = delete;
    CProcessMemoryUsage& operator=(CProcessMemoryUsage&&) = delete;

    //! Return the strategy used to calculate the memory used by the current process.
    //! \return Strategy used to calculate the process memory based on the current OS.
    static EMemoryStrategy memoryStrategy();
};
}
}

#endif //INCLUDED_ml_model_CSystemMemoryUsage_h
