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
#ifndef INCLUDED_ml_core_CStoredStringPtr_h
#define INCLUDED_ml_core_CStoredStringPtr_h

#include <core/CMemoryUsage.h>
#include <core/ImportExport.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

namespace ml {
namespace core {

// TODO(jan): rename
using CStoredStringPtr = std::optional<std::string>;

} // core
} // ml

#endif // INCLUDED_ml_core_CStoredStringPtr_h
