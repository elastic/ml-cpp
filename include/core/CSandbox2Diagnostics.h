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
#ifndef INCLUDED_ml_core_CSandbox2Diagnostics_h
#define INCLUDED_ml_core_CSandbox2Diagnostics_h

#include <core/ImportExport.h>

namespace ml {
namespace core {

//! Logs a one-time Sandbox2 environment self-check at INFO level. On platforms
//! without Sandbox2 support this is a no-op.
CORE_EXPORT void logSandbox2EnvironmentSelfCheck();

} // namespace core
} // namespace ml

#endif // INCLUDED_ml_core_CSandbox2Diagnostics_h
