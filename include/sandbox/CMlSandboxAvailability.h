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
#ifndef INCLUDED_ml_sandbox_CMlSandboxAvailability_h
#define INCLUDED_ml_sandbox_CMlSandboxAvailability_h

#include <core/CNonInstantiatable.h>

namespace ml {
namespace sandbox {

//! \brief
//! Reports whether this binary was built with Sandbox2 support.
//!
//! DESCRIPTION:\n
//! MlSandbox is a dormant dependency foundation: it links Sandbox2/Abseil
//! and builds a runnable forkserver on Linux, but nothing in the controller
//! or pytorch_inference wiring routes to it yet (see
//! docs/projects/mlcpp-sandbox2-pr2873 in the elastic-workspace harness).
//! This query is the only symbol callers outside this library may currently
//! depend on; the actual sandbox policy, spawner, and controller routing
//! land in later PRs of that plan.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Backed by the SANDBOX2_AVAILABLE compile definition set in
//! lib/sandbox/CMakeLists.txt, which is only defined when the Sandbox2
//! FetchContent target built successfully (Linux only).
class CMlSandboxAvailability : private core::CNonInstantiatable {
public:
    //! \return true if this binary was compiled with Sandbox2 linked in
    //! (Linux builds only); false on macOS/Windows or if the dependency
    //! foundation build step did not run.
    static bool isCompiledIn();
};
}
}

#endif // INCLUDED_ml_sandbox_CMlSandboxAvailability_h
