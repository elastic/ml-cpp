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
#ifndef INCLUDED_ml_sandbox_CChildIpcDirectoryReaper_h
#define INCLUDED_ml_sandbox_CChildIpcDirectoryReaper_h

#include <core/CProcess.h>
#include <core/ImportExport.h>

#include <map>
#include <mutex>
#include <string>

namespace ml {
namespace sandbox {

//! Removes a single per-deployment directory under $TMPDIR/ml-child-ipc/<child-id>
//! once the controller no longer has a live child bound to it.
//!
//! Lifecycle (paired with ensureChildIpcDirectory()):
//! - ensureChildIpcDirectory() creates $TMPDIR/ml-child-ipc/<child-id> (0700) before
//!   realpath()/policy construction; the intermediate ml-child-ipc directory is kept.
//! - noteSpawn() records which pid owns that canonical root for a successful launch.
//! - onChildExited() removes the directory only when the root is still bound to that pid
//!   (a fast restart reusing the same deployment id updates the binding first).
//! - onSpawnFailed() removes a root that was created but never got a live pid.
//!
//! Controller crash can still leave empty directories; there is no startup sweep because
//! that would race another controller on the same $TMPDIR.
class CORE_EXPORT CChildIpcDirectoryReaper {
public:
    void noteSpawn(core::CProcess::TPid pid, const std::string& canonicalChildRoot);

    void onChildExited(core::CProcess::TPid pid);

    void onSpawnFailed(const std::string& canonicalChildRoot);

private:
    static bool canonicalPerChildIpcRootHasExpectedShape(const std::string& canonicalChildRoot);

    static bool tryRemovePerChildIpcDirectory(const std::string& canonicalChildRoot);

    std::mutex m_Mutex;
    std::map<core::CProcess::TPid, std::string> m_PidToRoot;
    std::map<std::string, core::CProcess::TPid> m_RootToPid;
};

} // namespace sandbox
} // namespace ml

#endif // INCLUDED_ml_sandbox_CChildIpcDirectoryReaper_h
