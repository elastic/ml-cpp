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
#ifndef INCLUDED_ml_sandbox_CSandboxedProcessSpawner_h
#define INCLUDED_ml_sandbox_CSandboxedProcessSpawner_h

#include <core/CProcess.h>
#include <sandbox/ImportExport.h>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace sandbox2 {
class Sandbox2;
}

namespace ml {
namespace sandbox {

//! \brief
//! Spawn processes inside a Sandbox2 isolation boundary.
//!
//! DESCRIPTION:\n
//! Used by the ML controller to launch pytorch_inference with filesystem and
//! syscall restrictions. PID lifetime is tracked via Sandbox2 AwaitResult
//! monitor threads rather than waitpid(), because the sandboxee is a child of
//! the Sandbox2 forkserver rather than of the controller.
//!
class SANDBOX_EXPORT CSandboxedProcessSpawner {
public:
    using TStrVec = std::vector<std::string>;

public:
    CSandboxedProcessSpawner();
    ~CSandboxedProcessSpawner();

    //! Spawn a sandboxed process. Returns true on success.
    bool spawn(const std::string& processPath,
               const TStrVec& args,
               core::CProcess::TPid& childPid,
               std::string* failureReason = nullptr);

    //! Kill a sandboxed child process started by this object.
    bool terminateChild(core::CProcess::TPid pid);

    //! Returns true if this object spawned a sandboxed process with the given
    //! PID that is still running.
    bool hasChild(core::CProcess::TPid pid) const;

private:
    //! \brief A live sandboxed child and the handles needed to manage it safely.
    struct SSandboxedChild {
        std::uint64_t s_Generation{0};
        std::shared_ptr<sandbox2::Sandbox2> s_Sandbox;
        int s_PidFd{-1};
    };

    //! \brief The live sandboxed children, and the lock that guards them.
    //!
    //! DESCRIPTION:\n
    //! Held behind a shared_ptr because the monitor thread that removes a child
    //! outlives the spawn() call that started it, and can outlive this object:
    //! the controller may tear the spawner down while a sandboxed
    //! pytorch_inference is still running, and the monitor only learns that the
    //! sandboxee exited some time later. A raw pointer back to the spawner
    //! would be dangling by then, so the monitor co-owns the registry and a
    //! shared_ptr to the Sandbox2 instance instead, and the spawner needs no
    //! synchronisation in its destructor. Each entry carries a monotonic
    //! generation so a stale monitor cannot erase a re-registered PID, and a
    //! pidfd (when the kernel provides one) so terminateChild() can signal the
    //! exact process even after the PID has been reused.
    struct SPidRegistry {
        mutable std::mutex s_Mutex;
        std::uint64_t s_NextGeneration{0};
        std::map<core::CProcess::TPid, SSandboxedChild> s_Children;
    };
    using TPidRegistryPtr = std::shared_ptr<SPidRegistry>;

    const TPidRegistryPtr m_PidRegistry{std::make_shared<SPidRegistry>()};
};

} // namespace sandbox
} // namespace ml

#endif // INCLUDED_ml_sandbox_CSandboxedProcessSpawner_h
