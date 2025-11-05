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
#include <core/CDetachedProcessSpawner.h>

#include <core/CCondition.h>
#include <core/CLogger.h>
#include <core/CMutex.h>
#include <core/COsFileFuncs.h>
#include <core/CScopedLock.h>
#include <core/CThread.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <errno.h>
#include <fcntl.h>
#include <grp.h>
#include <pwd.h>
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

// Sandbox2 integration - use conditional compilation to avoid linking issues
#ifdef SANDBOX2_AVAILABLE
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <sandboxed_api/sandbox2/policy.h>
#include <sandboxed_api/sandbox2/policybuilder.h>
#include <sandboxed_api/sandbox2/result.h>
#include <sandboxed_api/sandbox2/sandbox2.h>
#include <sandboxed_api/sandbox2/util/bpf_helper.h>

// Define syscall numbers for x86_64
#ifndef __NR_mount
#define __NR_mount 165
#endif
#ifndef __NR_umount
#define __NR_umount 166
#endif
#ifndef __NR_umount2
#define __NR_umount2 166
#endif

// Additional syscall numbers for ML process filtering
#ifndef __NR_connect
#define __NR_connect 42 // x86_64
#endif
#ifdef __x86_64__
#ifndef __NR_mkdir
#define __NR_mkdir 83
#endif
#ifndef __NR_rmdir
#define __NR_rmdir 84
#endif
#ifndef __NR_unlink
#define __NR_unlink 87
#endif
#ifndef __NR_mknod
#define __NR_mknod 133
#endif
#ifndef __NR_getdents
#define __NR_getdents 78
#endif
#endif // __x86_64__
#ifndef __NR_mkdirat
#define __NR_mkdirat 258
#endif
#ifndef __NR_unlinkat
#define __NR_unlinkat 263
#endif
#ifndef __NR_mknodat
#define __NR_mknodat 259
#endif
#ifndef __NR_getdents64
#define __NR_getdents64 217
#endif
#endif // SANDBOX2_AVAILABLE

namespace ml {
namespace core {
namespace detail {

//! Structure to hold process paths for Sandbox2 policy
struct ProcessPaths {
    std::string pytorchLibDir;
    std::string modelPath; // renamed from modelDir
    std::string inputPipe;
    std::string outputPipe;
    std::string logPipe;       // new: --logPipe
    std::string logProperties; // new: --logProperties (config file)
};

//! Check if the process path is pytorch_inference
bool isPytorchInference(const std::string& processPath) {
    return processPath.find("pytorch_inference") != std::string::npos;
}

//! Parse command line arguments to extract file paths for sandbox policy
ProcessPaths parseProcessPaths(const std::vector<std::string>& args) {
    ProcessPaths paths;

    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];

        // Handle --arg=value format
        if (arg.find("--input=") == 0) {
            paths.inputPipe = arg.substr(8); // Skip "--input="
        } else if (arg.find("--output=") == 0) {
            paths.outputPipe = arg.substr(9); // Skip "--output="
        } else if (arg.find("--restore=") == 0) {
            paths.modelPath = arg.substr(10); // Skip "--restore="
        } else if (arg.find("--logPipe=") == 0) {
            paths.logPipe = arg.substr(10); // Skip "--logPipe="
        } else if (arg.find("--logProperties=") == 0) {
            paths.logProperties = arg.substr(16); // Skip "--logProperties="
        }
        // Handle --arg value format
        else if (arg == "--input" && i + 1 < args.size()) {
            paths.inputPipe = args[++i];
        } else if (arg == "--output" && i + 1 < args.size()) {
            paths.outputPipe = args[++i];
        } else if (arg == "--restore" && i + 1 < args.size()) {
            paths.modelPath = args[++i];
        } else if (arg == "--logPipe" && i + 1 < args.size()) {
            paths.logPipe = args[++i];
        } else if (arg == "--logProperties" && i + 1 < args.size()) {
            paths.logProperties = args[++i];
        }
    }

    return paths;
}

//! Calculate PyTorch library directory from executable path
std::string calculatePytorchLibDir(const std::string& processPath) {
    // Find the last directory separator
    size_t lastSlash = processPath.find_last_of('/');
    if (lastSlash == std::string::npos) {
        return ""; // Invalid path
    }

    // Get the directory containing the executable
    std::string exeDir = processPath.substr(0, lastSlash);

    // The lib directory is at ../lib relative to the executable
    // (since executables are typically in bin/ and libraries in lib/)
    size_t lastDirSlash = exeDir.find_last_of('/');
    if (lastDirSlash == std::string::npos) {
        return ""; // Invalid path
    }

    std::string parentDir = exeDir.substr(0, lastDirSlash);
    return parentDir + "/lib";
}

#ifndef SANDBOX2_DISABLED
#ifdef SANDBOX2_AVAILABLE
//! Apply standard ML process syscall restrictions to a Sandbox2 PolicyBuilder.
//!
//! This function implements the same security policy as CSystemCallFilter_Linux.cc
//! but using Sandbox2's PolicyBuilder API. The goal is to provide a consistent
//! syscall filtering policy that can be applied to all ML processes.
//!
//! DESIGN RATIONALE:
//! - Mirrors the whitelist approach from the seccomp filter for consistency
//! - Blocks dangerous syscalls: mount/umount, network (connect), file creation
//! - Allows essential syscalls: file I/O, memory management, threading, signals
//! - Parameterized to support different ML process needs (forecast temp storage)
//!
//! FUTURE MIGRATION PATH:
//! When other ML processes (autodetect, categorize, data_frame_analyzer, normalize)
//! are migrated to Sandbox2, they can use this same function with appropriate
//! parameters. For example:
//!   - autodetect: allowForecastTempStorage=true (needs mkdir/rmdir)
//!   - categorize: allowForecastTempStorage=false
//!   - data_frame_analyzer: allowForecastTempStorage=true
//!   - normalize: allowForecastTempStorage=false
//!
//! @param builder The PolicyBuilder to configure (modified in place)
//! @param allowForecastTempStorage If true, allow mkdir/rmdir/unlink for forecast temp storage
//! @param allowNetworkConnect If true, allow connect syscall (currently unused)
//! @return Reference to the builder for method chaining
sandbox2::PolicyBuilder& applyMlSyscallPolicy(sandbox2::PolicyBuilder& builder,
                                              bool allowForecastTempStorage = false,
                                              bool allowNetworkConnect = false) {
    // Block dangerous syscalls that no ML process should use
    builder.BlockSyscall(__NR_mount).BlockSyscall(__NR_umount).BlockSyscall(__NR_umount2);

    // Network access - currently no ML process needs this
    if (!allowNetworkConnect) {
        builder.BlockSyscall(__NR_connect);
    }

    // File/directory creation - only needed for forecast temp storage
    if (!allowForecastTempStorage) {
#ifdef __x86_64__
        builder.BlockSyscall(__NR_mkdir)
            .BlockSyscall(__NR_rmdir)
            .BlockSyscall(__NR_unlink)
            .BlockSyscall(__NR_mknod)
            .BlockSyscall(__NR_getdents);
#endif
        builder.BlockSyscall(__NR_mkdirat)
            .BlockSyscall(__NR_unlinkat)
            .BlockSyscall(__NR_mknodat)
            .BlockSyscall(__NR_getdents64);
    }

    // All other syscalls from the seccomp whitelist are implicitly allowed
    // by Sandbox2's default policy (read, write, mmap, futex, etc.)

    return builder;
}

//! Look up UID/GID for nobody user and nogroup
bool lookupNobodyUser(uid_t& uid, gid_t& gid) {
    struct passwd* pwd = getpwnam("nobody");
    if (!pwd) {
        LOG_ERROR(<< "Failed to lookup nobody user");
        return false;
    }
    uid = pwd->pw_uid;

    struct group* grp = getgrnam("nogroup");
    if (!grp) {
        LOG_ERROR(<< "Failed to lookup nogroup");
        return false;
    }
    gid = grp->gr_gid;

    LOG_DEBUG(<< "Found nobody user: UID=" << uid << ", GID=" << gid);
    return true;
}
//! Build Sandbox2 policy for pytorch_inference
std::unique_ptr<sandbox2::Policy>
buildSandboxPolicy(const ProcessPaths& paths, uid_t uid, gid_t gid) {
    auto builder = sandbox2::PolicyBuilder()
                       // Drop privileges to nobody:nogroup
                       .SetUserAndGroup(uid, gid)

                       // Allow essential system libraries (read-only)
                       .AddDirectoryAt("/lib", "/lib", true)
                       .AddDirectoryAt("/usr/lib", "/usr/lib", true)
                       .AddDirectoryAt("/lib64", "/lib64", true)
                       .AddDirectoryAt("/usr/lib64", "/usr/lib64", true)

                       // Allow minimal /tmp (private tmpfs)
                       .AddTmpfs("/tmp");

    // Apply standard ML syscall restrictions
    // pytorch_inference doesn't need forecast temp storage or network
    applyMlSyscallPolicy(builder,
                         /*allowForecastTempStorage=*/false,
                         /*allowNetworkConnect=*/false);

    // Allow PyTorch libraries (read-only)
    if (!paths.pytorchLibDir.empty()) {
        builder.AddDirectoryAt(paths.pytorchLibDir, paths.pytorchLibDir, true);
    }

    // Allow model file (read-only)
    if (!paths.modelPath.empty()) {
        builder.AddFileAt(paths.modelPath, paths.modelPath, true, false);
    }

    // Allow named pipes (read-write)
    if (!paths.inputPipe.empty()) {
        builder.AddFileAt(paths.inputPipe, paths.inputPipe, true, true);
    }
    if (!paths.outputPipe.empty()) {
        builder.AddFileAt(paths.outputPipe, paths.outputPipe, true, true);
    }
    if (!paths.logPipe.empty()) {
        builder.AddFileAt(paths.logPipe, paths.logPipe, true, true);
    }

    // Allow log properties file (read-only)
    if (!paths.logProperties.empty()) {
        builder.AddFileAt(paths.logProperties, paths.logProperties, true, false);
    }

    // Build the policy
    return builder.BuildOrDie();
}
#endif // SANDBOX2_AVAILABLE
#endif // SANDBOX2_DISABLED

#ifndef SANDBOX2_DISABLED
#ifdef SANDBOX2_AVAILABLE
//! Spawn process with Sandbox2
bool spawnWithSandbox2(const std::string& processPath,
                       const std::vector<std::string>& args,
                       ml::core::CProcess::TPid& childPid) {
    // Look up nobody user
    uid_t uid;
    gid_t gid;
    if (!lookupNobodyUser(uid, gid)) {
        return false;
    }

    // Parse process paths from command line arguments
    ProcessPaths paths = parseProcessPaths(args);

    // Calculate PyTorch library directory from executable path
    paths.pytorchLibDir = calculatePytorchLibDir(processPath);

    // Build Sandbox2 policy
    auto policy = buildSandboxPolicy(paths, uid, gid);

    // Create executor
    sandbox2::Sandbox2 sandbox(
        std::move(policy), std::make_unique<sandbox2::Executor>(processPath, args));

    // Launch sandboxed process
    auto result = sandbox.Run();
    if (!result.ok()) {
        LOG_ERROR(<< "Sandbox2 execution failed: " << result.status().message());
        return false;
    }

    // Get the PID from the result
    childPid = result->pid();

    LOG_DEBUG(<< "Spawned sandboxed '" << processPath << "' with PID " << childPid);
    return true;
}
#else
//! Fallback implementation when Sandbox2 is not available
bool spawnWithSandbox2(const std::string& processPath,
                       const std::vector<std::string>& args,
                       ml::core::CProcess::TPid& childPid) {
    LOG_DEBUG(<< "Sandbox2 not available, falling back to standard spawn for '"
              << processPath << "'");
    return false; // Indicates to use base implementation
}
#endif // SANDBOX2_AVAILABLE
#endif // SANDBOX2_DISABLED

//! FUTURE MIGRATION PLAN:
//!
//! Currently only pytorch_inference is spawned via Sandbox2. The long-term plan
//! is to migrate all ML processes to use Sandbox2 for consistent security:
//!
//! 1. pytorch_inference (CURRENT) - Spawned via CDetachedProcessSpawner
//!    - Uses: applyMlSyscallPolicy(builder, false, false)
//!    - No temp storage, no network
//!
//! 2. autodetect (FUTURE) - Will be spawned via CDetachedProcessSpawner
//!    - Uses: applyMlSyscallPolicy(builder, true, false)
//!    - Needs temp storage for forecasting
//!
//! 3. categorize (FUTURE) - Will be spawned via CDetachedProcessSpawner
//!    - Uses: applyMlSyscallPolicy(builder, false, false)
//!    - No temp storage, no network
//!
//! 4. data_frame_analyzer (FUTURE) - Will be spawned via CDetachedProcessSpawner
//!    - Uses: applyMlSyscallPolicy(builder, true, false)
//!    - Needs temp storage for forecasting
//!
//! 5. normalize (FUTURE) - Will be spawned via CDetachedProcessSpawner
//!    - Uses: applyMlSyscallPolicy(builder, false, false)
//!    - No temp storage, no network
//!
//! When migrating a process:
//! 1. Update CDetachedProcessSpawner::spawn() to detect the process type
//! 2. Create a process-specific buildSandboxPolicy() function
//! 3. Call applyMlSyscallPolicy() with appropriate parameters
//! 4. Conditionally disable seccomp in the process's Main.cc (like pytorch_inference)
//! 5. Update the process spawning code to use CDetachedProcessSpawner

} // namespace detail
} // namespace core
} // namespace ml

//! Linux-specific implementation of CDetachedProcessSpawner::spawn
bool ml::core::CDetachedProcessSpawner::spawn(const std::string& processPath,
                                              const std::vector<std::string>& args,
                                              ml::core::CProcess::TPid& childPid) {
#ifdef __linux__
    if (detail::isPytorchInference(processPath)) {
#ifdef SANDBOX2_DISABLED
        HANDLE_FATAL(<< "Sandbox2 is disabled but required for pytorch_inference process: "
                     << processPath);
        return false;
#elif !defined(SANDBOX2_AVAILABLE)
        HANDLE_FATAL(<< "Sandbox2 is not available but required for pytorch_inference process: "
                     << processPath);
        return false;
#else
        // Sandbox2 is available and enabled
        if (!detail::spawnWithSandbox2(processPath, args, childPid)) {
            HANDLE_FATAL(<< "Failed to spawn pytorch_inference with Sandbox2: " << processPath);
            return false;
        }
        return true;
#endif
    }
#endif // __linux__

    // For non-pytorch_inference processes, use standard posix_spawn
    // This will call the base implementation from CDetachedProcessSpawner.cc
    return false;
}
