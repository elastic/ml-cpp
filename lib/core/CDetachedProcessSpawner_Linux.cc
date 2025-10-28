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
#include <core/CScopedLock.h>
#include <core/CThread.h>
#include <core/COsFileFuncs.h>

#include <algorithm>
#include <set>
#include <vector>
#include <string>
#include <memory>
#include <functional>

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <pwd.h>
#include <grp.h>

// Sandbox2 integration - use conditional compilation to avoid linking issues
#ifdef SANDBOX2_AVAILABLE
#include <sandboxed_api/sandbox2/policy.h>
#include <sandboxed_api/sandbox2/policybuilder.h>
#include <sandboxed_api/sandbox2/result.h>
#include <sandboxed_api/sandbox2/sandbox2.h>
#include <sandboxed_api/sandbox2/util/bpf_helper.h>
#include <absl/status/status.h>
#include <absl/status/statusor.h>

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
#endif // SANDBOX2_AVAILABLE

namespace ml {
namespace core {
namespace detail {

//! Structure to hold process paths for Sandbox2 policy
struct ProcessPaths {
    std::string pytorchLibDir;
    std::string modelPath;      // renamed from modelDir
    std::string inputPipe;
    std::string outputPipe;
    std::string logPipe;        // new: --logPipe
    std::string logProperties;  // new: --logProperties (config file)
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

#ifndef SANDBOX2_DISABLED
#ifdef SANDBOX2_AVAILABLE
//! Build Sandbox2 policy for pytorch_inference
std::unique_ptr<sandbox2::Policy> buildSandboxPolicy(const ProcessPaths& paths, uid_t uid, gid_t gid) {
    auto builder = sandbox2::PolicyBuilder()
        // Drop privileges to nobody:nogroup
        .SetUserAndGroup(uid, gid)
        
        // Filesystem isolation - deny by default
        .BlockSyscall(__NR_mount)
        .BlockSyscall(__NR_umount)
        .BlockSyscall(__NR_umount2)
        
        // Allow essential system libraries (read-only)
        .AddDirectoryAt("/lib", "/lib", true)
        .AddDirectoryAt("/usr/lib", "/usr/lib", true)
        .AddDirectoryAt("/lib64", "/lib64", true)
        .AddDirectoryAt("/usr/lib64", "/usr/lib64", true)
        
        // Allow minimal /tmp (private tmpfs)
        .AddTmpfs("/tmp");
    
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
    sandbox2::Sandbox2 sandbox(std::move(policy), std::make_unique<sandbox2::Executor>(processPath, args));
    
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
    LOG_DEBUG(<< "Sandbox2 not available, falling back to standard spawn for '" << processPath << "'");
    return false; // Indicates to use base implementation
}
#endif // SANDBOX2_AVAILABLE
#endif // SANDBOX2_DISABLED

} // namespace detail
} // namespace core
} // namespace ml

//! Linux-specific implementation of CDetachedProcessSpawner::spawn
bool ml::core::CDetachedProcessSpawner::spawn(const std::string& processPath,
                                             const std::vector<std::string>& args,
                                             ml::core::CProcess::TPid& childPid) {
#ifdef __linux__
#ifndef SANDBOX2_DISABLED
    // Use Sandbox2 for pytorch_inference on Linux
    if (detail::isPytorchInference(processPath)) {
        if (detail::spawnWithSandbox2(processPath, args, childPid)) {
            // Note: PID tracking will be handled by the base implementation
            // when this function returns true, the base implementation
            // will call m_TrackerThread->addPid(childPid)
            return true;
        } else {
            LOG_ERROR(<< "Sandbox2 spawn failed for '" << processPath << "', falling back to posix_spawn");
            // Fall through to posix_spawn
        }
    }
#endif // SANDBOX2_DISABLED
#endif // __linux__

    // Fall back to standard posix_spawn implementation
    // This will call the base implementation from CDetachedProcessSpawner.cc
    return false; // Indicates to use base implementation
}
