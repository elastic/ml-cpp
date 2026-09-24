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
#include <sandbox/CSandbox2Diagnostics.h>

#include <core/CLogger.h>

// Portable half: the capability vocabulary every platform may print, and the
// no-op entry points a build without Sandbox2 links instead of the probe.
// Kept in this one file rather than a sibling CSandbox2Diagnostics.cc because
// ml_generate_platform_sources() substitutes Foo_Linux.cc for Foo.cc, so a
// pair of files under both names cannot both be compiled.
namespace ml {
namespace sandbox {

std::string describe(ESandbox2Capability capability) {
    switch (capability) {
    case ESandbox2Capability::E_Available:
        return "available (user namespace, id maps, mount/pid namespace, "
               "tmpfs and procfs mounts all permitted)";
    case ESandbox2Capability::E_UserNamespaceDenied:
        return "denied at unshare(CLONE_NEWUSER) - the container runtime's "
               "seccomp profile or kernel.unprivileged_userns_clone forbids "
               "user namespaces";
    case ESandbox2Capability::E_IdMapWriteDenied:
        return "denied at uid_map/gid_map write - the user namespace was "
               "created but cannot be given an identity mapping";
    case ESandbox2Capability::E_MountOrPidNamespaceDenied:
        return "denied at unshare(CLONE_NEWNS|CLONE_NEWPID) - user namespaces "
               "are permitted but mount/pid namespaces are not";
    case ESandbox2Capability::E_TmpfsMountDenied:
        return "denied at mount(tmpfs) inside the new namespaces - typically "
               "an LSM (AppArmor/SELinux) mount rule, since the namespaces "
               "themselves were created successfully";
    case ESandbox2Capability::E_ProcMountDenied:
        return "denied at mount(procfs) - typically because the runtime has "
               "bind-mounted over part of /proc, so the kernel refuses a new "
               "procfs mount that would unmask it";
    case ESandbox2Capability::E_ProbeFailed:
        return "unknown - the probe itself could not run, which says nothing "
               "about this host's capabilities";
    case ESandbox2Capability::E_ProbeUnsupported:
        return "not applicable - this build has no Sandbox2 support";
    }
    // No default: above, so a newly added enumerator is a compile-time
    // warning rather than a silently mislabelled log line. This is only
    // reached for a value outside the enumeration entirely.
    return "unrecognized capability value";
}

#if !defined(__linux__) || !defined(SANDBOX2_AVAILABLE)

ESandbox2Capability probeSandbox2Capability() {
    return ESandbox2Capability::E_ProbeUnsupported;
}

void logSandbox2EnvironmentSelfCheck() {
    // Deliberately silent rather than logging "not applicable" on every
    // controller start: a build with no Sandbox2 support never routes to it,
    // so the line would be noise on every non-Linux node.
}

#endif // !__linux__ || !SANDBOX2_AVAILABLE

} // namespace sandbox
} // namespace ml

#if defined(__linux__) && defined(SANDBOX2_AVAILABLE)

#include <fstream>
#include <string>

#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace ml {
namespace sandbox {
namespace {

//! Exit codes the probe children use to report which step failed. Small
//! distinct values, never 0 for a failure, so a child that dies on a signal
//! (status without WIFEXITED) is not mistaken for a pass.
enum EProbeExit : int {
    E_ExitOk = 0,
    E_ExitUserNs = 10,
    E_ExitIdMap = 11,
    E_ExitMountPidNs = 12,
    E_ExitTmpfs = 13,
    E_ExitProc = 14
};

//! Write \p content to \p path, returning false on any error. Used for the
//! uid_map/gid_map/setgroups triple, which must be written with a single
//! write() each - the kernel rejects partial or appended writes.
bool writeOnce(const char* path, const std::string& content) {
    const int fd{::open(path, O_WRONLY | O_CLOEXEC)};
    if (fd < 0) {
        return false;
    }
    const ssize_t written{::write(fd, content.data(), content.size())};
    ::close(fd);
    return written == static_cast<ssize_t>(content.size());
}

//! The innermost probe step, running inside the new user, mount and pid
//! namespaces. Mirrors sandboxed-api's
//! Namespace::InitializeInitialNamespaces(): a tmpfs for the future rootfs,
//! then a fresh procfs. Never returns - always _exit()s with an EProbeExit.
[[noreturn]] void runMountProbe(const std::string& scratchDir) {
    // MS_NOSUID|MS_NODEV mirrors what an unprivileged mount would get
    // anyway; passing them explicitly keeps the probe's request identical in
    // shape to the forkserver's.
    if (::mount("none", scratchDir.c_str(), "tmpfs", MS_NOSUID | MS_NODEV, nullptr) != 0) {
        ::_exit(E_ExitTmpfs);
    }

    // Mount the fresh procfs over the scratch tmpfs rather than over /proc
    // itself: the kernel's "locked mount" rule that rejects a new procfs
    // when the runtime has masked parts of /proc applies to the mount
    // request regardless of target, so this probes the same restriction
    // without disturbing the child's own /proc.
    const std::string procDir{scratchDir + "/proc"};
    if (::mkdir(procDir.c_str(), 0700) != 0) {
        ::_exit(E_ExitProc);
    }
    if (::mount("", procDir.c_str(), "proc", MS_NOSUID | MS_NODEV | MS_NOEXEC, nullptr) != 0) {
        ::_exit(E_ExitProc);
    }

    ::_exit(E_ExitOk);
}

//! Middle probe step: establish the user namespace and its identity maps,
//! then the mount and pid namespaces. CLONE_NEWPID only takes effect for
//! children, so this forks once more before the mount probe - matching the
//! forkserver, whose proc mount likewise happens in a process that is
//! already inside the new pid namespace. Never returns.
[[noreturn]] void runNamespaceProbe(const std::string& scratchDir, uid_t uid, gid_t gid) {
    // The controller marks itself non-dumpable (PR_SET_DUMPABLE=0) to harden
    // against same-uid /proc/<pid>/mem writes, and a forked child inherits
    // that. A non-dumpable process's /proc/self files are owned by root, so
    // open("/proc/self/uid_map") fails with EACCES and the probe would
    // misreport E_IdMapWriteDenied on a host that fully supports Sandbox2.
    // Sandbox2 itself is unaffected because its forkserver is freshly
    // exec()ed, which resets dumpability; restore the same state here. Safe:
    // this is a throwaway child that only probes and _exit()s, and the
    // caller's own dumpability is untouched.
    ::prctl(PR_SET_DUMPABLE, 1, 0, 0, 0);

    // unshare(CLONE_NEWUSER) requires a single-threaded caller; we are in a
    // freshly forked child, so that holds however many threads the
    // controller itself is running.
    if (::unshare(CLONE_NEWUSER) != 0) {
        ::_exit(E_ExitUserNs);
    }

    // setgroups must be denied before gid_map may be written by a process
    // with no CAP_SETGID in the parent namespace. A kernel too old to have
    // the setgroups file is fine - that predates the restriction.
    if (::access("/proc/self/setgroups", F_OK) == 0 &&
        writeOnce("/proc/self/setgroups", "deny") == false) {
        ::_exit(E_ExitIdMap);
    }
    const std::string idMap{"0 " + std::to_string(uid) + " 1\n"};
    if (writeOnce("/proc/self/uid_map", idMap) == false) {
        ::_exit(E_ExitIdMap);
    }
    const std::string gidMap{"0 " + std::to_string(gid) + " 1\n"};
    if (writeOnce("/proc/self/gid_map", gidMap) == false) {
        ::_exit(E_ExitIdMap);
    }

    if (::unshare(CLONE_NEWNS | CLONE_NEWPID) != 0) {
        ::_exit(E_ExitMountPidNs);
    }

    const pid_t inner{::fork()};
    if (inner < 0) {
        ::_exit(E_ExitMountPidNs);
    }
    if (inner == 0) {
        runMountProbe(scratchDir);
    }

    int status{0};
    if (::waitpid(inner, &status, 0) < 0 || WIFEXITED(status) == false) {
        ::_exit(E_ExitMountPidNs);
    }
    ::_exit(WEXITSTATUS(status));
}

//! Map a probe child's exit code back to the capability vocabulary.
ESandbox2Capability capabilityFromExit(int exitCode) {
    switch (exitCode) {
    case E_ExitOk:
        return ESandbox2Capability::E_Available;
    case E_ExitUserNs:
        return ESandbox2Capability::E_UserNamespaceDenied;
    case E_ExitIdMap:
        return ESandbox2Capability::E_IdMapWriteDenied;
    case E_ExitMountPidNs:
        return ESandbox2Capability::E_MountOrPidNamespaceDenied;
    case E_ExitTmpfs:
        return ESandbox2Capability::E_TmpfsMountDenied;
    case E_ExitProc:
        return ESandbox2Capability::E_ProcMountDenied;
    default:
        break;
    }
    return ESandbox2Capability::E_ProbeFailed;
}

std::string readProcSysValue(const char* path) {
    std::ifstream file{path};
    std::string value;
    if (file && std::getline(file, value)) {
        return value;
    }
    return std::string();
}

bool pathHasNoexecFlag(const char* path) {
    struct statfs mountInfo {};
    if (::statfs(path, &mountInfo) != 0) {
        return false;
    }
    return (mountInfo.f_flags & MS_NOEXEC) != 0;
}

} // namespace

ESandbox2Capability probeSandbox2Capability() {
    const char* tmpDirEnv{::getenv("TMPDIR")};
    const std::string base{tmpDirEnv != nullptr ? tmpDirEnv : "/tmp"};

    // A private scratch directory the probe mounts over. Created in the
    // parent so a failure to create it is reported as E_ProbeFailed (a
    // broken probe) rather than misattributed to a denied mount.
    std::string scratchDir{base + "/ml-sandbox2-probe-XXXXXX"};
    if (::mkdtemp(scratchDir.data()) == nullptr) {
        return ESandbox2Capability::E_ProbeFailed;
    }

    const uid_t uid{::getuid()};
    const gid_t gid{::getgid()};

    const pid_t child{::fork()};
    if (child < 0) {
        ::rmdir(scratchDir.c_str());
        return ESandbox2Capability::E_ProbeFailed;
    }
    if (child == 0) {
        runNamespaceProbe(scratchDir, uid, gid);
    }

    int status{0};
    const pid_t reaped{::waitpid(child, &status, 0)};
    // The child's tmpfs (if it got that far) lived in its own mount
    // namespace, which is gone with it, so the directory is empty again
    // here whatever happened inside.
    ::rmdir(scratchDir.c_str());

    if (reaped < 0 || WIFEXITED(status) == false) {
        return ESandbox2Capability::E_ProbeFailed;
    }
    return capabilityFromExit(WEXITSTATUS(status));
}

void logSandbox2EnvironmentSelfCheck() {
    static bool logged{false};
    if (logged) {
        return;
    }
    logged = true;

    const ESandbox2Capability capability{probeSandbox2Capability()};

    // Passive host facts alongside the active result. These are what the
    // frozen prior art (ml-cpp#2873's CSandbox2Diagnostics) reported on its
    // own; they are kept because they help interpret a denial, but they are
    // deliberately no longer the answer: both sysctls below are host-global
    // and are inherited unchanged by a container whose seccomp or LSM policy
    // denies the operation anyway, so on their own they report a healthy
    // environment on exactly the hosts where the sandbox cannot start.
    std::string usernsSysctl{readProcSysValue("/proc/sys/kernel/unprivileged_userns_clone")};
    if (usernsSysctl.empty()) {
        usernsSysctl = "absent";
    }
    std::string maxUserNamespaces{readProcSysValue("/proc/sys/user/max_user_namespaces")};
    if (maxUserNamespaces.empty()) {
        maxUserNamespaces = "absent";
    }

    const char* tmpDirEnv{::getenv("TMPDIR")};
    const std::string tmpDir{tmpDirEnv != nullptr ? tmpDirEnv : "/tmp"};

    const std::string message{
        "Sandbox2 environment self-check: capability=" + describe(capability) +
        ", unprivileged_userns_clone=" + usernsSysctl +
        ", max_user_namespaces=" + maxUserNamespaces + ", TMPDIR=" + tmpDir +
        ", TMPDIR writable=" + (::access(tmpDir.c_str(), W_OK) == 0 ? "yes" : "no") +
        ", TMPDIR noexec=" + (pathHasNoexecFlag(tmpDir.c_str()) ? "yes" : "no")};

    if (capability == ESandbox2Capability::E_Available) {
        LOG_INFO(<< message);
    } else {
        // Not fatal and not a launch failure: the controller only fails a
        // launch if Elasticsearch actually asks for the Sandbox2 route. A
        // node that never sets sandbox_enabled=true runs unaffected, so this
        // is a warning about what *would* happen, not an error that happened.
        LOG_WARN(<< message << " - a --requireSandbox launch on this host will fail closed");
    }
}

} // namespace sandbox
} // namespace ml

#endif // __linux__ && SANDBOX2_AVAILABLE
