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
#include <seccomp/CLandlockFilesystemPolicy.h>

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

ESandbox2Capability sandbox2Capability() {
    // Function-local static: initialised exactly once, thread-safely.
    static const ESandbox2Capability capability{probeSandbox2Capability()};
    return capability;
}

EConfinementLevel decideConfinement(ESandbox2Capability sandbox2, int landlockAbi) {
    if (sandbox2 == ESandbox2Capability::E_Available) {
        return EConfinementLevel::E_Sandbox2;
    }
    // A build without Sandbox2 never reaches the sandboxed route (the router
    // refuses it outright), so it is never offered the Landlock rung either.
    if (sandbox2 == ESandbox2Capability::E_ProbeUnsupported) {
        return EConfinementLevel::E_Unavailable;
    }
    // Any Sandbox2 denial - including E_ProbeFailed, where attempting
    // Sandbox2 anyway could deadlock in the forkserver's namespace setup -
    // steps down to Landlock if the kernel supports it.
    return landlockAbi >= 1 ? EConfinementLevel::E_Landlock : EConfinementLevel::E_Unavailable;
}

std::string describeLandlock(int landlockAbi) {
    if (landlockAbi >= 1) {
        return "available (ABI " + std::to_string(landlockAbi) + ")";
    }
    if (landlockAbi == 0) {
        return "not supported by this kernel";
    }
    return "blocked by a seccomp filter or LSM policy";
}

std::string fullSandboxRemedy(const SHostConfinement& host) {
    switch (host.s_Sandbox2) {
    case ESandbox2Capability::E_Available:
        return std::string{};
    case ESandbox2Capability::E_UserNamespaceDenied:
        if (host.s_UnprivilegedUsernsClone == "0") {
            return "For full Sandbox2 isolation, a system administrator must allow unprivileged "
                   "user namespaces by setting the kernel parameter "
                   "kernel.unprivileged_userns_clone=1 (for example with "
                   "'sysctl -w kernel.unprivileged_userns_clone=1', persisted in /etc/sysctl.d/).";
        }
        if (host.s_MaxUserNamespaces == "0") {
            return "For full Sandbox2 isolation, a system administrator must allow user "
                   "namespaces by setting the kernel parameter user.max_user_namespaces to a "
                   "non-zero value.";
        }
        return "For full Sandbox2 isolation, a system administrator must allow unprivileged "
               "user namespaces for this process. The kernel permits them "
               "(kernel.unprivileged_userns_clone is not 0), so they are being blocked by the "
               "container runtime - typically a seccomp profile that denies clone/unshare with "
               "CLONE_NEWUSER.";
    case ESandbox2Capability::E_IdMapWriteDenied:
    case ESandbox2Capability::E_MountOrPidNamespaceDenied:
        return "For full Sandbox2 isolation, a system administrator must allow this process "
               "to set up user, mount and PID namespaces; user namespaces can be created, but "
               "the container runtime blocks the later steps.";
    case ESandbox2Capability::E_TmpfsMountDenied:
        return "For full Sandbox2 isolation, a system administrator must allow mounts inside "
               "unprivileged user namespaces for this process; they are currently denied, "
               "typically by an AppArmor or SELinux policy.";
    case ESandbox2Capability::E_ProcMountDenied:
        return "For full Sandbox2 isolation, a system administrator must allow this process "
               "to mount a private /proc; the container runtime currently masks parts of /proc, "
               "which makes the kernel refuse it.";
    case ESandbox2Capability::E_ProbeFailed:
        return "The Sandbox2 capability probe itself could not run, so the reason is unknown; "
               "see the earlier ML controller log messages.";
    case ESandbox2Capability::E_ProbeUnsupported:
        return "This build does not include Sandbox2.";
    }
    return std::string{};
}

std::string landlockFallbackMessage(const SHostConfinement& host,
                                    const std::string& processPath) {
    return "Full Sandbox2 isolation is not available on this host (" +
           describe(host.s_Sandbox2) + "), so '" + processPath +
           "' is being launched with Landlock filesystem confinement and the seccomp system "
           "call filter instead. Landlock restricts which files the process can open but, "
           "unlike Sandbox2, does not isolate its view of processes, mounts or the network. " +
           fullSandboxRemedy(host);
}

std::string noConfinementMessage(const SHostConfinement& host, const std::string& processPath) {
    const std::string why{host.s_LandlockAbi == 0
                              ? "the operating system is too old or its kernel lacks the required "
                                "features (Landlock needs Linux 5.13 or later)"
                              : "Landlock is " + describeLandlock(host.s_LandlockAbi)};
    return "Refusing to launch '" + processPath +
           "': xpack.ml.trained_models.sandbox_enabled is true, but this host supports neither "
           "Sandbox2 isolation (" +
           describe(host.s_Sandbox2) +
           ") nor Landlock filesystem "
           "confinement - " +
           why +
           ". To run models on this node, deactivate the "
           "xpack.ml.trained_models.sandbox_enabled setting (set it to false); models then run "
           "with the seccomp system call filter only.";
}

#if !defined(__linux__) || !defined(SANDBOX2_AVAILABLE)

ESandbox2Capability probeSandbox2Capability() {
    return ESandbox2Capability::E_ProbeUnsupported;
}

const SHostConfinement& hostConfinement() {
    static const SHostConfinement host{};
    return host;
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

const SHostConfinement& hostConfinement() {
    static const SHostConfinement host{[] {
        SHostConfinement h;
        h.s_Sandbox2 = sandbox2Capability();
        h.s_LandlockAbi = seccomp::landlockAbiVersion();
        const std::string userns{readProcSysValue("/proc/sys/kernel/unprivileged_userns_clone")};
        const std::string maxUserns{readProcSysValue("/proc/sys/user/max_user_namespaces")};
        h.s_UnprivilegedUsernsClone = userns.empty() ? "absent" : userns;
        h.s_MaxUserNamespaces = maxUserns.empty() ? "absent" : maxUserns;
        h.s_Level = decideConfinement(h.s_Sandbox2, h.s_LandlockAbi);
        return h;
    }()};
    return host;
}

void logSandbox2EnvironmentSelfCheck() {
    static bool logged{false};
    if (logged) {
        return;
    }
    logged = true;

    const SHostConfinement& host{hostConfinement()};

    // The passive sysctl values are what the frozen prior art (ml-cpp#2873's
    // CSandbox2Diagnostics) reported on its own. They never decide anything
    // - both are host-global and are inherited unchanged by a container whose
    // seccomp or LSM policy denies user namespaces regardless - but they are
    // what tells an administrator which knob to turn.
    const char* tmpDirEnv{::getenv("TMPDIR")};
    const std::string tmpDir{tmpDirEnv != nullptr ? tmpDirEnv : "/tmp"};
    const std::string facts{
        "Sandbox2 environment self-check: sandbox2=" + describe(host.s_Sandbox2) +
        ", landlock=" + describeLandlock(host.s_LandlockAbi) +
        ", unprivileged_userns_clone=" + host.s_UnprivilegedUsernsClone +
        ", max_user_namespaces=" + host.s_MaxUserNamespaces + ", TMPDIR=" + tmpDir +
        ", TMPDIR writable=" + (::access(tmpDir.c_str(), W_OK) == 0 ? "yes" : "no") +
        ", TMPDIR noexec=" + (pathHasNoexecFlag(tmpDir.c_str()) ? "yes" : "no")};

    // Logged at controller start, before any launch, and regardless of
    // xpack.ml.trained_models.sandbox_enabled (which the controller only
    // learns per launch) - so each message says what *would* happen if the
    // setting is true.
    switch (host.s_Level) {
    case EConfinementLevel::E_Sandbox2:
        LOG_INFO(<< facts
                 << ". Models launched with xpack.ml.trained_models.sandbox_enabled=true "
                    "will run with full Sandbox2 isolation.");
        break;
    case EConfinementLevel::E_Landlock:
        // A supported, deliberate degradation - INFO, not WARN.
        LOG_INFO(<< facts
                 << ". Models launched with xpack.ml.trained_models.sandbox_enabled=true "
                    "will run with Landlock filesystem confinement, because full Sandbox2 "
                    "isolation is not available on this host. "
                 << fullSandboxRemedy(host));
        break;
    case EConfinementLevel::E_Unavailable:
        LOG_WARN(<< facts
                 << ". This host supports neither Sandbox2 nor Landlock, so every model "
                    "deployment on this node will fail to start while "
                    "xpack.ml.trained_models.sandbox_enabled is true; deactivate that "
                    "setting (set it to false) to run models here.");
        break;
    }
}

} // namespace sandbox
} // namespace ml

#endif // __linux__ && SANDBOX2_AVAILABLE
