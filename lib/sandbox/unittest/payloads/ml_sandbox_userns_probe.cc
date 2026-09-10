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

// Staged user-namespace capability probe for the ML_SANDBOX2_REQUIRE CI
// wiring. Unlike ml_sandbox_probe.cc (the typed filesystem/network launch
// policy's own *policy* mechanism probe, which runs inside an already-built
// Sandbox2 sandbox) this payload exercises the raw kernel primitives
// Sandbox2's own forkserver depends on - unshare(CLONE_NEWUSER), uid/gid
// mapping, unshare(CLONE_NEWNS | CLONE_NEWPID), and a proc mount inside the
// new namespaces - run directly by the host-process controller test, with no
// Sandbox2 policy involved at all. Its job is to pin down whether the
// *ambient CI environment* (e.g. a Buildkite k8s pod) permits userns
// operations, independent of any Sandbox2 policy's correctness. Deliberately
// dependency-free, like ml_sandbox_probe.cc and sandbox_smoke_payload.cc: no
// ml-cpp library dependencies, no sandbox policy of its own.
//
// Runs the following 7 stages in order and reports the first failed
// stage and errno on any failure; success only if all 7 complete:
//   1. probe pipe + fork
//   2. unshare(CLONE_NEWUSER)
//   3. uid/gid map writes, including setgroups
//   4. unshare(CLONE_NEWNS | CLONE_NEWPID)
//   5. fork into the new PID namespace
//   6. mount("/", MS_REC | MS_PRIVATE)
//   7. mount("proc", "/proc", "proc", ...)
//
// Stage 7 MUST run after the stage-5 fork, matching the existing fix
// (commit 50bacc2b) that mounts proc only after the fork into the new PID
// namespace. A proc mount issued by the stage-4 unshare()'d process itself,
// before forking into the namespace, would mount /proc for the wrong PID
// namespace view. Do not reorder stages 5 and 7.

// unshare() and the CLONE_NEWUSER/CLONE_NEWNS/CLONE_NEWPID constants are GNU
// extensions gated behind _GNU_SOURCE in glibc's <sched.h>; define it
// explicitly (must precede any system header include) rather than relying on
// libstdc++ defining it implicitly for this translation unit. Guarded
// because g++ already predefines it on glibc targets - an unconditional
// #define here would trigger a macro-redefinition warning.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sched.h>
#include <sys/mount.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

//! Wire format for reporting the probe's outcome back through the pipe
//! connecting the forked stages to this payload's own main(), which is the
//! only process that ever writes to stdout - the pipe is the only channel
//! available once fork() has split the staged work across processes that,
//! from stage 5 onward, live in a different PID namespace.
struct SStageResult {
    int s_FailedStage; // 0 means every stage succeeded.
    int s_Errno;
};

void writeResult(int pipeWriteFd, int failedStage, int errnoValue) {
    SStageResult result{failedStage, errnoValue};
    // Best-effort: if this write itself fails there is nothing more this
    // process can do to report - the reader treats EOF/a short read as a
    // failure of its own.
    static_cast<void>(::write(pipeWriteFd, &result, sizeof(result)));
}

//! Stages 6-7: mount("/", MS_REC | MS_PRIVATE) then mount("proc", ...).
//! Called only from the stage-5 grandchild, i.e. only once it is running as
//! the new PID namespace's own PID 1 - the ordering this whole probe exists
//! to pin down.
void runMountStages(int pipeWriteFd) {
    if (::mount(nullptr, "/", nullptr, MS_REC | MS_PRIVATE, nullptr) != 0) {
        writeResult(pipeWriteFd, 6, errno);
        return;
    }
    if (::mount("proc", "/proc", "proc", 0, nullptr) != 0) {
        writeResult(pipeWriteFd, 7, errno);
        return;
    }
    writeResult(pipeWriteFd, 0, 0);
}

//! Stages 2-5: unshare(CLONE_NEWUSER), uid/gid map writes (incl.
//! setgroups), unshare(CLONE_NEWNS | CLONE_NEWPID), then the stage-5 fork.
//! Called from the stage-1 fork's child.
void runNamespaceStages(int pipeWriteFd) {
    const uid_t uid = ::getuid();
    const gid_t gid = ::getgid();

    if (::unshare(CLONE_NEWUSER) != 0) {
        writeResult(pipeWriteFd, 2, errno);
        return;
    }

    // setgroups must be denied before the gid_map write below is permitted
    // for an unprivileged (non-CAP_SETGID) caller - kernel requirement
    // since Linux 3.19 (CVE-2014-8989 mitigation).
    int setgroupsFd = ::open("/proc/self/setgroups", O_WRONLY);
    if (setgroupsFd < 0 || ::write(setgroupsFd, "deny", 4) != 4) {
        const int savedErrno = errno;
        if (setgroupsFd >= 0) {
            ::close(setgroupsFd);
        }
        writeResult(pipeWriteFd, 3, savedErrno);
        return;
    }
    ::close(setgroupsFd);

    char uidMapBuf[64];
    const int uidMapLen = std::snprintf(uidMapBuf, sizeof(uidMapBuf),
                                        "0 %d 1\n", static_cast<int>(uid));
    int uidMapFd = ::open("/proc/self/uid_map", O_WRONLY);
    if (uidMapFd < 0 || ::write(uidMapFd, uidMapBuf, uidMapLen) != uidMapLen) {
        const int savedErrno = errno;
        if (uidMapFd >= 0) {
            ::close(uidMapFd);
        }
        writeResult(pipeWriteFd, 3, savedErrno);
        return;
    }
    ::close(uidMapFd);

    char gidMapBuf[64];
    const int gidMapLen = std::snprintf(gidMapBuf, sizeof(gidMapBuf),
                                        "0 %d 1\n", static_cast<int>(gid));
    int gidMapFd = ::open("/proc/self/gid_map", O_WRONLY);
    if (gidMapFd < 0 || ::write(gidMapFd, gidMapBuf, gidMapLen) != gidMapLen) {
        const int savedErrno = errno;
        if (gidMapFd >= 0) {
            ::close(gidMapFd);
        }
        writeResult(pipeWriteFd, 3, savedErrno);
        return;
    }
    ::close(gidMapFd);

    if (::unshare(CLONE_NEWNS | CLONE_NEWPID) != 0) {
        writeResult(pipeWriteFd, 4, errno);
        return;
    }

    // Stage 5: fork into the just-created PID namespace. unshare(CLONE_NEWPID)
    // does not move the calling process into the new namespace - only its
    // *next* forked child becomes that namespace's PID 1. Stages 6-7 (in
    // particular the stage-7 proc mount) must therefore run in this child,
    // never in the unshare()'d process itself.
    const pid_t pidNsChild = ::fork();
    if (pidNsChild < 0) {
        writeResult(pipeWriteFd, 5, errno);
        return;
    }
    if (pidNsChild == 0) {
        runMountStages(pipeWriteFd);
        ::_exit(0);
    }

    int status = 0;
    ::waitpid(pidNsChild, &status, 0);
}

} // namespace

int main() {
    int pipeFds[2];
    // Stage 1: probe pipe + fork.
    if (::pipe(pipeFds) != 0) {
        std::printf("ml_sandbox_userns_probe: outcome=failure stage=1 errno=%d detail=%s\n",
                    errno, std::strerror(errno));
        return EXIT_FAILURE;
    }

    const pid_t stage1Child = ::fork();
    if (stage1Child < 0) {
        const int savedErrno = errno;
        ::close(pipeFds[0]);
        ::close(pipeFds[1]);
        std::printf("ml_sandbox_userns_probe: outcome=failure stage=1 errno=%d detail=%s\n",
                    savedErrno, std::strerror(savedErrno));
        return EXIT_FAILURE;
    }

    if (stage1Child == 0) {
        ::close(pipeFds[0]);
        runNamespaceStages(pipeFds[1]);
        ::close(pipeFds[1]);
        ::_exit(0);
    }

    ::close(pipeFds[1]);
    SStageResult result{-1, 0};
    const ssize_t bytesRead = ::read(pipeFds[0], &result, sizeof(result));
    ::close(pipeFds[0]);

    int status = 0;
    ::waitpid(stage1Child, &status, 0);

    if (bytesRead != static_cast<ssize_t>(sizeof(result))) {
        // Short read/EOF: the staged process tree exited (or was killed)
        // before reporting a result - stage unknown, but still a failure.
        std::printf("ml_sandbox_userns_probe: outcome=failure stage=-1 errno=0 "
                    "detail=no_result_reported\n");
        return EXIT_FAILURE;
    }

    if (result.s_FailedStage == 0) {
        std::printf("ml_sandbox_userns_probe: outcome=success\n");
        return EXIT_SUCCESS;
    }

    std::printf("ml_sandbox_userns_probe: outcome=failure stage=%d errno=%d detail=%s\n",
                result.s_FailedStage, result.s_Errno, std::strerror(result.s_Errno));
    return EXIT_FAILURE;
}
