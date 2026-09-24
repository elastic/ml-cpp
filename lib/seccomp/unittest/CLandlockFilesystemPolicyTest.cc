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

#include <seccomp/CLandlockFilesystemPolicy.h>

#include <boost/filesystem.hpp>
#include <boost/system/error_code.hpp>
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <string>
#include <vector>

#ifdef Linux
#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

BOOST_AUTO_TEST_SUITE(CLandlockFilesystemPolicyTest)

BOOST_AUTO_TEST_CASE(testDescribeCoversEveryOutcome) {
    BOOST_TEST_REQUIRE(
        ml::seccomp::describe(ml::seccomp::ELandlockOutcome::E_Applied).empty() == false);
    BOOST_TEST_REQUIRE(
        ml::seccomp::describe(ml::seccomp::ELandlockOutcome::E_Unsupported).empty() == false);
    BOOST_TEST_REQUIRE(
        ml::seccomp::describe(ml::seccomp::ELandlockOutcome::E_Failed).empty() == false);
    BOOST_REQUIRE(ml::seccomp::describe(ml::seccomp::ELandlockOutcome::E_Applied) !=
                  ml::seccomp::describe(ml::seccomp::ELandlockOutcome::E_Failed));
}

// perChildIpcDirectory() is pure string logic, declared and defined inline in
// the header, so it is testable on every platform without forking or
// applying anything.
BOOST_AUTO_TEST_CASE(testPerChildIpcDirectoryPositiveCases) {
    using ml::seccomp::perChildIpcDirectory;

    BOOST_REQUIRE_EQUAL(std::string{"/tmp/ml-child-ipc/dep-1"},
                        perChildIpcDirectory("/tmp/ml-child-ipc/dep-1/logPipe"));
    // A deeper trusted base directory - only the last three components
    // matter.
    BOOST_REQUIRE_EQUAL(std::string{"/var/lib/es/tmp/ml-child-ipc/abc123"},
                        perChildIpcDirectory("/var/lib/es/tmp/ml-child-ipc/abc123/output"));
}

BOOST_AUTO_TEST_CASE(testPerChildIpcDirectoryNegativeCases) {
    using ml::seccomp::perChildIpcDirectory;

    // Empty input.
    BOOST_TEST_REQUIRE(perChildIpcDirectory("").empty());

    // Relative path: the Landlock rule must never depend on the process's
    // working directory.
    BOOST_TEST_REQUIRE(perChildIpcDirectory("ml-child-ipc/dep-1/logPipe").empty());

    // Flat $TMPDIR layout (the legacy/non-per-child layout) - no
    // "ml-child-ipc/<id>" shape at all.
    BOOST_TEST_REQUIRE(perChildIpcDirectory("/tmp/logPipe").empty());

    // Missing the <id> path component: the pipe sits directly under
    // ".../ml-child-ipc" rather than under a per-child subdirectory of it.
    BOOST_TEST_REQUIRE(perChildIpcDirectory("/tmp/ml-child-ipc/logPipe").empty());

    // A parent directory that merely ends in "ml-child-ipc" (e.g.
    // "xml-child-ipc") must NOT match - the comparison must be exact, not a
    // suffix match.
    BOOST_TEST_REQUIRE(perChildIpcDirectory("/tmp/xml-child-ipc/dep-1/logPipe").empty());
}

#ifdef Linux

namespace {

//! Exit codes a confined child uses to report what it observed. The ruleset
//! is irreversible, so every case below must run in its own forked child -
//! confining the test process itself would break every later test.
enum EChildExit : int {
    E_ChildOk = 0,
    E_ChildNotApplied = 20,
    E_ChildGrantedPathUnreadable = 21,
    E_ChildDeniedPathStillReadable = 22,
    E_ChildGrantedDirNotWritable = 23,
    E_ChildExecRefused = 24
};

//! Run \p body in a forked child and return its exit code, or -1 if the child
//! did not exit normally.
template<typename FUNC>
int runInChild(FUNC body) {
    const pid_t child{::fork()};
    if (child < 0) {
        return -1;
    }
    if (child == 0) {
        ::_exit(body());
    }
    int status{0};
    if (::waitpid(child, &status, 0) != child || WIFEXITED(status) == false) {
        return -1;
    }
    return WEXITSTATUS(status);
}

std::string makeScratchDirectory() {
    std::string path{"/tmp/ml-landlock-test-XXXXXX"};
    if (::mkdtemp(path.data()) == nullptr) {
        return std::string();
    }
    return path;
}

} // namespace

BOOST_AUTO_TEST_CASE(testRulesetGrantsTheAllowedPathAndDeniesEverythingElse) {
    // The point of the fallback is that it actually bounds the sandboxee, so
    // assert every half: the pipe directory still supports exactly what
    // CNamedPipeFactory does (mkfifo, open, unlink), it refuses anything else
    // (a regular file there would allow disk-filling or staging), and a path
    // that was never granted becomes unreadable *even though the uid still
    // owns it*. Without the negative halves a vacuously permissive ruleset
    // would "pass".
    const std::string scratch{makeScratchDirectory()};
    BOOST_TEST_REQUIRE(scratch.empty() == false);

    const std::string outside{scratch + "/outside.txt"};
    const int outsideFd{::open(outside.c_str(), O_CREAT | O_WRONLY, 0600)};
    BOOST_TEST_REQUIRE(outsideFd >= 0);
    ::close(outsideFd);

    const std::string pipes{scratch + "/pipes"};
    BOOST_TEST_REQUIRE(::mkdir(pipes.c_str(), 0700) == 0);

    const int childResult{runInChild([&] {
        ml::seccomp::SLandlockPaths paths;
        paths.s_PipeDirectories.push_back(pipes);

        const ml::seccomp::ELandlockOutcome outcome{
            ml::seccomp::applyLandlockFilesystemPolicy(paths)};
        if (outcome == ml::seccomp::ELandlockOutcome::E_Unsupported) {
            return static_cast<int>(E_ChildNotApplied);
        }
        if (outcome != ml::seccomp::ELandlockOutcome::E_Applied) {
            return static_cast<int>(E_ChildGrantedPathUnreadable);
        }

        // What CNamedPipeFactory does: mkfifo, open (O_RDWR so no peer is
        // needed), unlink.
        const std::string fifo{pipes + "/logPipe"};
        if (::mkfifo(fifo.c_str(), 0600) != 0) {
            return static_cast<int>(E_ChildGrantedDirNotWritable);
        }
        const int fifoFd{::open(fifo.c_str(), O_RDWR)};
        if (fifoFd < 0) {
            return static_cast<int>(E_ChildGrantedDirNotWritable);
        }
        ::close(fifoFd);
        if (::unlink(fifo.c_str()) != 0) {
            return static_cast<int>(E_ChildGrantedDirNotWritable);
        }

        // Anything other than a FIFO must be refused in the pipe directory.
        const int regularFd{::open((pipes + "/staged.bin").c_str(), O_CREAT | O_WRONLY, 0600)};
        if (regularFd >= 0) {
            ::close(regularFd);
            return static_cast<int>(E_ChildDeniedPathStillReadable);
        }

        // The sibling file, owned by this very uid, must now be unreachable.
        const int deniedFd{::open(outside.c_str(), O_RDONLY)};
        if (deniedFd >= 0) {
            ::close(deniedFd);
            return static_cast<int>(E_ChildDeniedPathStillReadable);
        }

        return static_cast<int>(E_ChildOk);
    })};

    ::unlink((pipes + "/staged.bin").c_str());
    ::unlink(outside.c_str());
    ::rmdir(pipes.c_str());
    ::rmdir(scratch.c_str());

    if (childResult == E_ChildNotApplied) {
        BOOST_TEST_MESSAGE("Landlock unsupported on this kernel - skipping enforcement assertions");
        return;
    }
    BOOST_REQUIRE_EQUAL(childResult, static_cast<int>(E_ChildOk));
}

BOOST_AUTO_TEST_CASE(testExecveIsRefusedEvenForAReadableBinary) {
    // EXECUTE is never granted, so Landlock alone refuses execve() - the
    // backstop if the seccomp filter (which also denies execve) ever failed
    // to install. Grant read on the binary's own directory to show that
    // readability does not imply executability.
    //
    // A negative control (EXECUTE granted) must also grant the directories
    // holding the ELF interpreter and libc: execve() needs EXECUTE on the
    // interpreter as well, so granting it on /usr/bin alone still fails, and
    // would make the control pass for the wrong reason.
    const int childResult{runInChild([] {
        ml::seccomp::SLandlockPaths paths;
        paths.s_ReadOnly.push_back("/usr/bin");
        if (ml::seccomp::applyLandlockFilesystemPolicy(paths) ==
            ml::seccomp::ELandlockOutcome::E_Unsupported) {
            return static_cast<int>(E_ChildNotApplied);
        }
        char* const argv[]{const_cast<char*>("true"), nullptr};
        ::execv("/usr/bin/true", argv);
        // Only reached if execv() failed. A *successful* exec replaces this
        // child with /usr/bin/true, which exits 0 - so the refusal must be
        // reported with a distinct non-zero code, or a broken ruleset that
        // allowed the exec would pass this test.
        return static_cast<int>(errno == EACCES ? E_ChildExecRefused
                                                : E_ChildDeniedPathStillReadable);
    })};

    if (childResult == E_ChildNotApplied) {
        BOOST_TEST_MESSAGE("Landlock unsupported on this kernel - skipping");
        return;
    }
    BOOST_REQUIRE_EQUAL(childResult, static_cast<int>(E_ChildExecRefused));
}

BOOST_AUTO_TEST_CASE(testPolicyIsIrreversibleWithinTheConfinedProcess) {
    // Landlock rulesets stack and can only narrow. Applying an empty second
    // ruleset must not restore access the first one removed - otherwise a
    // malicious model could simply re-apply a permissive policy.
    const std::string scratch{makeScratchDirectory()};
    BOOST_TEST_REQUIRE(scratch.empty() == false);
    const std::string probeFile{scratch + "/probe.txt"};
    const int fd{::open(probeFile.c_str(), O_CREAT | O_WRONLY, 0600)};
    BOOST_TEST_REQUIRE(fd >= 0);
    ::close(fd);

    const int childResult{runInChild([&] {
        ml::seccomp::SLandlockPaths empty;
        if (ml::seccomp::applyLandlockFilesystemPolicy(empty) ==
            ml::seccomp::ELandlockOutcome::E_Unsupported) {
            return static_cast<int>(E_ChildNotApplied);
        }
        // Now grant the scratch directory in a second ruleset; Landlock
        // composes by intersection, so this must NOT re-open access.
        ml::seccomp::SLandlockPaths permissive;
        permissive.s_PipeDirectories.push_back(scratch);
        permissive.s_ReadOnly.push_back(scratch);
        ml::seccomp::applyLandlockFilesystemPolicy(permissive);

        const int reopened{::open(probeFile.c_str(), O_RDONLY)};
        if (reopened >= 0) {
            ::close(reopened);
            return static_cast<int>(E_ChildDeniedPathStillReadable);
        }
        return static_cast<int>(E_ChildOk);
    })};

    ::unlink(probeFile.c_str());
    ::rmdir(scratch.c_str());

    if (childResult == E_ChildNotApplied) {
        BOOST_TEST_MESSAGE("Landlock unsupported on this kernel - skipping");
        return;
    }
    BOOST_REQUIRE_EQUAL(childResult, static_cast<int>(E_ChildOk));
}

BOOST_AUTO_TEST_CASE(testPytorchInferencePathsAreTheMeasuredMinimum) {
    // Pins the ruleset to the traced minimum so a later "just add the parent
    // directory" change is a visible test failure rather than a silent
    // widening. Every entry here is justified in
    // pytorchInferenceLandlockPaths().
    const ml::seccomp::SLandlockPaths paths{
        ml::seccomp::pytorchInferenceLandlockPaths("/app/tmp/ml-child-ipc/dep-1")};

    const auto contains = [](const std::vector<std::string>& haystack,
                             const std::string& needle) {
        return std::find(haystack.begin(), haystack.end(), needle) != haystack.end();
    };

    // The IPC directory holds pipes only, and is the sole modifiable path.
    BOOST_REQUIRE_EQUAL(paths.s_PipeDirectories.size(), 1);
    BOOST_TEST_REQUIRE(contains(paths.s_PipeDirectories, "/app/tmp/ml-child-ipc/dep-1"));
    BOOST_TEST_REQUIRE(contains(paths.s_ReadOnly, "/app/tmp/ml-child-ipc/dep-1") == false);

    // Sensitive trees are granted as exact files, never as directories.
    BOOST_TEST_REQUIRE(contains(paths.s_ReadOnly, "/proc/cpuinfo"));
    BOOST_TEST_REQUIRE(contains(paths.s_ReadOnly, "/proc/self/statm"));
    BOOST_TEST_REQUIRE(contains(paths.s_ReadOnly, "/proc/self/environ"));
    BOOST_TEST_REQUIRE(contains(paths.s_ReadOnly, "/etc/localtime"));
    for (const char* tooBroad :
         {"/", "/proc", "/proc/self", "/etc", "/tmp", "/app/tmp", "/lib",
          "/lib64", "/usr/lib", "/usr/lib64", "/usr", "/dev", "/sys", "/home"}) {
        BOOST_TEST_REQUIRE(contains(paths.s_ReadOnly, tooBroad) == false);
        BOOST_TEST_REQUIRE(contains(paths.s_PipeDirectories, tooBroad) == false);
    }
}

BOOST_AUTO_TEST_CASE(testRealPytorchPolicyDeniesTheExploitTargetWrite) {
    // End-to-end at the policy level, using the exact ruleset
    // pytorch_inference installs on the Landlock fallback route - not a
    // synthetic one. The attack-defense exploit model
    // (test/evil_model_generator.py) writes an -agentpath payload to
    // /usr/share/elasticsearch/config/jvm.options.d/gc.options; that path is
    // outside every grant pytorchInferenceLandlockPaths() produces, so the
    // real policy must deny a write there, while the per-child IPC directory
    // it does grant stays usable for what pytorch_inference actually does
    // there - create its own log FIFO. The grant is pipe-only (it may hold
    // nothing but this process's own FIFOs - see
    // SLandlockPaths::s_PipeDirectories), so unlike the original version of
    // this test, the positive half below uses mkfifo/open/unlink rather than
    // creating a regular file, which the real policy now refuses even inside
    // the granted directory. This is the same boundary the harness's ROP
    // exploit exercises, proven deterministically without a build-fragile ROP
    // chain.
    const std::string scratch{makeScratchDirectory()};
    BOOST_TEST_REQUIRE(scratch.empty() == false);

    // A stand-in for the operator TMPDIR, with the per-child IPC directory
    // laid out as the controller creates it: <tmp>/ml-child-ipc/<child-id>.
    const std::string ipcDir{scratch + "/ml-child-ipc/dep-e2e"};
    boost::system::error_code mkdirError;
    boost::filesystem::create_directories(ipcDir, mkdirError);
    BOOST_TEST_REQUIRE(mkdirError.value() == 0);

    // The exploit's hard-coded target, created here so the difference the
    // test observes is Landlock denying the write - not the parent directory
    // being absent. Its parent is deliberately outside every grant.
    const std::string forbiddenDir{scratch + "/config/jvm.options.d"};
    boost::filesystem::create_directories(forbiddenDir, mkdirError);
    BOOST_TEST_REQUIRE(mkdirError.value() == 0);
    const std::string forbiddenTarget{forbiddenDir + "/gc.options"};

    const int childResult{runInChild([&] {
        ml::seccomp::SLandlockPaths paths{ml::seccomp::pytorchInferenceLandlockPaths(ipcDir)};
        // Point the "config" grant nowhere near forbiddenTarget: the real
        // policy grants /etc etc., none of which cover this scratch config
        // path, so no extra removal is needed - forbiddenTarget is already
        // outside paths. Apply the real ruleset unchanged.
        const ml::seccomp::ELandlockOutcome outcome{
            ml::seccomp::applyLandlockFilesystemPolicy(paths)};
        if (outcome == ml::seccomp::ELandlockOutcome::E_Unsupported) {
            return static_cast<int>(E_ChildNotApplied);
        }
        if (outcome != ml::seccomp::ELandlockOutcome::E_Applied) {
            return static_cast<int>(E_ChildGrantedPathUnreadable);
        }

        // The per-child IPC directory the real policy grants must stay
        // usable for what pytorch_inference actually does there: mkfifo,
        // open, unlink - exactly what CNamedPipeFactory does. A regular file
        // is no longer valid here, since the grant is pipe-only.
        const std::string fifoStandin{ipcDir + "/logPipe.test"};
        if (::mkfifo(fifoStandin.c_str(), 0600) != 0) {
            return static_cast<int>(E_ChildGrantedDirNotWritable);
        }
        const int okFd{::open(fifoStandin.c_str(), O_RDWR)};
        if (okFd < 0) {
            return static_cast<int>(E_ChildGrantedDirNotWritable);
        }
        ::close(okFd);
        if (::unlink(fifoStandin.c_str()) != 0) {
            return static_cast<int>(E_ChildGrantedDirNotWritable);
        }

        // The exploit's target write must be denied by the real policy.
        const int deniedFd{::open(forbiddenTarget.c_str(), O_CREAT | O_WRONLY, 0600)};
        if (deniedFd >= 0) {
            ::close(deniedFd);
            return static_cast<int>(E_ChildDeniedPathStillReadable);
        }
        return static_cast<int>(E_ChildOk);
    })};

    boost::system::error_code rmError;
    boost::filesystem::remove_all(scratch, rmError);

    if (childResult == E_ChildNotApplied) {
        BOOST_TEST_MESSAGE("Landlock unsupported on this kernel - skipping");
        return;
    }
    BOOST_REQUIRE_EQUAL(childResult, static_cast<int>(E_ChildOk));
}

#endif // Linux

BOOST_AUTO_TEST_SUITE_END()
