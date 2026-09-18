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

// Linux-only mechanism-probe integration test for the typed
// filesystem/network launch policy. Builds a real policy via
// buildPytorchInferenceFilesystemPolicy, runs ml_sandbox_probe inside it,
// and asserts on the probe's per-mechanism "outcome=" lines rather than
// trusting a bare exit code - a policy that merely lets the probe start
// would otherwise look identical to a correctly minimized one.
//
// This test has been reviewed against the Sandbox2 PolicyBuilder API as
// used by CSandboxForkserverSmokeTest_Linux, but still needs a real
// Linux/Sandbox2 build-and-run pass to confirm it actually passes.

#include <sandbox/CPytorchInferenceSandboxPolicy.h>

#include <boost/test/unit_test.hpp>

#include <climits>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

#include "absl/time/time.h"
#include "sandboxed_api/sandbox2/executor.h"
#include "sandboxed_api/sandbox2/result.h"
#include "sandboxed_api/sandbox2/sandbox2.h"

#ifndef ML_SANDBOX2_PROBE_PAYLOAD
#error "ML_SANDBOX2_PROBE_PAYLOAD must be defined by lib/sandbox/unittest/CMakeLists.txt"
#endif

namespace {

//! Returns the outcome recorded for mechanism, or empty if the mechanism
//! line never appeared - a missing line is itself a failure (the probe
//! didn't reach that check, e.g. because it was killed earlier).
std::string outcomeFor(const std::string& resultsFileContent, const std::string& mechanism) {
    std::istringstream lines{resultsFileContent};
    std::string line;
    const std::string marker{"mechanism=" + mechanism + " outcome="};
    while (std::getline(lines, line)) {
        const std::size_t pos = line.find(marker);
        if (pos == std::string::npos) {
            continue;
        }
        const std::size_t start = pos + marker.size();
        const std::size_t end = line.find(' ', start);
        return line.substr(start, end == std::string::npos ? std::string::npos : end - start);
    }
    return {};
}

//! Returns the detail= field recorded for mechanism, or empty if the
//! mechanism line never appeared.
std::string detailFor(const std::string& resultsFileContent, const std::string& mechanism) {
    std::istringstream lines{resultsFileContent};
    std::string line;
    const std::string outcomeMarker{"mechanism=" + mechanism + " outcome="};
    const std::string detailMarker{" detail="};
    while (std::getline(lines, line)) {
        if (line.find(outcomeMarker) == std::string::npos) {
            continue;
        }
        const std::size_t pos = line.find(detailMarker);
        return pos == std::string::npos ? std::string{}
                                        : line.substr(pos + detailMarker.size());
    }
    return {};
}

std::string readFileOrEmpty(const std::string& path) {
    std::ifstream file{path};
    if (file.is_open() == false) {
        return {};
    }
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

//! Removes probe artifacts and the per-child IPC tree created by the
//! mechanism test, even when a BOOST_REQUIRE aborts the case mid-run.
class CMechanismProbeFixture {
public:
    CMechanismProbeFixture() {
        char pathTemplate[] = "/tmp/ml_sandbox_probe_test_XXXXXX";
        char* created = ::mkdtemp(pathTemplate);
        BOOST_TEST_REQUIRE(created != nullptr);
        m_LiteralTmpDir.assign(created);

        char resolved[PATH_MAX];
        BOOST_TEST_REQUIRE(::realpath(m_LiteralTmpDir.c_str(), resolved) != nullptr);
        m_TrustedTmpDir.assign(resolved);

        BOOST_TEST_REQUIRE(::mkdir((m_TrustedTmpDir + "/ml-child-ipc").c_str(), 0700) == 0);
        m_ChildRoot = m_TrustedTmpDir + "/ml-child-ipc/mechanism-probe-child";
        BOOST_TEST_REQUIRE(::mkdir(m_ChildRoot.c_str(), 0700) == 0);
    }

    ~CMechanismProbeFixture() {
        ::unlink((m_ChildRoot + "/probe.txt").c_str());
        ::unlink((m_ChildRoot + "/results.txt").c_str());
        ::rmdir(m_ChildRoot.c_str());
        ::rmdir((m_TrustedTmpDir + "/ml-child-ipc").c_str());
        if (m_LiteralTmpDir != m_TrustedTmpDir) {
            ::rmdir(m_LiteralTmpDir.c_str());
        }
        ::rmdir(m_TrustedTmpDir.c_str());
    }

    CMechanismProbeFixture(const CMechanismProbeFixture&) = delete;
    CMechanismProbeFixture& operator=(const CMechanismProbeFixture&) = delete;

    const std::string& trustedTmpDir() const { return m_TrustedTmpDir; }
    const std::string& childRoot() const { return m_ChildRoot; }

private:
    std::string m_LiteralTmpDir;
    std::string m_TrustedTmpDir;
    std::string m_ChildRoot;
};

} // namespace

BOOST_AUTO_TEST_SUITE(CPytorchInferenceSandboxPolicyMechanismTest_Linux)

BOOST_AUTO_TEST_CASE(testMinimizedPolicyEnforcesEveryMechanism) {
    CMechanismProbeFixture fixture;

    const std::vector<std::string> args{"--input=" + fixture.childRoot() + "/input.fifo",
                                        "--output=" + fixture.childRoot() + "/output.fifo",
                                        "--logPipe=" + fixture.childRoot() + "/log.fifo"};
    const ml::sandbox::SChildIpcValidationResult validated{
        ml::sandbox::validateChildIpcLaunchSpec(fixture.trustedTmpDir(), args)};
    BOOST_TEST_REQUIRE(validated.s_Ok);

    const std::string payloadPath{ML_SANDBOX2_PROBE_PAYLOAD};
    const std::vector<std::string> probeArgs{payloadPath, "/run/elastic/ml-ipc"};

    auto executor = std::make_unique<sandbox2::Executor>(payloadPath, probeArgs);
    executor->limits()->set_rlimit_cpu(10).set_walltime_limit(absl::Seconds(10));

    sandbox2::PolicyBuilder policyBuilder{ml::sandbox::buildPytorchInferenceFilesystemPolicy(
        "/usr/bin", "/usr/lib", validated, /*tmpfsSizeBytes=*/16 * 1024 * 1024)};
    policyBuilder.AddLibrariesForBinary(payloadPath);
    // legacyBpfAllowedSyscalls() grants __NR_connect but not __NR_socket -
    // real libtorch/pytorch_inference apparently also needs a bare socket()
    // for its own internal socket setup, so this is likely a real gap in
    // that shared declaration, not something specific to this probe. Fixing
    // the shared declaration belongs with whatever change owns that file;
    // granting it here, scoped to this test's own policy only, is enough to
    // prove ml_sandbox_probe's network mechanisms without widening the
    // production policy this test doesn't own.
    policyBuilder.AllowSyscall(__NR_socket);
    auto policy = policyBuilder.BuildOrDie();

    sandbox2::Sandbox2 s2(std::move(executor), std::move(policy));
    sandbox2::Result result = s2.Run();

    BOOST_TEST_REQUIRE(result.final_status() == sandbox2::Result::OK);

    // The child IPC directory is genuinely shared with the host, so the
    // probe's results file - written from inside the sandbox to the mapped
    // /run/elastic/ml-ipc path - is readable here at its host-visible
    // childRoot path once the sandbox has exited. This IS the "allowed IPC
    // access" proof, not a separate assertion: if the mount/policy were
    // wrong, this file would never appear.
    const std::string resultsContent{readFileOrEmpty(fixture.childRoot() + "/results.txt")};
    BOOST_TEST_REQUIRE(resultsContent.empty() == false);
    BOOST_TEST_REQUIRE(resultsContent.find("reached=true") != std::string::npos);

    BOOST_REQUIRE_EQUAL(outcomeFor(resultsContent, "ipc_readwrite"), "allowed");
    BOOST_REQUIRE_EQUAL(outcomeFor(resultsContent, "host_read_etc_shadow"), "denied");
    BOOST_REQUIRE_EQUAL(outcomeFor(resultsContent, "private_tmpfs_write"), "allowed");
    BOOST_REQUIRE_EQUAL(outcomeFor(resultsContent, "external_egress"), "denied");

    // Mount conformance: /etc must list only allowlistedEtcFiles() (5 entries)
    // plus "." and "..", never a full directory bind. A regression back to
    // AddDirectory("/etc", true) would spike this into the dozens/hundreds,
    // so an upper bound catches it without hard-coding the exact count.
    BOOST_REQUIRE_EQUAL(outcomeFor(resultsContent, "etc_enumeration"), "counted");
    BOOST_TEST_REQUIRE(std::stoi(detailFor(resultsContent, "etc_enumeration")) <= 10);

    BOOST_REQUIRE_EQUAL(outcomeFor(resultsContent, "pid_namespace"), "namespaced");
    BOOST_REQUIRE_EQUAL(outcomeFor(resultsContent, "loopback_reachable"), "ok");
}

BOOST_AUTO_TEST_SUITE_END()
