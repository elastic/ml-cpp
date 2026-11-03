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

#include <seccomp/CSystemCallFilter.h>

#include <boost/test/unit_test.hpp>

#include <set>
#include <vector>

#ifdef __linux__

// These must be included before BOOST_AUTO_TEST_SUITE() opens a namespace:
// BOOST_AUTO_TEST_SUITE(name) expands to `namespace name { ... }`, so any
// #include placed after it would get its declarations nested inside that
// namespace instead of at global scope, shadowing ::ml::seccomp with an
// incomplete duplicate.
#include <seccomp/CMlLegacyBpfSyscallAllowlist.h>
#include <seccomp/CSeccompFilterBuilder.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>

#endif // __linux__

BOOST_AUTO_TEST_SUITE(CSeccompFilterBuilderTest)

#ifdef __linux__

namespace {

//! Decodes the syscall numbers this builder actually applies, by walking the
//! generated program rather than re-reading the declaration it was built
//! from. This is the proof that the applied program matches the
//! declaration, not a comparison of two independently maintained lists.
//!
//! Rows before the syscall-number load (the arch load/check prefix) also use
//! BPF_JMP|BPF_JEQ|BPF_K, so decoding starts only once that load is seen.
std::set<int> decodeAppliedSyscalls(const std::vector<sock_filter>& program) {
    std::set<int> applied;
    bool sawNrLoad{false};
    for (const auto& instr : program) {
        if (instr.code == (BPF_LD | BPF_W | BPF_ABS) &&
            instr.k == offsetof(struct seccomp_data, nr)) {
            sawNrLoad = true;
            continue;
        }
        if (sawNrLoad && instr.code == (BPF_JMP | BPF_JEQ | BPF_K) && instr.jt > 0) {
            applied.insert(static_cast<int>(instr.k));
        }
    }
    return applied;
}

} // namespace

BOOST_AUTO_TEST_CASE(testAppliedProgramMatchesDeclaration) {
    const std::vector<int> declared{ml::seccomp::legacyBpfAllowedSyscalls()};
    const std::vector<sock_filter> program{ml::seccomp::buildSyscallAllowlistProgram(declared)};

    const std::set<int> declaredSet{declared.begin(), declared.end()};
    BOOST_REQUIRE_EQUAL(declaredSet.size(), declared.size()); // declaration has no duplicates
    const std::set<int> appliedSet{decodeAppliedSyscalls(program)};
    BOOST_REQUIRE_EQUAL_COLLECTIONS(declaredSet.begin(), declaredSet.end(),
                                    appliedSet.begin(), appliedSet.end());

    // Structural invariants that must hold regardless of declaration content:
    // native-arch load/check, syscall-number load, and a final deny/allow
    // pair. No index into this vector is hand-maintained anywhere in
    // production code.
    BOOST_TEST_REQUIRE(program.size() >= declared.size() + 4);
    BOOST_REQUIRE_EQUAL(static_cast<int>(BPF_RET | BPF_K),
                        static_cast<int>(program.back().code));
    BOOST_REQUIRE_EQUAL(static_cast<unsigned int>(SECCOMP_RET_ALLOW),
                        program.back().k);
    const auto& denyRow = program[program.size() - 2];
    BOOST_REQUIRE_EQUAL(static_cast<int>(BPF_RET | BPF_K),
                        static_cast<int>(denyRow.code));
    BOOST_TEST_REQUIRE(denyRow.k != SECCOMP_RET_ALLOW);
}

BOOST_AUTO_TEST_CASE(testJumpOffsetsAreDerivedNotHandMaintained) {
    // An arbitrary, deliberately unordered and out-of-production-order list.
    // If any jump offset were hand-maintained rather than derived from the
    // vector's size/index, reordering or resizing this list would desync it
    // from the generated rows; this test would fail with a stale allowlist
    // but pass immediately once regenerated, which is exactly the property
    // "no manual BPF jump offsets remain" requires.
    const std::vector<int> arbitrarySyscalls{200, 1, 57, 9, 300};
    const std::vector<sock_filter> program{
        ml::seccomp::buildSyscallAllowlistProgram(arbitrarySyscalls)};

    const std::size_t allowIndex{program.size() - 1};
    const std::size_t denyIndex{program.size() - 2};
    BOOST_REQUIRE_EQUAL(static_cast<unsigned int>(SECCOMP_RET_ALLOW),
                        program[allowIndex].k);
    BOOST_TEST_REQUIRE(program[denyIndex].k != SECCOMP_RET_ALLOW);

    // Every syscall row's own jt must land exactly on the allow row: for a
    // row at absolute index i, i + jt + 1 == allowIndex. The arch load/check
    // prefix also uses BPF_JMP|BPF_JEQ|BPF_K but targets the nr-load
    // instruction, not the allow row, so decoding starts only after the
    // syscall-number load is seen (mirrors decodeAppliedSyscalls() above).
    std::set<int> foundSyscalls;
    bool sawNrLoad{false};
    for (std::size_t i = 0; i < program.size(); ++i) {
        if (program[i].code == (BPF_LD | BPF_W | BPF_ABS) &&
            program[i].k == offsetof(struct seccomp_data, nr)) {
            sawNrLoad = true;
            continue;
        }
        if (sawNrLoad && program[i].code == (BPF_JMP | BPF_JEQ | BPF_K) &&
            program[i].jt > 0) {
            BOOST_REQUIRE_EQUAL(allowIndex, i + program[i].jt + 1);
            foundSyscalls.insert(static_cast<int>(program[i].k));
        }
    }
    const std::set<int> expected{arbitrarySyscalls.begin(), arbitrarySyscalls.end()};
    BOOST_REQUIRE_EQUAL_COLLECTIONS(expected.begin(), expected.end(),
                                    foundSyscalls.begin(), foundSyscalls.end());

#ifdef __x86_64__
    // The x32-ABI guard must jump to the deny row (numSyscalls rows ahead of
    // the JGT instruction), not hand-maintained like the old static FILTER[].
    constexpr std::uint32_t upperNrLimit{0x3FFFFFFF};
    bool sawX32Guard{false};
    sawNrLoad = false;
    for (std::size_t i = 0; i < program.size(); ++i) {
        if (program[i].code == (BPF_LD | BPF_W | BPF_ABS) &&
            program[i].k == offsetof(struct seccomp_data, nr)) {
            sawNrLoad = true;
            continue;
        }
        if (sawNrLoad && program[i].code == (BPF_JMP | BPF_JGT | BPF_K)) {
            BOOST_REQUIRE_EQUAL(upperNrLimit, program[i].k);
            BOOST_REQUIRE_EQUAL(denyIndex, i + program[i].jt + 1);
            sawX32Guard = true;
            break;
        }
    }
    BOOST_TEST_REQUIRE(sawX32Guard);
#endif
}

BOOST_AUTO_TEST_CASE(testAllowlistAtEightBitJumpLimitStillBuilds) {
    const std::vector<int> atLimit(std::numeric_limits<std::uint8_t>::max(), 1);
    const std::vector<sock_filter> program{ml::seccomp::buildSyscallAllowlistProgram(atLimit)};
    BOOST_TEST_REQUIRE(program.empty() == false);
    BOOST_REQUIRE_EQUAL(static_cast<unsigned int>(SECCOMP_RET_ALLOW),
                        program.back().k);
}

BOOST_AUTO_TEST_CASE(testOversizedAllowlistProducesEmptyProgram) {
    // The production declaration is compile-time capped at 255, but this
    // builder accepts an arbitrary vector. A wrapped jt would still look like
    // a well-formed program; fail closed with empty instead.
    const std::vector<int> oversized(
        static_cast<std::size_t>(std::numeric_limits<std::uint8_t>::max()) + 1, 1);
    const std::vector<sock_filter> program{ml::seccomp::buildSyscallAllowlistProgram(oversized)};
    BOOST_TEST_REQUIRE(program.empty());
}

BOOST_AUTO_TEST_CASE(testArchGuardRejectsNonNativeAbi) {
    const std::vector<sock_filter> program{
        ml::seccomp::buildSyscallAllowlistProgram(std::vector<int>{1})};

    BOOST_TEST_REQUIRE(program.size() >= 3);
    BOOST_REQUIRE_EQUAL(static_cast<int>(BPF_LD | BPF_W | BPF_ABS),
                        static_cast<int>(program[0].code));
    BOOST_REQUIRE_EQUAL(static_cast<unsigned int>(offsetof(struct seccomp_data, arch)),
                        program[0].k);
    BOOST_REQUIRE_EQUAL(static_cast<int>(BPF_JMP | BPF_JEQ | BPF_K),
                        static_cast<int>(program[1].code));
#ifdef __x86_64__
    BOOST_REQUIRE_EQUAL(static_cast<unsigned int>(AUDIT_ARCH_X86_64), program[1].k);
#elif defined(__aarch64__)
    BOOST_REQUIRE_EQUAL(static_cast<unsigned int>(AUDIT_ARCH_AARCH64),
                        program[1].k);
#endif
    BOOST_REQUIRE_EQUAL(static_cast<int>(BPF_RET | BPF_K),
                        static_cast<int>(program[2].code));
    BOOST_TEST_REQUIRE(program[2].k != SECCOMP_RET_ALLOW);
}

BOOST_AUTO_TEST_CASE(testCarryForwardSyscallsPresent) {
    // This declaration must
    // not silently drop pytorch_inference/libtorch compatibility fixes.
    // Each assertion below is a named regression test for one carried-forward
    // fix within this file's scope.
    const std::vector<int> syscalls{ml::seccomp::legacyBpfAllowedSyscalls()};
    const std::set<int> declared{syscalls.begin(), syscalls.end()};

    // 57f00ed1b: clone3 must be allowed by its literal syscall number (435 on
    // both x86_64 and aarch64), not only via __NR_clone3, because some build
    // images' kernel headers predate clone3 while the runtime glibc uses it.
    BOOST_TEST_REQUIRE(declared.count(435) == 1);

    // 03b1ee4a: prlimit64, queried by libtorch/the Sandbox2 monitor under
    // sustained load.
    BOOST_TEST_REQUIRE(declared.count(__NR_prlimit64) == 1);

#ifdef __x86_64__
    // ec7d3ed85: glibc's x86_64 file-system wrappers issue these legacy
    // syscalls (not their *at equivalents) when pytorch_inference creates
    // and tears down its named pipes.
    const int legacyFsSyscalls[]{__NR_mknod, __NR_unlink,   __NR_rmdir,
                                 __NR_mkdir, __NR_readlink, __NR_access,
                                 __NR_dup2};
    for (int nr : legacyFsSyscalls) {
        BOOST_TEST_REQUIRE(declared.count(nr) == 1);
    }
#endif
}

BOOST_AUTO_TEST_CASE(testSandbox2ExplicitSyscallsCarriedForwardFromPr2873) {
    // The clean rebuild's Sandbox2 policy builder originally granted only
    // legacyBpfAllowedSyscalls(), which is not sufficient: Sandbox2's
    // namespace/threading setup exercises syscalls (scheduling, epoll, pipes,
    // directory management) the legacy in-process filter never needed. PR
    // #2873's enhancement/sandbox2 branch already had a dedicated
    // sandbox2ExplicitSyscalls() list for exactly this; this regression test
    // keeps a future rewrite from dropping it again the same way.
    // sandbox2ExplicitSyscalls() returns by value, so it must be called once:
    // taking begin() and end() from two separate calls pairs iterators from
    // two different temporaries, which is undefined behaviour that passes or
    // crashes depending on heap layout.
    const std::vector<int> explicitSyscalls{ml::seccomp::sandbox2ExplicitSyscalls()};
    const std::set<int> explicitGrants{explicitSyscalls.begin(),
                                       explicitSyscalls.end()};

    BOOST_TEST_REQUIRE(explicitGrants.count(__NR_sched_getaffinity) == 1);
    BOOST_TEST_REQUIRE(explicitGrants.count(__NR_sched_setaffinity) == 1);
    BOOST_TEST_REQUIRE(explicitGrants.count(__NR_epoll_pwait) == 1);
    BOOST_TEST_REQUIRE(explicitGrants.count(__NR_pipe2) == 1);
}

#endif // __linux__

BOOST_AUTO_TEST_CASE(testDegradedModeAttestationMarker) {
    using ml::seccomp::ESystemCallFilterInstallOutcome;
    using ml::seccomp::degradedModeAttestationMarker;

    // The marker must be present and exact on success - this is what a
    // controller/Elasticsearch observer asserts, replacing "no fatal log
    // line appeared" as an implicit readiness signal.
    BOOST_REQUIRE_EQUAL(
        std::string("{\"ml_sandbox2_route\":\"legacy\",\"event\":\"seccomp_installed\"}"),
        degradedModeAttestationMarker(ESystemCallFilterInstallOutcome::E_Installed));

    // Every failure class must attest nothing - a caller that logged this
    // marker on a failed install would falsely claim protection that isn't
    // there.
    BOOST_TEST_REQUIRE(degradedModeAttestationMarker(ESystemCallFilterInstallOutcome::E_MechanismUnavailable)
                           .empty());
    BOOST_TEST_REQUIRE(degradedModeAttestationMarker(ESystemCallFilterInstallOutcome::E_PrivilegeRestrictionFailed)
                           .empty());
    BOOST_TEST_REQUIRE(degradedModeAttestationMarker(ESystemCallFilterInstallOutcome::E_FilterInstallFailed)
                           .empty());
}

BOOST_AUTO_TEST_CASE(testDecideDegradedModeActionFaultInjection) {
    using ml::seccomp::EDegradedModeAction;
    using ml::seccomp::ESystemCallFilterInstallOutcome;
    using ml::seccomp::decideDegradedModeAction;

    // Successful installation never terminates, regardless of the switch.
    BOOST_REQUIRE_EQUAL(static_cast<int>(EDegradedModeAction::E_ContinueDespiteFailure),
                        static_cast<int>(decideDegradedModeAction(
                            ESystemCallFilterInstallOutcome::E_Installed, false)));
    BOOST_REQUIRE_EQUAL(static_cast<int>(EDegradedModeAction::E_ContinueDespiteFailure),
                        static_cast<int>(decideDegradedModeAction(
                            ESystemCallFilterInstallOutcome::E_Installed, true)));

    // Every fault-injected failure class - capability probe failure,
    // PR_SET_NO_NEW_PRIVS, and filter installation - with the internal
    // switch off (today's production default), every call site continues;
    // with it on (the behaviour a later change activates), every one
    // terminates.
    const ESystemCallFilterInstallOutcome failureModes[]{
        ESystemCallFilterInstallOutcome::E_MechanismUnavailable,
        ESystemCallFilterInstallOutcome::E_PrivilegeRestrictionFailed,
        ESystemCallFilterInstallOutcome::E_FilterInstallFailed};

    for (const auto outcome : failureModes) {
        BOOST_REQUIRE_EQUAL(static_cast<int>(EDegradedModeAction::E_ContinueDespiteFailure),
                            static_cast<int>(decideDegradedModeAction(outcome, false)));
        BOOST_REQUIRE_EQUAL(static_cast<int>(EDegradedModeAction::E_TerminateBeforeIo),
                            static_cast<int>(decideDegradedModeAction(outcome, true)));
    }
}

BOOST_AUTO_TEST_CASE(testSandbox2LaunchedChildRecognisesOnlyExactlyOne) {
    using ml::seccomp::sandbox2LaunchedChild;

    // Exactly "1" - the value CSandboxedProcessSpawner_Linux.cc sets on a
    // sandboxee - and nothing else.
    BOOST_REQUIRE_EQUAL(true, sandbox2LaunchedChild("1"));

    BOOST_REQUIRE_EQUAL(false, sandbox2LaunchedChild(nullptr));
    BOOST_REQUIRE_EQUAL(false, sandbox2LaunchedChild(""));
    BOOST_REQUIRE_EQUAL(false, sandbox2LaunchedChild("0"));
    BOOST_REQUIRE_EQUAL(false, sandbox2LaunchedChild("true"));
    BOOST_REQUIRE_EQUAL(false, sandbox2LaunchedChild("10"));
    BOOST_REQUIRE_EQUAL(false, sandbox2LaunchedChild(" 1"));
}

BOOST_AUTO_TEST_CASE(testInProcessFilterSkippedEntirelyForSandbox2LaunchedChild) {
    using ml::seccomp::EDegradedModeAction;
    using ml::seccomp::ESystemCallFilterInstallOutcome;
    using ml::seccomp::applyInProcessSeccompFilter;

    // ML_SANDBOXED=1: the installer must never be invoked, no degraded-mode
    // termination may be derived and no attestation marker may be produced -
    // and that must hold for every outcome an installation attempt could
    // have returned, including the failure classes that would otherwise
    // terminate the launch once TERMINATE_ON_DEGRADED_SECCOMP_FAILURE is activated.
    const ESystemCallFilterInstallOutcome allOutcomes[]{
        ESystemCallFilterInstallOutcome::E_Installed,
        ESystemCallFilterInstallOutcome::E_MechanismUnavailable,
        ESystemCallFilterInstallOutcome::E_PrivilegeRestrictionFailed,
        ESystemCallFilterInstallOutcome::E_FilterInstallFailed};

    for (const auto wouldHaveReturned : allOutcomes) {
        bool installerCalled{false};
        const auto result = applyInProcessSeccompFilter(
            true, true, [&installerCalled, wouldHaveReturned] {
                installerCalled = true;
                return wouldHaveReturned;
            });

        BOOST_REQUIRE_EQUAL(false, installerCalled);
        BOOST_REQUIRE_EQUAL(false, result.s_Attempted);
        BOOST_REQUIRE_EQUAL(static_cast<int>(EDegradedModeAction::E_ContinueDespiteFailure),
                            static_cast<int>(result.s_Action));
        BOOST_TEST_REQUIRE(result.s_AttestationMarker.empty());
    }
}

BOOST_AUTO_TEST_CASE(testInProcessFilterUnchangedOnLegacyRoute) {
    using ml::seccomp::EDegradedModeAction;
    using ml::seccomp::ESystemCallFilterInstallOutcome;
    using ml::seccomp::applyInProcessSeccompFilter;

    // ML_SANDBOXED unset/not "1": behaviour is exactly the pre-existing
    // install + decide + attest sequence, i.e. the fault-injection coverage
    // above (testDecideDegradedModeActionFaultInjection) still describes
    // this path.
    bool installerCalled{false};
    const auto installed = applyInProcessSeccompFilter(false, true, [&installerCalled] {
        installerCalled = true;
        return ESystemCallFilterInstallOutcome::E_Installed;
    });
    BOOST_REQUIRE_EQUAL(true, installerCalled);
    BOOST_REQUIRE_EQUAL(true, installed.s_Attempted);
    BOOST_REQUIRE_EQUAL(static_cast<int>(EDegradedModeAction::E_ContinueDespiteFailure),
                        static_cast<int>(installed.s_Action));
    BOOST_REQUIRE_EQUAL(std::string("{\"ml_sandbox2_route\":\"legacy\",\"event\":\"seccomp_installed\"}"),
                        installed.s_AttestationMarker);

    const ESystemCallFilterInstallOutcome failureModes[]{
        ESystemCallFilterInstallOutcome::E_MechanismUnavailable,
        ESystemCallFilterInstallOutcome::E_PrivilegeRestrictionFailed,
        ESystemCallFilterInstallOutcome::E_FilterInstallFailed};

    for (const auto outcome : failureModes) {
        const auto failed = applyInProcessSeccompFilter(
            false, true, [outcome] { return outcome; });
        BOOST_REQUIRE_EQUAL(true, failed.s_Attempted);
        BOOST_REQUIRE_EQUAL(static_cast<int>(EDegradedModeAction::E_TerminateBeforeIo),
                            static_cast<int>(failed.s_Action));
        // A failed install attests nothing, exactly as before.
        BOOST_TEST_REQUIRE(failed.s_AttestationMarker.empty());
    }
}

BOOST_AUTO_TEST_CASE(testDegradedModeAttestationMarkerRouteLandlock) {
    using ml::seccomp::ESystemCallFilterInstallOutcome;
    using ml::seccomp::degradedModeAttestationMarker;

    // route must be threaded through verbatim - a controller/Elasticsearch
    // observer needs the marker's ml_sandbox2_route to agree with the
    // sandbox2_launch signal's own "route" field for the same launch.
    BOOST_REQUIRE_EQUAL(
        std::string("{\"ml_sandbox2_route\":\"landlock\",\"event\":\"seccomp_installed\"}"),
        degradedModeAttestationMarker(ESystemCallFilterInstallOutcome::E_Installed, "landlock"));

    // Default parameter is unchanged: omitting route still reports "legacy".
    BOOST_REQUIRE_EQUAL(
        std::string("{\"ml_sandbox2_route\":\"legacy\",\"event\":\"seccomp_installed\"}"),
        degradedModeAttestationMarker(ESystemCallFilterInstallOutcome::E_Installed));

    // A failed install attests nothing, regardless of which route asked for
    // the marker.
    BOOST_TEST_REQUIRE(degradedModeAttestationMarker(
                           ESystemCallFilterInstallOutcome::E_MechanismUnavailable, "landlock")
                           .empty());
    BOOST_TEST_REQUIRE(degradedModeAttestationMarker(
                           ESystemCallFilterInstallOutcome::E_PrivilegeRestrictionFailed, "landlock")
                           .empty());
    BOOST_TEST_REQUIRE(degradedModeAttestationMarker(
                           ESystemCallFilterInstallOutcome::E_FilterInstallFailed, "landlock")
                           .empty());
}

BOOST_AUTO_TEST_CASE(testApplyInProcessSeccompFilterPassesRouteThrough) {
    using ml::seccomp::ESystemCallFilterInstallOutcome;
    using ml::seccomp::applyInProcessSeccompFilter;

    // Not Sandbox2-launched (legacy/Landlock route child), successful
    // install, route == "landlock": the resulting marker must carry that
    // route, not the default "legacy".
    const auto result = applyInProcessSeccompFilter(
        false, true, [] { return ESystemCallFilterInstallOutcome::E_Installed; }, "landlock");

    BOOST_REQUIRE_EQUAL(true, result.s_Attempted);
    BOOST_REQUIRE_EQUAL(std::string("{\"ml_sandbox2_route\":\"landlock\",\"event\":\"seccomp_installed\"}"),
                        result.s_AttestationMarker);
}

BOOST_AUTO_TEST_CASE(testApplyInProcessSeccompFilterFailedInstallEmptyMarkerRegardlessOfRoute) {
    using ml::seccomp::ESystemCallFilterInstallOutcome;
    using ml::seccomp::applyInProcessSeccompFilter;

    const ESystemCallFilterInstallOutcome failureModes[]{
        ESystemCallFilterInstallOutcome::E_MechanismUnavailable,
        ESystemCallFilterInstallOutcome::E_PrivilegeRestrictionFailed,
        ESystemCallFilterInstallOutcome::E_FilterInstallFailed};
    const char* const routes[]{"legacy", "landlock"};

    for (const auto outcome : failureModes) {
        for (const char* route : routes) {
            const auto result = applyInProcessSeccompFilter(
                false, true, [outcome] { return outcome; }, route);
            BOOST_REQUIRE_EQUAL(true, result.s_Attempted);
            BOOST_TEST_REQUIRE(result.s_AttestationMarker.empty());
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
