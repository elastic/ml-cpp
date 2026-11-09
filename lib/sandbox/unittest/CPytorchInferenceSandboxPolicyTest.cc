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

// Exercises validateChildIpcLaunchSpec against the pinned child-root
// contract. This suite needs only realpath()/mkdtemp()/mkdir()/symlink(), not Sandbox2
// itself, so it runs on every POSIX ml-cpp CI platform (Linux and macOS),
// not just Linux - but not Windows, which has none of those APIs; see
// lib/sandbox/unittest/CMakeLists.txt's NOT WIN32 guard.

#include <sandbox/CChildIpcDirectoryReaper.h>
#include <sandbox/CPytorchInferenceSandboxPolicy.h>

#include <boost/test/unit_test.hpp>

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace {

//! Creates trustedTmpDir/ml-child-ipc/<childId> (mode 0700), mirroring the
//! native controller's pre-launch creation step, and returns the
//! *canonical* trusted base so literal test paths built from it never
//! diverge from realpath() output on hosts where /tmp is itself a symlink
//! (e.g. macOS's /tmp -> /private/tmp) - that divergence is a real
//! condition (E_MutableSymlinkOrAlias) this suite tests deliberately, so
//! setup must not trigger it by accident.
class CTempChildIpcFixture {
public:
    explicit CTempChildIpcFixture(const std::string& childId)
        : m_ChildId(childId) {
        char pathTemplate[] = "/tmp/ml_sandbox_policy_test_XXXXXX";
        char* created = ::mkdtemp(pathTemplate);
        BOOST_TEST_REQUIRE(created != nullptr);
        m_LiteralBase.assign(created);

        char resolved[PATH_MAX];
        BOOST_TEST_REQUIRE(::realpath(m_LiteralBase.c_str(), resolved) != nullptr);
        m_CanonicalBase.assign(resolved);

        BOOST_TEST_REQUIRE(::mkdir((m_CanonicalBase + "/ml-child-ipc").c_str(), 0700) == 0);
        m_ChildRoot = m_CanonicalBase + "/ml-child-ipc/" + m_ChildId;
        BOOST_TEST_REQUIRE(::mkdir(m_ChildRoot.c_str(), 0700) == 0);
    }

    ~CTempChildIpcFixture() {
        ::rmdir(m_ChildRoot.c_str());
        ::rmdir((m_CanonicalBase + "/ml-child-ipc").c_str());
        if (m_LiteralBase != m_CanonicalBase) {
            ::rmdir(m_LiteralBase.c_str());
        }
        ::rmdir(m_CanonicalBase.c_str());
    }

    const std::string& canonicalTrustedBase() const { return m_CanonicalBase; }
    const std::string& childRoot() const { return m_ChildRoot; }

private:
    std::string m_ChildId;
    std::string m_LiteralBase;
    std::string m_CanonicalBase;
    std::string m_ChildRoot;
};

//! Creates only the *trusted base* directory ($TMPDIR itself) - deliberately
//! leaving ml-child-ipc/<childId> absent, matching the real, pre-fix
//! production bug: Elasticsearch/CCommandProcessor only ever constructs the
//! --input=/--output=/--restore=/--logPipe= path *strings*; nothing had
//! created the directory those paths live in by the time
//! validateChildIpcLaunchSpec()'s realpath() calls ran. Tests using this
//! fixture drive ensureChildIpcDirectory() themselves, rather than
//! mkdir()-ing the child directory in setup the way CTempChildIpcFixture
//! does.
class CTrustedBaseOnlyFixture {
public:
    CTrustedBaseOnlyFixture() {
        char pathTemplate[] = "/tmp/ml_sandbox_policy_nodir_test_XXXXXX";
        char* created = ::mkdtemp(pathTemplate);
        BOOST_TEST_REQUIRE(created != nullptr);
        m_LiteralBase.assign(created);

        char resolved[PATH_MAX];
        BOOST_TEST_REQUIRE(::realpath(m_LiteralBase.c_str(), resolved) != nullptr);
        m_CanonicalBase.assign(resolved);
    }

    ~CTrustedBaseOnlyFixture() {
        ::rmdir((m_CanonicalBase + "/ml-child-ipc/child-ensure-1").c_str());
        ::rmdir((m_CanonicalBase + "/ml-child-ipc").c_str());
        if (m_LiteralBase != m_CanonicalBase) {
            ::rmdir(m_LiteralBase.c_str());
        }
        ::rmdir(m_CanonicalBase.c_str());
    }

    const std::string& canonicalTrustedBase() const { return m_CanonicalBase; }

private:
    std::string m_LiteralBase;
    std::string m_CanonicalBase;
};

} // namespace

BOOST_AUTO_TEST_SUITE(CPytorchInferenceSandboxPolicyTest)

BOOST_AUTO_TEST_CASE(testAcceptsAllFourPathOptionsUnderPinnedChildRoot) {
    CTempChildIpcFixture fixture{"child-1"};
    const std::vector<std::string> args{
        "--input=" + fixture.childRoot() + "/input.fifo",
        "--output=" + fixture.childRoot() + "/output.fifo",
        "--restore=" + fixture.childRoot() + "/restore.fifo",
        "--logPipe=" + fixture.childRoot() + "/log.fifo",
        "--someScalarOption=not-a-path",
    };

    const ml::sandbox::SChildIpcValidationResult result{
        ml::sandbox::validateChildIpcLaunchSpec(fixture.canonicalTrustedBase(), args)};

    BOOST_TEST_REQUIRE(result.s_Ok);
    BOOST_TEST_REQUIRE(result.s_Rejected.empty());
    BOOST_REQUIRE_EQUAL(result.s_Spec.s_ChildId, "child-1");
    BOOST_REQUIRE_EQUAL(result.s_Spec.s_ChildIpcRoot, fixture.childRoot());
    BOOST_REQUIRE_EQUAL(result.s_Spec.s_PipePaths.size(), 4);
}

BOOST_AUTO_TEST_CASE(testNoPathOptionsIsNotOk) {
    const ml::sandbox::SChildIpcValidationResult result{
        ml::sandbox::validateChildIpcLaunchSpec("/tmp", {"--foo=bar"})};

    BOOST_TEST_REQUIRE(result.s_Ok == false);
    BOOST_TEST_REQUIRE(result.s_Spec.s_ChildId.empty());
}

BOOST_AUTO_TEST_CASE(testRejectsRelativePath) {
    CTempChildIpcFixture fixture{"child-2"};
    const ml::sandbox::SChildIpcValidationResult result{ml::sandbox::validateChildIpcLaunchSpec(
        fixture.canonicalTrustedBase(), {"--input=relative/input.fifo"})};

    BOOST_TEST_REQUIRE(result.s_Ok == false);
    BOOST_REQUIRE_EQUAL(result.s_Rejected.size(), 1);
    BOOST_REQUIRE(result.s_Rejected[0].s_Reason ==
                  ml::sandbox::EChildIpcPathRejection::E_NotAbsolute);
}

BOOST_AUTO_TEST_CASE(testRejectsRootLevelPath) {
    CTempChildIpcFixture fixture{"child-3"};
    const ml::sandbox::SChildIpcValidationResult result{ml::sandbox::validateChildIpcLaunchSpec(
        fixture.canonicalTrustedBase(), {"--input=/input.fifo"})};

    BOOST_TEST_REQUIRE(result.s_Ok == false);
    BOOST_REQUIRE_EQUAL(result.s_Rejected.size(), 1);
    BOOST_REQUIRE(result.s_Rejected[0].s_Reason ==
                  ml::sandbox::EChildIpcPathRejection::E_RootLevelPath);
}

BOOST_AUTO_TEST_CASE(testRejectsDotDotEscape) {
    CTempChildIpcFixture fixture{"child-4"};
    const std::string escapingPath{fixture.childRoot() + "/../../../etc/passwd"};
    const ml::sandbox::SChildIpcValidationResult result{ml::sandbox::validateChildIpcLaunchSpec(
        fixture.canonicalTrustedBase(), {"--input=" + escapingPath})};

    BOOST_TEST_REQUIRE(result.s_Ok == false);
    BOOST_REQUIRE_EQUAL(result.s_Rejected.size(), 1);
    BOOST_REQUIRE(result.s_Rejected[0].s_Reason ==
                  ml::sandbox::EChildIpcPathRejection::E_ContainsDotDot);
}

BOOST_AUTO_TEST_CASE(testRejectsPathOutsideTrustedBase) {
    CTempChildIpcFixture fixture{"child-5"};
    const ml::sandbox::SChildIpcValidationResult result{ml::sandbox::validateChildIpcLaunchSpec(
        fixture.canonicalTrustedBase(), {"--input=/var/tmp/not-under-tmpdir/input.fifo"})};

    BOOST_TEST_REQUIRE(result.s_Ok == false);
    BOOST_REQUIRE_EQUAL(result.s_Rejected.size(), 1);
    BOOST_REQUIRE(result.s_Rejected[0].s_Reason ==
                      ml::sandbox::EChildIpcPathRejection::E_CanonicalizationFailed ||
                  result.s_Rejected[0].s_Reason ==
                      ml::sandbox::EChildIpcPathRejection::E_OutsideTrustedBase);
}

BOOST_AUTO_TEST_CASE(testRejectsWrongDepthDirectChildOfTrustedBase) {
    CTempChildIpcFixture fixture{"child-6"};
    // Direct child of $TMPDIR (missing the ml-child-ipc intermediate
    // directory) must fail, not silently be accepted as "close enough".
    const std::string tooShallow{fixture.canonicalTrustedBase() + "/input.fifo"};

    const ml::sandbox::SChildIpcValidationResult result{ml::sandbox::validateChildIpcLaunchSpec(
        fixture.canonicalTrustedBase(), {"--input=" + tooShallow})};

    BOOST_TEST_REQUIRE(result.s_Ok == false);
    BOOST_REQUIRE_EQUAL(result.s_Rejected.size(), 1);
    BOOST_REQUIRE(result.s_Rejected[0].s_Reason == ml::sandbox::EChildIpcPathRejection::E_RootLevelPath ||
                  result.s_Rejected[0].s_Reason ==
                      ml::sandbox::EChildIpcPathRejection::E_WrongDepth);
}

BOOST_AUTO_TEST_CASE(testRejectsTooDeepNestingUnderChildId) {
    CTempChildIpcFixture fixture{"child-7"};
    const std::string nestedDir{fixture.childRoot() + "/nested"};
    BOOST_TEST_REQUIRE(::mkdir(nestedDir.c_str(), 0700) == 0);

    const ml::sandbox::SChildIpcValidationResult result{ml::sandbox::validateChildIpcLaunchSpec(
        fixture.canonicalTrustedBase(), {"--input=" + nestedDir + "/input.fifo"})};

    BOOST_TEST_REQUIRE(result.s_Ok == false);
    BOOST_REQUIRE_EQUAL(result.s_Rejected.size(), 1);
    BOOST_REQUIRE(result.s_Rejected[0].s_Reason ==
                  ml::sandbox::EChildIpcPathRejection::E_WrongDepth);

    ::rmdir(nestedDir.c_str());
}

BOOST_AUTO_TEST_CASE(testRejectsDuplicateLiteralArgument) {
    CTempChildIpcFixture fixture{"child-8"};
    const std::string arg{"--input=" + fixture.childRoot() + "/input.fifo"};

    const ml::sandbox::SChildIpcValidationResult result{ml::sandbox::validateChildIpcLaunchSpec(
        fixture.canonicalTrustedBase(), {arg, arg})};

    BOOST_TEST_REQUIRE(result.s_Ok == false);
    BOOST_REQUIRE_EQUAL(result.s_Rejected.size(), 1);
    BOOST_REQUIRE(result.s_Rejected[0].s_Reason ==
                  ml::sandbox::EChildIpcPathRejection::E_Duplicate);
}

BOOST_AUTO_TEST_CASE(testRejectsMutableSymlinkAlias) {
    CTempChildIpcFixture fixture{"child-9"};
    const std::string aliasPath{fixture.canonicalTrustedBase() + "/ml-child-ipc/child-9-alias"};
    BOOST_TEST_REQUIRE(::symlink(fixture.childRoot().c_str(), aliasPath.c_str()) == 0);

    const ml::sandbox::SChildIpcValidationResult result{ml::sandbox::validateChildIpcLaunchSpec(
        fixture.canonicalTrustedBase(), {"--input=" + aliasPath + "/input.fifo"})};

    BOOST_TEST_REQUIRE(result.s_Ok == false);
    BOOST_REQUIRE_EQUAL(result.s_Rejected.size(), 1);
    BOOST_REQUIRE(result.s_Rejected[0].s_Reason ==
                  ml::sandbox::EChildIpcPathRejection::E_MutableSymlinkOrAlias);

    ::unlink(aliasPath.c_str());
}

BOOST_AUTO_TEST_CASE(testRejectsChildIdMismatchAcrossOptions) {
    // Both children must sit under the *same* trusted base for this to
    // actually exercise E_ChildIdMismatch - two independent
    // CTempChildIpcFixture instances each mkdtemp their own unrelated base,
    // so a second-fixture path would hit E_OutsideTrustedBase/E_WrongDepth
    // first and never reach the child-id comparison at all.
    CTempChildIpcFixture fixtureA{"child-10a"};
    const std::string siblingChildRoot{fixtureA.canonicalTrustedBase() + "/ml-child-ipc/child-10b"};
    BOOST_TEST_REQUIRE(::mkdir(siblingChildRoot.c_str(), 0700) == 0);

    const ml::sandbox::SChildIpcValidationResult result{ml::sandbox::validateChildIpcLaunchSpec(
        fixtureA.canonicalTrustedBase(),
        {"--input=" + fixtureA.childRoot() + "/input.fifo",
         "--output=" + siblingChildRoot + "/output.fifo"})};

    BOOST_TEST_REQUIRE(result.s_Ok == false);
    BOOST_REQUIRE_EQUAL(result.s_Rejected.size(), 1);
    BOOST_REQUIRE(result.s_Rejected[0].s_Reason ==
                  ml::sandbox::EChildIpcPathRejection::E_ChildIdMismatch);

    ::rmdir(siblingChildRoot.c_str());
}

BOOST_AUTO_TEST_CASE(testIgnoresScalarOptionsAsCandidatePaths) {
    CTempChildIpcFixture fixture{"child-11"};
    const ml::sandbox::SChildIpcValidationResult result{ml::sandbox::validateChildIpcLaunchSpec(
        fixture.canonicalTrustedBase(), {"--input=" + fixture.childRoot() + "/input.fifo",
                                         "--modelId=../../../etc/passwd", "--inputIsPipe"})};

    BOOST_TEST_REQUIRE(result.s_Ok);
    BOOST_TEST_REQUIRE(result.s_Rejected.empty());
}

BOOST_AUTO_TEST_CASE(testRejectsEmptyValueForRecognizedPathOptionEvenAmongValidOnes) {
    CTempChildIpcFixture fixture{"child-12"};
    // "--input=" (empty value) must be rejected, not silently skipped as if
    // the option were absent - even though "--output=..." for the same
    // child is otherwise valid. A prior version of the parser treated an
    // empty value identically to a missing "=" and never reached the
    // value.empty() rejection branch below it.
    const ml::sandbox::SChildIpcValidationResult result{ml::sandbox::validateChildIpcLaunchSpec(
        fixture.canonicalTrustedBase(),
        {"--input=", "--output=" + fixture.childRoot() + "/output.fifo"})};

    BOOST_TEST_REQUIRE(result.s_Ok == false);
    BOOST_REQUIRE_EQUAL(result.s_Rejected.size(), 1);
    BOOST_REQUIRE_EQUAL(result.s_Rejected[0].s_Arg, "--input=");
    BOOST_REQUIRE(result.s_Rejected[0].s_Reason ==
                  ml::sandbox::EChildIpcPathRejection::E_NotAbsolute);
}

BOOST_AUTO_TEST_CASE(testEnsureChildIpcDirectoryCreatesMissingDirectoryBeforeValidation) {
    // Reproduces the real bug: with neither ml-child-ipc nor the per-child
    // directory created yet, validateChildIpcLaunchSpec() must fail closed
    // (realpath() has nothing to resolve) - and after
    // ensureChildIpcDirectory() runs, the exact same validation call must
    // now succeed, proving the directory-creation step is what was missing,
    // not a mis-ordering of an already-existing step.
    CTrustedBaseOnlyFixture fixture;
    const std::string childRoot{fixture.canonicalTrustedBase() + "/ml-child-ipc/child-ensure-1"};
    const std::vector<std::string> args{"--input=" + childRoot + "/input.fifo",
                                        "--output=" + childRoot + "/output.fifo"};

    const ml::sandbox::SChildIpcValidationResult before{
        ml::sandbox::validateChildIpcLaunchSpec(fixture.canonicalTrustedBase(), args)};
    BOOST_TEST_REQUIRE(before.s_Ok == false);

    const ml::sandbox::EChildIpcDirectoryOutcome outcome{
        ml::sandbox::ensureChildIpcDirectory(fixture.canonicalTrustedBase(), args)};
    BOOST_REQUIRE(outcome == ml::sandbox::EChildIpcDirectoryOutcome::E_Ready);

    struct stat childRootStat;
    BOOST_TEST_REQUIRE(::stat(childRoot.c_str(), &childRootStat) == 0);
    BOOST_REQUIRE_EQUAL(static_cast<int>(childRootStat.st_mode & 0777), 0700);

    const ml::sandbox::SChildIpcValidationResult after{
        ml::sandbox::validateChildIpcLaunchSpec(fixture.canonicalTrustedBase(), args)};
    BOOST_TEST_REQUIRE(after.s_Ok);
    BOOST_TEST_REQUIRE(after.s_Rejected.empty());
    BOOST_REQUIRE_EQUAL(after.s_Spec.s_ChildId, "child-ensure-1");
}

BOOST_AUTO_TEST_CASE(testEnsureChildIpcDirectoryIsIdempotentAcrossRetries) {
    // A retry/restart for the same child-id must not fail just because the
    // directory from the earlier attempt is still there.
    CTrustedBaseOnlyFixture fixture;
    const std::string childRoot{fixture.canonicalTrustedBase() + "/ml-child-ipc/child-ensure-1"};
    const std::vector<std::string> args{"--input=" + childRoot + "/input.fifo"};

    BOOST_REQUIRE(ml::sandbox::ensureChildIpcDirectory(fixture.canonicalTrustedBase(), args) ==
                  ml::sandbox::EChildIpcDirectoryOutcome::E_Ready);
    BOOST_REQUIRE(ml::sandbox::ensureChildIpcDirectory(fixture.canonicalTrustedBase(), args) ==
                  ml::sandbox::EChildIpcDirectoryOutcome::E_Ready);

    const ml::sandbox::SChildIpcValidationResult result{
        ml::sandbox::validateChildIpcLaunchSpec(fixture.canonicalTrustedBase(), args)};
    BOOST_TEST_REQUIRE(result.s_Ok);
}

BOOST_AUTO_TEST_CASE(testEnsureChildIpcDirectoryFailsClosedOnCreationFailure) {
    // A creation failure (here: an unwritable trusted base, standing in for
    // permissions/ENOSPC on a real host) must report E_CreationFailed - not
    // crash, and not let validateChildIpcLaunchSpec() somehow still pass.
    CTrustedBaseOnlyFixture fixture;
    BOOST_TEST_REQUIRE(::chmod(fixture.canonicalTrustedBase().c_str(), 0500) == 0);

    const std::string childRoot{fixture.canonicalTrustedBase() + "/ml-child-ipc/child-ensure-1"};
    const std::vector<std::string> args{"--input=" + childRoot + "/input.fifo"};

    const ml::sandbox::EChildIpcDirectoryOutcome outcome{
        ml::sandbox::ensureChildIpcDirectory(fixture.canonicalTrustedBase(), args)};
    BOOST_REQUIRE(outcome == ml::sandbox::EChildIpcDirectoryOutcome::E_CreationFailed);

    const ml::sandbox::SChildIpcValidationResult result{
        ml::sandbox::validateChildIpcLaunchSpec(fixture.canonicalTrustedBase(), args)};
    BOOST_TEST_REQUIRE(result.s_Ok == false);

    // Restore write permission so the fixture destructor can clean up.
    ::chmod(fixture.canonicalTrustedBase().c_str(), 0700);
}

BOOST_AUTO_TEST_CASE(testEnsureChildIpcDirectoryRejectsRegularFileInTheWay) {
    CTrustedBaseOnlyFixture fixture;
    const std::string childRoot{fixture.canonicalTrustedBase() + "/ml-child-ipc/child-ensure-1"};
    BOOST_TEST_REQUIRE(
        ::mkdir((fixture.canonicalTrustedBase() + "/ml-child-ipc").c_str(), 0700) == 0);
    FILE* file{::fopen(childRoot.c_str(), "w")};
    BOOST_TEST_REQUIRE(file != nullptr);
    ::fclose(file);

    const std::vector<std::string> args{"--input=" + childRoot + "/input.fifo"};
    BOOST_REQUIRE(ml::sandbox::ensureChildIpcDirectory(fixture.canonicalTrustedBase(), args) ==
                  ml::sandbox::EChildIpcDirectoryOutcome::E_CreationFailed);

    ::unlink(childRoot.c_str());
}

#ifndef _WIN32
BOOST_AUTO_TEST_CASE(testChildIpcDirectoryReaperRemovesEmptyPerChildDirectory) {
    CTrustedBaseOnlyFixture fixture;
    const std::string childRoot{fixture.canonicalTrustedBase() + "/ml-child-ipc/reap-empty"};
    const std::vector<std::string> args{"--input=" + childRoot + "/input.fifo"};
    BOOST_REQUIRE(ml::sandbox::ensureChildIpcDirectory(fixture.canonicalTrustedBase(), args) ==
                  ml::sandbox::EChildIpcDirectoryOutcome::E_Ready);

    ml::sandbox::CChildIpcDirectoryReaper reaper;
    reaper.noteSpawn(1001, childRoot);
    reaper.onChildExited(1001);

    struct stat childRootStat;
    BOOST_REQUIRE_EQUAL(-1, ::stat(childRoot.c_str(), &childRootStat));
    BOOST_REQUIRE_EQUAL(ENOENT, errno);

    struct stat parentStat;
    BOOST_REQUIRE_EQUAL(
        0, ::stat((fixture.canonicalTrustedBase() + "/ml-child-ipc").c_str(), &parentStat));
}

BOOST_AUTO_TEST_CASE(testChildIpcDirectoryReaperKeepsDirectoryWithUnexpectedFile) {
    CTrustedBaseOnlyFixture fixture;
    const std::string childRoot{fixture.canonicalTrustedBase() + "/ml-child-ipc/reap-nonempty"};
    const std::vector<std::string> args{"--input=" + childRoot + "/input.fifo"};
    BOOST_REQUIRE(ml::sandbox::ensureChildIpcDirectory(fixture.canonicalTrustedBase(), args) ==
                  ml::sandbox::EChildIpcDirectoryOutcome::E_Ready);
    FILE* file{::fopen((childRoot + "/leftover.txt").c_str(), "w")};
    BOOST_TEST_REQUIRE(file != nullptr);
    ::fclose(file);

    ml::sandbox::CChildIpcDirectoryReaper reaper;
    reaper.noteSpawn(1002, childRoot);
    reaper.onChildExited(1002);

    struct stat childRootStat;
    BOOST_REQUIRE_EQUAL(0, ::stat(childRoot.c_str(), &childRootStat));
    ::unlink((childRoot + "/leftover.txt").c_str());
}

BOOST_AUTO_TEST_CASE(testChildIpcDirectoryReaperPidGenerationGuard) {
    CTrustedBaseOnlyFixture fixture;
    const std::string childRoot{fixture.canonicalTrustedBase() + "/ml-child-ipc/reap-gen"};
    const std::vector<std::string> args{"--input=" + childRoot + "/input.fifo"};
    BOOST_REQUIRE(ml::sandbox::ensureChildIpcDirectory(fixture.canonicalTrustedBase(), args) ==
                  ml::sandbox::EChildIpcDirectoryOutcome::E_Ready);

    ml::sandbox::CChildIpcDirectoryReaper reaper;
    reaper.noteSpawn(2001, childRoot);
    reaper.noteSpawn(2002, childRoot);
    reaper.onChildExited(2001);

    struct stat childRootStat;
    BOOST_REQUIRE_EQUAL(0, ::stat(childRoot.c_str(), &childRootStat));

    reaper.onChildExited(2002);
    BOOST_REQUIRE_EQUAL(-1, ::stat(childRoot.c_str(), &childRootStat));
    BOOST_REQUIRE_EQUAL(ENOENT, errno);
}

BOOST_AUTO_TEST_CASE(testChildIpcDirectoryReaperOnSpawnFailedSkipsLivePid) {
    CTrustedBaseOnlyFixture fixture;
    const std::string childRoot{fixture.canonicalTrustedBase() + "/ml-child-ipc/reap-live"};
    const std::vector<std::string> args{"--input=" + childRoot + "/input.fifo"};
    BOOST_REQUIRE(ml::sandbox::ensureChildIpcDirectory(fixture.canonicalTrustedBase(), args) ==
                  ml::sandbox::EChildIpcDirectoryOutcome::E_Ready);

    ml::sandbox::CChildIpcDirectoryReaper reaper;
    reaper.noteSpawn(3001, childRoot);
    reaper.onSpawnFailed(childRoot);

    struct stat childRootStat;
    BOOST_REQUIRE_EQUAL(0, ::stat(childRoot.c_str(), &childRootStat));
    reaper.onChildExited(3001);
    BOOST_REQUIRE_EQUAL(-1, ::stat(childRoot.c_str(), &childRootStat));
}
#endif // !_WIN32

BOOST_AUTO_TEST_CASE(testEnsureChildIpcDirectoryRejectsLoosePermissionsOnExistingDirectory) {
    CTrustedBaseOnlyFixture fixture;
    const std::string mlChildIpc{fixture.canonicalTrustedBase() + "/ml-child-ipc"};
    const std::string childRoot{mlChildIpc + "/child-ensure-1"};
    BOOST_TEST_REQUIRE(::mkdir(mlChildIpc.c_str(), 0700) == 0);
    BOOST_TEST_REQUIRE(::mkdir(childRoot.c_str(), 0755) == 0);

    const std::vector<std::string> args{"--input=" + childRoot + "/input.fifo"};
    BOOST_REQUIRE(ml::sandbox::ensureChildIpcDirectory(fixture.canonicalTrustedBase(), args) ==
                  ml::sandbox::EChildIpcDirectoryOutcome::E_CreationFailed);
}

BOOST_AUTO_TEST_SUITE_END()
