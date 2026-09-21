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

// Purpose-built allowlisted payload for the typed filesystem/network launch
// policy's mechanism probe. Runs *inside* the sandbox under the policy built
// by buildPytorchInferenceFilesystemPolicy and prints
// one "mechanism=... outcome=..." line per check to stdout, which the
// controller-side test (CPytorchInferenceSandboxPolicyMechanismTest_Linux)
// asserts on directly - a wrong-but-still-startable policy would otherwise
// look identical to a correct one if the test only checked the exit code.
// Deliberately dependency-free, like sandbox_smoke_payload.cc: no ml-cpp
// library dependencies, no policy of its own.

#include <arpa/inet.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <linux/magic.h>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <sys/vfs.h>
#include <unistd.h>

namespace {

//! File descriptor for the results file this probe writes into the mapped
//! per-child IPC directory. The host-side test reads that file directly
//! from the *host* path after the sandbox exits - the IPC directory is
//! genuinely shared, so this doubles as the "allowed IPC access" proof and
//! as this probe's only result channel (no stdout capture plumbing exists
//! yet; that lands once a real process spawner owns pipe plumbing for
//! sandboxed children).
int g_ResultsFd = -1;

void report(const char* mechanism, const char* outcome, const std::string& detail = "") {
    std::printf("ml_sandbox_probe: mechanism=%s outcome=%s detail=%s\n",
                mechanism, outcome, detail.c_str());
    std::fflush(stdout);
    if (g_ResultsFd >= 0) {
        std::string line{std::string("mechanism=") + mechanism +
                         " outcome=" + outcome + " detail=" + detail + "\n"};
        ::write(g_ResultsFd, line.c_str(), line.size());
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: ml_sandbox_probe <child-ipc-dir>\n");
        return EXIT_FAILURE;
    }
    const std::string ipcDir{argv[1]};

    g_ResultsFd = ::open((ipcDir + "/results.txt").c_str(),
                         O_CREAT | O_WRONLY | O_TRUNC, 0600);

    std::printf("ml_sandbox_probe: reached\n");
    std::fflush(stdout);
    if (g_ResultsFd >= 0) {
        const std::string reachedLine{"reached=true\n"};
        ::write(g_ResultsFd, reachedLine.c_str(), reachedLine.size());
    }

    // Allowed IPC access (positive control): write then read back a file
    // inside the mapped per-child IPC directory.
    const std::string ipcFile{ipcDir + "/probe.txt"};
    int writeFd = ::open(ipcFile.c_str(), O_CREAT | O_WRONLY, 0600);
    if (writeFd >= 0) {
        ::write(writeFd, "probe", 5);
        ::close(writeFd);
        int readFd = ::open(ipcFile.c_str(), O_RDONLY);
        char buf[8]{};
        const bool readBack = readFd >= 0 && ::read(readFd, buf, sizeof(buf)) == 5 &&
                              std::strncmp(buf, "probe", 5) == 0;
        if (readFd >= 0) {
            ::close(readFd);
        }
        report("ipc_readwrite", readBack ? "allowed" : "denied");
    } else {
        report("ipc_readwrite", "denied", std::strerror(errno));
    }

    // Denied host read (negative control): /etc/shadow must not be
    // readable even though narrow, individually justified /etc files are
    // allowlisted (allowlistedEtcFiles()).
    int shadowFd = ::open("/etc/shadow", O_RDONLY);
    if (shadowFd < 0) {
        report("host_read_etc_shadow", "denied", std::strerror(errno));
    } else {
        ::close(shadowFd);
        report("host_read_etc_shadow", "allowed");
    }

    // Private tmpfs: writability alone doesn't prove /tmp is a private
    // tmpfs rather than a host bind - a regressed policy that AddDirectory's
    // the real host /tmp would still pass a plain write check on any
    // world-writable host. statfs()'s f_type is the actual mechanism
    // distinguishing tmpfs from a bind-mounted host directory.
    struct statfs tmpStatfs {};
    const bool isTmpfs = ::statfs("/tmp", &tmpStatfs) == 0 && tmpStatfs.f_type == TMPFS_MAGIC;
    const std::string privateTmpFile{"/tmp/ml_sandbox_probe_private_tmp_test"};
    int tmpFd = ::open(privateTmpFile.c_str(), O_CREAT | O_WRONLY, 0600);
    if (tmpFd >= 0) {
        ::close(tmpFd);
        ::unlink(privateTmpFile.c_str());
        report("private_tmpfs_write", isTmpfs ? "allowed" : "denied",
               isTmpfs ? "" : "writable but not tmpfs-backed");
    } else {
        report("private_tmpfs_write", "denied", std::strerror(errno));
    }

    // Mount enumeration conformance: /etc must list only the
    // allowlisted files, never a full directory bind.
    DIR* etcDir = ::opendir("/etc");
    if (etcDir != nullptr) {
        int entryCount = 0;
        while (::readdir(etcDir) != nullptr) {
            ++entryCount;
        }
        ::closedir(etcDir);
        report("etc_enumeration", "counted", std::to_string(entryCount));
    } else {
        report("etc_enumeration", "denied", std::strerror(errno));
    }

    // Private PID namespace: this process should be (close to) the
    // sandbox's own init, not a real-looking host PID.
    report("pid_namespace", (::getpid() <= 2) ? "namespaced" : "not_namespaced",
           std::to_string(::getpid()));

    // External egress denial (negative control): an outbound connect to
    // a guaranteed non-routable test address (TEST-NET-1, RFC 5737) must
    // fail - Sandbox2's network namespace has no route out. Using a
    // non-routable address instead of a real host keeps this check
    // hermetic and independent of network availability in CI.
    int egressSocket = ::socket(AF_INET, SOCK_STREAM, 0);
    if (egressSocket >= 0) {
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(80);
        ::inet_pton(AF_INET, "192.0.2.1", &addr.sin_addr);
        const int rc = ::connect(egressSocket, reinterpret_cast<sockaddr*>(&addr),
                                 sizeof(addr));
        const int connectErrno = errno;
        // A namespace with no route out fails synchronously with
        // ENETUNREACH/EHOSTUNREACH before any packet leaves the sandbox.
        // ECONNREFUSED would mean a packet actually reached something that
        // sent back RST - a routing leak, not isolation - so only the
        // no-route errnos count as "denied"; anything else (including
        // success) is reported "allowed" to keep that distinction visible.
        const bool denied = rc != 0 && (connectErrno == ENETUNREACH ||
                                        connectErrno == EHOSTUNREACH);
        report("external_egress", denied ? "denied" : "allowed", std::strerror(connectErrno));
        ::close(egressSocket);
    } else {
        report("external_egress", "denied", std::strerror(errno));
    }

    // Local operation success (positive control): loopback must remain
    // reachable at the network-namespace level. Connection-refused (nobody
    // listening on this port) still counts as "reachable" - only a
    // namespace-level error (e.g. ENETUNREACH) means loopback itself broke.
    int loopbackSocket = ::socket(AF_INET, SOCK_STREAM, 0);
    if (loopbackSocket >= 0) {
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(1);
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        const int rc = ::connect(loopbackSocket,
                                 reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        const bool loopbackReachable = rc == 0 || errno == ECONNREFUSED;
        report("loopback_reachable", loopbackReachable ? "ok" : "broken",
               std::strerror(errno));
        ::close(loopbackSocket);
    } else {
        report("loopback_reachable", "broken", std::strerror(errno));
    }

    std::printf("ml_sandbox_probe: done\n");
    std::fflush(stdout);
    if (g_ResultsFd >= 0) {
        ::close(g_ResultsFd);
    }
    return EXIT_SUCCESS;
}
