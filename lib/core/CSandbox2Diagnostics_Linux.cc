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
#include <core/CSandbox2Diagnostics.h>

#include <core/CLogger.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/vfs.h>
#include <unistd.h>

namespace ml {
namespace core {

namespace {

const int FORKSERVER_TMPDIR_WARN_CHARS{80};
const int FORKSERVER_SOCKET_PATH_LIMIT_CHARS{108};

std::string readProcSysValue(const char* path) {
    std::ifstream stream(path);
    std::string value;
    if (stream.good()) {
        std::getline(stream, value);
    }
    return value;
}

bool pathIsWritable(const char* path) {
    return ::access(path, W_OK) == 0;
}

bool pathHasNoexecFlag(const char* path) {
    struct statfs mountInfo;
    if (::statfs(path, &mountInfo) != 0) {
        return false;
    }
    return (mountInfo.f_flags & MS_NOEXEC) != 0;
}

std::string describeTmpdir(const char* tmpdir) {
    if (tmpdir == nullptr) {
        return "unset";
    }
    std::string description{"len="};
    description += std::to_string(::strlen(tmpdir));
    description += ", value=";
    description += tmpdir;
    description += ", forkserver_override_needed=";
    description += (::strlen(tmpdir) > FORKSERVER_TMPDIR_WARN_CHARS) ? "yes" : "no";
    return description;
}

} // namespace

void logSandbox2EnvironmentSelfCheck() {
    static bool logged{false};
    if (logged) {
        return;
    }
    logged = true;

    std::string usernsStatus;
    const std::string usernsClone{readProcSysValue("/proc/sys/kernel/unprivileged_userns_clone")};
    if (usernsClone.empty()) {
        usernsStatus = "unprivileged_userns_clone=absent (assume available)";
    } else {
        usernsStatus = "unprivileged_userns_clone=" + usernsClone;
    }

    const std::string maxUserNamespaces{readProcSysValue("/proc/sys/user/max_user_namespaces")};
    if (!maxUserNamespaces.empty()) {
        usernsStatus += ", max_user_namespaces=" + maxUserNamespaces;
    }

    const bool tmpWritable{pathIsWritable("/tmp")};
    const bool tmpNoexec{pathHasNoexecFlag("/tmp")};

    const char* tmpdir{::getenv("TMPDIR")};

    LOG_INFO(<< "Sandbox2 environment self-check: " << usernsStatus
             << ", /tmp writable=" << (tmpWritable ? "yes" : "no")
             << ", /tmp noexec=" << (tmpNoexec ? "yes" : "no")
             << ", forkserver_socket_path_limit=" << FORKSERVER_SOCKET_PATH_LIMIT_CHARS
             << ", TMPDIR " << describeTmpdir(tmpdir));
}

} // namespace core
} // namespace ml
