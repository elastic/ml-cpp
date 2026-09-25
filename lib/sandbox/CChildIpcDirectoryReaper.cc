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
#include <sandbox/CChildIpcDirectoryReaper.h>

#include <core/CLogger.h>

#ifndef _WIN32
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <cerrno>
#include <cstring>
#include <vector>

namespace {

#ifndef _WIN32
std::vector<std::string> splitPathComponents(const std::string& path) {
    std::vector<std::string> components;
    std::size_t start{0};
    while (start < path.size()) {
        const std::size_t end{path.find('/', start)};
        if (end == start) {
            start = end + 1;
            continue;
        }
        if (end == std::string::npos) {
            components.push_back(path.substr(start));
            break;
        }
        components.push_back(path.substr(start, end - start));
        start = end + 1;
    }
    return components;
}
#endif

} // namespace

namespace ml {
namespace sandbox {

bool CChildIpcDirectoryReaper::canonicalPerChildIpcRootHasExpectedShape(const std::string& canonicalChildRoot) {
#ifndef _WIN32
    if (canonicalChildRoot.empty()) {
        return false;
    }
    const std::vector<std::string> components{splitPathComponents(canonicalChildRoot)};
    if (components.size() < 2) {
        return false;
    }
    return components[components.size() - 2] == "ml-child-ipc";
#else
    return false;
#endif
}

bool CChildIpcDirectoryReaper::tryRemovePerChildIpcDirectory(const std::string& canonicalChildRoot) {
#ifndef _WIN32
    if (canonicalPerChildIpcRootHasExpectedShape(canonicalChildRoot) == false) {
        LOG_WARN(<< "Refusing to remove per-child IPC directory with unexpected shape: "
                 << canonicalChildRoot);
        return false;
    }

    static const char* const KNOWN_FIFO_LEAVES[]{"input", "output", "restore", "logPipe"};
    for (const char* leaf : KNOWN_FIFO_LEAVES) {
        const std::string fifoPath{canonicalChildRoot + '/' + leaf};
        struct stat pathStat {};
        if (::lstat(fifoPath.c_str(), &pathStat) != 0) {
            continue;
        }
        if (S_ISFIFO(pathStat.st_mode)) {
            if (::unlink(fifoPath.c_str()) != 0 && errno != ENOENT) {
                LOG_WARN(<< "Failed to unlink FIFO " << fifoPath << ": "
                         << ::strerror(errno));
            }
        }
    }

    DIR* dir{::opendir(canonicalChildRoot.c_str())};
    if (dir == nullptr) {
        return errno == ENOENT;
    }
    bool unexpectedEntry{false};
    for (;;) {
        errno = 0;
        const dirent* entry{::readdir(dir)};
        if (entry == nullptr) {
            break;
        }
        const char* name{entry->d_name};
        if (name[0] == '.' && (name[1] == '\0' || (name[1] == '.' && name[2] == '\0'))) {
            continue;
        }
        unexpectedEntry = true;
        LOG_DEBUG(<< "Leaving per-child IPC directory " << canonicalChildRoot
                  << " in place: unexpected entry '" << name << '\'');
        break;
    }
    ::closedir(dir);
    if (unexpectedEntry) {
        return false;
    }

    if (::rmdir(canonicalChildRoot.c_str()) != 0) {
        if (errno == ENOENT) {
            return true;
        }
        LOG_WARN(<< "Failed to remove per-child IPC directory "
                 << canonicalChildRoot << ": " << ::strerror(errno));
        return false;
    }
    LOG_DEBUG(<< "Removed per-child IPC directory " << canonicalChildRoot);
    return true;
#else
    return false;
#endif
}

void CChildIpcDirectoryReaper::noteSpawn(core::CProcess::TPid pid,
                                         const std::string& canonicalChildRoot) {
    if (pid <= 0 || canonicalChildRoot.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_Mutex);
    const auto previousRootIt{m_PidToRoot.find(pid)};
    if (previousRootIt != m_PidToRoot.end() && previousRootIt->second != canonicalChildRoot) {
        const auto rootOwnerIt{m_RootToPid.find(previousRootIt->second)};
        if (rootOwnerIt != m_RootToPid.end() && rootOwnerIt->second == pid) {
            m_RootToPid.erase(rootOwnerIt);
        }
    }
    m_PidToRoot[pid] = canonicalChildRoot;
    m_RootToPid[canonicalChildRoot] = pid;
}

void CChildIpcDirectoryReaper::onChildExited(core::CProcess::TPid pid) {
    if (pid <= 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_Mutex);
    const auto pidIt{m_PidToRoot.find(pid)};
    if (pidIt == m_PidToRoot.end()) {
        return;
    }
    const std::string root{pidIt->second};
    m_PidToRoot.erase(pidIt);
    const auto rootIt{m_RootToPid.find(root)};
    if (rootIt == m_RootToPid.end() || rootIt->second != pid) {
        // A newer spawn reused this deployment id; do not remove its directory.
        return;
    }
    m_RootToPid.erase(rootIt);
    tryRemovePerChildIpcDirectory(root);
}

void CChildIpcDirectoryReaper::onSpawnFailed(const std::string& canonicalChildRoot) {
    if (canonicalChildRoot.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_Mutex);
    const auto rootIt{m_RootToPid.find(canonicalChildRoot)};
    if (rootIt != m_RootToPid.end()) {
        return;
    }
    tryRemovePerChildIpcDirectory(canonicalChildRoot);
}

} // namespace sandbox
} // namespace ml
