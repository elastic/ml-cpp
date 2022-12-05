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
#include <core/CDetachedProcessSpawner.h>

#include <core/CCondition.h>
#include <core/CLogger.h>
#include <core/CMutex.h>
#include <core/CScopedLock.h>
#include <core/CThread.h>

#include <algorithm>
#include <set>

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

// environ is a global variable from the C runtime library
extern char** environ;

namespace {

//! Maximum number of newly opened files between calls to setupFileActions().
const int MAX_NEW_OPEN_FILES{10};

//! Attempt to close all file descriptors except the standard ones.  The
//! standard file descriptors will be reopened on /dev/null in the spawned
//! process.  Returns false and sets errno if the actions cannot be initialised
//! at all, but other errors are ignored.
bool setupFileActions(posix_spawn_file_actions_t* fileActions, int& maxFdHint) {
    if (::posix_spawn_file_actions_init(fileActions) != 0) {
        return false;
    }

    struct rlimit rlim;
    ::memset(&rlim, 0, sizeof(struct rlimit));
    if (::getrlimit(RLIMIT_NOFILE, &rlim) != 0) {
        rlim.rlim_cur = 36; // POSIX default
    }

    // Assume only a handful of new files have been opened since the last time
    // this function was called. Doing this means we learn the practical limit
    // on the number of open files, which will be a lot less than the enforced
    // limit, and avoids making masses of expensive fcntl() calls.
    int maxFdToTest{std::min(static_cast<int>(rlim.rlim_cur), maxFdHint + MAX_NEW_OPEN_FILES)};
    for (int fd = 0; fd <= maxFdToTest; ++fd) {
        if (fd == STDIN_FILENO) {
            ::posix_spawn_file_actions_addopen(fileActions, fd, "/dev/null", O_RDONLY, S_IRUSR);
            maxFdHint = fd;
        } else if (fd == STDOUT_FILENO || fd == STDERR_FILENO) {
            ::posix_spawn_file_actions_addopen(fileActions, fd, "/dev/null", O_WRONLY, S_IWUSR);
            maxFdHint = fd;
        } else {
            // Close other files that are open.  There is a race condition here,
            // in that files could be opened or closed between this code running
            // and the posix_spawn() function being called.  However, this would
            // violate the restrictions stated in the contract detailed in the
            // Doxygen description of this class.
            if (::fcntl(fd, F_GETFL) != -1) {
                ::posix_spawn_file_actions_addclose(fileActions, fd);
                maxFdHint = fd;
            }
        }
    }

    return true;
}
}

namespace ml {
namespace core {
namespace detail {

class CTrackerThread : public CThread {
public:
    using TPidSet = std::set<CProcess::TPid>;

public:
    CTrackerThread() : m_Shutdown(false), m_Condition(m_Mutex) {}

    //! Mutex is accessible so the code outside the class can avoid race
    //! conditions.
    CMutex& mutex() { return m_Mutex; }

    //! Add a PID to track.
    void addPid(CProcess::TPid pid) {
        CScopedLock lock(m_Mutex);
        m_Pids.insert(pid);
        m_Condition.signal();
    }

    bool terminatePid(CProcess::TPid pid) {
        if (!this->havePid(pid)) {
            LOG_ERROR(<< "Will not attempt to kill process " << pid << ": not a child process");
            return false;
        }

        if (::kill(pid, SIGTERM) == -1) {
            // Don't log an error if the process exited normally in between
            // checking whether it was our child process and killing it
            if (errno != ESRCH) {
                LOG_ERROR(<< "Failed to kill process " << pid << ": " << ::strerror(errno));
            } else {
                // But log at debug in case there's a bug in this area
                LOG_DEBUG(<< "No such process while trying to kill PID " << pid);
            }
            return false;
        }

        return true;
    }

    bool havePid(CProcess::TPid pid) const {
        if (pid <= 0) {
            return false;
        }

        CScopedLock lock(m_Mutex);
        // Do an extra cycle of waiting for zombies, so we give the most
        // up-to-date answer possible
        const_cast<CTrackerThread*>(this)->checkForDeadChildren();
        return m_Pids.find(pid) != m_Pids.end();
    }

protected:
    void run() override {
        CScopedLock lock(m_Mutex);

        while (!m_Shutdown) {
            // Reap zombies every 50ms if child processes are running,
            // otherwise wait for a child process to start.
            if (m_Pids.empty()) {
                m_Condition.wait();
            } else {
                m_Condition.wait(50);
            }

            this->checkForDeadChildren();
        }
    }

    void shutdown() override {
        LOG_DEBUG(<< "Shutting down spawned process tracker thread");
        CScopedLock lock(m_Mutex);
        m_Shutdown = true;
        m_Condition.signal();
    }

private:
    //! Reap zombie child processes and adjust the set of live child PIDs
    //! accordingly.  MUST be called with m_Mutex locked.
    void checkForDeadChildren() {
        int status = 0;
        for (;;) {
            CProcess::TPid pid = ::waitpid(-1, &status, WNOHANG);
            // 0 means there are child processes but none have died
            if (pid == 0) {
                break;
            }
            // -1 means error
            if (pid == -1) {
                if (errno != EINTR) {
                    break;
                }
            } else {
                if (WIFSIGNALED(status)) {
                    int signal = WTERMSIG(status);
                    if (signal == SIGTERM) {
                        // We expect this when a job is force-closed, so log
                        // at a lower level
                        LOG_INFO(<< "Child process with PID " << pid
                                 << " was terminated by signal " << signal);
                    } else {
                        // This should never happen if the system is working
                        // normally - possible reasons are the Linux OOM
                        // killer, manual intervention and bugs that cause
                        // access violations
                        LOG_ERROR(<< "Child process with PID " << pid
                                  << " was terminated by signal " << signal);
                    }
                } else {
                    int exitCode = WEXITSTATUS(status);
                    if (exitCode == 0) {
                        // This is the happy case
                        LOG_DEBUG(<< "Child process with PID " << pid << " has exited");
                    } else {
                        LOG_WARN(<< "Child process with PID " << pid
                                 << " has exited with exit code " << exitCode);
                    }
                }
                m_Pids.erase(pid);
            }
        }
    }

private:
    bool m_Shutdown;
    TPidSet m_Pids;
    mutable CMutex m_Mutex;
    CCondition m_Condition;
};
}

CDetachedProcessSpawner::CDetachedProcessSpawner(const TStrVec& permittedProcessPaths)
    : m_PermittedProcessPaths(permittedProcessPaths),
      m_TrackerThread(std::make_shared<detail::CTrackerThread>()) {
    if (m_TrackerThread->start() == false) {
        LOG_ERROR(<< "Failed to start spawned process tracker thread");
    }
}

CDetachedProcessSpawner::~CDetachedProcessSpawner() {
    if (m_TrackerThread->stop() == false) {
        LOG_ERROR(<< "Failed to stop spawned process tracker thread");
    }
}

bool CDetachedProcessSpawner::spawn(const std::string& processPath, const TStrVec& args) {
    CProcess::TPid dummy(0);
    return this->spawn(processPath, args, dummy);
}

bool CDetachedProcessSpawner::spawn(const std::string& processPath,
                                    const TStrVec& args,
                                    CProcess::TPid& childPid) {
    if (std::find(m_PermittedProcessPaths.begin(), m_PermittedProcessPaths.end(),
                  processPath) == m_PermittedProcessPaths.end()) {
        LOG_ERROR(<< "Spawning process '" << processPath << "' is not permitted");
        return false;
    }

    if (::access(processPath.c_str(), X_OK) != 0) {
        LOG_ERROR(<< "Cannot execute '" << processPath << "': " << ::strerror(errno));
        return false;
    }

    using TCharPVec = std::vector<char*>;
    // Size of argv is two bigger than the number of arguments because:
    // 1) We add the program name at the beginning
    // 2) The list of arguments must be terminated by a NULL pointer
    TCharPVec argv;
    argv.reserve(args.size() + 2);

    // These const_casts may cause const data to get modified BUT only in the
    // child post-fork, so this won't corrupt parent process data
    argv.push_back(const_cast<char*>(processPath.c_str()));
    for (size_t index = 0; index < args.size(); ++index) {
        argv.push_back(const_cast<char*>(args[index].c_str()));
    }
    argv.push_back(static_cast<char*>(nullptr));

    posix_spawn_file_actions_t fileActions;
    if (setupFileActions(&fileActions, m_MaxObservedFd) == false) {
        LOG_ERROR(<< "Failed to set up file actions prior to spawn of '"
                  << processPath << "': " << ::strerror(errno));
        return false;
    }
    posix_spawnattr_t spawnAttributes;
    if (::posix_spawnattr_init(&spawnAttributes) != 0) {
        LOG_ERROR(<< "Failed to set up spawn attributes prior to spawn of '"
                  << processPath << "': " << ::strerror(errno));
        return false;
    }
    ::posix_spawnattr_setflags(&spawnAttributes, POSIX_SPAWN_SETPGROUP);

    {
        // Hold the tracker thread mutex until the PID is added to the tracker
        // to avoid a race condition if the process is started but dies really
        // quickly
        CScopedLock lock(m_TrackerThread->mutex());

        int err(::posix_spawn(&childPid, processPath.c_str(), &fileActions,
                              &spawnAttributes, &argv[0], environ));

        ::posix_spawn_file_actions_destroy(&fileActions);
        ::posix_spawnattr_destroy(&spawnAttributes);

        if (err != 0) {
            LOG_ERROR(<< "Failed to spawn '" << processPath << "': " << ::strerror(err));
            return false;
        }

        m_TrackerThread->addPid(childPid);
    }

    LOG_DEBUG(<< "Spawned '" << processPath << "' with PID " << childPid);

    return true;
}

bool CDetachedProcessSpawner::terminateChild(CProcess::TPid pid) {
    return m_TrackerThread->terminatePid(pid);
}

bool CDetachedProcessSpawner::hasChild(CProcess::TPid pid) const {
    return m_TrackerThread->havePid(pid);
}
}
}
