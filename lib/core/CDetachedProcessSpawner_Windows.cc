/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#include <core/CDetachedProcessSpawner.h>

#include <core/CCondition.h>
#include <core/CLogger.h>
#include <core/CMutex.h>
#include <core/CScopedLock.h>
#include <core/CShellArgQuoter.h>
#include <core/CThread.h>
#include <core/CWindowsError.h>
#include <core/WindowsSafe.h>

#include <boost/make_shared.hpp>

#include <map>


namespace ml
{
namespace core
{
namespace detail
{

class CTrackerThread : public CThread
{
    public:
        typedef std::map<CProcess::TPid, HANDLE> TPidHandleMap;

    public:
        CTrackerThread(void)
            : m_Shutdown(false),
              m_Condition(m_Mutex)
        {
        }

        virtual ~CTrackerThread(void)
        {
            // Close the handles to any child processes that outlived us
            CScopedLock lock(m_Mutex);

            for (const auto &entry : m_Pids)
            {
                CloseHandle(entry.second);
            }
        }

        //! Mutex is accessible so the code outside the class can avoid race
        //! conditions.
        CMutex &mutex(void)
        {
            return m_Mutex;
        }

        //! Add a PID to track, together with its corresponding process handle.
        void addPid(CProcess::TPid pid, HANDLE processHandle)
        {
            CScopedLock lock(m_Mutex);
            m_Pids.insert({ pid, processHandle });
            m_Condition.signal();
        }

        bool terminatePid(CProcess::TPid pid)
        {
            HANDLE handle = this->handleForPid(pid);
            if (handle == INVALID_HANDLE_VALUE)
            {
                LOG_ERROR("Will not attempt to kill process " << pid <<
                          ": not a child process");
                return false;
            }

            UINT exitCode = 0;
            if (TerminateProcess(handle, exitCode) == FALSE)
            {
                LOG_ERROR("Failed to kill process " << pid << ": " <<
                          CWindowsError());
                return false;
            }

            return true;
        }

        //! Given a process ID, return the corresponding process handle.
        HANDLE handleForPid(CProcess::TPid pid) const
        {
            if (pid == 0)
            {
                return INVALID_HANDLE_VALUE;
            }

            CScopedLock lock(m_Mutex);
            // Do an extra cycle of waiting for zombies, so we give the most
            // up-to-date answer possible
            const_cast<CTrackerThread *>(this)->checkForDeadChildren();
            auto iter = m_Pids.find(pid);
            return iter == m_Pids.end() ? INVALID_HANDLE_VALUE : iter->second;
        }

    protected:
        virtual void run(void)
        {
            CScopedLock lock(m_Mutex);

            while (!m_Shutdown)
            {
                // Reap zombies every 50ms if child processes are running,
                // otherwise wait for a child process to start.
                if (m_Pids.empty())
                {
                    m_Condition.wait();
                }
                else
                {
                    m_Condition.wait(50);
                }

                this->checkForDeadChildren();
            }
        }

        virtual void shutdown(void)
        {
            LOG_DEBUG("Shutting down spawned process tracker thread");
            CScopedLock lock(m_Mutex);
            m_Shutdown = true;
            m_Condition.signal();
        }

    private:
        //! Reap zombie child processes and adjust the set of live child PIDs
        //! accordingly.  MUST be called with m_Mutex locked.
        void checkForDeadChildren(void)
        {
            auto iter = m_Pids.begin();
            while (iter != m_Pids.end())
            {
                // The reason for using WaitForSingleObject() here instead of
                // WaitForMultipleObjects() (which would avoid the need to wait
                // on a condition variable above) is that the latter function
                // can only wait for 64 objects simultaneously.  We could easily
                // have more child processes than this, so it would lead to code
                // complexity and headaches getting test coverage to use
                // WaitForMultipleObjects().
                HANDLE processHandle = iter->second;
                if (WaitForSingleObject(processHandle, 0) == WAIT_OBJECT_0)
                {
                    CloseHandle(processHandle);
                    iter = m_Pids.erase(iter);
                }
                else
                {
                    ++iter;
                }
            }
        }

    private:
        bool           m_Shutdown;
        TPidHandleMap  m_Pids;
        mutable CMutex m_Mutex;
        CCondition     m_Condition;
};

}


CDetachedProcessSpawner::CDetachedProcessSpawner(const TStrVec &permittedProcessPaths)
    : m_PermittedProcessPaths(permittedProcessPaths),
      m_TrackerThread(boost::make_shared<detail::CTrackerThread>())
{
    if (m_TrackerThread->start() == false)
    {
        LOG_ERROR("Failed to start spawned process tracker thread");
    }
}

CDetachedProcessSpawner::~CDetachedProcessSpawner(void)
{
    if (m_TrackerThread->stop() == false)
    {
        LOG_ERROR("Failed to stop spawned process tracker thread");
    }
}

bool CDetachedProcessSpawner::spawn(const std::string &processPath, const TStrVec &args)
{
    CProcess::TPid dummy(0);
    return this->spawn(processPath, args, dummy);
}

bool CDetachedProcessSpawner::spawn(const std::string &processPath,
                                    const TStrVec &args,
                                    CProcess::TPid &childPid)
{
    if (std::find(m_PermittedProcessPaths.begin(),
                  m_PermittedProcessPaths.end(),
                  processPath) == m_PermittedProcessPaths.end())
    {
        LOG_ERROR("Spawning process '" << processPath << "' is not permitted");
        return false;
    }

    bool processPathHasExeExt(processPath.length() > 4 &&
                              processPath.compare(processPath.length() - 4, 4, ".exe") == 0);

    // Windows takes command lines as a single string
    std::string cmdLine(CShellArgQuoter::quote(processPath));
    for (size_t index = 0; index < args.size(); ++index)
    {
        cmdLine += ' ';
        cmdLine += CShellArgQuoter::quote(args[index]);
    }

    STARTUPINFO startupInfo;
    ::memset(&startupInfo, 0, sizeof(STARTUPINFO));
    startupInfo.cb = sizeof(STARTUPINFO);

    PROCESS_INFORMATION processInformation;
    ::memset(&processInformation, 0, sizeof(PROCESS_INFORMATION));

    {
        // Hold the tracker thread mutex until the PID is added to the tracker
        // to avoid a race condition if the process is started but dies really
        // quickly
        CScopedLock lock(m_TrackerThread->mutex());

        if (CreateProcess((processPathHasExeExt ? processPath : processPath + ".exe").c_str(),
                          const_cast<char *>(cmdLine.c_str()),
                          0,
                          0,
                          FALSE,
                          CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS,
                          0,
                          0,
                          &startupInfo,
                          &processInformation) == FALSE)
        {
            LOG_ERROR("Failed to spawn '" << processPath << "': " << CWindowsError());
            return false;
        }

        childPid = GetProcessId(processInformation.hProcess);
        m_TrackerThread->addPid(childPid, processInformation.hProcess);
    }

    LOG_DEBUG("Spawned '" << processPath << "' with PID " << childPid);

    CloseHandle(processInformation.hThread);

    return true;
}

bool CDetachedProcessSpawner::terminateChild(CProcess::TPid pid)
{
    return m_TrackerThread->terminatePid(pid);
}

bool CDetachedProcessSpawner::hasChild(CProcess::TPid pid) const
{
    return m_TrackerThread->handleForPid(pid) != INVALID_HANDLE_VALUE;
}


}
}

