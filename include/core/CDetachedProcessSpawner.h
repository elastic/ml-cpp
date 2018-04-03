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
#ifndef INCLUDED_ml_core_CDetachedProcessSpawner_h
#define INCLUDED_ml_core_CDetachedProcessSpawner_h

#include <core/CProcess.h>
#include <core/ImportExport.h>

#include <boost/shared_ptr.hpp>

#include <string>
#include <vector>


namespace ml
{
namespace core
{
namespace detail
{
class CTrackerThread;
}


//! \brief
//! Spawn detached processes.
//!
//! DESCRIPTION:\n
//! Spawns processes that do not need to communicate with the parent
//! process once they're running.  (If you need to spawn processes
//! that communicate with the parent process, look at the core::CPOpen
//! class.)
//!
//! A list of permitted processes must be supplied to the constructor.
//! Requests to start processes not on this list will fail.
//!
//! On Windows, only .exe files may be spawned.
//!
//! Can also kill processes that it started.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Closes parent process file descriptors in the spawned children.
//!
//! There are a couple of limitations with the way this is done on *nix:
//! 1) File descriptors above a million are not closed - the assumption
//!    is that the process running this code will not run with such a high
//!    ulimit, or that if it is such high numbered files will not be open
//!    at the time it is started and that the process itself will not have
//!    opened them.
//! 2) It is assumed that no files will be opened or closed within the
//!    process running this code while the spawn() method of this class is
//!    running.  In reality, this means that this class cannot be used in
//!    an arbitrary multi-threaded process.  It can only be used in a
//!    process dedicated to spawning processes and where this restriction
//!    can be managed.
//!
//! To enforce the constraint that only processes started by this class
//! can be killed, a lookup of all the spawned process IDs is kept, and
//! a thread tracks deaths of spawned processes to keep this lookup
//! up-to-date.  (If we simply kept track of spawned process IDs but
//! didn't adjust for deaths of processes that exited without being
//! killed by this class then after a few days there'd be many obsolete
//! entires in the lookup, and this could represent a security risk
//! given how operating systems recycle process IDs.)
//!
class CORE_EXPORT CDetachedProcessSpawner
{
    public:
        using TStrVec = std::vector<std::string>;

        using TTrackerThreadP = boost::shared_ptr<detail::CTrackerThread>;

    public:
        //! Permitted paths may be relative or absolute, but each process must
        //! be invoked using the exact path supplied.  For example, if
        //! /usr/bin/grep is permitted then you cannot invoke it as ./grep
        //! while the current working directory is /usr/bin.  On Windows,
        //! the supplied names should NOT have the .exe extension.
        CDetachedProcessSpawner(const TStrVec &permittedProcessPaths);

        ~CDetachedProcessSpawner();

        //! Spawn a process.  Returns true on success or false on error,
        //! however, it is important to realise that if the spawned process
        //! itself crashes this will not be detected as a failure by this
        //! method.  On Windows, the supplied process path should NOT have the
        //! .exe extension.
        bool spawn(const std::string &processPath, const TStrVec &args);

        //! As above, but, on success, returns the PID of the process that was
        //! started.
        bool spawn(const std::string &processPath,
                   const TStrVec &args,
                   CProcess::TPid &childPid);

        //! Kill the child process with the specified PID.  If there is a
        //! process running with the specified PID that was not spawned by this
        //! object then it will NOT be killed.
        bool terminateChild(CProcess::TPid pid);

        //! Returns true if this object spawned a process with the given PID
        //! that is still running.
        bool hasChild(CProcess::TPid pid) const;

    private:
        //! Paths to processes that may be spawned.
        TStrVec         m_PermittedProcessPaths;

        //! Thread to track which processes that have been created are still
        //! alive.
        TTrackerThreadP m_TrackerThread;
};


}
}

#endif // INCLUDED_ml_core_CDetachedProcessSpawner_h

