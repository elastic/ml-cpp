/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CBackgroundPersister_h
#define INCLUDED_ml_api_CBackgroundPersister_h

#include <core/AtomicTypes.h>
#include <core/CDataAdder.h>
#include <core/CFastMutex.h>
#include <core/CNonCopyable.h>
#include <core/CThread.h>
#include <core/CoreTypes.h>

#include <api/ImportExport.h>

#include <functional>
#include <list>

class CBackgroundPersisterTest;

namespace ml {
namespace api {

//! \brief
//! Enables a data adder to run in a different thread.
//!
//! DESCRIPTION:\n
//! A wrapper around core::CThread to hide the gory details of
//! running a data adder in the background.
//!
//! Only one background persistence may run at any time.  This
//! is partly to avoid clashes in whatever external data store
//! state is being persisted to, and partly because the chances
//! are that a lot of memory is being used by the temporary
//! copy of the data to be persisted.
//!
//! Persistence happens in a background thread and further
//! persistence requests via the startBackgroundPersist*() methods
//! will be rejected if the background thread is executing.
//! However, calls to startBackgroundPersist() and
//! startBackgroundPersistIfAppropriate() are not thread safe and
//! must not be made concurrently.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This class expects to call a persistence function taking
//! just the data adder as an argument.  It's easy to wrap up
//! extra data to be passed to a function that requires more by
//! using a boost:bind.  boost::bind copies its arguments, which
//! is generally what is required for access in a separate
//! thread.  However, note that a couple of copies are made, so
//! if the bound data is very large then binding a
//! std::shared_ptr may be more appropriate than binding
//! values.
//!
//! A data adder must be supplied to the constructor, and, since
//! this is held by reference it must outlive this object.  If
//! the data adder is not thread safe then it may not be used by
//! any other object until after this object is destroyed.
//!
class API_EXPORT CBackgroundPersister : private core::CNonCopyable {
public:
    using TFirstProcessorPeriodicPersistFunc = std::function<bool(CBackgroundPersister&)>;

public:
    //! The supplied data adder must outlive this object.  If the data
    //! adder is not thread safe then it may not be used by any other
    //! object until after this object is destroyed.  When using this
    //! constructor the first processor persistence function must be
    //! set before the object is used.
    CBackgroundPersister(core_t::TTime periodicPersistInterval, core::CDataAdder& dataAdder);

    //! As above, but also supply the first processor persistence
    //! function at construction time.
    CBackgroundPersister(core_t::TTime periodicPersistInterval,
                         const TFirstProcessorPeriodicPersistFunc& firstProcessorPeriodicPersistFunc,
                         core::CDataAdder& dataAdder);

    ~CBackgroundPersister();

    //! Is background persistence currently in progress?
    bool isBusy() const;

    //! Wait for any background persistence currently in progress to
    //! complete
    bool waitForIdle();

    //! Add a function to be called when the background persist is started.
    //! This will be rejected if a background persistence is currently in
    //! progress.  It is likely that the supplied \p persistFunc will have
    //! data bound into it that will be used by the function it calls, i.e. the
    //! called function will take more arguments than just the data adder.
    //! \return true if the function was added; false if not.
    bool addPersistFunc(core::CDataAdder::TPersistFunc persistFunc);

    //! Set the first processor persist function, which is used to start the
    //! chain of background persistence.  This will be rejected if a
    //! background persistence is currently in progress.
    //! This should be set once before startBackgroundPersistIfAppropriate is
    //! called.
    bool firstProcessorPeriodicPersistFunc(const TFirstProcessorPeriodicPersistFunc& firstProcessorPeriodicPersistFunc);

    //! Start a background persist is one is not running.
    //! Calls the first processor periodic persist function first.
    //! Concurrent calls to this method are not threadsafe.
    bool startBackgroundPersist();

    //! If the periodic persist interval has passed since the last persist
    //! then it is appropriate to persist now.  Start it by calling the
    //! first processor periodic persist function.
    //! Concurrent calls to this method are not threadsafe.
    bool startBackgroundPersistIfAppropriate();

private:
    //! Implementation of the background thread
    class CBackgroundThread : public core::CThread {
    public:
        CBackgroundThread(CBackgroundPersister& owner);

    protected:
        //! Inherited virtual interface
        virtual void run();
        virtual void shutdown();

    private:
        //! Reference to the owning background persister
        CBackgroundPersister& m_Owner;
    };

private:
    //! Persist in the background setting the last persist time
    //! to timeOfPersistence
    bool startBackgroundPersist(core_t::TTime timeOfPersistence);

    //! When this function is called a background persistence will be
    //! triggered unless there is already one in progress.
    bool startPersist();

    //! Clear any persistence functions that have been added but not yet
    //! invoked.  This will be rejected if a background persistence is
    //! currently in progress.
    //! \return true if the list of functions is clear; false if not.
    bool clear();

private:
    //! How frequently should background persistence be attempted?
    core_t::TTime m_PeriodicPersistInterval;

    //! What was the wall clock time when we started our last periodic
    //! persistence?
    core_t::TTime m_LastPeriodicPersistTime;

    //! The function that will be called to start the chain of background
    //! persistence.
    TFirstProcessorPeriodicPersistFunc m_FirstProcessorPeriodicPersistFunc;

    //! Reference to the data adder to be used by the background thread.
    //! The data adder refered to must outlive this object.  If the data
    //! adder is not thread safe then it may not be used by any other
    //! object until after this object is destroyed.
    core::CDataAdder& m_DataAdder;

    //! Mutex to ensure atomicity of operations where required.
    core::CFastMutex m_Mutex;

    //! Is the background thread currently busy persisting data?
    atomic_t::atomic_bool m_IsBusy;

    //! Have we been told to shut down?
    atomic_t::atomic_bool m_IsShutdown;

    using TPersistFuncList = std::list<core::CDataAdder::TPersistFunc>;

    //! Function to call in the background thread to do persistence.
    TPersistFuncList m_PersistFuncs;

    //! Thread used to do the background work
    CBackgroundThread m_BackgroundThread;

    // Allow the background thread to access the member variables of the owning
    // object
    friend class CBackgroundThread;

    // For testing
    friend class ::CBackgroundPersisterTest;
};
}
}

#endif // INCLUDED_ml_api_CBackgroundPersister_h
