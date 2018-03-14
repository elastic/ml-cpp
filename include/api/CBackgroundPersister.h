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
#ifndef INCLUDED_ml_api_CBackgroundPersister_h
#define INCLUDED_ml_api_CBackgroundPersister_h

#include <core/CDataAdder.h>
#include <core/CFastMutex.h>
#include <core/CNonCopyable.h>
#include <core/CThread.h>
#include <core/CoreTypes.h>

#include <api/ImportExport.h>

#include <functional>
#include <list>

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
//! IMPLEMENTATION DECISIONS:\n
//! This class expects to call a persistence function taking
//! just the data adder as an argument.  It's easy to wrap up
//! extra data to be passed to a function that requires more by
//! using a boost:bind.  boost::bind copies its arguments, which
//! is generally what is required for access in a separate
//! thread.  However, note that a couple of copies are made, so
//! if the bound data is very large then binding a
//! boost::shared_ptr may be more appropriate than binding
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
    CBackgroundPersister(
        core_t::TTime periodicPersistInterval,
        const TFirstProcessorPeriodicPersistFunc& firstProcessorPeriodicPersistFunc,
        core::CDataAdder& dataAdder);

    ~CBackgroundPersister(void);

    //! Is background persistence currently in progress?
    bool isBusy(void) const;

    //! Wait for any background persistence currently in progress to
    //! complete
    bool waitForIdle(void);

    //! Add a function to be called when the background persist is started.
    //! This will be rejected if a background persistence is currently in
    //! progress.  It is likely that the supplied \p persistFunc will have
    //! data bound into it that will be used by the function it calls, i.e. the
    //! called function will take more arguments than just the data adder.
    //! \return true if the function was added; false if not.
    bool addPersistFunc(core::CDataAdder::TPersistFunc persistFunc);

    //! When this function is called a background persistence will be
    //! triggered unless there is already one in progress.
    bool startPersist(void);

    //! Clear any persistence functions that have been added but not yet
    //! invoked.  This will be rejected if a background persistence is
    //! currently in progress.
    //! \return true if the list of functions is clear; false if not.
    bool clear(void);

    //! Set the first processor persist function, which is used to start the
    //! chain of background persistence.  This will be rejected if a
    //! background persistence is currently in progress.
    bool firstProcessorPeriodicPersistFunc(
        const TFirstProcessorPeriodicPersistFunc& firstProcessorPeriodicPersistFunc);

    //! Check whether a background persist is appropriate now, and if it is
    //! then start it by calling the first processor periodic persist
    //! function.
    bool startBackgroundPersistIfAppropriate(void);

private:
    //! Implementation of the background thread
    class CBackgroundThread : public core::CThread {
    public:
        CBackgroundThread(CBackgroundPersister& owner);

    protected:
        //! Inherited virtual interface
        virtual void run(void);
        virtual void shutdown(void);

    private:
        //! Reference to the owning background persister
        CBackgroundPersister& m_Owner;
    };

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
    volatile bool m_IsBusy;

    //! Have we been told to shut down?
    volatile bool m_IsShutdown;

    using TPersistFuncList = std::list<core::CDataAdder::TPersistFunc>;

    //! Function to call in the background thread to do persistence.
    TPersistFuncList m_PersistFuncs;

    //! Thread used to do the background work
    CBackgroundThread m_BackgroundThread;

    // Allow the background thread to access the member variables of the owning
    // object
    friend class CBackgroundThread;
};
}
}

#endif // INCLUDED_ml_api_CBackgroundPersister_h
