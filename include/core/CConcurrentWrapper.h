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

#ifndef INCLUDED_ml_core_CConcurrentWrapper_h
#define INCLUDED_ml_core_CConcurrentWrapper_h

#include <core/CConcurrentQueue.h>
#include <core/CNonCopyable.h>

#include <functional>
#include <thread>

namespace ml {
namespace core {

//! \brief
//! A thread safe concurrent wrapper.
//!
//! DESCRIPTION:\n
//! A thread safe concurrent wrapper based on Herb Sutters Active Object Pattern.
//!
//! See https://channel9.msdn.com/Shows/Going+Deep/C-and-Beyond-2012-Herb-Sutter-Concurrency-and-Parallelism
//!
//! @tparam T the wrapped object
template<typename T>
class CConcurrentWrapper final : private CNonCopyable {
public:
    //! Wrap and return the wrapped object
    //!
    //! The object has to wrapped once and only once, pass the reference around in your code.
    //! This starts a background thread.
    explicit CConcurrentWrapper(T& resource,
                                std::size_t queueCapacity = 100,
                                std::size_t notifyCapacity = 50)
        : m_Queue(queueCapacity, notifyCapacity), m_Resource(resource), m_Done(false) {
        m_Worker = std::thread([this] {
            while (!m_Done) {
                m_Queue.pop()();
            }
        });
    }

    ~CConcurrentWrapper() {
        m_Queue.push([this] { m_Done = true; });

        m_Worker.join();
    }

    //! Push something into the queue of the wrapped object
    //! The code inside of this lambda is guaranteed to be executed in an atomic fashion.
    template<typename F>
    void operator()(F f) const {
        m_Queue.push([=] { f(m_Resource); });
    }

    //! Debug the memory used by this component.
    void debugMemoryUsage(const CMemoryUsage::TMemoryUsagePtr& mem) const {
        mem->setName("CConcurrentWrapper");
        m_Queue.debugMemoryUsage(mem->addChild());
    }

    //! Get the memory used by this component.
    std::size_t memoryUsage() const { return m_Queue.memoryUsage(); }

private:
    //! Queue for the tasks
    mutable CConcurrentQueue<std::function<void()>> m_Queue;

    //! The wrapped resource
    T& m_Resource;

    //! thread for the worker
    std::thread m_Worker;

    //! boolean for stopping the worker
    //! never touched outside of the main thread
    bool m_Done;
};
}
}

#endif // INCLUDED_ml_core_CConcurrentWrapper_h
