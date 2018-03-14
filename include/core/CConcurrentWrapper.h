/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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

#ifndef INCLUDED_ml_core_CConcurrentWrapper_h
#define INCLUDED_ml_core_CConcurrentWrapper_h

#include <core/CConcurrentQueue.h>
#include <core/CMemory.h>
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
//! See
//! https://channel9.msdn.com/Shows/Going+Deep/C-and-Beyond-2012-Herb-Sutter-Concurrency-and-Parallelism
//!
//! @tparam T the wrapped object
//! @tparam QUEUE_CAPACITY internal queue capacity
//! @tparam NOTIFY_CAPACITY special parameter, for signaling the producer in blocking case
template <typename T, size_t QUEUE_CAPACITY = 100, size_t NOTIFY_CAPACITY = 50>
class CConcurrentWrapper final : private CNonCopyable {
public:
    //! Wrap and return the wrapped object
    //!
    //! The object has to wrapped once and only once, pass the reference around in your code.
    //! This starts a background thread.
    CConcurrentWrapper(T& resource) : m_Resource(resource), m_Done(false) {
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
    template <typename F>
    void operator()(F f) const {
        m_Queue.push([=] { f(m_Resource); });
    }

    //! Debug the memory used by this component.
    void debugMemoryUsage(CMemoryUsage::TMemoryUsagePtr mem) const {
        mem->setName("CConcurrentWrapper");
        m_Queue.debugMemoryUsage(mem->addChild());
    }

    //! Get the memory used by this component.
    std::size_t memoryUsage(void) const { return m_Queue.memoryUsage(); }

private:
    //! Queue for the tasks
    mutable CConcurrentQueue<std::function<void()>, QUEUE_CAPACITY, NOTIFY_CAPACITY> m_Queue;

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

#endif /* INCLUDED_ml_core_CConcurrentWrapper_h */
