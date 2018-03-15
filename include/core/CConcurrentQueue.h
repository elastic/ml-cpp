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
#ifndef INCLUDED_ml_core_CConcurrentQueue_h
#define INCLUDED_ml_core_CConcurrentQueue_h

#include <core/CMemory.h>
#include <core/CNonCopyable.h>

#include <boost/circular_buffer.hpp>

#include <condition_variable>
#include <mutex>

namespace ml {
namespace core {

//! \brief
//! A thread safe concurrent queue.
//!
//! DESCRIPTION:\n
//! A thread safe concurrent queue.
//!
//! This is a simple wrapper around boost::circular_buffer
//! in order to make it thread-safe. As boost circular buffer is bounded and start overwriting
//! when the buffer gets full, the queue blocks if it reaches the capacity, so no message is lost
//! for the price of blocking.
//!
//! @tparam T the objects of the queue
//! @tparam QUEUE_CAPACITY fixed queue capacity
//! @tparam NOTIFY_CAPACITY special parameter, for signaling the producer in blocking case
template<typename T, size_t QUEUE_CAPACITY, size_t NOTIFY_CAPACITY = QUEUE_CAPACITY>
class CConcurrentQueue final : private CNonCopyable {
public:
    CConcurrentQueue(void) : m_Queue(QUEUE_CAPACITY) {
        static_assert(NOTIFY_CAPACITY > 0, "NOTIFY_CAPACITY must be positive");
        static_assert(QUEUE_CAPACITY >= NOTIFY_CAPACITY,
                      "QUEUE_CAPACITY cannot be less than NOTIFY_CAPACITY");
    }

    //! Pop an item out of the queue, this blocks until an item is available
    T pop() {
        std::unique_lock<std::mutex> lock(m_Mutex);
        while (m_Queue.empty()) {
            m_ConsumerCondition.wait(lock);
        }
        size_t oldSize = m_Queue.size();
        auto val = m_Queue.front();
        m_Queue.pop_front();

        // notification in case buffer was full
        if (oldSize >= NOTIFY_CAPACITY) {
            lock.unlock();
            m_ProducerCondition.notify_all();
        }
        return val;
    }

    //! Pop an item out of the queue, this blocks until an item is available
    void pop(T& item) {
        std::unique_lock<std::mutex> lock(m_Mutex);
        while (m_Queue.empty()) {
            m_ConsumerCondition.wait(lock);
        }

        size_t oldSize = m_Queue.size();
        item = m_Queue.front();
        m_Queue.pop_front();

        // notification in case buffer was full
        if (oldSize >= NOTIFY_CAPACITY) {
            lock.unlock();
            m_ProducerCondition.notify_all();
        }
    }

    //! Pop an item out of the queue, this blocks if the queue is full
    //! which means it can deadlock if no one consumes items (implementor's responsibility)
    void push(const T& item) {
        std::unique_lock<std::mutex> lock(m_Mutex);
        size_t pending = m_Queue.size();
        // block if buffer is full, this can deadlock if no one consumes items,
        // implementor has to take care
        while (pending >= QUEUE_CAPACITY) {
            m_ProducerCondition.wait(lock);
            pending = m_Queue.size();
        }

        m_Queue.push_back(item);

        lock.unlock();
        if (pending == 0) {
            m_ConsumerCondition.notify_all();
        }
    }

    //! Debug the memory used by this component.
    void debugMemoryUsage(CMemoryUsage::TMemoryUsagePtr mem) const {
        mem->setName("CConcurrentQueue");
        CMemoryDebug::dynamicSize("m_Queue", m_Queue, mem);
    }

    //! Get the memory used by this component.
    std::size_t memoryUsage(void) const { return CMemory::dynamicSize(m_Queue); }

    // ! Return the number of items currently in the queue
    size_t size() const { return m_Queue.size(); }

private:
    //! The internal queue
    boost::circular_buffer<T> m_Queue;

    //! Mutex
    std::mutex m_Mutex;

    //! Condition variable for consumer
    std::condition_variable m_ConsumerCondition;

    //! Condition variable for producers
    std::condition_variable m_ProducerCondition;
};
}
}

#endif /* INCLUDED_ml_core_CConcurrentQueue_h */
