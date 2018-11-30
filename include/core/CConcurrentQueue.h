/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CConcurrentQueue_h
#define INCLUDED_ml_core_CConcurrentQueue_h

#include <core/CMemory.h>
#include <core/CNonCopyable.h>

#include <boost/circular_buffer.hpp>
#include <boost/optional.hpp>

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
    using TOptional = boost::optional<T>;

public:
    CConcurrentQueue() : m_Queue(QUEUE_CAPACITY) {
        static_assert(NOTIFY_CAPACITY > 0, "NOTIFY_CAPACITY must be positive");
        static_assert(QUEUE_CAPACITY >= NOTIFY_CAPACITY,
                      "QUEUE_CAPACITY cannot be less than NOTIFY_CAPACITY");
    }

    //! Pop an item out of the queue, this blocks until an item is available
    T pop() {
        std::unique_lock<std::mutex> lock(m_Mutex);
        this->waitWhileEmpty(lock);

        size_t oldSize{m_Queue.size()};
        T result{std::move(m_Queue.front())};
        m_Queue.pop_front();

        this->notifyIfNoLongerFull(lock, oldSize);
        return result;
    }

    //! Pop an item out of the queue, this returns none if an item isn't available
    TOptional tryPop() {
        std::unique_lock<std::mutex> lock(m_Mutex);
        if (m_Queue.empty()) {
            return boost::none;
        }

        size_t oldSize{m_Queue.size()};
        TOptional result{std::move(m_Queue.front())};
        m_Queue.pop_front();

        this->notifyIfNoLongerFull(lock, oldSize);
        return result;
    }

    //! Push a copy of \p item onto the queue, this blocks if the queue is full which
    //! means it can deadlock if no one consumes items (implementor's responsibility)
    void push(const T& item) {
        std::unique_lock<std::mutex> lock(m_Mutex);
        std::size_t pending{m_Queue.size()};
        this->waitWhileFull(lock, pending);

        m_Queue.push_back(item);

        this->notifyIfNoLongerEmpty(lock, pending);
    }

    //! Forward \p item to the queue, this blocks if the queue is full which means
    //! it can deadlock if no one consumes items (implementor's responsibility)
    void push(T&& item) {
        std::unique_lock<std::mutex> lock(m_Mutex);
        size_t pending{m_Queue.size()};
        this->waitWhileFull(lock, pending);

        m_Queue.push_back(std::forward<T>(item));

        this->notifyIfNoLongerEmpty(lock, pending);
    }

    //! Forward \p item to the queue, if the queue is full this fails and returns false
    bool tryPush(T&& item) {
        std::unique_lock<std::mutex> lock(m_Mutex);
        std::size_t pending{m_Queue.size()};
        if (pending == QUEUE_CAPACITY) {
            return false;
        }

        m_Queue.push_back(std::forward<T>(item));

        this->notifyIfNoLongerEmpty(lock, pending);
        return true;
    }

    //! Debug the memory used by this component.
    void debugMemoryUsage(CMemoryUsage::TMemoryUsagePtr mem) const {
        mem->setName("CConcurrentQueue");
        CMemoryDebug::dynamicSize("m_Queue", m_Queue, mem);
    }

    //! Get the memory used by this component.
    std::size_t memoryUsage() const { return CMemory::dynamicSize(m_Queue); }

    //! Return the number of items currently in the queue
    size_t size() const { return m_Queue.size(); }

private:
    void waitWhileEmpty(std::unique_lock<std::mutex>& lock) {
        while (m_Queue.empty()) {
            m_ConsumerCondition.wait(lock);
        }
    }
    void waitWhileFull(std::unique_lock<std::mutex>& lock, std::size_t& pending) {
        while (pending >= QUEUE_CAPACITY) {
            m_ProducerCondition.wait(lock);
            pending = m_Queue.size();
        }
    }

    void notifyIfNoLongerFull(std::unique_lock<std::mutex>& lock, std::size_t oldSize) {
        lock.unlock();
        if (oldSize >= NOTIFY_CAPACITY) {
            m_ProducerCondition.notify_all();
        }
    }
    void notifyIfNoLongerEmpty(std::unique_lock<std::mutex>& lock, std::size_t pending) {
        lock.unlock();
        if (pending == 0) {
            m_ConsumerCondition.notify_all();
        }
    }

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
