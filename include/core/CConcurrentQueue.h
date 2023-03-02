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
#ifndef INCLUDED_ml_core_CConcurrentQueue_h
#define INCLUDED_ml_core_CConcurrentQueue_h

#include <core/CMemoryDec.h>

#include <boost/circular_buffer.hpp>

#include <condition_variable>
#include <mutex>
#include <optional>

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
template<typename T>
class CConcurrentQueue final {
public:
    using TOptional = std::optional<T>;

public:
    explicit CConcurrentQueue(std::size_t queueCapacity, std::size_t notifyCapacity)
        : m_QueueCapacity(queueCapacity), m_NotifyCapacity(notifyCapacity),
          m_Queue(queueCapacity) {}

    explicit CConcurrentQueue(std::size_t queueCapacity)
        : CConcurrentQueue(queueCapacity, queueCapacity) {}

    CConcurrentQueue(const CConcurrentQueue&) = delete;
    CConcurrentQueue& operator=(const CConcurrentQueue&) = delete;
    // The use of std::mutex prevents this class from being moveable
    CConcurrentQueue(CConcurrentQueue&&) = delete;
    CConcurrentQueue& operator=(CConcurrentQueue&&) = delete;

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
    //! or the pop isn't allowed
    template<typename PREDICATE>
    TOptional tryPop(PREDICATE allowed) {
        std::unique_lock<std::mutex> lock(m_Mutex);
        if (m_Queue.empty() || allowed(m_Queue.front()) == false) {
            return std::nullopt;
        }

        size_t oldSize{m_Queue.size()};
        TOptional result{std::move(m_Queue.front())};
        m_Queue.pop_front();

        this->notifyIfNoLongerFull(lock, oldSize);
        return result;
    }

    //! Pop an item out of the queue, this returns none if an item isn't available
    TOptional tryPop() { return this->tryPop(always); }

    //! Push a copy of \p item onto the queue, this blocks if the queue is full which
    //! means it can deadlock if no one consumes items (implementor's responsibility)
    void push(const T& item) {
        std::unique_lock<std::mutex> lock(m_Mutex);
        this->waitWhileFull(lock);

        std::size_t oldSize{m_Queue.size()};
        m_Queue.push_back(item);

        this->notifyIfNoLongerEmpty(lock, oldSize);
    }

    //! Forward \p item to the queue, this blocks if the queue is full which means
    //! it can deadlock if no one consumes items (implementor's responsibility)
    void push(T&& item) {
        std::unique_lock<std::mutex> lock(m_Mutex);
        this->waitWhileFull(lock);

        size_t oldSize{m_Queue.size()};
        m_Queue.push_back(std::forward<T>(item));

        this->notifyIfNoLongerEmpty(lock, oldSize);
    }

    //! Forward \p item to the queue, if the queue is full this fails and returns false
    bool tryPush(T&& item) {
        std::unique_lock<std::mutex> lock(m_Mutex);
        if (m_Queue.size() >= m_QueueCapacity) {
            return false;
        }

        size_t oldSize{m_Queue.size()};
        m_Queue.push_back(std::forward<T>(item));

        this->notifyIfNoLongerEmpty(lock, oldSize);
        return true;
    }

    //! Debug the memory used by this component.
    void debugMemoryUsage(const CMemoryUsage::TMemoryUsagePtr& mem) const {
        mem->setName("CConcurrentQueue");
        memory_debug::dynamicSize("m_Queue", m_Queue, mem);
    }

    //! Get the memory used by this component.
    std::size_t memoryUsage() const { return memory::dynamicSize(m_Queue); }

    //! Return the number of items currently in the queue
    size_t size() {
        std::unique_lock<std::mutex> lock(m_Mutex);
        return m_Queue.size();
    }

private:
    void waitWhileEmpty(std::unique_lock<std::mutex>& lock) {
        m_ConsumerCondition.wait(lock, [this] { return m_Queue.size() > 0; });
    }
    void waitWhileFull(std::unique_lock<std::mutex>& lock) {
        m_ProducerCondition.wait(
            lock, [this] { return m_Queue.size() < m_QueueCapacity; });
    }

    void notifyIfNoLongerFull(std::unique_lock<std::mutex>& lock, std::size_t oldSize) {
        lock.unlock();
        if (oldSize >= m_NotifyCapacity) {
            m_ProducerCondition.notify_all();
        }
    }
    void notifyIfNoLongerEmpty(std::unique_lock<std::mutex>& lock, std::size_t oldSize) {
        lock.unlock();
        if (oldSize == 0) {
            m_ConsumerCondition.notify_all();
        }
    }

    static bool always(const T&) { return true; }

private:
    //! Fixed queue capacity
    std::size_t m_QueueCapacity;

    //! For signaling the producer in blocking case
    std::size_t m_NotifyCapacity;

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
