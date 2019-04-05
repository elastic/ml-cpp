/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CConcurrentQueue_h
#define INCLUDED_ml_core_CConcurrentQueue_h

#include <core/CMemory.h>
#include <core/CNonCopyable.h>

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <mutex>

namespace ml {
namespace core {

//! \brief A (mostly) non-blocking thread safe multi-producer multi-consumer
//! bounded queue.
//!
//! DESCRIPTION:\n
//! This blocks when pushing to a full queue and popping from an empty queue.
//! However, locks are only taken under an atomic check that the slot being
//! written (or read) is full (or empty) so provided consumers keep up with
//! producers or vice versa this doesn't block.
//!
//! It's the caller's responsibility to ensure that number of items pushed is
//! equal to number of items popped or this will deadlock.
//!
//! The supplied queue size is rounded up to the next power of 2 so modulo can
//! be implemented with a bit mask.
//!
//! The pushEmplace method forwards the constructor arguments and constructs
//! the object in-place. Example usage,
//! \code
//! CConcurrentQueue<std::pair<double, double>> queue;
//! queue.pushEmplace(0.0, 1.0);
//! \endcode
//!
//! \tparam T the type of the objects of the queue.
//! \tparam CAPACITY the target queue capacity.
template<typename T, std::size_t CAPACITY>
class CConcurrentQueue final {
public:
    CConcurrentQueue(std::size_t capacityMultiplier = 1)
        : m_Capacity{nextPow2(CAPACITY * capacityMultiplier)}, m_Front{0}, m_Back{0},
          m_Elements(m_Capacity) {}

    CConcurrentQueue(const CConcurrentQueue& other) = delete;
    CConcurrentQueue(CConcurrentQueue&& other) = delete;
    CConcurrentQueue& operator=(const CConcurrentQueue& other) = delete;
    CConcurrentQueue& operator=(CConcurrentQueue&& other) = delete;

    double load() const {
        // This is only a ballpark estimate, since we can't simultaneously
        // atomically read both the front and back, so we only ensure the
        // reads are atomic (and not that they'll see all writes).
        std::size_t front{m_Front.load(std::memory_order_relaxed)};
        std::size_t back{m_Back.load(std::memory_order_relaxed)};
        return static_cast<double>(back < front ? back + m_Capacity - front : back - front) /
               static_cast<double>(m_Capacity);
    }

    void push(T value) {
        std::size_t next{m_Back.fetch_add(1, std::memory_order_release)};
        m_Elements[next & (m_Capacity - 1)].push(std::move(value));
    }

    template<typename... ARGS>
    void pushEmplace(ARGS&&... args) {
        std::size_t next{m_Back.fetch_add(1, std::memory_order_release)};
        m_Elements[next & (m_Capacity - 1)].push(std::forward<ARGS>(args)...);
    }

    T pop() {
        std::size_t next{m_Front.fetch_add(1, std::memory_order_release)};
        return m_Elements[next & (m_Capacity - 1)].pop();
    }

    //! Debug the memory used by this component.
    void debugMemoryUsage(CMemoryUsage::TMemoryUsagePtr mem) const {
        mem->setName("CConcurrentQueue");
        CMemoryDebug::dynamicSize("m_Elements", m_Elements, mem);
    }

    //! Get the memory used by this component.
    std::size_t memoryUsage() const { return CMemory::dynamicSize(m_Elements); }

private:
    // We insert pads at various points to avoid independent atomic operations
    // thrashing the cache.
    using TCacheLine = char[64];

    class CElement {
    public:
        static const std::uint8_t READ;
        static const std::uint8_t WRITTEN;
        static const std::uint8_t READING;
        static const std::uint8_t WRITING;

        CElement() : value{}, m_State{READ} {}
        CElement(const CElement&) = delete;
        CElement(CElement&&) = delete;
        CElement& operator=(const CElement&) = delete;
        CElement& operator=(CElement&& other) = delete;

        template<typename... ARGS>
        void push(ARGS&&... args) {
            m_ProducersWaiting.fetch_add(1, std::memory_order_release);

            while (true) {
                std::uint8_t state{READ};
                if (this->swapIfStateIs(state, WRITING)) {
                    break;
                } else if (state != READING) {
                    std::unique_lock<std::mutex> lock{m_Mutex};
                    state = READ;
                    if (this->swapIfStateIs(state, WRITING)) {
                        break;
                    }
                    m_ProducerCondition.wait(lock);
                }
            }

            m_ProducersWaiting.fetch_sub(1, std::memory_order_release);
            value = toQueueType(std::forward<ARGS>(args)...);
            m_State.store(WRITTEN, std::memory_order_release);

            if (m_ConsumersWaiting.load(std::memory_order_acquire) > 0) {
                std::unique_lock<std::mutex> lock{m_Mutex};
                m_ConsumerCondition.notify_one();
            }
        }

        T pop() {
            m_ConsumersWaiting.fetch_add(1, std::memory_order_release);

            while (true) {
                std::uint8_t state{WRITTEN};
                if (this->swapIfStateIs(state, READING)) {
                    break;
                } else if (state != WRITING) {
                    std::unique_lock<std::mutex> lock{m_Mutex};
                    state = WRITTEN;
                    if (this->swapIfStateIs(state, READING)) {
                        break;
                    }
                    m_ConsumerCondition.wait(lock);
                }
            }

            m_ConsumersWaiting.fetch_sub(1, std::memory_order_release);
            T result = std::move(value);
            m_State.store(READ, std::memory_order_release);

            if (m_ProducersWaiting.load(std::memory_order_acquire) > 0) {
                std::unique_lock<std::mutex> lock{m_Mutex};
                m_ProducerCondition.notify_one();
            }

            return result;
        }

    private:
        bool swapIfStateIs(std::uint8_t& oldState, std::uint8_t newState) {
            return m_State.compare_exchange_weak(oldState, newState, // if in old state
                                                 std::memory_order_release,
                                                 std::memory_order_relaxed);
        }

        static T&& toQueueType(T&& t) { return std::forward<T>(t); }

        template<typename... ARGS>
        static T toQueueType(ARGS&&... args) {
            return T{std::forward<ARGS>(args)...};
        }

    private:
        T value;
        std::atomic<std::uint8_t> m_State;
        std::atomic_size_t m_ConsumersWaiting;
        std::atomic_size_t m_ProducersWaiting;
        std::mutex m_Mutex;
        std::condition_variable m_ConsumerCondition;
        std::condition_variable m_ProducerCondition;
        TCacheLine m_Pad;
    };

private:
    static std::size_t nextPow2(std::size_t n) {
        return 1 << static_cast<std::size_t>(std::ceil(std::log2(static_cast<double>(n))));
    }

private:
    std::size_t m_Capacity;
    std::atomic_size_t m_Front;
    TCacheLine m_Pad0;
    std::atomic_size_t m_Back;
    TCacheLine m_Pad1;
    std::vector<CElement> m_Elements;
};

template<typename T, std::size_t CAPACITY>
const std::uint8_t CConcurrentQueue<T, CAPACITY>::CElement::READ{0};
template<typename T, std::size_t CAPACITY>
const std::uint8_t CConcurrentQueue<T, CAPACITY>::CElement::WRITTEN{1};
template<typename T, std::size_t CAPACITY>
const std::uint8_t CConcurrentQueue<T, CAPACITY>::CElement::READING{2};
template<typename T, std::size_t CAPACITY>
const std::uint8_t CConcurrentQueue<T, CAPACITY>::CElement::WRITING{3};
}
}

#endif // INCLUDED_ml_core_CConcurrentQueue_h
