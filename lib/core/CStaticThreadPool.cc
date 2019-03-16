/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CStaticThreadPool.h>

namespace ml {
namespace core {
namespace {
std::size_t computeSize(std::size_t hint) {
    std::size_t bound{std::thread::hardware_concurrency()};
    std::size_t size{bound > 0 ? std::min(hint, bound) : hint};
    return std::max(size, std::size_t{1});
}
}

CStaticThreadPool::CStaticThreadPool(std::size_t size)
    : m_Busy{false}, m_TaskQueue(computeSize(size)) {
    size = computeSize(size);
    m_Pool.reserve(size);
    for (std::size_t id = 0; id < size; ++id) {
        try {
            m_Pool.emplace_back([this, id] { this->worker(id); });
        } catch (...) {
            this->shutdown();
            throw;
        }
    }
}

CStaticThreadPool::~CStaticThreadPool() {
    this->shutdown();
}

void CStaticThreadPool::schedule(TTask&& task_) {
    m_TaskQueue.push(CWrappedTask{std::forward<TTask>(task_)});
}

bool CStaticThreadPool::busy() const {
    return m_Busy.load();
}

void CStaticThreadPool::busy(bool value) {
    m_Busy.store(value);
}

void CStaticThreadPool::shutdown() {

    // Signal to each thread that it is finished.
    for (std::size_t id = 0; id < m_Pool.size(); ++id) {
        m_TaskQueue.push(CWrappedTask{[&] { m_Done = true; }});
    }

    for (auto& thread : m_Pool) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    m_Pool.clear();
}

void CStaticThreadPool::worker(std::size_t id) {
    for (std::size_t i{id}; m_Done == false; ++i) {
        CWrappedTask task{m_TaskQueue.pop()};
        task();
        // One gets the highest throughput without yielding if the task size is
        // large enough, but the performance is unstable if it is too small (due
        // to contention on the queue when it is empty). This monitors the fraction
        // of full slots in the queue and starts yielding if there are too few.
        // This gives the feeding thread(s) the chance to replenish it before it
        // becomes empty. I tested a range of loads at which to yield and the
        // performance was pretty stable for values between 0.1 and 0.5.
        if (m_TaskQueue.load() < 0.2) {
            std::this_thread::yield();
        }
    }
}

CStaticThreadPool::CWrappedTask::CWrappedTask(TTask&& task)
    : m_Task{std::forward<TTask>(task)} {
}

void CStaticThreadPool::CWrappedTask::operator()() {
    if (m_Task != nullptr) {
        try {
            m_Task();
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Failed executing task with error '" << e.what() << "'");
        }
    }
}
}
}
