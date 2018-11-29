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
    : m_Cursor{0}, m_TaskQueues{computeSize(size)} {
    m_Pool.reserve(m_TaskQueues.size());
    for (std::size_t id = 0; id < m_TaskQueues.size(); ++id) {
        try {
            m_Pool.emplace_back([&, id] { worker(id); });
        } catch (...) {
            this->shutdown();
            throw;
        }
    }
}

CStaticThreadPool::~CStaticThreadPool() {
    this->shutdown();
}

void CStaticThreadPool::schedule(TTask&& task) {
    m_TaskQueues[m_Cursor.fetch_add(1) % m_TaskQueues.size()].push(std::forward<TTask>(task));
}

void CStaticThreadPool::schedule(std::function<void()>&& f) {
    this->schedule(TTask{[g = std::move(f)] {
        g();
        return boost::any{};
}
});
}

void CStaticThreadPool::shutdown() {
    // Signal to each thread that it is finished.
    for (auto& queue : m_TaskQueues) {
        queue.push(TTask{[&] {
            m_Done = true;
            return boost::any{};
        }});
    }
    for (auto& thread : m_Pool) {
        thread.join();
    }
}

void CStaticThreadPool::worker(std::size_t id) {

    auto noThrowExecute = [](TOptionalTask& task) {
        try {
            (*task)();
        } catch (const std::future_error& e) {
            LOG_ERROR(<< "Failed executing packaged task: " << e.code() << " "
                      << e.what());
        }
    };

    while (m_Done == false) {
        // We maintain "worker count" queues and each worker has an affinity to a
        // different queue. We don't immediately block if the worker's "queue" is
        // empty because different tasks can have different duration and we could
        // assign imbalanced work to the queues. However, this arrangement means
        // if everything is working well we have essentially no contention between
        // workers on queue reads.

        TOptionalTask task;
        for (std::size_t i = 0; i < m_TaskQueues.size(); ++i) {
            task = m_TaskQueues[(id + i) % m_TaskQueues.size()].try_pop();
            if (task != boost::none) {
                break;
            }
        }
        if (task == boost::none) {
            task = m_TaskQueues[id].pop();
        }

        noThrowExecute(task);
    }

    // Drain this thread's queue before exiting.
    for (auto task = m_TaskQueues[id].try_pop(); task; task = m_TaskQueues[id].try_pop()) {
        noThrowExecute(task);
    }
}
}
}
