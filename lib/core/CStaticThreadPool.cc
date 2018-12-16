/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CStaticThreadPool.h>

#include <chrono>

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
    : m_Busy{false}, m_Cursor{0}, m_TaskQueues{computeSize(size)} {
    m_Pool.reserve(m_TaskQueues.size());
    for (std::size_t id = 0; id < m_TaskQueues.size(); ++id) {
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
    // Only block if every queue is full.
    std::size_t size{m_TaskQueues.size()};
    std::size_t i{m_Cursor.load()};
    std::size_t end{i + size};
    CWrappedTask task{std::forward<TTask>(task_)};
    for (/**/; i < end; ++i) {
        if (m_TaskQueues[i % size].tryPush(std::move(task))) {
            break;
        }
    }
    if (i == end) {
        m_TaskQueues[i % size].push(std::move(task));
    }
    m_Cursor.store(i + 1);
}

void CStaticThreadPool::schedule(std::function<void()>&& f) {
    this->schedule(TTask([g = std::move(f)] {
        g();
        return boost::any{};
    }));
}

bool CStaticThreadPool::busy() const {
    return m_Busy.load();
}

void CStaticThreadPool::busy(bool value) {
    m_Busy.store(value);
}

void CStaticThreadPool::shutdown() {

    // Drain the queues before starting to shut down in order to maximise throughput.
    this->drainQueuesWithoutBlocking();

    // Signal to each thread that it is finished. We bind each task to a thread so
    // so each thread executes exactly one shutdown task.
    for (std::size_t id = 0; id < m_TaskQueues.size(); ++id) {
        TTask done{[&] {
            m_Done = true;
            return boost::any{};
        }};
        m_TaskQueues[id].push(CWrappedTask{std::move(done), id});
    }

    for (auto& thread : m_Pool) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    m_TaskQueues.clear();
    m_Pool.clear();
}

void CStaticThreadPool::worker(std::size_t id) {

    auto ifAllowed = [id](const CWrappedTask& task) {
        return task.executableOnThread(id);
    };

    TOptionalTask task;

    while (m_Done == false) {
        // We maintain "worker count" queues and each worker has an affinity to a
        // different queue. We don't immediately block if the worker's "queue" is
        // empty because different tasks can have different duration and we could
        // assign imbalanced work to the queues. However, this arrangement means
        // if everything is working well we have essentially no contention between
        // workers on queue reads.

        std::size_t size{m_TaskQueues.size()};
        for (std::size_t i = 0; i < size; ++i) {
            task = m_TaskQueues[(id + i) % size].tryPop(ifAllowed);
            if (task != boost::none) {
                break;
            }
        }
        if (task == boost::none) {
            task = m_TaskQueues[id].pop();
        }

        (*task)();

        // In the typical situation that the thread(s) adding tasks to the queues can
        // do this much faster than the threads consuming them, all queues will be full
        // and the producer(s) will be waiting to add a task as each one is consumed.
        // By switching to work on a new queue here we minimise contention between the
        // producers and consumers. Testing on bare metal (OSX) the overhead per task
        // dropped from around 2.2 microseconds to 1.5 microseconds by yielding here.
        std::this_thread::yield();
    }
}

void CStaticThreadPool::drainQueuesWithoutBlocking() {
    TOptionalTask task;
    auto popTask = [&] {
        for (auto& queue : m_TaskQueues) {
            task = queue.tryPop();
            if (task != boost::none) {
                (*task)();
                return true;
            }
        }
        return false;
    };
    while (popTask()) {
    }
}

CStaticThreadPool::CWrappedTask::CWrappedTask(TTask&& task, TOptionalSize threadId)
    : m_Task{std::forward<TTask>(task)}, m_ThreadId{threadId} {
}

bool CStaticThreadPool::CWrappedTask::executableOnThread(std::size_t id) const {
    return m_ThreadId == boost::none || *m_ThreadId == id;
}

void CStaticThreadPool::CWrappedTask::operator()() {
    try {
        m_Task();
    } catch (const std::future_error& e) {
        LOG_ERROR(<< "Failed executing packaged task: '" << e.code() << "' "
                  << "with error '" << e.what() << "'");
    }
}
}
}
