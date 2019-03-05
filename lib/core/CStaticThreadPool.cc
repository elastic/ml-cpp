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

    // For many small tasks the best strategy for minimising contention between the
    // producers and consumers is to 1) not yield, 2) set the cursor to add tasks on
    // the queue for the thread on which a task is popped. This gives about twice the
    // throughput of this strategy. However, if there are a small number of large
    // tasks they must be added to different queues to ensure the work is parallelised.
    // For a general purpose thread pool we must avoid that pathology. If we need a
    // better pool for very fine grained threading we'd be better investing in a
    // somewhat lock free bounded queue: for a fixed memory buffer it is possible to
    // safely add and remove elements at different ends of a queue of length greater
    // than one.
    m_Cursor.store(i + 1);
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
        TTask done{[&] { m_Done = true; }};
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
    std::size_t size{m_TaskQueues.size()};

    while (m_Done == false) {
        // We maintain "worker count" queues and each worker has an affinity to a
        // different queue. We don't immediately block if the worker's "queue" is
        // empty because different tasks can have different duration and we could
        // assign imbalanced work to the queues. However, this arrangement means
        // if everything is working well we have essentially no contention between
        // workers on queue reads.

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
        // dropped by around 120% by yielding here.
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
