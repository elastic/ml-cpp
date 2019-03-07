/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CStaticThreadPool_h
#define INCLUDED_ml_core_CStaticThreadPool_h

#include <core/CConcurrentQueue.h>
#include <core/Concurrency.h>
#include <core/ImportExport.h>

#include <boost/optional.hpp>

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

namespace ml {
namespace core {

//! \brief A minimal fixed size thread pool for implementing CThreadPoolExecutor.
//!
//! IMPLEMENTATION:\n
//! This purposely has very limited interface and is intended to mainly support
//! CThreadPoolExecutor which provides the mechanism by which we expose the thread
//! pool to the rest of the code via calls core::async.
class CORE_EXPORT CStaticThreadPool {
public:
    using TTask = std::function<void()>;

public:
    explicit CStaticThreadPool(std::size_t size);

    ~CStaticThreadPool();

    CStaticThreadPool(const CStaticThreadPool&) = delete;
    CStaticThreadPool(CStaticThreadPool&&) = delete;
    CStaticThreadPool& operator=(const CStaticThreadPool&) = delete;
    CStaticThreadPool& operator=(CStaticThreadPool&&) = delete;

    //! Schedule a Callable type to be executed by a thread in the pool.
    //!
    //! \note This forwards the task to the queue.
    //! \note This can block (if the task queues are full). This is intentional
    //! and is suitable for our use case where we don't need to guaranty that this
    //! always returns immediately and instead want to exert back pressure on the
    //! thread scheduling tasks if the pool can't keep up.
    void schedule(TTask&& task);

    //! Check if the thread pool has been marked as busy.
    bool busy() const;

    //! Check if the thread pool has been marked as busy.
    void busy(bool busy);

private:
    using TOptionalSize = boost::optional<std::size_t>;
    class CWrappedTask {
    public:
        explicit CWrappedTask(TTask&& task, TOptionalSize threadId = boost::none);

        bool executableOnThread(std::size_t id) const;
        void operator()();

    private:
        TTask m_Task;
        TOptionalSize m_ThreadId;
    };
    using TOptionalTask = boost::optional<CWrappedTask>;
    using TWrappedTaskQueue = CConcurrentQueue<CWrappedTask, 50>;
    using TWrappedTaskQueueVec = std::vector<TWrappedTaskQueue>;
    using TThreadVec = std::vector<std::thread>;

private:
    void shutdown();
    void worker(std::size_t id);
    void drainQueuesWithoutBlocking();

private:
    // This doesn't have to be atomic because it is always only set to true,
    // always set straight before it is checked on each worker in the pool
    // and tearing can't happen for single byte writes.
    bool m_Done = false;
    std::atomic_bool m_Busy;
    std::atomic<std::uint64_t> m_Cursor;
    TWrappedTaskQueueVec m_TaskQueues;
    TThreadVec m_Pool;
};
}
}

#endif // INCLUDED_ml_core_CStaticThreadPool_h
