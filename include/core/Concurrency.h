/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_core_Concurrency_h
#define INCLUDED_ml_core_Concurrency_h

#include <core/CLogger.h>
#include <core/CLoopProgress.h>
#include <core/ImportExport.h>

#include <boost/any.hpp>

#include <algorithm>
#include <functional>
#include <future>
#include <thread>
#include <type_traits>

namespace ml {
namespace core {
namespace concurrency_detail {
template<typename FUNCTION, typename BOUND_STATE>
struct SCallableWithBoundState {
    SCallableWithBoundState(FUNCTION&& function, BOUND_STATE&& functionState)
        : s_Function{std::forward<FUNCTION>(function)},
          s_FunctionState{std::forward<BOUND_STATE>(functionState)} {}
    template<typename... ARGS>
    void operator()(ARGS&&... args) const {
        s_Function(s_FunctionState, std::forward<ARGS>(args)...);
    }

    mutable FUNCTION s_Function;
    mutable BOUND_STATE s_FunctionState;
};
}

//! This provides a utility to bind **retrievable** state by copy to an arbitary
//! Callable.
//!
//! The intention of this class is to extend lambdas to be stateful with the state
//! held by **value** and accessible from the resulting Callable type. This can be
//! used with parallel_for_each when one wants a copy of some state for each thread
//! which can be retrieved from the returned Callable objects. The canonical usage
//! is as follows:
//! \code{.cpp}
//! std::vector<unsigned> foo;
//! ...
//! auto results = parallel_for_each(foo.begin(), foo.end(),
//!                                  bindRetrievableState([](unsigned& max, unsigned value) {
//!                                                           max = std::max(max, value); },
//!                                                       unsigned{0}));
//! unsigned overallMax{0};
//! for (const auto& result : results) {
//!     overallMax = std::max(overallMax, result.s_State);
//! }
//! \endcode
//!
//! The state can be any type and is always supplied as the first argument to the
//! Callable it binds to. The copy acted on by the thread is accessed as a public
//! member called s_State of the returned function object. The state object must be
//! copy constructible and movable.
template<typename FUNCTION, typename STATE>
auto bindRetrievableState(FUNCTION&& function, STATE&& state) {
    return concurrency_detail::SCallableWithBoundState<FUNCTION, STATE>(
        std::forward<FUNCTION>(function), std::forward<STATE>(state));
}

//! \brief The base executor hierarchy.
class CExecutor {
public:
    virtual ~CExecutor() = default;
    virtual void schedule(std::packaged_task<boost::any()>&& f) = 0;
    virtual bool busy() const = 0;
    virtual void busy(bool value) = 0;
};

//! Setup the global default executor for async.
//!
//! \note This is not thread safe as the intention is that it is invoked once,
//! usually at the beginning of main or in single threaded test code.
//! \note If this is called with threads equal to zero it defaults to calling
//! std::thread::hardware_concurrency to size the thread pool.
CORE_EXPORT
void startDefaultAsyncExecutor(std::size_t threadPoolSize = 0);

//! Shutdown the thread pool and reset the executor to sequential in the same thread.
//!
//! \note This is not thread safe as the intention is that it is invoked in single
//! threaded test code
CORE_EXPORT
void stopDefaultAsyncExecutor();

//! The default async executor.
//!
//! This gets the default parallel executor created by startDefaultAsyncExecutor.
//! If this hasn't been started execution happens serially in the same thread.
CORE_EXPORT
CExecutor& defaultAsyncExecutor();

//! Get the default async executor thread pool size.
CORE_EXPORT
std::size_t defaultAsyncThreadPoolSize();

namespace concurrency_detail {
template<typename F>
boost::any resultToAny(F& f, const std::false_type&) {
    return boost::any{f()};
}
template<typename F>
boost::any resultToAny(F& f, const std::true_type&) {
    f();
    return boost::any{};
}

template<typename R>
class CTypedFutureAnyWrapper {
public:
    CTypedFutureAnyWrapper() = default;
    CTypedFutureAnyWrapper(std::future<boost::any>&& future)
        : m_Future{std::forward<std::future<boost::any>>(future)} {}

    bool valid() const { return m_Future.valid(); }
    void wait() const { m_Future.wait(); }
    R get() { return boost::any_cast<R>(m_Future.get()); }

private:
    std::future<boost::any> m_Future;
};
}

template<typename R>
using future = concurrency_detail::CTypedFutureAnyWrapper<R>;

//! An version of std::async which uses a specified executor.
//!
//! \note f must be copy constructible.
//! \note f must be thread safe.
//! \note If f throws this will throw.
//! \warning This behaves differently from std::async which can lead to deadlocks
//! in situations where launching a new thread would not. For example, if you're
//! using the defaultAsyncExecutor you need to be careful calling async nested as
//! it is easy to create a deadlock if tasks wait on other tasks enqueued after
//! them. Prefer using high level primitives, such as parallel_for_each, which are
//! safer.
template<typename FUNCTION, typename... ARGS>
future<std::result_of_t<std::decay_t<FUNCTION>(std::decay_t<ARGS>...)>>
async(CExecutor& executor, FUNCTION&& f, ARGS&&... args) {
    using R = std::result_of_t<std::decay_t<FUNCTION>(std::decay_t<ARGS>...)>;

    // Note g stores copies of the arguments in the pack, which are moved into place
    // if possible, so this is safe to invoke later in the context of a packaged task.
    auto g = std::bind<R>(std::forward<FUNCTION>(f), std::forward<ARGS>(args)...);

    std::packaged_task<boost::any()> task([g_ = std::move(g)]() mutable {
        return concurrency_detail::resultToAny(g_, std::is_same<R, void>{});
    });
    auto result = task.get_future();

    // Schedule the task to compute the result.
    executor.schedule(std::move(task));

    return std::move(result);
}

//! Wait for all \p futures to be available.
template<typename T>
void wait_for_all(const std::vector<future<T>>& futures) {
    std::for_each(futures.begin(), futures.end(),
                  [](const future<T>& future) { future.wait(); });
}

//! Wait for a valid future to be available otherwise return immediately.
template<typename T>
void wait_for_valid(const future<T>& future) {
    if (future.valid()) {
        future.wait();
    }
}

//! Wait for all valid \p futures to be available.
template<typename T>
void wait_for_all_valid(const std::vector<future<T>>& futures) {
    std::for_each(futures.begin(), futures.end(), wait_for_valid);
}

//! \brief Waits for a future to complete when the object is destroyed.
template<typename T>
class CWaitIfValidWhenExitingScope {
public:
    CWaitIfValidWhenExitingScope(future<T>& future) : m_Future{future} {}
    ~CWaitIfValidWhenExitingScope() { wait_for_valid(m_Future); }
    CWaitIfValidWhenExitingScope(const CWaitIfValidWhenExitingScope&) = delete;
    CWaitIfValidWhenExitingScope& operator=(const CWaitIfValidWhenExitingScope&) = delete;

private:
    future<T>& m_Future;
};

//! Get the conjunction of all \p futures.
CORE_EXPORT
bool get_conjunction_of_all(std::vector<future<bool>>& futures);

namespace concurrency_detail {
class CORE_EXPORT CDefaultAsyncExecutorBusyForScope {
public:
    CDefaultAsyncExecutorBusyForScope();
    ~CDefaultAsyncExecutorBusyForScope();
    bool wasBusy() const;

private:
    bool m_WasBusy;
};

CORE_EXPORT
void noop(double);
}

//! Run \p f in parallel using async.
//!
//! This executes \p f on each index in the range [\p start, \p end) using the
//! default async executor.
//!
//! \param[in] partitions The number of tasks into which to partition the range.
//! \param[in] start The first index for which to execute \p f.
//! \param[in] end The end of the indices for which to execute \p f.
//! \param[in,out] f The function to execute on each index. This expected to
//! be a Callable equivalent to std::function<void(std::size_t)>.
//! \note f must be copy constructible.
//! \note f must be thread safe.
//! \note If f throws this will throw.
template<typename FUNCTION>
std::vector<FUNCTION>
parallel_for_each(std::size_t partitions,
                  std::size_t start,
                  std::size_t end,
                  FUNCTION&& f,
                  const std::function<void(double)>& recordProgress = concurrency_detail::noop) {

    if (end <= start) {
        recordProgress(1.0);
        return {std::forward<FUNCTION>(f)};
    }

    partitions = std::min(partitions, end - start);

    // This function is not re-entrant without the chance of deadlock if we always
    // execute in parallel. In particular, we wait on the completion of tasks which
    // are enqueued in the global thread pool. However, if this has already been
    // called further up the stack we will enqueue functions which will wait on the
    // results of these and will block the tasks they need to complete behind them
    // in the queues. There are a couple strategies we could take, two obvious ones
    // in order of increasing complexity and decreasing pessimism are:
    //   1) Execute sequentially if there are any tasks in the thread pool, we then
    //      won't wait here and nothing will block in the queue,
    //   2) Assign tasks a priority and ensure in the queue that tasks are execute
    //      in priority order. Then we can assign these tasks "highest priority in
    //      queue" + 1.
    //
    // This implements 1) because it is significantly simpler to and in practice we
    // probably don't actually need to nest parallel_for_each. We're just interested
    // in guarding against accidental deadlock in the case someone inadvertantly calls
    // parallelised code in the function to execute.

    concurrency_detail::CDefaultAsyncExecutorBusyForScope scope{};

    if (partitions < 2 || scope.wasBusy()) {
        CLoopProgress progress{end - start, recordProgress};
        for (std::size_t i = start; i < end; ++i, progress.increment()) {
            f(i);
        }
        return {std::forward<FUNCTION>(f)};
    }

    std::vector<FUNCTION> functions{partitions, std::forward<FUNCTION>(f)};

    // Threads access the indices in the following pattern:
    //   [0, m,   2*m,   ...]
    //   [1, m+1, 2*m+1, ...]
    //   [2, m+2, 2*m+2, ...]
    //   ...
    // for m partitions. This is choosen because it is expected that the index
    // will often be used to access elements in a contiguous block of memory.
    // Provided the threads are doing roughly equal work per call this should
    // ensure the best possible locality of reference for reads which occur
    // at a similar time in the different threads.

    std::vector<future<bool>> tasks;

    for (std::size_t offset = 0; offset < partitions; ++offset, ++start) {
        // Note there is one copy of g for each thread so capture by reference
        // is thread safe provided f is thread safe.

        CLoopProgress progress{end - offset, recordProgress,
                               1.0 / static_cast<double>(partitions)};

        auto& g = functions[offset];
        tasks.emplace_back(
            async(defaultAsyncExecutor(),
                  [&g, partitions, progress](std::size_t start_, std::size_t end_) mutable {
                      for (std::size_t i = start_; i < end_;
                           i += partitions, progress.increment(partitions)) {
                          g(i);
                      }
                      return true; // So we can check for exceptions via get.
                  },
                  start, end));
    }

    get_conjunction_of_all(tasks);

    return functions;
}

//! Overload with number of threads partitions.
template<typename FUNCTION>
std::vector<FUNCTION>
parallel_for_each(std::size_t start,
                  std::size_t end,
                  FUNCTION&& f,
                  const std::function<void(double)>& recordProgress = concurrency_detail::noop) {
    return parallel_for_each(defaultAsyncThreadPoolSize(), start, end,
                             std::forward<FUNCTION>(f), recordProgress);
}

//! Run \p f in parallel using async.
//!
//! This executes \p f on each dereferenced iterator value in the range
//! [\p start, \p end) using the default async executor.
//!
//! \param[in] partitions The number of tasks into which to partition the range.
//! \param[in] start The first iterator for the values on which to execute \p f.
//! \param[in] end The end iterator for the values on which to execute \p f.
//! \param[in,out] f The function to execute on each value. This expected to
//! be a Callable equivalent to std::function<void (decltype(*start))>.
//! \note f must be copy constructible.
//! \note f must be thread safe.
//! \note If f throws this will throw.
template<typename ITR, typename FUNCTION>
std::vector<FUNCTION>
parallel_for_each(std::size_t partitions,
                  ITR start,
                  ITR end,
                  FUNCTION&& f,
                  const std::function<void(double)>& recordProgress = concurrency_detail::noop) {

    std::size_t size(std::distance(start, end));
    if (size == 0) {
        recordProgress(1.0);
        return {std::forward<FUNCTION>(f)};
    }

    partitions = std::min(partitions, size);

    // See above for details.

    concurrency_detail::CDefaultAsyncExecutorBusyForScope scope{};

    if (partitions < 2 || scope.wasBusy()) {

        CLoopProgress progress{size, recordProgress};
        for (ITR i = start; i != end; ++i, progress.increment()) {
            f(*i);
        }
        return {std::forward<FUNCTION>(f)};
    }

    std::vector<FUNCTION> functions{partitions, std::forward<FUNCTION>(f)};

    // See above for the rationale for this access pattern.

    std::vector<future<bool>> tasks;

    for (std::size_t offset = 0; offset < partitions; ++offset, ++start) {
        // Note there is one copy of g for each thread so capture by reference
        // is thread safe provided f is thread safe.

        CLoopProgress progress{size - offset, recordProgress,
                               1.0 / static_cast<double>(partitions)};

        auto& g = functions[offset];
        tasks.emplace_back(async(
            defaultAsyncExecutor(),
            [&g, partitions, offset, size, progress](ITR start_) mutable {

                std::size_t i{offset};

                auto incrementByPartitions = [&i, partitions, size](ITR& j) {
                    if (i < size) {
                        std::advance(j, partitions);
                    }
                };

                for (ITR j = start_; i < size; i += partitions,
                         incrementByPartitions(j), progress.increment(partitions)) {
                    g(*j);
                }
                return true; // So we can check for exceptions via get.
            },
            start));
    }

    get_conjunction_of_all(tasks);

    return functions;
}

//! Overload with number of threads partitions.
template<typename ITR, typename FUNCTION>
std::vector<FUNCTION>
parallel_for_each(ITR start,
                  ITR end,
                  FUNCTION&& f,
                  const std::function<void(double)>& recordProgress = concurrency_detail::noop) {
    return parallel_for_each(defaultAsyncThreadPoolSize(), start, end,
                             std::forward<FUNCTION>(f), recordProgress);
}
}
}

#endif // INCLUDED_ml_core_Concurrency_h
