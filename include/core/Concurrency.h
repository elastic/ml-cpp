/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_core_Concurrency_h
#define INCLUDED_ml_core_Concurrency_h

#include <core/ImportExport.h>

#include <transwarp/transwarp.h>

#include <functional>
#include <thread>

namespace ml {
namespace core {
using transwarp::executor;
using transwarp::task;
using static_thread_pool = transwarp::detail::thread_pool;

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

    FUNCTION s_Function;
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

//! Setup the global default executor for async.
//!
//! \note This is not thread safe as the intention is that it is invoked once at
//! the beginning of main.
//! \note If this is called with threads = 0 it defaults threads to using calling
//! std::thread::hardware_concurrency to size the thread pool.
CORE_EXPORT
void startDefaultAsyncExecutor(std::size_t threadPoolSize = 0);

//! Shutdown the thread pool and reset the executor to sequential in the same thread.
//!
//! \note This is not thread safe as the intention is that it is invoked in single
//! threaded test code.
CORE_EXPORT
void stopDefaultAsyncExecutor();

//! The default async executor.
//!
//! This gets the default parallel executor created by startDefaultAsyncExecutor.
//! If this hasn't been started execution happens serially in the same thread.
CORE_EXPORT
executor& defaultAsyncExecutor();

//! Get the default async executor thread pool size.
CORE_EXPORT
std::size_t defaultAsyncThreadPoolSize();

//! An version of std::async which uses a specified executor.
template<typename FUNCTION, typename... ARGS>
std::shared_ptr<task<std::result_of_t<std::decay_t<FUNCTION>(std::decay_t<ARGS>...)>>>
async(executor& exec, FUNCTION&& f, ARGS&&... args) {
    // Note g stores copies of the arguments in the pack, which are moved into place
    // if possible, so this is safe to invoke in the context of a scheduled task.
    auto g = std::bind(std::forward<FUNCTION>(f), std::forward<ARGS>(args)...);

    // Create and schedule the task for the supplied executor.
    auto task = transwarp::make_task(transwarp::root, "async_task", g);
    task->schedule(exec);

    return task;
}

//! Wait for \p task to finish if there is one.
template<typename RESULT>
void await(const std::shared_ptr<task<RESULT>>& task) {
    if (task != nullptr) {
        task->wait();
    }
}

//! Wait for a collection of tasks to finish.
template<typename RESULT>
void await(const std::vector<std::shared_ptr<task<RESULT>>>& tasks) {
    for (const auto& task : tasks) {
        await(task);
    }
}

//! Run \p f in parallel using async.
//!
//! This executes \p f on each index in the range [\p start, \p end) using the
//! default async executor.
//!
//! \param[in] start The first index for which to execute \p f.
//! \param[in] end The end of the indices for which to execute \p f.
//! \param[in,out] f The function to execute on each index. This expected to
//! be a Callable equivalent to std::function<void(std::size_t)>.
//! \note f must be copy constructible.
//! \note f must be thread safe.
template<typename FUNCTION>
std::vector<FUNCTION> parallel_for_each(std::size_t start, std::size_t end, FUNCTION&& f) {

    if (end <= start) {
        return {std::forward<FUNCTION>(f)};
    }

    std::size_t threads{std::min(defaultAsyncThreadPoolSize(), end - start)};

    if (threads == 0) {
        for (std::size_t i = start; i < end; ++i) {
            f(i);
        }
        return {std::forward<FUNCTION>(f)};
    }

    std::vector<FUNCTION> functions{threads, std::forward<FUNCTION>(f)};

    // Threads access the indices in the following pattern:
    //   [0, m,   2*m,   ...]
    //   [1, m+1, 2*m+1, ...]
    //   [2, m+2, 2*m+2, ...]
    //   ...
    // for m threads. This is choosen because it is expected that the index
    // will often be used to access elements in a contiguous block of memory.
    // Provided the threads are doing roughly equal work per call this should
    // ensure the best possible locality of reference for reads which occur
    // at a similar time in the different threads.

    std::vector<std::shared_ptr<task<decltype(f(0))>>> tasks;

    for (std::size_t offset = 0; offset < threads; ++offset, ++start) {
        // Note there is one copy of g for each thread so capture by reference
        // is thread safe provided f is thread safe.

        auto& g = functions[offset];
        tasks.emplace_back(async(defaultAsyncExecutor(),
                                 [&g, threads](std::size_t start_, std::size_t end_) {
                                     for (std::size_t i = start_; i < end_; i += threads) {
                                         g(i);
                                     }
                                 },
                                 start, end));
    }

    await(tasks);

    return functions;
}

//! Run \p f in parallel using async.
//!
//! This executes \p f on each dereferenced iterator value in the range
//! [\p start, \p end) using the default async executor.
//!
//! \param[in] start The first iterator for the values on which to execute \p f.
//! \param[in] end The end iterator for the values on which to execute \p f.
//! \param[in,out] f The function to execute on each value. This expected to
//! be a Callable equivalent to std::function<void (decltype(*start))>.
//! \note f must be copy constructible.
//! \note f must be thread safe.
template<typename ITR, typename FUNCTION>
std::vector<FUNCTION> parallel_for_each(ITR start, ITR end, FUNCTION&& f) {

    std::size_t size(std::distance(start, end));
    if (size == 0) {
        return {std::forward<FUNCTION>(f)};
    }

    std::size_t threads{std::min(defaultAsyncThreadPoolSize(), size)};

    if (threads == 0) {
        return {std::for_each(start, end, std::forward<FUNCTION>(f))};
    }

    std::vector<FUNCTION> functions{threads, std::forward<FUNCTION>(f)};

    // See above for the rationale for this access pattern.

    std::vector<std::shared_ptr<task<decltype(f(*start))>>> tasks;

    for (std::size_t offset = 0; offset < threads; ++offset, ++start) {
        // Note there is one copy of g for each thread so capture by reference
        // is thread safe provided f is thread safe.

        auto& g = functions[offset];
        tasks.emplace_back(async(
            defaultAsyncExecutor(),
            [&g, threads, offset, size](ITR start_) {
                std::size_t i{offset};
                auto incrementByThreads = [&i, threads, size](ITR& j) {
                    if (i < size) {
                        std::advance(j, threads);
                    }
                };
                for (ITR j = start_; i < size; i += threads, incrementByThreads(j)) {
                    g(*j);
                }
            },
            start));
    }

    await(tasks);

    return functions;
}
}
}

#endif // INCLUDED_ml_core_Concurrency_h
