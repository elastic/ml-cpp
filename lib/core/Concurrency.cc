/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/Concurrency.h>

#include <core/CLogger.h>
#include <core/CStaticThreadPool.h>

#include <memory>
#include <numeric>
#include <thread>

namespace ml {
namespace core {
namespace {
//! \brief Executes a function immediately (on the calling thread).
class CImmediateExecutor final : public CExecutor {
public:
    void schedule(std::function<void()>&& f) override { f(); }
    bool busy() const override { return false; }
    void busy(bool) override {}
};

//! \brief Executes a function in a thread pool.
class CThreadPoolExecutor final : public CExecutor {
public:
    explicit CThreadPoolExecutor(std::size_t size) : m_ThreadPool{size} {}

    void schedule(std::function<void()>&& f) override {
        m_ThreadPool.schedule(std::forward<std::function<void()>>(f));
    }
    bool busy() const override { return m_ThreadPool.busy(); }
    void busy(bool value) override { return m_ThreadPool.busy(value); }

private:
    CStaticThreadPool m_ThreadPool;
};

class CExecutorHolder {
public:
    CExecutorHolder()
        : m_ThreadPoolSize{0}, m_Executor(std::make_unique<CImmediateExecutor>()) {}

    static CExecutorHolder makeThreadPool(std::size_t threadPoolSize) {
        if (threadPoolSize == 0) {
            threadPoolSize = std::thread::hardware_concurrency();
        }

        if (threadPoolSize > 0) {
            try {
                return CExecutorHolder{threadPoolSize};
            } catch (const std::exception& e) {
                LOG_ERROR(<< "Failed to create thread pool with '" << e.what()
                          << "'. Falling back to running single threaded");
            }
        } else {
            LOG_ERROR(<< "Failed to determine hardware concurrency and no "
                      << "thread count provided. Falling back to running "
                      << "single threaded");
        }

        // We'll just have to execute in the main thread.
        return CExecutorHolder{};
    }

    CExecutor& get() const { return *m_Executor; }
    std::size_t threadPoolSize() const { return m_ThreadPoolSize; }

private:
    CExecutorHolder(std::size_t threadPoolSize)
        : m_ThreadPoolSize{threadPoolSize},
          m_Executor(std::make_unique<CThreadPoolExecutor>(threadPoolSize)) {}

private:
    std::size_t m_ThreadPoolSize;
    std::unique_ptr<CExecutor> m_Executor;
};

CExecutorHolder singletonExecutor;
}

void startDefaultAsyncExecutor(std::size_t threadPoolSize) {
    // This is purposely not thread safe. This is only meant to be called once from
    // the main thread, typically from main of an executable or in single threaded
    // test code.
    singletonExecutor = CExecutorHolder::makeThreadPool(threadPoolSize);
}

void stopDefaultAsyncExecutor() {
    // This is purposely not thread safe. It is only meant to be called from single
    // threaded test code.
    singletonExecutor = CExecutorHolder{};
}

std::size_t defaultAsyncThreadPoolSize() {
    return singletonExecutor.threadPoolSize();
}

CExecutor& defaultAsyncExecutor() {
    return singletonExecutor.get();
}

bool get_conjunction_of_all(std::vector<std::future<bool>>& futures) {

    // This waits until results are present. If we get an exception we still want
    // to wait until all results are ready in case continuing destroys state access
    // which a worker thread reads. We just rethrow the _last_ exception we received.
    std::exception_ptr e;
    bool result{std::accumulate(futures.begin(), futures.end(), true,
                                [&e](bool conjunction, std::future<bool>& future) {
                                    try {
                                        // Don't shortcircuit
                                        bool value = future.get();
                                        conjunction &= value;
                                    } catch (...) {
                                        e = std::current_exception();
                                    }
                                    return conjunction;
                                })};

    if (e != nullptr) {
        std::rethrow_exception(e);
    }

    return result;
}

namespace concurrency_detail {
CDefaultAsyncExecutorBusyForScope::CDefaultAsyncExecutorBusyForScope()
    : m_WasBusy{defaultAsyncExecutor().busy()} {
    if (m_WasBusy == false) {
        defaultAsyncExecutor().busy(true);
    }
}

CDefaultAsyncExecutorBusyForScope::~CDefaultAsyncExecutorBusyForScope() {
    if (m_WasBusy == false) {
        defaultAsyncExecutor().busy(false);
    }
}

bool CDefaultAsyncExecutorBusyForScope::wasBusy() const {
    return m_WasBusy;
}

void noop(double) {
}
}
}
}
