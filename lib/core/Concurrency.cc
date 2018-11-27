/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/Concurrency.h>

#include <core/CLogger.h>

#include <boost/make_unique.hpp>

#include <memory>
#include <thread>

namespace ml {
namespace core {
namespace {
class CExecutorHolder {
public:
    CExecutorHolder()
        : m_ThreadPoolSize{0}, m_Executor{boost::make_unique<transwarp::sequential>()} {}

    static CExecutorHolder makeThreadPool(std::size_t threadPoolSize) {
        if (threadPoolSize == 0) {
            threadPoolSize = std::thread::hardware_concurrency();
        }

        if (threadPoolSize > 0) {
            try {
                return CExecutorHolder{threadPoolSize};
            } catch (const std::exception& e) {
                LOG_ERROR(<< "Failed to create thread pool: " << e.what());
            }
        } else {
            LOG_WARN(<< "Failed to determine hardware concurrency");
        }

        // We'll just have to execute in the main thread.
        return CExecutorHolder{};
    }

    executor& executor() { return *m_Executor; }
    std::size_t threadPoolSize() const { return m_ThreadPoolSize; }

private:
    CExecutorHolder(std::size_t threadPoolSize)
        : m_ThreadPoolSize{threadPoolSize},
          m_Executor{boost::make_unique<transwarp::parallel>(threadPoolSize)} {}

private:
    std::size_t m_ThreadPoolSize;
    std::unique_ptr<core::executor> m_Executor;
};

CExecutorHolder singletonExecutor;
}

void startDefaultAsyncExecutor(std::size_t threadPoolSize) {
    singletonExecutor = CExecutorHolder::makeThreadPool(threadPoolSize);
}

void stopDefaultAsyncExecutor() {
    singletonExecutor = CExecutorHolder{};
}

std::size_t defaultAsyncThreadPoolSize() {
    return singletonExecutor.threadPoolSize();
}

//! The global default thread pool.
//!
//! This gets the default parallel executor specified by the create_default_executor.
executor& defaultAsyncExecutor() {
    return singletonExecutor.executor();
}
}
}
