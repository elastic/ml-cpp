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
std::size_t THREAD_POOL_SIZE{0};

std::unique_ptr<executor> makeThreadPool(std::size_t threads) {
    if (threads == 0) {
        threads = std::thread::hardware_concurrency();
    }

    THREAD_POOL_SIZE = threads;

    if (THREAD_POOL_SIZE > 0) {
        try {
            return boost::make_unique<transwarp::parallel>(THREAD_POOL_SIZE);
        } catch (const std::exception& e) {
            LOG_WARN(<< "Failed to create thread pool: " << e.what());
        }
    } else {
        LOG_WARN(<< "Failed to determine hardware concurrency");
    }

    // We'll just have to execute in the main thread.
    return boost::make_unique<transwarp::sequential>();
}

std::unique_ptr<executor> THREAD_POOL{boost::make_unique<transwarp::sequential>()};
}

void startDefaultAsyncExecutor(std::size_t threads) {
    THREAD_POOL = makeThreadPool(threads);
}

void stopDefaultAsyncExecutor() {
    THREAD_POOL_SIZE = 0;
    THREAD_POOL = boost::make_unique<transwarp::sequential>();
}

std::size_t defaultAsyncThreadPoolSize() {
    return THREAD_POOL_SIZE;
}

//! The global default thread pool.
//!
//! This gets the default parallel executor specified by the create_default_executor.
executor& defaultAsyncExecutor() {
    return *THREAD_POOL;
}
}
}
