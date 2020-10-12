/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CBlockingCallCancellingTimer.h>
#include <core/CNamedPipeFactory.h>
#include <core/COsFileFuncs.h>
#include <core/CThread.h>

#include <boost/test/unit_test.hpp>

#include <cerrno>
#include <chrono>
#include <thread>

BOOST_AUTO_TEST_SUITE(CBlockingCallCancellingTimerTest)

BOOST_AUTO_TEST_CASE(testCancelBlock) {

    ml::core::CBlockingCallCancellingTimer cancellerThread{
        ml::core::CThread::currentThreadId(), std::chrono::seconds{1}};
    BOOST_TEST_REQUIRE(cancellerThread.start());

    // The CBlockingCallCancellingTimer should wake up the blocking open
    // of the named pipe "test_pipe".  Without this wake up, it would block
    // indefinitely as nothing will ever connect to the other end.  The wake up
    // happens after 1 second.

    std::string testPipeName{ml::core::CNamedPipeFactory::defaultPath() + "test_pipe"};
    ml::core::CNamedPipeFactory::TIStreamP pipeStrm{ml::core::CNamedPipeFactory::openPipeStreamRead(
        testPipeName, cancellerThread.hasCancelledBlockingCall())};
    BOOST_TEST_REQUIRE(pipeStrm == nullptr);

    BOOST_TEST_REQUIRE(cancellerThread.stop());

    BOOST_REQUIRE_EQUAL(true, cancellerThread.hasCancelledBlockingCall().load());

    // Confirm that cancellation of the named pipe connection deleted the pipe
    BOOST_REQUIRE_EQUAL(-1, ml::core::COsFileFuncs::access(
                                testPipeName.c_str(), ml::core::COsFileFuncs::EXISTS));
    BOOST_REQUIRE_EQUAL(ENOENT, errno);
}

BOOST_AUTO_TEST_CASE(testNoCancelAfterCancellerShutdown) {

    ml::core::CBlockingCallCancellingTimer cancellerThread{
        ml::core::CThread::currentThreadId(), std::chrono::seconds{100}};
    BOOST_TEST_REQUIRE(cancellerThread.start());

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    BOOST_TEST_REQUIRE(cancellerThread.stop());

    BOOST_REQUIRE_EQUAL(false, cancellerThread.hasCancelledBlockingCall().load());
}

BOOST_AUTO_TEST_SUITE_END()
