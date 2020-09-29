/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDualThreadStreamBuf.h>
#include <core/CNamedPipeFactory.h>
#include <core/COsFileFuncs.h>
#include <core/CThread.h>

#include "../CBlockingCallCancellingStreamMonitor.h"

#include <boost/test/unit_test.hpp>

#include <cerrno>
#include <chrono>
#include <istream>
#include <thread>

BOOST_AUTO_TEST_SUITE(CBlockingCallCancellingStreamMonitorTest)

namespace {

class CEofThread : public ml::core::CThread {
public:
    CEofThread(ml::core::CDualThreadStreamBuf& buf) : m_Buf{buf} {}

protected:
    void run() override {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        m_Buf.signalEndOfFile();
    }

    void shutdown() override {}

private:
    ml::core::CDualThreadStreamBuf& m_Buf;
};
}

BOOST_AUTO_TEST_CASE(testCancelBlock) {
    ml::core::CDualThreadStreamBuf buf;
    std::istream monStrm{&buf};

    ml::controller::CBlockingCallCancellingStreamMonitor cancellerThread{
        ml::core::CThread::currentThreadId(), monStrm};
    BOOST_TEST_REQUIRE(cancellerThread.start());

    // The CBlockingCallCancellingStreamMonitor should wake up the blocking open
    // of the named pipe "test_pipe".  Without this wake up, it would block
    // indefinitely as nothing will ever connect to the other end.  The wake up
    // happens when a stream being monitored encounters end-of-file.  In the
    // real program this would be STDIN, but in this test another thread is the
    // source, and it runs out of data after 0.2 seconds.

    CEofThread eofThread{buf};
    BOOST_TEST_REQUIRE(eofThread.start());

    std::string testPipeName{ml::core::CNamedPipeFactory::defaultPath() + "test_pipe"};
    ml::core::CNamedPipeFactory::TIStreamP pipeStrm{ml::core::CNamedPipeFactory::openPipeStreamRead(
        testPipeName, cancellerThread.hasCancelledBlockingCall())};
    BOOST_TEST_REQUIRE(pipeStrm == nullptr);

    BOOST_TEST_REQUIRE(cancellerThread.stop());

    BOOST_REQUIRE_EQUAL(true, cancellerThread.hasCancelledBlockingCall().load());

    BOOST_TEST_REQUIRE(eofThread.stop());

    // Confirm that cancellation of the named pipe connection deleted the pipe
    BOOST_REQUIRE_EQUAL(-1, ml::core::COsFileFuncs::access(
                                testPipeName.c_str(), ml::core::COsFileFuncs::EXISTS));
    BOOST_REQUIRE_EQUAL(ENOENT, errno);
}

BOOST_AUTO_TEST_SUITE_END()
