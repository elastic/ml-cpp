/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDualThreadStreamBuf.h>
#include <core/CNamedPipeFactory.h>
#include <core/CSleep.h>
#include <core/CThread.h>

#include "../CBlockingCallCancellerThread.h"

#include <boost/test/unit_test.hpp>

#include <istream>

BOOST_AUTO_TEST_SUITE(CBlockingCallCancellerThreadTest)

namespace {

class CEofThread : public ml::core::CThread {
public:
    CEofThread(ml::core::CDualThreadStreamBuf& buf) : m_Buf(buf) {}

protected:
    virtual void run() {
        ml::core::CSleep::sleep(200);

        m_Buf.signalEndOfFile();
    }

    virtual void shutdown() {}

private:
    ml::core::CDualThreadStreamBuf& m_Buf;
};
}


BOOST_AUTO_TEST_CASE(testCancelBlock) {
    ml::core::CDualThreadStreamBuf buf;
    std::istream monStrm(&buf);

    ml::controller::CBlockingCallCancellerThread cancellerThread(
        ml::core::CThread::currentThreadId(), monStrm);
    BOOST_TEST(cancellerThread.start());

    // The CBlockingCallCancellerThread should wake up the blocking open of the
    // named pipe "test_pipe".  Without this wake up, it would block
    // indefinitely as nothing will ever connect to the other end.  The wake up
    // happens when a stream being monitored encounters end-of-file.  In the
    // real program this would be STDIN, but in this test another thread is the
    // source, and it runs out of data after 0.2 seconds.

    CEofThread eofThread(buf);
    BOOST_TEST(eofThread.start());

    ml::core::CNamedPipeFactory::TIStreamP pipeStrm = ml::core::CNamedPipeFactory::openPipeStreamRead(
        ml::core::CNamedPipeFactory::defaultPath() + "test_pipe");
    BOOST_TEST(pipeStrm == nullptr);

    BOOST_TEST(cancellerThread.stop());

    BOOST_TEST(eofThread.stop());
}

BOOST_AUTO_TEST_SUITE_END()
