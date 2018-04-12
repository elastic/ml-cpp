/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CBlockingCallCancellerThreadTest.h"

#include <core/CDualThreadStreamBuf.h>
#include <core/CNamedPipeFactory.h>
#include <core/CSleep.h>
#include <core/CThread.h>

#include "../CBlockingCallCancellerThread.h"

#include <istream>

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

CppUnit::Test* CBlockingCallCancellerThreadTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CBlockingCallCancellerThreadTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CBlockingCallCancellerThreadTest>(
        "CBlockingCallCancellerThreadTest::testCancelBlock",
        &CBlockingCallCancellerThreadTest::testCancelBlock));

    return suiteOfTests;
}

void CBlockingCallCancellerThreadTest::testCancelBlock() {
    ml::core::CDualThreadStreamBuf buf;
    std::istream monStrm(&buf);

    ml::controller::CBlockingCallCancellerThread cancellerThread(
        ml::core::CThread::currentThreadId(), monStrm);
    CPPUNIT_ASSERT(cancellerThread.start());

    // The CBlockingCallCancellerThread should wake up the blocking open of the
    // named pipe "test_pipe".  Without this wake up, it would block
    // indefinitely as nothing will ever connect to the other end.  The wake up
    // happens when a stream being monitored encounters end-of-file.  In the
    // real program this would be STDIN, but in this test another thread is the
    // source, and it runs out of data after 0.2 seconds.

    CEofThread eofThread(buf);
    CPPUNIT_ASSERT(eofThread.start());

    ml::core::CNamedPipeFactory::TIStreamP pipeStrm = ml::core::CNamedPipeFactory::openPipeStreamRead(
        ml::core::CNamedPipeFactory::defaultPath() + "test_pipe");
    CPPUNIT_ASSERT(pipeStrm == nullptr);

    CPPUNIT_ASSERT(cancellerThread.stop());

    CPPUNIT_ASSERT(eofThread.stop());
}
