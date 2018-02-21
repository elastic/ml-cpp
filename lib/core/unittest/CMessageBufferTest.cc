/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#include "CMessageBufferTest.h"

#include <core/CLogger.h>
#include <core/CMessageBuffer.h>

#include <vector>


CppUnit::Test   *CMessageBufferTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CMessageBufferTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CMessageBufferTest>(
                                   "CMessageBufferTest::testAll",
                                   &CMessageBufferTest::testAll) );

    return suiteOfTests;
}

namespace
{
    class CBuffer
    {
        public:
            typedef std::vector<std::string>    TStrVec;

        public:
            CBuffer(uint32_t flushInterval) : m_FlushInterval(flushInterval)
            {
            }

            void    addMessage(const std::string &str)
            {
                if((m_Buffer.size() % 1000) == 0)
                {
                    LOG_DEBUG("Received " << m_Buffer.size() << " strings");
                }
                m_Buffer.push_back(str);
            }
    
            uint32_t    flushInterval(void) const
            {
                return m_FlushInterval;
            }

            ml::core_t::TTime flushMessages(TStrVec &messages)
            {
                LOG_DEBUG("Flush messages " << m_Buffer.size());

                messages = m_Buffer;

                m_Buffer.clear();

                // For time sensitive buffers, this value can provide the
                // current time for example, but for this simple test it's not
                // used
                return 0;
            }

            void    flushAllMessages(TStrVec &messages)
            {
                this->flushMessages(messages);
            }

            void    processMessages(const TStrVec &messages, ml::core_t::TTime)
            {
                m_Results.insert(m_Results.end(), messages.begin(), messages.end());

                LOG_DEBUG("Processed " << messages.size() << " " << m_Results.size() << " messages");
            }

            size_t  size(void) const
            {
                return m_Results.size();
            }

        private:
            uint32_t    m_FlushInterval;
            TStrVec     m_Buffer;
            TStrVec     m_Results;
    };
}

void    CMessageBufferTest::testAll(void)
{
    CBuffer buffer(10);

    ml::core::CMessageBuffer<std::string, CBuffer>   queue(buffer);

    CPPUNIT_ASSERT(queue.start());

    size_t  max(100000);

    LOG_DEBUG("Sending " << max << " strings");

    for(size_t i = 0; i < max; ++i)
    {
        queue.addMessage("Test string");
    }

    LOG_DEBUG("Sent all strings");

    queue.stop();

    CPPUNIT_ASSERT_EQUAL(max, buffer.size()); 
}
