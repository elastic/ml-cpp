/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
            using TStrVec = std::vector<std::string>;

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
    
            uint32_t    flushInterval() const
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

            size_t  size() const
            {
                return m_Results.size();
            }

        private:
            uint32_t    m_FlushInterval;
            TStrVec     m_Buffer;
            TStrVec     m_Results;
    };
}

void    CMessageBufferTest::testAll()
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
