/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CLogger.h>
#include <string>

namespace ml {
namespace core {

const char CJsonOutputStreamWrapper::JSON_ARRAY_START('[');
const char CJsonOutputStreamWrapper::JSON_ARRAY_END(']');
const char CJsonOutputStreamWrapper::JSON_ARRAY_DELIMITER(',');

CJsonOutputStreamWrapper::CJsonOutputStreamWrapper(std::ostream& outStream)
    : m_ConcurrentOutputStream(outStream), m_FirstObject(true) {
    // initialize the bufferpool
    for (size_t i = 0; i < BUFFER_POOL_SIZE; ++i) {
        m_StringBuffers[i].Reserve(BUFFER_START_SIZE);
        m_StringBufferQueue.push(&m_StringBuffers[i]);
    }

    m_ConcurrentOutputStream([](std::ostream& o) { o.put(JSON_ARRAY_START); });
}

CJsonOutputStreamWrapper::~CJsonOutputStreamWrapper() {
    m_ConcurrentOutputStream([](std::ostream& o) { o.put(JSON_ARRAY_END); });
}

void CJsonOutputStreamWrapper::acquireBuffer(TGenericLineWriter& writer,
                                             rapidjson::StringBuffer*& buffer) {
    buffer = m_StringBufferQueue.pop();
    writer.Reset(*buffer);
}

void CJsonOutputStreamWrapper::releaseBuffer(TGenericLineWriter& writer,
                                             rapidjson::StringBuffer* buffer) {
    writer.Flush();

    // check for data that has to be written
    if (buffer->GetLength() > 0) {
        m_ConcurrentOutputStream([this, buffer](std::ostream& o) {
            if (m_FirstObject) {
                m_FirstObject = false;
            } else {
                o.put(JSON_ARRAY_DELIMITER);
            }

            o.write(buffer->GetString(), buffer->GetLength());
            o.flush();
            this->returnAndCheckBuffer(buffer);
        });
    } else {
        m_StringBufferQueue.push(buffer);
    }
}

void CJsonOutputStreamWrapper::flushBuffer(TGenericLineWriter& writer,
                                           rapidjson::StringBuffer*& buffer) {
    writer.Flush();

    m_ConcurrentOutputStream([this, buffer](std::ostream& o) {
        if (m_FirstObject) {
            m_FirstObject = false;
        } else {
            o.put(JSON_ARRAY_DELIMITER);
        }
        o.write(buffer->GetString(), buffer->GetLength());
        this->returnAndCheckBuffer(buffer);
    });

    acquireBuffer(writer, buffer);
}

void CJsonOutputStreamWrapper::returnAndCheckBuffer(rapidjson::StringBuffer* buffer) {
    buffer->Clear();

    if (buffer->stack_.GetCapacity() > BUFFER_REALLOC_TRIGGER_SIZE) {
        // we have to free and realloc
        buffer->ShrinkToFit();
        buffer->Reserve(BUFFER_START_SIZE);
    }

    m_StringBufferQueue.push(buffer);
}

void CJsonOutputStreamWrapper::flush() {
    m_ConcurrentOutputStream([](std::ostream& o) { o.flush(); });
}

void CJsonOutputStreamWrapper::syncFlush() {
    std::mutex m;
    std::condition_variable c;
    std::unique_lock<std::mutex> lock(m);

    m_ConcurrentOutputStream([&m, &c](std::ostream& o) {
        o.flush();
        std::unique_lock<std::mutex> waitLock(m);
        c.notify_all();
    });

    c.wait(lock);
}

void CJsonOutputStreamWrapper::debugMemoryUsage(CMemoryUsage::TMemoryUsagePtr mem) const {
    std::size_t bufferSize = 0;
    for (size_t i = 0; i < BUFFER_POOL_SIZE; ++i) {
        // GetSize() returns the length of the string, not the used memory, need to inspect
        // internals
        bufferSize += m_StringBuffers[i].stack_.GetCapacity();
    }

    mem->addItem("m_StringBuffers", bufferSize);

    // we can not use dynamic size methods as it would stumble upon the pointers
    // basically estimating the size of the circular buffer
    std::size_t queueSize = BUFFER_POOL_SIZE * sizeof(rapidjson::StringBuffer*);

    mem->addItem("m_StringBufferQueue", queueSize);

    m_ConcurrentOutputStream.debugMemoryUsage(mem->addChild());
}

std::size_t CJsonOutputStreamWrapper::memoryUsage() const {
    std::size_t memoryUsage = 0;
    for (size_t i = 0; i < BUFFER_POOL_SIZE; ++i) {
        // GetSize() returns the length of the string, not the used memory, need to inspect
        // internals
        memoryUsage += m_StringBuffers[i].stack_.GetCapacity();
    }

    // we can not use dynamic size methods as it would stumble upon the pointers
    // basically estimating the size of the circular buffer
    memoryUsage += BUFFER_POOL_SIZE * sizeof(rapidjson::StringBuffer*);

    memoryUsage += m_ConcurrentOutputStream.memoryUsage();
    return memoryUsage;
}
}
}
