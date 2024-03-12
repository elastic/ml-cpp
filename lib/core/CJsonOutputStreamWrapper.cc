/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CLogger.h>
#include <core/CMemoryDef.h>

#include <string>

namespace ml {
namespace core {

const char CJsonOutputStreamWrapper::JSON_ARRAY_START('[');
const char CJsonOutputStreamWrapper::JSON_ARRAY_END(']');
const char CJsonOutputStreamWrapper::JSON_ARRAY_DELIMITER(',');

CJsonOutputStreamWrapper::CJsonOutputStreamWrapper(std::ostream& outStream)
    : m_StringBufferQueue(CJsonOutputStreamWrapper::BUFFER_POOL_SIZE),
      m_ConcurrentOutputStream(outStream), m_FirstObject(true) {
    // initialize the bufferpool
    for (auto& stringBuffer : m_StringBuffers) {
        stringBuffer.reserve(BUFFER_START_SIZE);
        m_StringBufferQueue.push(&stringBuffer);
    }

    m_ConcurrentOutputStream([](std::ostream& o) { o.put(JSON_ARRAY_START); });
}

CJsonOutputStreamWrapper::~CJsonOutputStreamWrapper() {
    m_ConcurrentOutputStream([](std::ostream& o) {
        o.put(JSON_ARRAY_END);
        o.flush();
    });
}

void CJsonOutputStreamWrapper::acquireBuffer(TGenericLineWriter& writer,
                                             std::string*& buffer) {
    buffer = m_StringBufferQueue.pop();
    writer.reset(*buffer);
}

void CJsonOutputStreamWrapper::releaseBuffer(TGenericLineWriter& writer, std::string* buffer) {
    writer.flush();

    // check for data that has to be written
    if (buffer->size() > 0) {
        m_ConcurrentOutputStream([this, buffer](std::ostream& o) {
            if (m_FirstObject) {
                m_FirstObject = false;
            } else {
                o.put(JSON_ARRAY_DELIMITER);
            }

            o << *buffer;
            o.flush();
            this->returnAndCheckBuffer(buffer);
        });
    } else {
        m_StringBufferQueue.push(buffer);
    }
}

void CJsonOutputStreamWrapper::flushBuffer(TGenericLineWriter& writer, std::string*& buffer) {
    writer.flush();

    m_ConcurrentOutputStream([this, buffer](std::ostream& o) {
        if (m_FirstObject) {
            m_FirstObject = false;
        } else {
            o.put(JSON_ARRAY_DELIMITER);
        }
        o << *buffer;
        this->returnAndCheckBuffer(buffer);
    });

    acquireBuffer(writer, buffer);
}

void CJsonOutputStreamWrapper::returnAndCheckBuffer(std::string* buffer) {
    std::string& str = *buffer;
    str.clear();

    if (str.capacity() > BUFFER_REALLOC_TRIGGER_SIZE) {
        // we have to free and realloc
        str.shrink_to_fit();
        str.reserve(BUFFER_START_SIZE);
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

void CJsonOutputStreamWrapper::writeJson(std::string json) {
    m_ConcurrentOutputStream([ this, json_ = std::move(json) ](std::ostream & o) {
        if (m_FirstObject) {
            m_FirstObject = false;
        } else {
            o.put(JSON_ARRAY_DELIMITER);
        }
        o << json_;
    });
}

void CJsonOutputStreamWrapper::debugMemoryUsage(const CMemoryUsage::TMemoryUsagePtr& mem) const {
    std::size_t bufferSize{0};
    for (const auto& stringBuffer : m_StringBuffers) {
        // GetSize() returns the length of the string, not the used memory, need to inspect internals
        bufferSize += stringBuffer.capacity();
    }

    mem->addItem("m_StringBuffers", bufferSize);

    // we can not use dynamic size methods as it would stumble upon the pointers
    // basically estimating the size of the circular buffer
    std::size_t queueSize = BUFFER_POOL_SIZE * sizeof(std::string*);

    mem->addItem("m_StringBufferQueue", queueSize);

    m_ConcurrentOutputStream.debugMemoryUsage(mem->addChild());
}

std::size_t CJsonOutputStreamWrapper::memoryUsage() const {
    std::size_t memoryUsage{0};
    for (const auto& stringBuffer : m_StringBuffers) {
        // GetSize() returns the length of the string, not the used memory, need to inspect internals
        memoryUsage += stringBuffer.capacity();
    }

    // we can not use dynamic size methods as it would stumble upon the pointers
    // basically estimating the size of the circular buffer
    memoryUsage += BUFFER_POOL_SIZE * sizeof(std::string*);

    memoryUsage += m_ConcurrentOutputStream.memoryUsage();
    return memoryUsage;
}
}
}
