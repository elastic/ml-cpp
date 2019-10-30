/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <test/CThreadDataReader.h>

#include <core/CNamedPipeFactory.h>
#include <core/CScopedLock.h>
#include <core/CSleep.h>

#include <fstream>

namespace ml {
namespace test {

CThreadDataReader::CThreadDataReader(std::uint32_t sleepTimeMs,
                                     std::size_t maxAttempts,
                                     const std::string& fileName)
    : m_Shutdown(false), m_StreamWentBad(false), m_SleepTimeMs(sleepTimeMs),
      m_MaxAttempts(maxAttempts), m_Attempt(0), m_FileName(fileName) {
}

const std::string& CThreadDataReader::data() const {
    // The memory barriers associated with the mutex lock should ensure
    // the thread calling this method has as up-to-date view of m_Data's
    // member variables as the thread that updated it
    core::CScopedLock lock{m_Mutex};
    return m_Data;
}

std::size_t CThreadDataReader::attemptsTaken() const {
    return m_Attempt;
}

bool CThreadDataReader::streamWentBad() const {
    return m_StreamWentBad;
}

void CThreadDataReader::run() {
    m_StreamWentBad = false;
    m_Attempt = 0;
    m_Data.clear();

    std::ifstream strm;

    // Try to open the file repeatedly to allow time for the other
    // thread to create it
    do {
        if (m_Shutdown) {
            return;
        }
        if (++m_Attempt > m_MaxAttempts) {
            return;
        }
        core::CSleep::sleep(m_SleepTimeMs);
        strm.open(m_FileName);
    } while (strm.is_open() == false);

    static const std::streamsize BUF_SIZE{512};
    char buffer[BUF_SIZE];
    while (strm.good()) {
        if (m_Shutdown) {
            return;
        }
        strm.read(buffer, BUF_SIZE);
        if (strm.bad()) {
            m_StreamWentBad = true;
            return;
        }
        if (strm.gcount() > 0) {
            core::CScopedLock lock(m_Mutex);
            // This code deals with the test character we write to
            // detect the short-lived connection problem on Windows
            const char* copyFrom{buffer};
            std::size_t copyLen{static_cast<std::size_t>(strm.gcount())};
            if (m_Data.empty() && *buffer == core::CNamedPipeFactory::TEST_CHAR) {
                ++copyFrom;
                --copyLen;
            }
            if (copyLen > 0) {
                m_Data.append(copyFrom, copyLen);
            }
        }
    }
}

void CThreadDataReader::shutdown() {
    m_Shutdown = true;
}
}
}
