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
#include <test/CThreadDataWriter.h>

#include <chrono>
#include <fstream>
#include <thread>

namespace ml {
namespace test {

CThreadDataWriter::CThreadDataWriter(std::uint32_t sleepTimeMs,
                                     const std::string& fileName,
                                     char testChar,
                                     std::size_t size)
    : m_SleepTimeMs(sleepTimeMs), m_FileName(fileName), m_TestChar(testChar),
      m_Size(size) {
}

void CThreadDataWriter::run() {
    std::this_thread::sleep_for(std::chrono::milliseconds(m_SleepTimeMs));

    std::ofstream strm{m_FileName};
    for (std::size_t i = 0; i < m_Size && strm.good(); ++i) {
        strm << m_TestChar;
    }
}

void CThreadDataWriter::shutdown() {
}
}
}
