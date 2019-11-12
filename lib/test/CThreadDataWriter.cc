/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <test/CThreadDataWriter.h>

#include <core/CSleep.h>

#include <fstream>

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
    core::CSleep::sleep(m_SleepTimeMs);

    std::ofstream strm{m_FileName};
    for (std::size_t i = 0; i < m_Size && strm.good(); ++i) {
        strm << m_TestChar;
    }
}

void CThreadDataWriter::shutdown() {
}
}
}
