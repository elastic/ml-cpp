/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_test_CThreadDataWriter_h
#define INCLUDED_ml_test_CThreadDataWriter_h

#include <core/CThread.h>

#include <test/ImportExport.h>

#include <cstdint>
#include <string>

namespace ml {
namespace test {

//! \brief
//! Writes to a file in a thread.
//!
//! DESCRIPTION:\n
//! When the thread is started it waits a short time, then writes
//! data of a specified size to a file.  The data consists of a
//! single specified character repeated.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Using core::CThread rather than std::thread to minimise risk
//! as it was factored out of existing code using core::CThread.
//!
class TEST_EXPORT CThreadDataWriter : public core::CThread {
public:
    CThreadDataWriter(std::uint32_t sleepTimeMs,
                      const std::string& fileName,
                      char testChar,
                      std::size_t size);

protected:
    void run() override;
    void shutdown() override;

private:
    const std::uint32_t m_SleepTimeMs;
    const std::string m_FileName;
    const char m_TestChar;
    const std::size_t m_Size;
};
}
}

#endif // INCLUDED_ml_test_CThreadDataWriter_h
