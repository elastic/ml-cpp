/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_test_CThreadDataReader_h
#define INCLUDED_ml_test_CThreadDataReader_h

#include <core/CMutex.h>
#include <core/CThread.h>

#include <test/ImportExport.h>

#include <atomic>
#include <cstdint>
#include <string>

namespace ml {
namespace test {

//! \brief
//! Reads a file in a thread.
//!
//! DESCRIPTION:\n
//! When the thread is started it makes several attempts to open
//! the file, separated by the specified sleep.  This gives some
//! other thread time to create the file.  The data is read into
//! a string which can be accessed safely after the reader thread
//! has completed (so this class should not be used to read
//! collosal files that will exhaust memory).
//!
//! IMPLEMENTATION DECISIONS:\n
//! Using core::CThread rather than std::thread to minimise risk
//! as it was factored out of existing code using core::CThread.
//!
class TEST_EXPORT CThreadDataReader : public core::CThread {
public:
    CThreadDataReader(std::uint32_t sleepTimeMs, std::size_t maxAttempts, const std::string& fileName);

    const std::string& data() const;
    std::size_t attemptsTaken() const;
    bool streamWentBad() const;

protected:
    void run() override;
    void shutdown() override;

private:
    mutable core::CMutex m_Mutex;
    std::atomic_bool m_Shutdown;
    std::atomic_bool m_StreamWentBad;
    const std::uint32_t m_SleepTimeMs;
    const std::size_t m_MaxAttempts;
    std::atomic_size_t m_Attempt;
    const std::string m_FileName;
    std::string m_Data;
};
}
}

#endif // INCLUDED_ml_test_CThreadDataReader_h
