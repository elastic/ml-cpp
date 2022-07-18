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
#ifndef INCLUDED_ml_core_CJsonOutputStreamWrapper_h
#define INCLUDED_ml_core_CJsonOutputStreamWrapper_h

#include <core/CConcurrentQueue.h>
#include <core/CConcurrentWrapper.h>
#include <core/CMemory.h>
#include <core/CNonCopyable.h>
#include <core/CRapidJsonLineWriter.h>
#include <core/ImportExport.h>

#include <rapidjson/stringbuffer.h>

#include <ostream>

namespace ml {
namespace core {

//! \brief
//! A wrapper around a shared output stream for concurrent writes, handling
//! writes.
//!
//! DESCRIPTION:\n
//! Takes an output stream and wraps it, provides buffers and pooling.
//!
//! Consider not to use this directly but CRapidJsonConcurrentLineWriter.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Pool and buffer sizes are hardcoded.
class CORE_EXPORT CJsonOutputStreamWrapper final : CNonCopyable {
private:
    //! Number of buffers in the pool.
    static constexpr size_t BUFFER_POOL_SIZE{16};
    //! Size of 1 buffer in the pool.
    //! \note This size is not fixed but might get enlarged at runtime.
    static constexpr size_t BUFFER_START_SIZE{1024};
    //! Upper boundary for buffer size, if above buffer gets automatically shrunk
    //! back to BUFFER_START_SIZE after last usage.
    static constexpr size_t BUFFER_REALLOC_TRIGGER_SIZE{4096};

    static const char JSON_ARRAY_START;
    static const char JSON_ARRAY_END;
    static const char JSON_ARRAY_DELIMITER;

public:
    using TOStreamConcurrentWrapper = core::CConcurrentWrapper<std::ostream>;
    using TGenericLineWriter = core::CRapidJsonLineWriter<rapidjson::StringBuffer>;

public:
    //! Wrap a given ostream for concurrent access.
    //! \param[in] outStream The stream to write to
    explicit CJsonOutputStreamWrapper(std::ostream& outStream);

    ~CJsonOutputStreamWrapper();

    //! Acquires a buffer from the pool and attaches it to the given writer object.
    void acquireBuffer(TGenericLineWriter& writer, rapidjson::StringBuffer*& buffer);

    //! Releases a buffer from the pool, remaining data will be written before returning it.
    void releaseBuffer(TGenericLineWriter& writer, rapidjson::StringBuffer* buffer);

    //! Flush the buffer/writer if necessary, keeps the logic when to flush in here.
    //! \param writer A rapidjson writer object
    //! \param buffer The buffer for writing
    //! Side-effect: the writer as well as the buffer are altered.
    void flushBuffer(TGenericLineWriter& writer, rapidjson::StringBuffer*& buffer);

    //! Flush the wrapped outputstream.
    //!
    //! \note This is still async
    void flush();

    //! A sync flush that blocks until flush has actually happened.
    void syncFlush();

    //! Write preformatted JSON.
    void writeJson(std::string json);

    //! Debug the memory used by this component.
    void debugMemoryUsage(const CMemoryUsage::TMemoryUsagePtr& mem) const;

    //! Get the memory used by this component.
    std::size_t memoryUsage() const;

private:
    void returnAndCheckBuffer(rapidjson::StringBuffer* buffer);

private:
    //! the pool of buffers
    rapidjson::StringBuffer m_StringBuffers[BUFFER_POOL_SIZE];

    //! the pool of available buffers
    CConcurrentQueue<rapidjson::StringBuffer*, BUFFER_POOL_SIZE> m_StringBufferQueue;

    //! the stream object wrapped by CConcurrentWrapper
    TOStreamConcurrentWrapper m_ConcurrentOutputStream;

    //! whether we wrote the first element
    bool m_FirstObject;
};
}
}

#endif // INCLUDED_ml_core_CJsonOutputStreamWrapper_h
