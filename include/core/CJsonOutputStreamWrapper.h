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
    //! number of buffers in the pool
    static const size_t BUFFER_POOL_SIZE = 16;
    //! size of 1 buffer in the pool
    //! Note: this size is not fixed but might get enlarged at runtime
    static const size_t BUFFER_START_SIZE = 1024;
    //! Upper boundary for buffer size, if above buffer gets automatically shrunk
    //! back to BUFFER_START_SIZE after last usage
    static const size_t BUFFER_REALLOC_TRIGGER_SIZE = 4096;

    static const char JSON_ARRAY_START;
    static const char JSON_ARRAY_END;
    static const char JSON_ARRAY_DELIMITER;

public:
    using TOStreamConcurrentWrapper = core::CConcurrentWrapper<std::ostream>;
    using TGenericLineWriter = core::CRapidJsonLineWriter<rapidjson::StringBuffer>;

public:
    //! wrap a given ostream for concurrent access
    //! \param[in] outStream The stream to write to
    explicit CJsonOutputStreamWrapper(std::ostream &outStream);

    ~CJsonOutputStreamWrapper();

    //! acquires a buffer from the pool and attaches it to the given writer object
    void acquireBuffer(TGenericLineWriter &writer, rapidjson::StringBuffer *&buffer);

    //! releases a buffer from the pool, remaining data will be written before returning it
    void releaseBuffer(TGenericLineWriter &writer, rapidjson::StringBuffer *buffer);

    //! flush the buffer/writer if necessary, keeps the logic when to flush in here
    //! \param writer A rapidjson writer object
    //! \param buffer The buffer for writing
    //! side-effect: the writer as well as the buffer are altered
    void flushBuffer(TGenericLineWriter &writer, rapidjson::StringBuffer *&buffer);

    //! flush the wrapped outputstream
    //! note: this is still async
    void flush();

    //! a sync flush, that blocks until flush has actually happened
    void syncFlush();

    //! Debug the memory used by this component.
    void debugMemoryUsage(CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the memory used by this component.
    std::size_t memoryUsage(void) const;

private:
    void returnAndCheckBuffer(rapidjson::StringBuffer *buffer);

private:
    //! the pool of buffers
    rapidjson::StringBuffer m_StringBuffers[BUFFER_POOL_SIZE];

    //! the pool of available buffers
    CConcurrentQueue<rapidjson::StringBuffer *, BUFFER_POOL_SIZE> m_StringBufferQueue;

    //! the stream object wrapped by CConcurrentWrapper
    TOStreamConcurrentWrapper m_ConcurrentOutputStream;

    //! whether we wrote the first element
    bool m_FirstObject;
};
}
}

#endif /* INCLUDED_ml_core_CJsonOutputStreamWrapper_h */
