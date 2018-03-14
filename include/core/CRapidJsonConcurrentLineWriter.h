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

#ifndef INCLUDED_ml_core_CRapidJsonConcurrentLineWriter_h
#define INCLUDED_ml_core_CRapidJsonConcurrentLineWriter_h

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CRapidJsonLineWriter.h>

namespace ml {
namespace core {

//! \brief
//! A Json line writer for concurrently writing to a shared output stream
//!
//! DESCRIPTION:\n
//! Takes a wrapped output stream, hides all buffering/pooling/concurrency.
//!
//! IMPLEMENTATION DECISIONS:\n
//! hard code encoding and stream type
//!
class CORE_EXPORT CRapidJsonConcurrentLineWriter
    : public CRapidJsonLineWriter<rapidjson::StringBuffer> {
public:
    using TRapidJsonLineWriterBase = CRapidJsonLineWriter<rapidjson::StringBuffer>;

public:
    //! Take a wrapped stream and provide a json writer object
    //! \p outStream reference to an wrapped output stream
    CRapidJsonConcurrentLineWriter(CJsonOutputStreamWrapper& outStream);

    ~CRapidJsonConcurrentLineWriter();

    //! Flush buffers, including the output stream.
    //! Note: flush still happens asynchronous
    void flush();

    //! Hooks into end object to automatically flush if json object is complete
    //! Note: This is a non-virtual overwrite
    bool EndObject(rapidjson::SizeType memberCount = 0);

    //! Debug the memory used by this component.
    void debugMemoryUsage(CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the memory used by this component.
    std::size_t memoryUsage(void) const;

    //! Write JSON document to outputstream
    //! Note this non-virtual overwrite is needed to avoid slicing of the writer
    //! and hence ensure the correct EndObject is called
    //! \p doc reference to rapidjson document value
    void write(rapidjson::Value& doc) { doc.Accept(*this); }

private:
    //! The stream object
    CJsonOutputStreamWrapper& m_OutputStreamWrapper;

    //! internal buffer, managed by the stream wrapper
    rapidjson::StringBuffer* m_StringBuffer;
};
}
}

#endif /* INCLUDED_ml_core_CRapidJsonConcurrentLineWriter_h */
