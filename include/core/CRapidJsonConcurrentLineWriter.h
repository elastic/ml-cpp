/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_core_CRapidJsonConcurrentLineWriter_h
#define INCLUDED_ml_core_CRapidJsonConcurrentLineWriter_h

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CRapidJsonLineWriter.h>

namespace ml {
namespace core {

//! \brief
//! A Json line writer for concurrently writing to a shared output stream.
//!
//! DESCRIPTION:\n
//! Takes a wrapped output stream, hides all buffering/pooling/concurrency.
//! CRapidJsonConcurrentLineWriter objects must not be shared between threads.
//! The intended usage is as follows:
//! \code{.cpp}
//! std::ostringstream stream;
//! core::CJsonOutputStreamWrapper streamWrapper{stream};
//! std::thread thread{[&streamWrapper]() {
//!     core::CRapidJsonConcurrentLineWriter writer{streamWrapper};
//!     writer.StartObject();
//!     writer.Key("foo");
//!     writer.Int(1)
//!     writer.EndObject();
//! }};
//! ...
//! \endcode
//!
//! IMPLEMENTATION DECISIONS:\n
//! Hardcode encoding and stream type.
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
    void debugMemoryUsage(const CMemoryUsage::TMemoryUsagePtr& mem) const;

    //! Get the memory used by this component.
    std::size_t memoryUsage() const;

    //! Write JSON document to outputstream
    //! \note This overwrite is needed because the members of rapidjson::Writer
    //! are not virtual and we need to avoid "slicing" the writer to ensure that
    //! that the correct StartObject/EndObject functions are called when this is
    //! passed to \p doc Accept.
    void write(const rapidjson::Value& doc) { doc.Accept(*this); }

private:
    //! The stream object
    CJsonOutputStreamWrapper& m_OutputStreamWrapper;

    //! internal buffer, managed by the stream wrapper
    rapidjson::StringBuffer* m_StringBuffer;
};
}
}

#endif /* INCLUDED_ml_core_CRapidJsonConcurrentLineWriter_h */
