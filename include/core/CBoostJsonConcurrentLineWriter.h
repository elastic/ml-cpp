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

#ifndef INCLUDED_ml_core_CBoostJsonConcurrentLineWriter_h
#define INCLUDED_ml_core_CBoostJsonConcurrentLineWriter_h

#include <core/CBoostJsonLineWriter.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CStringBufWriter.h>

namespace json = boost::json;

namespace ml {
namespace core {

//! \brief
//! A Json line writer for concurrently writing to a shared output stream.
//!
//! DESCRIPTION:\n
//! Takes a wrapped output stream, hides all buffering/pooling/concurrency.
//! CBoostJsonConcurrentLineWriter objects must not be shared between threads.
//! The intended usage is as follows:
//! \code{.cpp}
//! std::ostringstream stream;
//! core::CJsonOutputStreamWrapper streamWrapper{stream};
//! std::thread thread{[&streamWrapper]() {
//!     core::CBoostJsonConcurrentLineWriter writer{streamWrapper};
//!     writer.onObjectBegin();
//!     writer.onKey("foo");
//!     writer.onInt(1);
//!     writer.onObjectEnd();
//! }};
//! ...
//! \endcode
//!
//! IMPLEMENTATION DECISIONS:\n
//! Hardcode encoding and stream type.
//!
class CORE_EXPORT CBoostJsonConcurrentLineWriter : public CStringBufWriter {
public:
    using TBoostJsonLineWriterBase = CBoostJsonLineWriter<std::string>;

public:
    //! Take a wrapped stream and provide a json writer object
    //! \p outStream reference to an wrapped output stream
    explicit CBoostJsonConcurrentLineWriter(CJsonOutputStreamWrapper& outStream);

    ~CBoostJsonConcurrentLineWriter() override;

    //! Flush buffers, including the output stream.
    //! Note: flush still happens asynchronous
    void flush() override;

    //! Hooks into end object to automatically flush if json object is complete
    //! Note: This is a non-virtual overwrite
    bool onObjectEnd(std::size_t memberCount = 0) override;

    //! Debug the memory used by this component.
    void debugMemoryUsage(const CMemoryUsage::TMemoryUsagePtr& mem) const;

    //! Get the memory used by this component.
    std::size_t memoryUsage() const;

private:
    //! The stream object
    CJsonOutputStreamWrapper& m_OutputStreamWrapper;

    //! internal buffer, managed by the stream wrapper
    std::string* m_StringBuffer;
};
}
}

#endif /* INCLUDED_ml_core_CBoostJsonConcurrentLineWriter_h */
