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
#ifndef INCLUDED_ml_api_CSingleStreamDataAdder_h
#define INCLUDED_ml_api_CSingleStreamDataAdder_h

#include <core/CDataAdder.h>

#include <api/ImportExport.h>

#include <string>

namespace ml {
namespace api {

//! \brief
//! Persists data to a single C++ stream.
//!
//! DESCRIPTION:\n
//! Persists data to a single C++ stream in the format required by
//! Elasticsearch's bulk API, namely:
//! { metadata document }
//! { document to index }
//!
//! Each time streamComplete() is called a zero byte ('\0') is appended
//! to the stream.  This class will return the same stream for every
//! search.  The caller must not close the stream.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Not thread safe - each object of this class must only be accessed from
//! one thread at a time.
//!
//! The single stream must be already open when passed to the constructor.
//!
class API_EXPORT CSingleStreamDataAdder : public core::CDataAdder {
public:
    //! The \p stream must already be open when the constructor is
    //! called.
    CSingleStreamDataAdder(const TOStreamP& stream);

    //! Returns a stream that can be used to persist data to a C++
    //! stream, or NULL if this is not possible.  Many errors cannot
    //! be detected by this method, so the stream will go into the
    //! "bad" state if an error occurs during upload.  The caller
    //! must check for this.
    //! \param id ID to add to metadata document
    TOStreamP addStreamed(const std::string& id) override;

    //! Clients that get a stream using addStreamed() must call this
    //! method one they've finished sending data to the stream.
    //! \param stream The completed data stream
    //! \param force If true the stream is flushed
    bool streamComplete(TOStreamP& stream, bool force) override;

    std::size_t maxDocumentSize() const override;

private:
    //! Recommended maximum Elasticsearch document size
    static const std::size_t MAX_DOCUMENT_SIZE;

private:
    //! The stream we're writing to.
    TOStreamP m_Stream;
};
}
}

#endif // INCLUDED_ml_api_CSingleStreamDataAdder_h
