/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
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
        CSingleStreamDataAdder(const TOStreamP &stream);

        //! Returns a stream that can be used to persist data to a C++
        //! stream, or NULL if this is not possible.  Many errors cannot
        //! be detected by this method, so the stream will go into the
        //! "bad" state if an error occurs during upload.  The caller
        //! must check for this.
        //! \param index Index to add to metadata document
        //! \param id ID to add to metadata document
        virtual TOStreamP addStreamed(const std::string &index,
                                      const std::string &id);

        //! Clients that get a stream using addStreamed() must call this
        //! method one they've finished sending data to the stream.
        //! \param stream The completed data stream
        //! \param force If true the stream is flushed
        virtual bool streamComplete(TOStreamP &stream,
                                    bool force);

        virtual std::size_t maxDocumentSize(void) const;

    private:
        //! Recommended maximum Elasticsearch document size
        static const size_t MAX_DOCUMENT_SIZE;

    private:
        //! The stream we're writing to.
        TOStreamP           m_Stream;
};


}
}

#endif // INCLUDED_ml_api_CSingleStreamDataAdder_h

