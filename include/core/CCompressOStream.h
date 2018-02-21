/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CCompressOStream_h
#define INCLUDED_ml_core_CCompressOStream_h

#include <core/CDualThreadStreamBuf.h>
#include <core/CStateCompressor.h>
#include <core/CThread.h>
#include <core/ImportExport.h>

namespace ml
{
namespace core
{

//! \brief
//! An output stream that writes to a boost filtering_stream endpoint.
//!
//! DESCRIPTION:\n
//! An extension of the C++ streams library that uploads data
//! that is written to the stream to a supplied boost::iostreams::
//! filtering_stream.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The upload is done in a separate thread to avoid blocking
//! the thread that creates the stream object.  A CDualThreadStreamBuf
//! manages the buffering of data between the client thread and
//! the upload thread.
//!
class CORE_EXPORT CCompressOStream : public std::ostream
{
    public:
        //! Constructor
        CCompressOStream(CStateCompressor::CChunkFilter &filter);

        //! Destructor will close the stream
        virtual ~CCompressOStream(void);

        //! Close the stream
        void close(void);

    private:
        class CCompressThread : public CThread
        {
            public:
                CCompressThread(CCompressOStream &stream,
                                CDualThreadStreamBuf &streamBuf,
                                CStateCompressor::CChunkFilter &filter);

            protected:
                //! Implementation of inherited interface
                virtual void run(void);
                virtual void shutdown(void);

            public:
                //! Reference to the owning stream
                CCompressOStream                 &m_Stream;

                //! Reference to the owning stream's buffer
                CDualThreadStreamBuf             &m_StreamBuf;

            private:
                //! Reference to the output sink - this handles
                //! downstream writing to datastore
                CStateCompressor::CChunkFilter   &m_FilterSink;

                //! The gzip filter to live within the new thread
                CStateCompressor::TFilteredOutput m_OutFilter;
        };

    private:
        //! The stream buffer
        CDualThreadStreamBuf m_StreamBuf;

        //! Thread used for the upload
        CCompressThread      m_UploadThread;
};

} // core
} // ml

#endif // INCLUDED_ml_core_CCompressOStream_h
