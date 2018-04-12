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
#ifndef INCLUDED_ml_core_CStateCompressor_h
#define INCLUDED_ml_core_CStateCompressor_h

#include <core/CDataAdder.h>
#include <core/ImportExport.h>

#include <boost/iostreams/filtering_stream.hpp>

#include <ios>
#include <ostream>

namespace ml {
namespace core {

class CCompressOStream;

//! \brief
//! A CDataAdder-derived class that compresses and chunks data
//!
//! DESCRIPTION:\n
//! A CDataAdder-derived class that compresses and chunks data
//! suitable for decompressing by CStateDecompressor.
//! This class implements the CDataAdder interface, and
//! is initialised with a CDataAdder object that contains
//! the underlying data store - so this class basically fits on top
//! of an existing CDataAdder (and it's counterpart CStateDecompressor
//! on top of a CDataSearcher) to enable compression for the
//! ml persist interface
//!
//! IMPLEMENTATION DECISIONS:\n
//! Not copyable.
//! Uses boost filtering_stream internally, and has a
//! dependency on the CBase64Filter class, as it is expected
//! that downstream CDataAdder/CDataSearcher store will
//! support strings of Base64 encoded data
//!
class CORE_EXPORT CStateCompressor : public CDataAdder {
public:
    static const std::string COMPRESSED_ATTRIBUTE;
    static const std::string END_OF_STREAM_ATTRIBUTE;

public:
    using TFilteredOutput = boost::iostreams::filtering_stream<boost::iostreams::output>;
    using TFilteredOutputP = boost::shared_ptr<TFilteredOutput>;
    using TCompressOStreamP = boost::shared_ptr<CCompressOStream>;

    // Implements the boost::iostreams Sink template interface
    class CChunkFilter {
    public:
        using char_type = char;

        //! Inform the filtering_stream owning object what this is capable of
        struct category : public boost::iostreams::sink_tag,
                          public boost::iostreams::closable_tag {};

    public:
        //! Constructor
        CChunkFilter(CDataAdder& adder);

        //! Interface method: accept n bytes from s
        std::streamsize write(const char* s, std::streamsize n);

        //! Interface method: flush the output and close the stream
        void close();

        //! Set the search ID to use
        void index(const std::string& index, const std::string& id);

        //! True if all of the chunked writes were successful.
        //! If one or any of the writes failed the result is false
        bool allWritesSuccessful();

        //! How many compressed documents have been generated?
        size_t numCompressedDocs() const;

    private:
        //! Handle the details of writing a stream of bytes to the internal
        //! CDataAdder object
        void writeInternal(const char* s, std::streamsize& written, std::streamsize& n);

        //! Close stream - end the JSON output
        void closeStream(bool isFinal);

    private:
        //! The underlying datastore
        CDataAdder& m_Adder;

        //! The filtering_stream compressor given to external clients
        CDataAdder::TOStreamP m_OStream;

        //! The sequential document number currently being written to
        std::size_t m_CurrentDocNum;

        //! The number of bytes written to the current CDataAdder stream
        std::size_t m_BytesDone;

        //! The largest document size permitted by the downstream CDataAdder
        std::size_t m_MaxDocSize;

        //! The search index to use - set by the upstream CDataAdder
        std::string m_Index;

        //! The base ID
        std::string m_BaseId;

        //! true if all the writes were successfull
        bool m_WritesSuccessful;
    };

public:
    //! Constructor: take a reference to the underlying downstream datastore
    CStateCompressor(CDataAdder& compressedAdder);

    //! Add streamed data - return of NULL stream indicates failure.
    //! Since the data to be written isn't known at the time this function
    //! returns it is not possible to detect all error conditions
    //! immediately.  If the stream goes bad whilst being written to then
    //! this also indicates failure.
    //! As this class compresses incoming stream data, it is responsible for
    //! dealing with the underlying storage layer, so only 1 stream will ever
    //! be given out to clients.
    virtual TOStreamP addStreamed(const std::string& index, const std::string& id);

    //! Clients that get a stream using addStreamed() must call this
    //! method one they've finished sending data to the stream.
    //! They should set force to true.
    //! Returns true if all of the chunked uploads were
    //! successful
    virtual bool streamComplete(TOStreamP& strm, bool force);

    //! How many compressed documents have been generated?
    size_t numCompressedDocs() const;

private:
    //! The chunking part of the iostreams filter chain
    CChunkFilter m_FilterSink;

    //! The iostreams filter chain that handles compression/chunking
    TFilteredOutputP m_OutFilter;

    TCompressOStreamP m_OutStream;
};
}
}

#endif // INCLUDED_ml_core_CStateCompressor_h
