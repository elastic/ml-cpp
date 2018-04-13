/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CStateDecompressor_h
#define INCLUDED_ml_core_CStateDecompressor_h

#include <core/CDataSearcher.h>
#include <core/ImportExport.h>

#include <boost/iostreams/filtering_stream.hpp>

#include <rapidjson/reader.h>
#include <rapidjson/istreamwrapper.h>

namespace ml
{
namespace core
{

//! \brief
//! A CDataSearcher-derived class that decompresses chunked and compressed data
//!
//! DESCRIPTION:\n
//! A CDataSearcher-derived class that decompresses data
//! that has been compressed and chunked by CStateCompressor.
//! This class implements the CDataSearcher interface, and
//! is initialised with a CDataSearcher object that contains
//! the underlying data - so this class basically fits on top
//! of an existing CDataSearcher (and it's counterpart CStateCompressor
//! on top of a CDataAdder) to enable compression for the
//! ml persist interface
//!
//! IMPLEMENTATION DECISIONS:\n
//! Not copyable.
//! Uses boost filtering_stream internally, and has a
//! dependency on the CBase64Filter class, as it is expected
//! that downstream CDataAdder/CDataSearcher store will
//! support strings of Base64 encoded data
//! Parses JSON from the downstream store, using a stream
//! interface.
//!
class CORE_EXPORT CStateDecompressor : public CDataSearcher
{
    public:
        using TFilteredInput = boost::iostreams::filtering_stream<boost::iostreams::input>;
        using TFilteredInputP = boost::shared_ptr<TFilteredInput>;

        static const std::string EMPTY_DATA;

        // Implements the boost::iostreams Source template interface
        class CDechunkFilter
        {
            public:
                using char_type = char;

                //! Inform the filtering_stream owning object what this is capable of
                struct category :
                                public boost::iostreams::source_tag
                {};

            public:
                //! Constructor
                CDechunkFilter(CDataSearcher &searcher);

                //! Interface method: read up to n bytes from the downstream
                //! datastore, decompress them and put them into s
                std::streamsize read(char *s, std::streamsize n);

                //! Interface method: close the downstream stream
                void close();

            private:
                //! Find the JSON header
                //! Read until the array field CStateCompressor::COMPRESSED is found
                bool readHeader();

                //! Manage the reading of bytes from the stream
                void handleRead(char *s, std::streamsize n, std::streamsize &bytesDone);

                //! Write a footer at the end of the document or stream
                std::streamsize endOfStream(char *s, std::streamsize n, std::streamsize bytesDone);

                //! Parse the next json object
                bool parseNext();

            private:
                //! <a href="http://rapidjson.org/classrapidjson_1_1_handler.html">Handler</a>
                //! for events fired by rapidjson during parsing.
                //! Note: using the base handler, so we only need to implement what is needed
                struct SRapidJsonHandler final : public rapidjson::BaseReaderHandler<>
                {
                    bool Bool(bool b);
                    bool String(const char *str, rapidjson::SizeType length, bool);
                    bool StartObject();
                    bool Key(const char *str, rapidjson::SizeType length, bool);
                    bool EndObject(rapidjson::SizeType);
                    bool StartArray();
                    bool EndArray(rapidjson::SizeType);

                    enum ETokenType
                    {
                        E_TokenKey = 1,
                        E_TokenBool = 2,
                        E_TokenString = 3,
                        E_TokenObjectStart = 4,
                        E_TokenObjectEnd = 5,
                        E_TokenArrayStart = 6,
                        E_TokenArrayEnd = 7
                    };

                    //! the last token type extracted
                    ETokenType                   s_Type;

                    //! the last string (c string) as pointer (only valid till next call)
                    const char                   *s_CompressedChunk;

                    //! the last string length (only valid till next call)
                    rapidjson::SizeType          s_CompressedChunkLength;
                };


                //! Has a valid document been seen?
                bool m_Initialised;

                //! Has any data been written downstream?
                bool m_SentData;

                //! The downstream data store to read from
                CDataSearcher &m_Searcher;

                //! The stream given to clients to read from
                CDataSearcher::TIStreamP m_IStream;

                //! The sequential document number currently being written to
                std::size_t m_CurrentDocNum;

                //! Have we read all the data possible from downstream?
                bool m_EndOfStream;

                //! The search configuration parameter set by the upstream caller
                std::string m_SearchString;

                //! Wrapper around the downstream reader
                boost::shared_ptr<rapidjson::IStreamWrapper> m_InputStreamWrapper;

                //! JSON reader for the downstream stream
                boost::shared_ptr<rapidjson::Reader> m_Reader;

                SRapidJsonHandler                     m_Handler;

                //! The offset into the current token that has been read
                std::streamsize m_BufferOffset;

                //! Level of nested objects, used to unwind later on.
                size_t  m_NestedLevel;
        };

    public:
        //! Constructor - take a CDataSearcher for the downstream data store
        CStateDecompressor(CDataSearcher &compressedSearcher);

        //! CDataSearcher interface method - transparently read compressed
        //! data and return it in an uncompressed stream
        virtual TIStreamP search(size_t currentDocNum, size_t limit);

        virtual void setStateRestoreSearch(const std::string &index);

        //! CDataSearcher interface method - specify the search strings to use
        virtual void setStateRestoreSearch(const std::string &index,
                                           const std::string &id);

    private:
        //! Reference to the downstream data store
        CDataSearcher &m_Searcher;

        //! The dechunker object
        CDechunkFilter m_FilterSource;

        //! The boost filtering_stream object that handles decompression
        TFilteredInputP m_InFilter;
};

}
}

#endif // INCLUDED_ml_core_CStateDecompressor_h

