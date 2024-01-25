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
#ifndef INCLUDED_ml_core_CStateDecompressor_h
#define INCLUDED_ml_core_CStateDecompressor_h

#include <core/CBoostJsonUnbufferedIStreamWrapper.h>
#include <core/CDataSearcher.h>
#include <core/ImportExport.h>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/json.hpp>

namespace json = boost::json;

namespace ml {
namespace core {

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
class CORE_EXPORT CStateDecompressor : public CDataSearcher {
public:
    using TFilteredInput = boost::iostreams::filtering_stream<boost::iostreams::input>;
    using TFilteredInputP = std::shared_ptr<TFilteredInput>;

    static const std::string EMPTY_DATA;

    // Implements the boost::iostreams Source template interface
    class CDechunkFilter {
    public:
        using char_type = char;

        //! Inform the filtering_stream owning object what this is capable of
        struct category : public boost::iostreams::source_tag {};

    public:
        //! Constructor
        CDechunkFilter(CDataSearcher& searcher);

        //! Interface method: read up to n bytes from the downstream
        //! datastore, decompress them and put them into s
        std::streamsize read(char* s, std::streamsize n);

        //! Interface method: close the downstream stream
        void close();

    private:
        //! Find the JSON header
        //! Read until the array field CStateCompressor::COMPRESSED is found
        bool readHeader();

        //! Manage the reading of bytes from the stream
        void handleRead(char* s, std::streamsize n, std::streamsize& bytesDone);

        //! Write a footer at the end of the document or stream
        std::streamsize endOfStream(char* s, std::streamsize n, std::streamsize bytesDone);

        //! Parse the next json object
        bool parseNext();

    private:
        //! <a href="http://rapidjson.org/classrapidjson_1_1_handler.html">Handler</a>
        //! for events fired by rapidjson during parsing.
        //! Note: using the base handler, so we only need to implement what is needed
        //! <a https://www.boost.org/doc/libs/1_83_0/libs/json/doc/html/json/ref/boost__json__basic_parser.html#json.ref.boost__json__basic_parser.handler0">Handler</a>
        //! for events fired by rapidjson during parsing.
        struct SBaseBoostJsonHandler {

            /// The maximum number of elements allowed in an array
            static constexpr std::size_t max_array_size = -1;

            /// The maximum number of elements allowed in an object
            static constexpr std::size_t max_object_size = -1;

            /// The maximum number of characters allowed in a string
            static constexpr std::size_t max_string_size = -1;

            /// The maximum number of characters allowed in a key
            static constexpr std::size_t max_key_size = -1;

            /// Called once when the JSON parsing begins.
            ///
            /// @return `true` on success.
            /// @param ec Set to the error, if any occurred.
            ///
            bool on_document_begin(json::error_code& ec) { return true; }

            /// Called when the JSON parsing is done.
            ///
            /// @return `true` on success.
            /// @param ec Set to the error, if any occurred.
            ///
            bool on_document_end(json::error_code& ec) { return true; }

            /// Called when the beginning of an array is encountered.
            ///
            /// @return `true` on success.
            /// @param ec Set to the error, if any occurred.
            ///
            bool on_array_begin(json::error_code& ec);

            /// Called when the end of the current array is encountered.
            ///
            /// @return `true` on success.
            /// @param n The number of elements in the array.
            /// @param ec Set to the error, if any occurred.
            ///
            bool on_array_end(std::size_t n, json::error_code& ec);

            /// Called when the beginning of an object is encountered.
            ///
            /// @return `true` on success.
            /// @param ec Set to the error, if any occurred.
            ///
            bool on_object_begin(json::error_code& ec);

            /// Called when the end of the current object is encountered.
            ///
            /// @return `true` on success.
            /// @param n The number of elements in the object.
            /// @param ec Set to the error, if any occurred.
            ///
            bool on_object_end(std::size_t n, json::error_code& ec);

            /// Called with characters corresponding to part of the current string.
            ///
            /// @return `true` on success.
            /// @param s The partial characters
            /// @param n The total size of the string thus far
            /// @param ec Set to the error, if any occurred.
            ///
            bool on_string_part(std::string_view s, std::size_t n, json::error_code& ec);

            /// Called with the last characters corresponding to the current string.
            ///
            /// @return `true` on success.
            /// @param s The remaining characters
            /// @param n The total size of the string
            /// @param ec Set to the error, if any occurred.
            ///
            bool on_string(std::string_view s, std::size_t n, json::error_code& ec);

            /// Called with characters corresponding to part of the current key.
            ///
            /// @return `true` on success.
            /// @param s The partial characters
            /// @param n The total size of the key thus far
            /// @param ec Set to the error, if any occurred.
            ///
            bool on_key_part(std::string_view s, std::size_t n, json::error_code& ec);

            /// Called with the last characters corresponding to the current key.
            ///
            /// @return `true` on success.
            /// @param s The remaining characters
            /// @param n The total size of the key
            /// @param ec Set to the error, if any occurred.
            ///
            bool on_key(std::string_view s, std::size_t n, json::error_code& ec);

            /// Called with the characters corresponding to part of the current number.
            ///
            /// @return `true` on success.
            /// @param s The partial characters
            /// @param ec Set to the error, if any occurred.
            ///
            bool on_number_part(std::string_view s, json::error_code& ec) {
                return true;
            }

            /// Called when a signed integer is parsed.
            ///
            /// @return `true` on success.
            /// @param i The value
            /// @param s The remaining characters
            /// @param ec Set to the error, if any occurred.
            ///
            bool on_int64(int64_t i, std::string_view s, json::error_code& ec) {
                return true;
            }

            /// Called when an unsigend integer is parsed.
            ///
            /// @return `true` on success.
            /// @param u The value
            /// @param s The remaining characters
            /// @param ec Set to the error, if any occurred.
            ///
            bool on_uint64(uint64_t u, std::string_view s, json::error_code& ec) {
                return true;
            }

            /// Called when a double is parsed.
            ///
            /// @return `true` on success.
            /// @param d The value
            /// @param s The remaining characters
            /// @param ec Set to the error, if any occurred.
            ///
            bool on_double(double d, std::string_view s, json::error_code& ec) {
                return true;
            }

            /// Called when a boolean is parsed.
            ///
            /// @return `true` on success.
            /// @param b The value
            /// @param s The remaining characters
            /// @param ec Set to the error, if any occurred.
            ///
            bool on_bool(bool b, json::error_code& ec);

            /// Called when a null is parsed.
            ///
            /// @return `true` on success.
            /// @param ec Set to the error, if any occurred.
            ///
            bool on_null(json::error_code& ec) { return true; }

            /// Called with characters corresponding to part of the current comment.
            ///
            /// @return `true` on success.
            /// @param s The partial characters.
            /// @param ec Set to the error, if any occurred.
            ///
            bool on_comment_part(std::string_view s, json::error_code& ec) {
                return true;
            }

            /// Called with the last characters corresponding to the current comment.
            ///
            /// @return `true` on success.
            /// @param s The remaining characters
            /// @param ec Set to the error, if any occurred.
            ///
            bool on_comment(std::string_view s, json::error_code& ec) {
                return true;
            }

            size_t s_Level[2];
            bool s_IsEndOfLevel[2];
            std::string s_Name[2];
            std::string s_Value[2];

            //! Setting m_NextIndex = (1 - m_NextIndex) advances the
            //! stored details.
            size_t s_NextIndex;

            bool s_RememberValue;
        };

        struct SBoostJsonHandler final : public SBaseBoostJsonHandler {
            constexpr static std::size_t max_object_size = std::size_t(-1);
            constexpr static std::size_t max_array_size = std::size_t(-1);
            constexpr static std::size_t max_key_size = std::size_t(-1);
            constexpr static std::size_t max_string_size = std::size_t(-1);

            bool on_bool(bool b, json::error_code& ec);
            bool on_string(std::string_view s, std::size_t n, json::error_code& ec);
            bool on_string_part(std::string_view s, std::size_t n, json::error_code& ec);
            bool on_object_begin(json::error_code& ec);
            bool on_key(std::string_view s, std::size_t n, json::error_code& ec);
            bool on_key_part(std::string_view s, std::size_t n, json::error_code& ec);
            bool on_object_end(std::size_t n, json::error_code& ec);
            bool on_array_begin(json::error_code& ec);
            bool on_array_end(std::size_t n, json::error_code& ec);

            enum ETokenType {
                ETokenNull = 0,
                E_TokenKey = 1,
                E_TokenBool = 2,
                E_TokenString = 3,
                E_TokenObjectStart = 4,
                E_TokenObjectEnd = 5,
                E_TokenArrayStart = 6,
                E_TokenArrayEnd = 7,
                E_TokenStringPart = 8,
                E_TokenKeyPart = 9,
                E_TokenComma = 10,
                E_TokenColon = 11,
                E_TokenSpace = 12
            };

            //! the last token type extracted
            ETokenType s_Type;

            //! the last string (c string) as pointer (only valid till next call)
            char s_CompressedChunk[4096 * 400];

            //            const char* s_CompressedChunk;
            //            char* s_CompressedChunk;

            //! the last string length (only valid till next call)
            std::streamsize s_CompressedChunkLength;

            bool s_NewToken{true};
            bool s_StringEnd{false};
            bool s_IsObject{false};
            bool s_IsArray{false};
        };

        //! Has a valid document been seen?
        bool m_Initialised;

        //! Has any data been written downstream?
        bool m_SentData;

        //! The downstream data store to read from
        CDataSearcher& m_Searcher;

        //! The stream given to clients to read from
        CDataSearcher::TIStreamP m_IStream;

        //! The sequential document number currently being written to
        std::size_t m_CurrentDocNum;

        //! Have we read all the data possible from downstream?
        bool m_EndOfStream;

        //! The search configuration parameter set by the upstream caller
        std::string m_SearchString;

        //! Wrapper around the downstream reader
        std::shared_ptr<CBoostJsonUnbufferedIStreamWrapper> m_InputStreamWrapper;

        //! JSON reader for the downstream stream
        //        std::shared_ptr<json::basic_parser<SBoostJsonHandler>> m_Reader;
        json::basic_parser<SBoostJsonHandler> m_Reader;

        //        SBoostJsonHandler m_Handler;

        //! The offset into the current token that has been read
        std::streamsize m_BufferOffset;

        //! Level of nested objects, used to unwind later on.
        std::size_t m_NestedLevel;

        //! Flag to indicate that non null character has been seen by the parser
        bool m_ParsingStarted{false};
    };

public:
    //! Constructor - take a CDataSearcher for the downstream data store
    CStateDecompressor(CDataSearcher& compressedSearcher);

    //! CDataSearcher interface method - transparently read compressed
    //! data and return it in an uncompressed stream
    TIStreamP search(std::size_t currentDocNum, std::size_t limit) override;

private:
    //! The dechunker object
    CDechunkFilter m_FilterSource;

    //! The boost filtering_stream object that handles decompression
    TFilteredInputP m_InFilter;
};
}
}

#endif // INCLUDED_ml_core_CStateDecompressor_h
