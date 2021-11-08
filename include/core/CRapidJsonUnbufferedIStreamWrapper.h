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

#ifndef INCLUDED_ml_core_CRapidJsonUnbufferedIStreamWrapper_h
#define INCLUDED_ml_core_CRapidJsonUnbufferedIStreamWrapper_h

#include <core/ImportExport.h>

#include <rapidjson/rapidjson.h>

#include <istream>

namespace ml {
namespace core {

//! \brief
//! An unbuffered RapidJSON istream wrapper.
//!
//! DESCRIPTION:\n
//! RapidJSON has a class IStreamWrapper that must be used to
//! wrap istreams before they can be accessed by the parsing
//! functions.
//!
//! In the 2017 versions of RapidJSON, IStreamWrapper was
//! unbuffered. We came to depend on it being unbuffered because
//! we use it to parse documents from a stream that contains
//! additional data after the JSON document currently being
//! parsed, and hence we don't want the wrapper to consume any
//! characters beyond those in the document it's parsing.
//!
//! In the 2021 versions of RapidJSON, IStreamWrapper was
//! changed to read multiple characters from the underlying
//! istream and buffer them. This broke our code. It's also
//! not possible to simply use a single character buffer, as
//! in debug mode RapidJSON asserts that the buffer is at least
//! 4 characters in size.
//!
//! This class is an unbuffered istream wrapper compatible
//! with the RapidJSON parser functions, similar to the class
//! that existed in RapidJSON itself in 2017. It should be
//! more efficient that a buffered wrapper with a single
//! character buffer, as it avoids the extra function call to
//! fill that buffer per character in the stream.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Only works with char, not wchar_t.
//!
//! Method names have to match those required by RapidJSON,
//! which is why they do not conform to our coding standards.
//!
//! The Peek4() method used to detect character encodings
//! pretends that the stream contains fewer than 4 characters
//! in total, which disables character encoding detection.
//! This is not a problem for us as we know our input is
//! always UTF-8.
//!
class CORE_EXPORT CRapidJsonUnbufferedIStreamWrapper {
public:
    //! The stream's char type must be available as Ch.
    using Ch = char;

public:
    CRapidJsonUnbufferedIStreamWrapper(std::istream& strm);

    //! No default constructor.
    CRapidJsonUnbufferedIStreamWrapper() = delete;

    //! No copying.
    CRapidJsonUnbufferedIStreamWrapper(const CRapidJsonUnbufferedIStreamWrapper&) = delete;
    CRapidJsonUnbufferedIStreamWrapper&
    operator=(const CRapidJsonUnbufferedIStreamWrapper&) = delete;

    //! Peek the next character. Returns '\0' when the end of the stream is
    //! reached.
    char Peek() const {
        int c{m_Stream.peek()};
        return RAPIDJSON_LIKELY(c != std::istream::traits_type::eof())
                   ? static_cast<char>(c)
                   : '\0';
    }

    //! Take the next character. Returns '\0' when the end of the stream is
    //! reached.
    char Take();

    //! Return the number of characters taken.
    std::size_t Tell() const { return m_Count; }

    //! Not implemented.
    void Put(char) { RAPIDJSON_ASSERT(false); }

    //! Not implemented.
    void Flush() { RAPIDJSON_ASSERT(false); }

    //! Not implemented.
    char* PutBegin() {
        RAPIDJSON_ASSERT(false);
        return nullptr;
    }

    //! Not implemented.
    std::size_t PutEnd(char*) {
        RAPIDJSON_ASSERT(false);
        return 0;
    }

    //! For encoding detection only. In this implementation we pretend there are
    //! fewer than four characters remaining in the stream, which disables
    //! encoding detection. This is not a problem for our use case as we always
    //! work in UTF-8.
    const char* Peek4() const { return nullptr; }

private:
    //! Reference to the stream.
    std::istream& m_Stream;

    //! Count of characters taken.
    std::size_t m_Count{0};
};
}
}

#endif /*  INCLUDED_ml_core_CRapidJsonUnbufferedIStreamWrapper_h */
