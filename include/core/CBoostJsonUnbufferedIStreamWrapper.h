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

#ifndef INCLUDED_ml_core_CBoostJsonUnbufferedIStreamWrapper_h
#define INCLUDED_ml_core_CBoostJsonUnbufferedIStreamWrapper_h

#include <core/ImportExport.h>

#include <istream>

namespace ml {
namespace core {

//! \brief
//! An unbuffered istream wrapper.
//!
//! DESCRIPTION:\n
//! This class is an unbuffered istream wrapper. It should be
//! more efficient that a buffered wrapper with a single
//! character buffer, as it avoids the extra function call to
//! fill that buffer per character in the stream.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Only works with char, not wchar_t.
//!
class CORE_EXPORT CBoostJsonUnbufferedIStreamWrapper {
public:
    //! The stream's char type must be available as Ch.
    using Ch = char;

public:
    explicit CBoostJsonUnbufferedIStreamWrapper(std::istream& strm);

    //! No default constructor.
    CBoostJsonUnbufferedIStreamWrapper() = delete;

    //! No copying.
    CBoostJsonUnbufferedIStreamWrapper(const CBoostJsonUnbufferedIStreamWrapper&) = delete;
    CBoostJsonUnbufferedIStreamWrapper&
    operator=(const CBoostJsonUnbufferedIStreamWrapper&) = delete;

    //! Peek the next character. Returns '\0' when the end of the stream is
    //! reached.
    char peek() const {
        int c{m_Stream.peek()};
        return (c != std::istream::traits_type::eof()) ? static_cast<char>(c) : '\0';
    }

    //! Take the next character. Returns '\0' when the end of the stream is
    //! reached.
    char take();

    //! Return the number of characters taken.
    std::size_t tell() const { return m_Count; }

private:
    //! Reference to the stream.
    std::istream& m_Stream;

    //! Count of characters taken.
    std::size_t m_Count{0};
};
}
}

#endif /*  INCLUDED_ml_core_CBoostJsonUnbufferedIStreamWrapper_h */
