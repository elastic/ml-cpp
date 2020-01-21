/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CCsvLineParser_h
#define INCLUDED_ml_core_CCsvLineParser_h

#include <core/CMemoryUsage.h>
#include <core/ImportExport.h>

#include <boost/scoped_array.hpp>

#include <string>

namespace ml {
namespace core {

//! \brief
//! Parses single lines of CSV formatted data.
//!
//! DESCRIPTION:\n
//! A class for parsing individual lines of CSV data.
//! Used in the implementation of the overall CSV input
//! parser, but also publicly available for use in other
//! situations.
//!
//! IMPLEMENTATION DECISIONS:\n
//! It seems like overkill to be writing a bespoke CSV parser, but none of the
//! open source options really works well:
//! - boost::escaped_list_separator doesn't cope with the fact that Excel style
//!   CSV escapes double quotes by doubling them up (i.e. "" means ")
//! - boost::spirit is just too complicated
//! - libcsv_parser++ is GPL
//! - bcsv involves pulling in a whole new string library (bstrlib)
//! - libcsv is the best of the bunch, but is LGPL which is not ideal, and
//!   although it uses the convention that quotes are escaped by doubling them
//!   up, it expects fields containing quotes to be quoted, whereas Excel format
//!   only quotes fields that contain commas or new lines
//!
class CORE_EXPORT CCsvLineParser {
public:
    //! Default CSV separator
    static const char COMMA;

    //! CSV quote character
    static const char QUOTE;

public:
    //! Construct, optionally supplying a non-standard separator.
    //! The string to be parsed must be supplied by calling the
    //! reset() method.
    CCsvLineParser(char separator = COMMA);

    //! Supply a new CSV string to be parsed.
    void reset(const std::string& line);

    //! Parse the next token from the current line.
    bool parseNext(std::string& value);

    //! Are we at the end of the current line?
    bool atEnd() const;

    //! Debug the memory used by this parser.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const;

    //! Get the memory used by this parser.
    std::size_t memoryUsage() const;

private:
    //! Attempt to parse the next token from the working record
    //! into the working field.
    bool parseNextToken(const char* end, const char*& current);

private:
    using TScopedCharArray = boost::scoped_array<char>;

private:
    //! Input field separator by default this is ',' but can be
    //! overridden in the constructor.
    const char m_Separator;

    //! Did the separator character appear after the last CSV field
    //! we parsed?
    bool m_SeparatorAfterLastField;

    //! The line to be parsed.  Held as a pointer that must outlive
    //! use of this class to avoid copying.
    const std::string* m_Line;

    //! Pointers to the current position and end of the line being
    //! parsed.
    const char* m_LineCurrent;
    const char* m_LineEnd;

    //! The working field is a raw character array rather than a
    //! string because it is built up one character at a time, and
    //! when you append a character to a string the following
    //! character has to be set to the zero terminator.  The array
    //! of characters is NOT zero terminated and hence avoids this
    //! overhead.  This is something to be aware of when accessing
    //! it, but improves performance of the parsing by about 20%.
    //! The character array is always big enough to hold the entire
    //! current row string such that the code that pulls out
    //! individual fields doesn't need to check the capacity - even
    //! if the current row has just one field, the working field
    //! array will be big enough to hold it.
    TScopedCharArray m_WorkField;
    char* m_WorkFieldEnd;
    std::size_t m_WorkFieldCapacity;
};
}
}

#endif // INCLUDED_ml_core_CCsvLineParser_h
