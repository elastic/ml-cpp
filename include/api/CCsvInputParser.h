/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CCsvInputParser_h
#define INCLUDED_ml_api_CCsvInputParser_h

#include <api/CInputParser.h>
#include <api/ImportExport.h>

#include <boost/scoped_array.hpp>

#include <iosfwd>
#include <sstream>
#include <string>

namespace ml {
namespace api {

//! \brief
//! Parse the CSV formatted input data
//!
//! DESCRIPTION:\n
//! Parse the CSV input passing each data row to a supplied callback function.
//!
//! The Input format consists of:
//! 1) Input field names as Excel style CSV
//! 2) Data rows, each in Excel style CSV
//!
//! The default separator (,) can be changed to any other character e.g. \t
//! which would make this tab separated value input parser.
//!
//! The parser for individual lines of CSV is also publicly available in the
//! nested CCsvLineParser class.
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
class API_EXPORT CCsvInputParser : public CInputParser {
public:
    //! Default CSV separator
    static const char COMMA;

    //! CSV quote character
    static const char QUOTE;

    //! CSV record end character
    static const char RECORD_END;

    //! Character to ignore at the end of lines
    static const char STRIP_BEFORE_END;

private:
    using TScopedCharArray = boost::scoped_array<char>;

public:
    //! A class for parsing individual lines of CSV data.
    //! Used in the implementation of the overall CSV input
    //! parser, but also publicly available for use in other
    //! situations.
    class API_EXPORT CCsvLineParser {
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

    private:
        //! Attempt to parse the next token from the working record
        //! into the working field.
        bool parseNextToken(const char* end, const char*& current);

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
        size_t m_WorkFieldCapacity;
    };

public:
    //! Construct with a string to be parsed
    CCsvInputParser(const std::string& input, char separator = COMMA);

    //! Construct with an input stream to be parsed.  Once a stream is
    //! passed to this constructor, no other object should read from it.
    //! For example, if std::cin is passed, no other object should read from
    //! std::cin, otherwise unpredictable and incorrect results will be
    //! generated.
    CCsvInputParser(std::istream& strmIn, char separator = COMMA);

    //! Get field name row exactly as it was in the input
    const std::string& fieldNameStr() const;

    //! Read records from the stream.  The supplied reader function is called
    //! once per record.  If the supplied reader function returns false, reading
    //! will stop.  This method keeps reading until it reaches the end of the
    //! stream or an error occurs.  If it successfully reaches the end of
    //! the stream it returns true, otherwise it returns false.
    bool readStreamAsMaps(const TMapReaderFunc& readerFunc) override;

    //! Read records from the stream.  The supplied reader function is called
    //! once per record.  If the supplied reader function returns false, reading
    //! will stop.  This method keeps reading until it reaches the end of the
    //! stream or an error occurs.  If it successfully reaches the end of
    //! the stream it returns true, otherwise it returns false.
    bool readStreamAsVecs(const TVecReaderFunc& readerFunc) override;

private:
    //! Attempt to parse a single CSV record that contains the field
    //! names for the rest of the stream.
    bool readFieldNames();

    //! Read records from the stream.  Relies on the field names having been
    //! previously read successfully.  The same working vector is populated
    //! for every record.
    template<typename READER_FUNC, typename STR_VEC>
    bool parseRecordLoop(const READER_FUNC& readerFunc, STR_VEC& workSpace);

    //! Attempt to parse a single CSV record from the stream into the
    //! working record.  The CSV is assumed to be in the Excel style.
    bool parseCsvRecordFromStream();

    //! Attempt to parse the field names from the working record.
    bool parseFieldNames();

    //! Attempt to parse the current working record into data fields.
    template<typename STR_VEC>
    bool parseDataRecord(STR_VEC& values);

    //! Wrapper around std::getline() that removes carriage returns
    //! preceding the linefeed that breaks the line.  This means that we
    //! never get confused by carriage returns in field values, whether
    //! we're running on Unix or Windows.
    std::istream& getline(std::string& str);

private:
    //! Allocate this much memory for the working buffer
    static const size_t WORK_BUFFER_SIZE;

    //! If we've been initialised with a string, this object is used to read
    //! the string
    std::istringstream m_StringInputBuf;

    //! Reference to the stream we're going to read from
    std::istream& m_StrmIn;

    //! Hold this as a member, so that its capacity adjusts to a reasonable
    //! size for the input rather than repeatedly having to allocate new
    //! string buffers.
    std::string m_CurrentRowStr;

    //! Similar to the current row string, the working buffer is also held
    //! as a member to avoid constantly reallocating it.  However, the
    //! working buffer is a raw character array rather than a string to
    //! facilitate the use of std::istream::read() to obtain input rather
    //! than std::getline().  std::getline() is efficient in the GNU STL but
    //! sadly not in the Microsoft or Apache STLs, where it copies one
    //! character at a time.  std::istream::read() uses memcpy() to shuffle
    //! data around on all platforms, and is hence an order of magnitude
    //! faster.  (This is the sort of optimisation to be used ONLY after
    //! careful profiling in the rare cases where the reduction in code
    //! clarity yields a large performance benefit.)  The array of
    //! characters is NOT zero terminated, which is something to be aware of
    //! when accessing it.
    TScopedCharArray m_WorkBuffer;
    const char* m_WorkBufferPtr;
    const char* m_WorkBufferEnd;
    bool m_NoMoreRecords;

    //! Field name row exactly as it appears in the input
    std::string m_FieldNameStr;

    //! Parser used to parse the individual lines
    CCsvLineParser m_LineParser;
};
}
}

#endif // INCLUDED_ml_api_CCsvInputParser_h
