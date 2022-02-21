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
#ifndef INCLUDED_ml_api_CCsvInputParser_h
#define INCLUDED_ml_api_CCsvInputParser_h

#include <core/CCsvLineParser.h>

#include <api/CInputParser.h>
#include <api/ImportExport.h>

#include <boost/scoped_array.hpp>

#include <iosfwd>
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
    //! CSV record end character
    static const char RECORD_END;

    //! Character to ignore at the end of lines
    static const char STRIP_BEFORE_END;

public:
    //! Construct with an input stream to be parsed.  Once a stream is
    //! passed to this constructor, no other object should read from it.
    //! For example, if std::cin is passed, no other object should read from
    //! std::cin, otherwise unpredictable and incorrect results will be
    //! generated.
    CCsvInputParser(std::istream& strmIn, char separator = core::CCsvLineParser::COMMA);

    //! As above but also provide some mutable field names
    CCsvInputParser(TStrVec mutableFieldNames,
                    std::istream& strmIn,
                    char separator = core::CCsvLineParser::COMMA);

    //! Read records from the stream.  The supplied reader function is called
    //! once per record.  If the supplied reader function returns false, reading
    //! will stop.  This method keeps reading until it reaches the end of the
    //! stream or an error occurs.  If it successfully reaches the end of
    //! the stream it returns true, otherwise it returns false.
    bool readStreamIntoMaps(const TMapReaderFunc& readerFunc,
                            const TRegisterMutableFieldFunc& registerFunc) override;

    //! Read records from the stream.  The supplied reader function is called
    //! once per record.  If the supplied reader function returns false, reading
    //! will stop.  This method keeps reading until it reaches the end of the
    //! stream or an error occurs.  If it successfully reaches the end of
    //! the stream it returns true, otherwise it returns false.
    bool readStreamIntoVecs(const TVecReaderFunc& readerFunc,
                            const TRegisterMutableFieldFunc& registerFunc) override;

    // Bring the other overloads into scope
    using CInputParser::readStreamIntoMaps;
    using CInputParser::readStreamIntoVecs;

private:
    using TScopedCharArray = boost::scoped_array<char>;

private:
    //! Attempt to parse a single CSV record that contains the field
    //! names for the rest of the stream.
    bool readFieldNames();

    //! Read records from the stream.  Relies on the field names having been
    //! previously read successfully.  The same working vector is populated
    //! for every record.
    template<typename READER_FUNC>
    bool parseRecordLoop(const READER_FUNC& readerFunc, TStrRefVec& workSpace);

    //! Attempt to parse a single CSV record from the stream into the
    //! working record.  The CSV is assumed to be in the Excel style.
    bool parseCsvRecordFromStream();

    //! Attempt to parse the field names from the working record.
    bool parseFieldNames();

    //! Attempt to parse the current working record into data fields.
    bool parseDataRecord(TStrRefVec& values);

private:
    //! Allocate this much memory for the working buffer
    static const std::size_t WORK_BUFFER_SIZE;

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
    const char* m_WorkBufferPtr = nullptr;
    const char* m_WorkBufferEnd = nullptr;
    bool m_NoMoreRecords = false;

    //! Field name row exactly as it appears in the input
    std::string m_FieldNameStr;

    //! Parser used to parse the individual lines
    core::CCsvLineParser m_LineParser;
};
}
}

#endif // INCLUDED_ml_api_CCsvInputParser_h
