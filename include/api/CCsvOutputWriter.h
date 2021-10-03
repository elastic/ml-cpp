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
#ifndef INCLUDED_ml_api_CCsvOutputWriter_h
#define INCLUDED_ml_api_CCsvOutputWriter_h

#include <api/CSimpleOutputWriter.h>
#include <api/ImportExport.h>

#include <iosfwd>
#include <sstream>
#include <string>

namespace ml {
namespace api {

//! \brief
//! Write output data in CSV format
//!
//! DESCRIPTION:\n
//! Write output data in the CSV format (Excel style by default) either
//! to a stream or a string.
//!
//! The output format consists of:
//! 1) Field names
//! 2) Data rows
//!
//! The default separator (,) can be changed to any other character e.g. \t
//! which would make this a tab separated value output writer.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Using the Excel style CSV format by default.  So, by default:
//! - Fields are only quoted if they contain a quote, the separator or
//!   the record end character
//! - Quotes are escaped by doubling them up
//!
//! It is not acceptable to have the separator character be the same as the
//! escape character, the quote character or the record end character.
//!
class API_EXPORT CCsvOutputWriter : public CSimpleOutputWriter {
public:
    //! CSV separator
    static const char COMMA;

    //! CSV quote character
    static const char QUOTE;

    //! CSV record end character
    static const char RECORD_END;

public:
    //! Constructor that causes output to be written to the internal string
    //! stream
    CCsvOutputWriter(bool outputHeader = true, char escape = QUOTE, char separator = COMMA);

    //! Constructor that causes output to be written to the specified stream
    CCsvOutputWriter(std::ostream& strmOut,
                     bool outputHeader = true,
                     char escape = QUOTE,
                     char separator = COMMA);

    //! Destructor flushes the stream
    ~CCsvOutputWriter() override;

    // Bring the other overload of fieldNames() into scope
    using CSimpleOutputWriter::fieldNames;

    //! Set field names, adding extra field names if they're not already
    //! present - this is only allowed once
    bool fieldNames(const TStrVec& fieldNames, const TStrVec& extraFieldNames) override;

    // Bring the other overloads of writeRow() into scope
    using CSimpleOutputWriter::writeRow;

    //! Write a row to the stream, optionally overriding some of the
    //! original field values.  Where the same field is present in both
    //! overrideDataRowFields and dataRowFields, the value in
    //! overrideDataRowFields will be written.
    bool writeRow(const TStrStrUMap& dataRowFields,
                  const TStrStrUMap& overrideDataRowFields) override;

    //! Get the contents of the internal string stream - for use with the
    //! zero argument constructor
    std::string internalString() const;

protected:
    //! Output stream accessor
    std::ostream& outputStream();

private:
    //! Append a field to the work record, quoting it if required, and
    //! escaping embedded quotes
    void appendField(const std::string& field);

private:
    //! If we've been initialised without a specific stream, output is
    //! written to this string stream
    std::ostringstream m_StringOutputBuf;

    //! Reference to the stream we're going to write to
    std::ostream& m_StrmOut;

    //! Should we output a row containing the CSV column names?
    bool m_OutputHeader;

    //! CSV field names in the order they are to be written to the output
    TStrVec m_FieldNames;

    //! Pre-computed hashes for each field name.  The pre-computed hashes
    //! are at the same index in this vector as the corresponding field name
    //! in the m_FieldNames vector.
    TPreComputedHashVec m_Hashes;

    //! Used to build up output records before writing them to the output
    //! stream, so that invalid write requests can have no effect on the
    //! output stream.  Held as a member so that the capacity adjusts to
    //! an appropriate level, avoiding regular memory allocations.
    std::string m_WorkRecord;

    //! Character to use for escaping quotes (const to allow compiler
    //! optimisations, since the value can't be changed after construction)
    const char m_Escape;

    //! Output field separator by default this is ',' but can be
    //! overridden in the constructor
    const char m_Separator;
};
}
}

#endif // INCLUDED_ml_api_CCsvOutputWriter_h
