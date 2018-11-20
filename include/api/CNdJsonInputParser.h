/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CNdJsonInputParser_h
#define INCLUDED_ml_api_CNdJsonInputParser_h

#include <api/CNdInputParser.h>
#include <api/ImportExport.h>

#include <rapidjson/document.h>

#include <iosfwd>
#include <string>

namespace ml {
namespace api {

//! \brief
//! Parse JSON input where each line is a separate JSON document
//!
//! DESCRIPTION:\n
//! Since newline characters within values are represented as \n in JSON, it
//! is always possible to write a whole JSON document to a single line.  Doing
//! this makes it easy to put many JSON documents into a file or data stream
//! and easily tell where one ends and the next begins.
//!
//! This class is designed to parse such data.  Each line is expected to be
//! a complete single JSON document that will be converted to a single event
//! for processing.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Using the RapidJson library to do the heavy lifting, but copying output
//! to standard STL/Boost data structures.
//!
//! It is possible to tell the parser that all documents have exactly the
//! same structure, i.e. the same field names in the same order.  In
//! this case the field names vector is only populated once.  If the
//! documents turn out to have different structures then parsing may
//! detect an error but may instead pass incorrect records to the handler
//! function.  The default is the less efficient but safer option of
//! parsing the field names separately from each document.
//!
class API_EXPORT CNdJsonInputParser : public CNdInputParser {
public:
    //! Construct with an input stream to be parsed.  Once a stream is
    //! passed to this constructor, no other object should read from it.
    //! For example, if std::cin is passed, no other object should read from
    //! std::cin, otherwise unpredictable and incorrect results will be
    //! generated.
    CNdJsonInputParser(std::istream& strmIn, bool allDocsSameStructure = false);

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
    //! Attempt to parse the current working record into data fields.
    bool parseDocument(char* begin, rapidjson::Document& document);

    bool decodeDocumentWithCommonFields(const rapidjson::Document& document,
                                        TStrVec& fieldNames,
                                        TStrRefVec& fieldValRefs,
                                        TStrStrUMap& recordFields);

    bool decodeDocumentWithCommonFields(const rapidjson::Document& document,
                                        TStrVec& fieldNames,
                                        TStrVec& fieldValues);

    bool decodeDocumentWithArbitraryFields(const rapidjson::Document& document,
                                           TStrVec& fieldNames,
                                           TStrStrUMap& recordFields);

    bool decodeDocumentWithArbitraryFields(const rapidjson::Document& document,
                                           TStrVec& fieldNames,
                                           TStrVec& fieldValues);

    static bool jsonValueToString(const std::string& fieldName,
                                  const rapidjson::Value& jsonValue,
                                  std::string& fieldValueStr);

private:
    //! Are all JSON documents expected to contain the same fields in the
    //! same order?
    bool m_AllDocsSameStructure;
};
}
}

#endif // INCLUDED_ml_api_CNdJsonInputParser_h
