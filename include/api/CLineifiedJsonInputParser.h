/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CLineifiedJsonInputParser_h
#define INCLUDED_ml_api_CLineifiedJsonInputParser_h

#include <api/CLineifiedInputParser.h>
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
class API_EXPORT CLineifiedJsonInputParser : public CLineifiedInputParser {
public:
    //! Construct with an input stream to be parsed.  Once a stream is
    //! passed to this constructor, no other object should read from it.
    //! For example, if std::cin is passed, no other object should read from
    //! std::cin, otherwise unpredictable and incorrect results will be
    //! generated.
    CLineifiedJsonInputParser(std::istream& strmIn, bool allDocsSameStructure = false);

    //! Read records from the stream. The supplied reader function is called
    //! once per record.  If the supplied reader function returns false,
    //! reading will stop.  This method keeps reading until it reaches the
    //! end of the stream or an error occurs.  If it successfully reaches
    //! the end of the stream it returns true, otherwise it returns false.
    virtual bool readStream(const TReaderFunc& readerFunc);

private:
    //! Attempt to parse the current working record into data fields.
    bool parseDocument(char* begin, rapidjson::Document& document);

    bool decodeDocumentWithCommonFields(const rapidjson::Document& document,
                                        TStrVec& fieldNames,
                                        TStrRefVec& fieldValRefs,
                                        TStrStrUMap& recordFields);

    bool decodeDocumentWithArbitraryFields(const rapidjson::Document& document, TStrVec& fieldNames, TStrStrUMap& recordFields);

private:
    //! Are all JSON documents expected to contain the same fields in the
    //! same order?
    bool m_AllDocsSameStructure;
};
}
}

#endif // INCLUDED_ml_api_CLineifiedJsonInputParser_h
