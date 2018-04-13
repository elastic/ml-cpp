/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CLineifiedXmlInputParser_h
#define INCLUDED_ml_api_CLineifiedXmlInputParser_h

#include <api/CLineifiedInputParser.h>
#include <api/ImportExport.h>

#include <iosfwd>

namespace ml {
namespace core {
class CXmlParserIntf;
}
namespace api {

//! \brief
//! Parse XML input where each line is a separate XML document.
//!
//! DESCRIPTION:\n
//! Since newline characters within values can be represented as &#xA; in
//! XML, it is always possible to write a whole XML document to a single
//! line.  Doing this makes it easy to put many XML documents into a file
//! or data stream and easily tell where one ends and the next begins.
//!
//! This class is designed to parse such data.  Each line is expected to be
//! a complete single XML document that will be converted to a single event
//! for processing.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Takes an interface to the XML parser as a constructor argument to allow
//! performance comparisons between different XML parsers.
//!
class API_EXPORT CLineifiedXmlInputParser : public CLineifiedInputParser {
public:
    //! Construct with an XML parser interface and an input stream to be
    //! parsed.  Once a stream is passed to this constructor, no other
    //! object should read from it.  For example, if std::cin is passed, no
    //! other object should read from std::cin, otherwise unpredictable and
    //! incorrect results will be generated.
    CLineifiedXmlInputParser(core::CXmlParserIntf& parser,
                             std::istream& strmIn,
                             bool allDocsSameStructure = false);

    //! Read records from the stream. The supplied reader function is called
    //! once per record.  If the supplied reader function returns false,
    //! reading will stop.  This method keeps reading until it reaches the
    //! end of the stream or an error occurs.  If it successfully reaches
    //! the end of the stream it returns true, otherwise it returns false.
    virtual bool readStream(const TReaderFunc& readerFunc);

private:
    //! Attempt to parse the current working record into data fields.
    bool decodeDocumentWithCommonFields(TStrVec& fieldNames,
                                        TStrRefVec& fieldValRefs,
                                        TStrStrUMap& recordFields);

    void decodeDocumentWithArbitraryFields(TStrVec& fieldNames, TStrStrUMap& recordFields);

private:
    //! Reference to the parser we're going to use
    core::CXmlParserIntf& m_Parser;

    //! Are all XML documents expected to contain the same fields in the
    //! same order?
    bool m_AllDocsSameStructure;
};
}
}

#endif // INCLUDED_ml_api_CLineifiedXmlInputParser_h
