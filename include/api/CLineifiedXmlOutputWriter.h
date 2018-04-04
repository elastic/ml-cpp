/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#ifndef INCLUDED_ml_api_CLineifiedXmlOutputWriter_h
#define INCLUDED_ml_api_CLineifiedXmlOutputWriter_h

#include <core/CXmlNodeWithChildrenPool.h>

#include <api/COutputHandler.h>
#include <api/ImportExport.h>

#include <iosfwd>
#include <sstream>
#include <string>

namespace ml {
namespace api {

//! \brief
//! Write output data in XML format, one document per line
//!
//! DESCRIPTION:\n
//! This class writes every result passed to it as a separate XML
//! document.  Each document is restricted to a single line so that
//! whatever process consumes the output can determine where one
//! document ends and the next starts.
//!
//! IMPLEMENTATION:\n
//! Using RapidXml to do the heavy lifting.
//!
class API_EXPORT CLineifiedXmlOutputWriter : public COutputHandler {
public:
    //! Constructor that causes output to be written to the internal string
    //! stream
    CLineifiedXmlOutputWriter(const std::string& rootName);

    //! Constructor that causes output to be written to the specified stream
    CLineifiedXmlOutputWriter(const std::string& rootName, std::ostream& strmOut);

    //! Destructor flushes the stream
    virtual ~CLineifiedXmlOutputWriter();

    //! Set field names - this function has no affect it always
    //! returns true
    virtual bool fieldNames(const TStrVec& fieldNames, const TStrVec& extraFieldNames);

    //! Returns an empty vector
    virtual const TStrVec& fieldNames() const;

    // Bring the other overload of fieldNames() into scope
    using COutputHandler::fieldNames;

    //! Write the data row fields as an XML document
    virtual bool writeRow(const TStrStrUMap& dataRowFields, const TStrStrUMap& overrideDataRowFields);

    // Bring the other overload of writeRow() into scope
    using COutputHandler::writeRow;

    //! Get the contents of the internal string stream - for use with the
    //! zero argument constructor
    std::string internalString() const;

private:
    //! Name of the root element in which the fields to be output will be
    //! nested
    std::string m_RootName;

    //! If we've been initialised without a specific stream, output is
    //! written to this string stream
    std::ostringstream m_StringOutputBuf;

    //! Reference to the stream we're going to write to
    std::ostream& m_OutStream;

    //! XML node pool for efficiency
    core::CXmlNodeWithChildrenPool m_Pool;
};
}
}

#endif // INCLUDED_ml_api_CLineifiedXmlOutputWriter_h
