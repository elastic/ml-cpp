/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CLineifiedJsonOutputWriter_h
#define INCLUDED_ml_api_CLineifiedJsonOutputWriter_h

#include <core/CRapidJsonLineWriter.h>

#include <api/COutputHandler.h>
#include <api/ImportExport.h>

#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>

#include <iosfwd>
#include <set>
#include <sstream>
#include <string>


namespace ml
{
namespace api
{

//! \brief
//! Write output data in JSON format, one document per line
//!
//! DESCRIPTION:\n
//! This class writes every result passed to it as a separate JSON
//! document.  Each document is restricted to a single line so that
//! whatever process consumes the output can determine where one
//! document ends and the next starts.
//!
//! IMPLEMENTATION:\n
//! Using RapidJson to do the heavy lifting.
//!
class API_EXPORT CLineifiedJsonOutputWriter : public COutputHandler
{
    public:
        typedef std::set<std::string> TStrSet;

    public:
        //! Constructor that causes output to be written to the internal string
        //! stream
        CLineifiedJsonOutputWriter(void);

        //! Constructor that causes output to be written to the internal string
        //! stream, with some numeric fields
        CLineifiedJsonOutputWriter(const TStrSet &numericFields);

        //! Constructor that causes output to be written to the specified stream
        CLineifiedJsonOutputWriter(std::ostream &strmOut);

        //! Constructor that causes output to be written to the specified stream
        CLineifiedJsonOutputWriter(const TStrSet &numericFields, std::ostream &strmOut);

        //! Destructor flushes the stream
        virtual ~CLineifiedJsonOutputWriter(void);

        // Bring the other overload of fieldNames() into scope
        using COutputHandler::fieldNames;

        //! Set field names - this function has no affect it always
        //! returns true
        virtual bool fieldNames(const TStrVec &fieldNames,
                                const TStrVec &extraFieldNames);

        // Bring the other overload of writeRow() into scope
        using COutputHandler::writeRow;

        //! Write the data row fields as a JSON object
        virtual bool writeRow(const TStrStrUMap &dataRowFields,
                              const TStrStrUMap &overrideDataRowFields);

        //! Get the contents of the internal string stream - for use with the
        //! zero argument constructor
        std::string internalString(void) const;

    private:
        //! Write a single field to the document
        void writeField(const std::string &name, 
                        const std::string &value,
                        rapidjson::Document &doc) const;

    private:
        //! Which output fields are numeric?
        TStrSet                       m_NumericFields;

        //! If we've been initialised without a specific stream, output is
        //! written to this string stream
        std::ostringstream            m_StringOutputBuf;

        //! Reference to the stream we're going to write to
        std::ostream                  &m_OutStream;

        //! JSON writer ostream wrapper
        rapidjson::OStreamWrapper     m_WriteStream;

        typedef core::CRapidJsonLineWriter<rapidjson::OStreamWrapper> TGenericLineWriter;

        //! JSON writer
        TGenericLineWriter            m_Writer;
};


}
}

#endif // INCLUDED_ml_api_CLineifiedJsonOutputWriter_h

