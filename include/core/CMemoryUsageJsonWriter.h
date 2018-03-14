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
#ifndef INCLUDED_ml_core_CMemoryUsageJsonWriter_h
#define INCLUDED_ml_core_CMemoryUsageJsonWriter_h

#include <core/CMemoryUsage.h>
#include <core/CRapidJsonLineWriter.h>
#include <core/ImportExport.h>

#include <rapidjson/ostreamwrapper.h>

#include <iosfwd>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace ml {
namespace core {

//! \brief A lightweight wrapper over rapidjson::LineWriter
//! to be used by CMemoryUsage to format DebugMemoryUsage info
//! in a JSON string.
//!
//! DESCRIPTION:\n
//! This class is used by CMemoryUsage to format DebugMemoryUsage info
//! in a JSON string. A stream is passed into the class at construction
//! and this is then filled with JSON objects and arrays.
//! Finalise should be called to flush the stream before
//! downstream use.
class CORE_EXPORT CMemoryUsageJsonWriter {
    public:
        //! Constructor
        CMemoryUsageJsonWriter(std::ostream &outStream);

        //! Destructor
        ~CMemoryUsageJsonWriter();

        //! Flush the underlying stream, which we only hold by reference
        void finalise();

        //! Calls underlying JSON writer startObject
        void startObject();

        //! Calls underlying JSON writer endObject()
        void endObject();

        //! Calls underlying JSON writer startArray, with a string name
        void startArray(const std::string &description);

        //! Calls underlying JSON writer endArray
        void endArray();

        //! Add a memory description item to the writer
        void addItem(const CMemoryUsage::SMemoryUsage &item);

    private:
        //! JSON writer ostream wrapper
        rapidjson::OStreamWrapper m_WriteStream;

        typedef CRapidJsonLineWriter<rapidjson::OStreamWrapper> TGenericLineWriter;

        //! JSON writer
        TGenericLineWriter        m_Writer;

        //! Have we finalised the stream?
        bool                      m_Finalised;

};


} // core
} // ml

#endif // INCLUDED_ml_core_CMemoryUsageJsonWriter_h
