/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CNdJsonOutputWriter.h>

#include <core/CScopedRapidJsonPoolAllocator.h>
#include <core/CSleep.h>
#include <core/CStringUtils.h>

#include <ostream>

namespace ml {
namespace api {

CNdJsonOutputWriter::CNdJsonOutputWriter()
    : m_OutStream(m_StringOutputBuf), m_WriteStream(m_OutStream),
      m_Writer(m_WriteStream) {
}

CNdJsonOutputWriter::CNdJsonOutputWriter(const TStrSet& numericFields)
    : m_NumericFields(numericFields), m_OutStream(m_StringOutputBuf),
      m_WriteStream(m_OutStream), m_Writer(m_WriteStream) {
}

CNdJsonOutputWriter::CNdJsonOutputWriter(std::ostream& strmOut)
    : m_OutStream(strmOut), m_WriteStream(m_OutStream), m_Writer(m_WriteStream) {
}

CNdJsonOutputWriter::CNdJsonOutputWriter(const TStrSet& numericFields, std::ostream& strmOut)
    : m_NumericFields(numericFields), m_OutStream(strmOut),
      m_WriteStream(m_OutStream), m_Writer(m_WriteStream) {
}

CNdJsonOutputWriter::~CNdJsonOutputWriter() {
    // Since we didn't flush the stream whilst working, we flush it on
    // destruction
    m_WriteStream.Flush();

    // We don't want the program to die before the remote end of the link has
    // had a chance to read from any pipe to which our output stream might be
    // connected, so sleep briefly here
    core::CSleep::sleep(20);
}

bool CNdJsonOutputWriter::fieldNames(const TStrVec& /*fieldNames*/,
                                     const TStrVec& /*extraFieldNames*/) {
    return true;
}

bool CNdJsonOutputWriter::writeRow(const TStrStrUMap& dataRowFields,
                                   const TStrStrUMap& overrideDataRowFields) {
    using TScopedAllocator = core::CScopedRapidJsonPoolAllocator<TGenericLineWriter>;
    TScopedAllocator scopedAllocator("CNdJsonOutputWriter::writeRow", m_Writer);

    rapidjson::Document doc = m_Writer.makeDoc();

    // Write all the fields to the document as strings
    // No need to copy the strings as the doc is written straight away
    TStrStrUMapCItr fieldValueIter = dataRowFields.begin();
    for (; fieldValueIter != dataRowFields.end(); ++fieldValueIter) {
        const std::string& name = fieldValueIter->first;
        const std::string& value = fieldValueIter->second;

        // Only output fields that aren't overridden
        if (overrideDataRowFields.find(name) == overrideDataRowFields.end()) {
            this->writeField(name, value, doc);
        }
    }

    fieldValueIter = overrideDataRowFields.begin();
    for (; fieldValueIter != overrideDataRowFields.end(); ++fieldValueIter) {
        const std::string& name = fieldValueIter->first;
        const std::string& value = fieldValueIter->second;

        this->writeField(name, value, doc);
    }

    m_Writer.write(doc);
    m_Writer.Reset(m_WriteStream);

    return true;
}

std::string CNdJsonOutputWriter::internalString() const {
    const_cast<rapidjson::OStreamWrapper&>(m_WriteStream).Flush();

    // This is only of any value if the first constructor was used - it's up to
    // the caller to know this
    return m_StringOutputBuf.str();
}

void CNdJsonOutputWriter::writeField(const std::string& name,
                                     const std::string& value,
                                     rapidjson::Document& doc) const {
    if (m_NumericFields.find(name) != m_NumericFields.end()) {
        double numericValue(0.0);
        if (core::CStringUtils::stringToType(value, numericValue) == false) {
            LOG_WARN(<< "Non-numeric value output in numeric JSON document");
            // Write a 0 instead of returning
        }
        m_Writer.addDoubleFieldToObj(name, numericValue, doc);
    } else {
        m_Writer.addStringFieldCopyToObj(name, value, doc, true);
    }
}
}
}
