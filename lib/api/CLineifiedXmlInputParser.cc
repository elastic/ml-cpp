/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CLineifiedXmlInputParser.h>

#include <core/CLogger.h>
#include <core/CXmlParserIntf.h>

#include <sstream>

namespace ml {
namespace api {

CLineifiedXmlInputParser::CLineifiedXmlInputParser(core::CXmlParserIntf& parser, std::istream& strmIn, bool allDocsSameStructure)
    : CLineifiedInputParser(strmIn), m_Parser(parser), m_AllDocsSameStructure(allDocsSameStructure) {
}

bool CLineifiedXmlInputParser::readStream(const TReaderFunc& readerFunc) {
    TStrVec& fieldNames = this->fieldNames();
    TStrRefVec fieldValRefs;

    // Reset the record buffer pointers in case we're reading a new stream
    this->resetBuffer();

    // We reuse the same field map for every record
    TStrStrUMap recordFields;

    TCharPSizePr beginLenPair(this->parseLine());
    while (beginLenPair.first != 0) {
        if (m_Parser.parseBufferInSitu(beginLenPair.first, beginLenPair.second) == false) {
            LOG_ERROR("Failed to parse XML document");
            return false;
        }

        if (m_Parser.navigateRoot() == false || m_Parser.navigateFirstChild() == false) {
            LOG_ERROR("XML document has unexpected structure");
            return false;
        }

        if (m_AllDocsSameStructure) {
            if (this->decodeDocumentWithCommonFields(fieldNames, fieldValRefs, recordFields) == false) {
                LOG_ERROR("Failed to decode XML document");
                return false;
            }
        } else {
            this->decodeDocumentWithArbitraryFields(fieldNames, recordFields);
        }

        if (readerFunc(recordFields) == false) {
            LOG_ERROR("Record handler function forced exit");
            return false;
        }

        beginLenPair = this->parseLine();
    }

    return true;
}

bool CLineifiedXmlInputParser::decodeDocumentWithCommonFields(TStrVec& fieldNames, TStrRefVec& fieldValRefs, TStrStrUMap& recordFields) {
    if (fieldValRefs.empty()) {
        // We haven't yet decoded any documents, so decode the first one long-hand
        this->decodeDocumentWithArbitraryFields(fieldNames, recordFields);

        // Cache references to the strings in the map corresponding to each field
        // name for next time
        fieldValRefs.reserve(fieldNames.size());
        for (TStrVecCItr iter = fieldNames.begin(); iter != fieldNames.end(); ++iter) {
            fieldValRefs.push_back(boost::ref(recordFields[*iter]));
        }

        return true;
    }

    size_t i(0);
    bool more(true);
    do {
        m_Parser.currentNodeValue(fieldValRefs[i]);
        ++i;
        more = m_Parser.navigateNext();
    } while (i < fieldValRefs.size() && more);

    if (i < fieldValRefs.size() || more) {
        while (more) {
            ++i;
            more = m_Parser.navigateNext();
        }

        LOG_ERROR("Incorrect number of fields: expected " << fieldValRefs.size() << ", got " << i);
        return false;
    }

    return true;
}

void CLineifiedXmlInputParser::decodeDocumentWithArbitraryFields(TStrVec& fieldNames, TStrStrUMap& recordFields) {
    // The major drawback of having self-describing messages is that we can't
    // make assumptions about what fields exist or what order they're in
    fieldNames.clear();
    recordFields.clear();

    do {
        fieldNames.push_back(std::string());
        std::string& name = fieldNames.back();
        m_Parser.currentNodeName(name);
        m_Parser.currentNodeValue(recordFields[name]);
    } while (m_Parser.navigateNext());

    this->gotFieldNames(true);
    this->gotData(true);
}
}
}
