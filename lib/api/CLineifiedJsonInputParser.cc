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
#include <api/CLineifiedJsonInputParser.h>

#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <sstream>

namespace ml {
namespace api {

CLineifiedJsonInputParser::CLineifiedJsonInputParser(std::istream& strmIn, bool allDocsSameStructure)
    : CLineifiedInputParser(strmIn), m_AllDocsSameStructure(allDocsSameStructure) {
}

bool CLineifiedJsonInputParser::readStream(const TReaderFunc& readerFunc) {
    TStrVec& fieldNames = this->fieldNames();
    TStrRefVec fieldValRefs;

    // Reset the record buffer pointers in case we're reading a new stream
    this->resetBuffer();

    // We reuse the same field map for every record
    TStrStrUMap recordFields;

    char* begin(this->parseLine().first);
    while (begin != nullptr) {
        rapidjson::Document document;
        if (this->parseDocument(begin, document) == false) {
            LOG_ERROR(<< "Failed to parse JSON document");
            return false;
        }

        if (m_AllDocsSameStructure) {
            if (this->decodeDocumentWithCommonFields(
                    document, fieldNames, fieldValRefs, recordFields) == false) {
                LOG_ERROR(<< "Failed to decode JSON document");
                return false;
            }
        } else {
            if (this->decodeDocumentWithArbitraryFields(document, fieldNames,
                                                        recordFields) == false) {
                LOG_ERROR(<< "Failed to decode JSON document");
                return false;
            }
        }

        if (readerFunc(recordFields) == false) {
            LOG_ERROR(<< "Record handler function forced exit");
            return false;
        }

        begin = this->parseLine().first;
    }

    return true;
}

bool CLineifiedJsonInputParser::parseDocument(char* begin, rapidjson::Document& document) {
    // Parse JSON string using Rapidjson
    if (document.ParseInsitu<rapidjson::kParseStopWhenDoneFlag>(begin).HasParseError()) {
        LOG_ERROR(<< "JSON parse error: " << document.GetParseError());
        return false;
    }

    if (!document.IsObject()) {
        LOG_ERROR(<< "Top level of JSON document must be an object: "
                  << document.GetType());
        return false;
    }

    return true;
}

bool CLineifiedJsonInputParser::decodeDocumentWithCommonFields(const rapidjson::Document& document,
                                                               TStrVec& fieldNames,
                                                               TStrRefVec& fieldValRefs,
                                                               TStrStrUMap& recordFields) {
    if (fieldValRefs.empty()) {
        // We haven't yet decoded any documents, so decode the first one long-hand
        if (this->decodeDocumentWithArbitraryFields(document, fieldNames, recordFields) == false) {
            return false;
        }

        // Cache references to the strings in the map corresponding to each field
        // name for next time
        fieldValRefs.reserve(fieldNames.size());
        for (TStrVecCItr iter = fieldNames.begin(); iter != fieldNames.end(); ++iter) {
            fieldValRefs.push_back(boost::ref(recordFields[*iter]));
        }

        return true;
    }

    TStrRefVecItr refIter = fieldValRefs.begin();
    for (rapidjson::Value::ConstMemberIterator iter = document.MemberBegin();
         iter != document.MemberEnd(); ++iter, ++refIter) {
        if (refIter == fieldValRefs.end()) {
            LOG_ERROR(<< "More fields than field references");
            return false;
        }

        switch (iter->value.GetType()) {
        case rapidjson::kNullType:
            refIter->get().clear();
            break;
        case rapidjson::kFalseType:
            refIter->get() = '0';
            break;
        case rapidjson::kTrueType:
            refIter->get() = '1';
            break;
        case rapidjson::kObjectType:
        case rapidjson::kArrayType:
            LOG_ERROR(
                << "Can't handle nested objects/arrays in JSON documents: "
                << fieldNames.back());
            return false;
        case rapidjson::kStringType:
            refIter->get().assign(iter->value.GetString(), iter->value.GetStringLength());
            break;
        case rapidjson::kNumberType:
            core::CStringUtils::typeToString(iter->value.GetDouble())
                .swap(refIter->get());
            break;
        }
    }

    return true;
}

bool CLineifiedJsonInputParser::decodeDocumentWithArbitraryFields(const rapidjson::Document& document,
                                                                  TStrVec& fieldNames,
                                                                  TStrStrUMap& recordFields) {
    // The major drawback of having self-describing messages is that we can't
    // make assumptions about what fields exist or what order they're in
    fieldNames.clear();
    recordFields.clear();

    for (rapidjson::Value::ConstMemberIterator iter = document.MemberBegin();
         iter != document.MemberEnd(); ++iter) {
        fieldNames.push_back(
            std::string(iter->name.GetString(), iter->name.GetStringLength()));

        switch (iter->value.GetType()) {
        case rapidjson::kNullType:
            recordFields[fieldNames.back()];
            break;
        case rapidjson::kFalseType:
            recordFields[fieldNames.back()] = '0';
            break;
        case rapidjson::kTrueType:
            recordFields[fieldNames.back()] = '1';
            break;
        case rapidjson::kObjectType:
        case rapidjson::kArrayType:
            LOG_ERROR(
                << "Can't handle nested objects/arrays in JSON documents: "
                << fieldNames.back());
            fieldNames.pop_back();
            return false;
        case rapidjson::kStringType:
            recordFields[fieldNames.back()].assign(iter->value.GetString(),
                                                   iter->value.GetStringLength());
            break;
        case rapidjson::kNumberType:
            core::CStringUtils::typeToString(iter->value.GetDouble())
                .swap(recordFields[fieldNames.back()]);
            break;
        }
    }

    this->gotFieldNames(true);
    this->gotData(true);

    return true;
}
}
}
