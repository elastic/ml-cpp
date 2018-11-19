/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CNdJsonInputParser.h>

#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <sstream>

namespace ml {
namespace api {

CNdJsonInputParser::CNdJsonInputParser(std::istream& strmIn, bool allDocsSameStructure)
    : CNdInputParser(strmIn), m_AllDocsSameStructure(allDocsSameStructure) {
}

bool CNdJsonInputParser::readStreamAsMaps(const TMapReaderFunc& readerFunc) {
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

bool CNdJsonInputParser::readStreamAsVecs(const TVecReaderFunc& readerFunc) {
    TStrVec& fieldNames = this->fieldNames();

    // Reset the record buffer pointers in case we're reading a new stream
    this->resetBuffer();

    // We reuse the same field vector for every record
    TStrVec fieldValues;

    char* begin(this->parseLine().first);
    while (begin != nullptr) {
        rapidjson::Document document;
        if (this->parseDocument(begin, document) == false) {
            LOG_ERROR(<< "Failed to parse JSON document");
            return false;
        }

        if (m_AllDocsSameStructure) {
            if (this->decodeDocumentWithCommonFields(document, fieldNames, fieldValues) == false) {
                LOG_ERROR(<< "Failed to decode JSON document");
                return false;
            }
        } else {
            if (this->decodeDocumentWithArbitraryFields(document, fieldNames,
                                                        fieldValues) == false) {
                LOG_ERROR(<< "Failed to decode JSON document");
                return false;
            }
        }

        if (readerFunc(fieldNames, fieldValues) == false) {
            LOG_ERROR(<< "Record handler function forced exit");
            return false;
        }

        begin = this->parseLine().first;
    }

    return true;
}

bool CNdJsonInputParser::parseDocument(char* begin, rapidjson::Document& document) {
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

bool CNdJsonInputParser::decodeDocumentWithCommonFields(const rapidjson::Document& document,
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
        for (const auto& fieldName : fieldNames) {
            fieldValRefs.emplace_back(recordFields[fieldName]);
        }

        return true;
    }

    auto nameIter = fieldNames.begin();
    auto refIter = fieldValRefs.begin();
    for (auto iter = document.MemberBegin(); iter != document.MemberEnd(); ++iter, ++refIter) {
        if (nameIter == fieldNames.end() || refIter == fieldValRefs.end()) {
            LOG_ERROR(<< "More fields than field references");
            return false;
        }

        if (this->jsonValueToString(*nameIter, iter->value, *refIter) == false) {
            return false;
        }
    }

    return true;
}

bool CNdJsonInputParser::decodeDocumentWithCommonFields(const rapidjson::Document& document,
                                                        TStrVec& fieldNames,
                                                        TStrVec& fieldValues) {
    if (fieldValues.empty()) {
        // We haven't yet decoded any documents, so decode the first one long-hand
        return this->decodeDocumentWithArbitraryFields(document, fieldNames, fieldValues);
    }

    auto nameIter = fieldNames.begin();
    auto valueIter = fieldValues.begin();
    for (auto iter = document.MemberBegin(); iter != document.MemberEnd();
         ++iter, ++nameIter, ++valueIter) {
        if (nameIter == fieldNames.end() || valueIter == fieldValues.end()) {
            LOG_ERROR(<< "More fields than fields");
            return false;
        }

        if (this->jsonValueToString(*nameIter, iter->value, *valueIter) == false) {
            return false;
        }
    }

    return true;
}

bool CNdJsonInputParser::decodeDocumentWithArbitraryFields(const rapidjson::Document& document,
                                                           TStrVec& fieldNames,
                                                           TStrStrUMap& recordFields) {
    // The major drawback of having self-describing messages is that we can't
    // make assumptions about what fields exist or what order they're in
    fieldNames.clear();
    recordFields.clear();

    for (auto iter = document.MemberBegin(); iter != document.MemberEnd(); ++iter) {
        fieldNames.emplace_back(iter->name.GetString(), iter->name.GetStringLength());
        const std::string& fieldName = fieldNames.back();
        if (this->jsonValueToString(fieldName, iter->value, recordFields[fieldName]) == false) {
            return false;
        }
    }

    this->gotFieldNames(true);
    this->gotData(true);

    return true;
}

bool CNdJsonInputParser::decodeDocumentWithArbitraryFields(const rapidjson::Document& document,
                                                           TStrVec& fieldNames,
                                                           TStrVec& fieldValues) {
    // The major drawback of having self-describing messages is that we can't
    // make assumptions about what fields exist or what order they're in
    fieldNames.clear();
    fieldValues.clear();

    for (auto iter = document.MemberBegin(); iter != document.MemberEnd(); ++iter) {
        fieldNames.emplace_back(iter->name.GetString(), iter->name.GetStringLength());
        fieldValues.emplace_back();
        const std::string& fieldName = fieldNames.back();
        std::string& fieldValue = fieldValues.back();
        if (this->jsonValueToString(fieldName, iter->value, fieldValue) == false) {
            return false;
        }
    }

    this->gotFieldNames(true);
    this->gotData(true);

    return true;
}

bool CNdJsonInputParser::jsonValueToString(const std::string& fieldName,
                                           const rapidjson::Value& jsonValue,
                                           std::string& fieldValueStr) {
    switch (jsonValue.GetType()) {
    case rapidjson::kNullType:
        fieldValueStr.clear();
        break;
    case rapidjson::kFalseType:
        fieldValueStr = '0';
        break;
    case rapidjson::kTrueType:
        fieldValueStr = '1';
        break;
    case rapidjson::kObjectType:
    case rapidjson::kArrayType:
        LOG_ERROR(<< "Can't handle nested objects/arrays in JSON documents: " << fieldName);
        return false;
    case rapidjson::kStringType:
        fieldValueStr.assign(jsonValue.GetString(), jsonValue.GetStringLength());
        break;
    case rapidjson::kNumberType:
        fieldValueStr = core::CStringUtils::typeToString(jsonValue.GetDouble());
        break;
    }

    return true;
}
}
}
