/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */
#include <api/CNdJsonInputParser.h>

#include <core/CLogger.h>
#include <core/CStringUtils.h>

namespace ml {
namespace api {

CNdJsonInputParser::CNdJsonInputParser(std::istream& strmIn, bool allDocsSameStructure)
    : CNdInputParser{TStrVec{}, strmIn}, m_AllDocsSameStructure{allDocsSameStructure} {
}

CNdJsonInputParser::CNdJsonInputParser(TStrVec mutableFieldNames, std::istream& strmIn, bool allDocsSameStructure)
    : CNdInputParser{std::move(mutableFieldNames), strmIn}, m_AllDocsSameStructure{allDocsSameStructure} {
}

bool CNdJsonInputParser::readStreamIntoMaps(const TMapReaderFunc& readerFunc,
                                            const TRegisterMutableFieldFunc& registerFunc) {
    TStrVec& fieldNames = this->fieldNames();
    TStrRefVec fieldValRefs;

    // Reset the record buffer pointers in case we're reading a new stream
    this->resetBuffer();

    // We reuse the same field map for every record
    TStrStrUMap recordFields;

    char* begin(this->parseLine().first);
    while (begin != nullptr) {
        json::value document;
        if (this->parseDocument(begin, document) == false) {
            LOG_ERROR(<< "Failed to parse JSON document");
            return false;
        }

        if (m_AllDocsSameStructure) {
            if (this->decodeDocumentWithCommonFields(registerFunc, document, fieldNames,
                                                     fieldValRefs, recordFields) == false) {
                LOG_ERROR(<< "Failed to decode JSON document");
                return false;
            }
        } else {
            if (this->decodeDocumentWithArbitraryFields(
                    registerFunc, document, fieldNames, recordFields) == false) {
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

bool CNdJsonInputParser::readStreamIntoVecs(const TVecReaderFunc& readerFunc,
                                            const TRegisterMutableFieldFunc& registerFunc) {
    TStrVec& fieldNames{this->fieldNames()};

    // Reset the record buffer pointers in case we're reading a new stream
    this->resetBuffer();

    // We reuse the same field vector for every record
    TStrVec fieldValues;

    char* begin{this->parseLine().first};
    while (begin != nullptr) {
        json::value document;
        if (this->parseDocument(begin, document) == false) {
            LOG_ERROR(<< "Failed to parse JSON document");
            return false;
        }

        if (m_AllDocsSameStructure) {
            if (this->decodeDocumentWithCommonFields(
                    registerFunc, document, fieldNames, fieldValues) == false) {
                LOG_ERROR(<< "Failed to decode JSON document");
                return false;
            }
        } else {
            if (this->decodeDocumentWithArbitraryFields(
                    registerFunc, document, fieldNames, fieldValues) == false) {
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

bool CNdJsonInputParser::parseDocument(char* begin, json::value& document) {
    // Parse JSON string
    json::error_code ec;
    json::stream_parser p;
    std::string line;
    char buffer[4096];
    size_t bytesRead{0};
    while (begin != nullptr) {
        buffer[bytesRead] = *begin++;
        ++bytesRead;
        if (bytesRead == 4096) {
            p.write_some(buffer, sizeof(buffer), ec);
            if (ec) {
                LOG_ERROR(<< "JSON parse error: " << ec.message());
                return false;
            }
        }
    }
    p.write_some(buffer, sizeof(buffer), ec);
    if (ec) {
        LOG_ERROR(<< "JSON parse error: " << ec.message());
        return false;
    }
    p.finish( ec );
    if( ec ) {
        LOG_ERROR(<< "JSON parse error: " << ec.message());
        return false;
    }
    document = p.release();

    if (document.is_object() == false) {
        LOG_ERROR(<< "Top level of JSON document must be an object: "
                  << document.kind());
        return false;
    }

    return true;
}

bool CNdJsonInputParser::decodeDocumentWithCommonFields(const TRegisterMutableFieldFunc& registerFunc,
                                                        const json::value& document,
                                                        TStrVec& fieldNames,
                                                        TStrRefVec& fieldValRefs,
                                                        TStrStrUMap& recordFields) {
    if (fieldValRefs.empty()) {
        // We haven't yet decoded any documents, so decode the first one long-hand
        if (this->decodeDocumentWithArbitraryFields(registerFunc, document, fieldNames,
                                                    recordFields) == false) {
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
    for (auto iter = document.as_object().begin(); iter != document.as_object().end(); ++iter, ++refIter) {
        if (nameIter == fieldNames.end() || refIter == fieldValRefs.end()) {
            LOG_ERROR(<< "More fields than field references");
            return false;
        }

        if (this->jsonValueToString(*nameIter, iter->value(), *refIter) == false) {
            return false;
        }
    }

    return true;
}

bool CNdJsonInputParser::decodeDocumentWithCommonFields(const TRegisterMutableFieldFunc& registerFunc,
                                                        const json::value& document,

                                                        TStrVec& fieldNames,
                                                        TStrVec& fieldValues) {
    if (fieldValues.empty()) {
        // We haven't yet decoded any documents, so decode the first one long-hand
        return this->decodeDocumentWithArbitraryFields(registerFunc, document,
                                                       fieldNames, fieldValues);
    }

    auto nameIter = fieldNames.begin();
    auto valueIter = fieldValues.begin();
    for (auto iter = document.as_object().begin(); iter != document.as_object().end();
         ++iter, ++nameIter, ++valueIter) {
        if (nameIter == fieldNames.end() || valueIter == fieldValues.end()) {
            LOG_ERROR(<< "More fields in document than common fields");
            return false;
        }

        if (this->jsonValueToString(*nameIter, iter->value(), *valueIter) == false) {
            return false;
        }
    }

    return true;
}

bool CNdJsonInputParser::decodeDocumentWithArbitraryFields(const TRegisterMutableFieldFunc& registerFunc,
                                                           const json::value& document,
                                                           TStrVec& fieldNames,
                                                           TStrStrUMap& recordFields) {
    // The major drawback of having self-describing messages is that we can't
    // make assumptions about what fields exist or what order they're in
    fieldNames.clear();
    recordFields.clear();

    for (auto iter = document.as_object().begin(); iter != document.as_object().end(); ++iter) {
        fieldNames.emplace_back(iter->key(), iter->key().size());
        const std::string& fieldName = fieldNames.back();
        if (this->jsonValueToString(fieldName, iter->value(), recordFields[fieldName]) == false) {
            return false;
        }
    }

    this->registerMutableFields(registerFunc, recordFields);

    return true;
}

bool CNdJsonInputParser::decodeDocumentWithArbitraryFields(const TRegisterMutableFieldFunc& registerFunc,
                                                           const json::value& document,
                                                           TStrVec& fieldNames,
                                                           TStrVec& fieldValues) {
    // The major drawback of having self-describing messages is that we can't
    // make assumptions about what fields exist or what order they're in
    fieldNames.clear();
    fieldValues.clear();

    for (auto iter = document.as_object().begin(); iter != document.as_object().end(); ++iter) {
        fieldNames.emplace_back(iter->key(), iter->key().size());
        fieldValues.emplace_back();
        const std::string& fieldName = fieldNames.back();
        std::string& fieldValue = fieldValues.back();
        if (this->jsonValueToString(fieldName, iter->value(), fieldValue) == false) {
            return false;
        }
    }

    this->registerMutableFields(registerFunc, fieldNames, fieldValues);

    return true;
}

bool CNdJsonInputParser::jsonValueToString(const std::string& fieldName,
                                           const json::value& jsonValue,
                                           std::string& fieldValueStr) {
    fieldValueStr = json::serialize(jsonValue);
    return true;
}
}
}
