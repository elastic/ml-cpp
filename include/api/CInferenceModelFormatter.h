/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CInferenceModelFormatter_h
#define INCLUDED_ml_api_CInferenceModelFormatter_h

#include <api/CInferenceModelDefinition.h>

#include <rapidjson/document.h>

#include <string>

namespace ml {
namespace api {

class CInferenceModelFormatter {
public:
    using TRapidjsonUPtr = std::unique_ptr<rapidjson::Document>;
    using TStrVec = std::vector<std::string>;

public:
    explicit CInferenceModelFormatter(const std::string& str, const TStrVec& fieldNames);
    explicit CInferenceModelFormatter(const rapidjson::Document& doc);

    std::string toString();

private:
    void initInput();
    void initPreprocessing();
    void initEvaluation();

private:
    std::string m_String;
    rapidjson::Document m_JsonDoc;

    ml::api::CInferenceModelDefinition m_Definition;
    std::vector<std::string> m_FieldNames;
};
}
}

#endif // INCLUDED_ml_api_CInferenceModelFormatter_h
