/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CInferenceModelFormatter_h
#define INCLUDED_ml_api_CInferenceModelFormatter_h

#include <api/SInferenceModelDefinition.h>

#include <rapidjson/document.h>

#include <string>

namespace ml {
namespace api {

class CInferenceModelFormatter {
public:
    using TRapidjsonUPtr = std::unique_ptr<rapidjson::Document>;
public:
    explicit CInferenceModelFormatter(const std::string& str);
    explicit CInferenceModelFormatter(const rapidjson::Document& doc);

    std::string toString();

private:
    void initInput();
    void initPreprocessing();
    void initEvaluation();

private:
    std::string m_String;
    rapidjson::Document m_JsonDoc;

    ml::api::SInferenceModelDefinition m_Definition;

};

}
}

#endif // INCLUDED_ml_api_CInferenceModelFormatter_h
