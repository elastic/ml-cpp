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

class CBoostedTreeRegressionInferenceModelFormatter {
public:
    using TRapidjsonUPtr = std::unique_ptr<rapidjson::Document>;
    using TStrVec = std::vector<std::string>;
    using TStrSizeUMap = std::unordered_map<std::string, std::size_t>;
    using TStrSizeUMapVec = std::vector<TStrSizeUMap>;

public:
    explicit CBoostedTreeRegressionInferenceModelFormatter(const std::string& str,
                                                           const TStrVec& fieldNames,
                                                           const TStrSizeUMapVec& categoryNameMap);

    std::string toString();

private:
    ml::api::CInferenceModelDefinition m_Definition;

public:
    const CInferenceModelDefinition& definition() const;
};
}
}

#endif // INCLUDED_ml_api_CInferenceModelFormatter_h
