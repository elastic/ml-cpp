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

//! \brief This class contains logic of creating an inference model definition
//! for boosted tree regression.
class CBoostedTreeRegressionInferenceModelFormatter {
public:
    using TStrVec = std::vector<std::string>;
    using TStrSizeUMap = std::unordered_map<std::string, std::size_t>;
    using TStrSizeUMapVec = std::vector<TStrSizeUMap>;

public:
    //! \param persistenceString string serialization of the persisted regression object.
    //! \param fieldNames names of the input fields.
    //! \param categoryNameMap mapping from string to sequence number for categorical fields.
    explicit CBoostedTreeRegressionInferenceModelFormatter(const std::string& persistenceString,
                                                           const TStrVec& fieldNames,
                                                           const TStrSizeUMapVec& categoryNameMap);

    //! Inference model definition object.
    const CInferenceModelDefinition& definition() const;
    //! Format inference model definition as a JSON string.
    std::string toString();

private:
    CInferenceModelDefinition m_Definition;
};
}
}

#endif // INCLUDED_ml_api_CInferenceModelFormatter_h
