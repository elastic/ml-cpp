/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CDataSummarization_h
#define INCLUDED_ml_api_CDataSummarization_h

#include <api/CInferenceModelDefinition.h>

#include <sstream>
#include <string>

namespace ml {
namespace api {
class API_EXPORT CDataSummarization : public CSerializableToJsonStream {
public:
    using TRapidJsonWriter = core::CRapidJsonConcurrentLineWriter;

public:
    void addToJsonStream(TGenericLineWriter& writer) const override;
    void addToDocumentCompressed(TRapidJsonWriter& writer) const;
    std::string jsonString() const;
    void jsonStream(std::ostream& jsonStrm) const;
    std::stringstream jsonCompressedStream() const;
};
}
}
#endif //INCLUDED_ml_api_CDataSummarization_h
