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
namespace core {
class CDataFrame;
class CPackedBitVector;
}
namespace api {

//! \brief Class generates a compressed and chunked JSON blob that contains
// selected data frame rows.
class API_EXPORT CDataSummarization : public CSerializableToJsonDocumentCompressed {
public:
    using TRapidJsonWriter = core::CRapidJsonConcurrentLineWriter;

public:
    CDataSummarization(const core::CDataFrame& frame, core::CPackedBitVector rowMask);

    CDataSummarization(const CDataSummarization&) = delete;
    CDataSummarization& operator=(const CDataSummarization&) = delete;

    void addToJsonStream(TGenericLineWriter& writer) const override;
    void addToDocumentCompressed(TRapidJsonWriter& writer) const override;
    std::string jsonString() const;

private:
    core::CPackedBitVector m_RowMask;
    const core::CDataFrame& m_Frame;
};
}
}
#endif //INCLUDED_ml_api_CDataSummarization_h
