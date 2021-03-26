/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CDataSummarizationJsonSerializer_h
#define INCLUDED_ml_api_CDataSummarizationJsonSerializer_h

#include <core/CPackedBitVector.h>

#include <api/CSerializableToJson.h>

#include <sstream>
#include <string>

namespace ml {
namespace core {
class CDataFrame;
}
namespace api {

//! \brief Class generates a compressed and chunked JSON blob that contains
// selected data frame rows.
class API_EXPORT CDataSummarizationJsonSerializer final
    : public CSerializableToJsonDocumentCompressed {
public:
    using TRapidJsonWriter = core::CRapidJsonConcurrentLineWriter;

public:
    CDataSummarizationJsonSerializer(const core::CDataFrame& frame,
                                     core::CPackedBitVector rowMask);

    CDataSummarizationJsonSerializer(const CDataSummarizationJsonSerializer&) = delete;
    CDataSummarizationJsonSerializer&
    operator=(const CDataSummarizationJsonSerializer&) = delete;

    void addToJsonStream(TGenericLineWriter& writer) const override;
    void addToDocumentCompressed(TRapidJsonWriter& writer) const override;
    std::string jsonString() const;

private:
    core::CPackedBitVector m_RowMask;
    const core::CDataFrame& m_Frame;
};
}
}
#endif //INCLUDED_ml_api_CDataSummarizationJsonSerializer_h
