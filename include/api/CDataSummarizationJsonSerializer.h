/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CDataSummarizationJsonSerializer_h
#define INCLUDED_ml_api_CDataSummarizationJsonSerializer_h

#include <core/CPackedBitVector.h>

#include <maths/CBoostedTreeFactory.h>

#include <api/CSerializableToJson.h>

#include <istream>
#include <memory>
#include <sstream>
#include <string>

namespace ml {
namespace core {
class CDataFrame;
}
namespace api {

//! \brief Class serializes and deserializes data summarization using chunked JSON blob.
//! Data summarization contains data rows as well as categorical encoding information.
// TODO #1849 chunking is not supported yet.
class API_EXPORT CDataSummarizationJsonSerializer final
    : public CSerializableToJsonDocumentCompressed {
public:
    CDataSummarizationJsonSerializer(const core::CDataFrame& frame,
                                     core::CPackedBitVector rowMask,
                                     std::size_t numberColumns,
                                     std::stringstream encodings);

    CDataSummarizationJsonSerializer(const CDataSummarizationJsonSerializer&) = delete;
    CDataSummarizationJsonSerializer&
    operator=(const CDataSummarizationJsonSerializer&) = delete;

    void addToJsonStream(TGenericLineWriter& writer) const override;
    void addToDocumentCompressed(TRapidJsonWriter& writer) const override;
    std::string jsonString() const;

private:
    core::CPackedBitVector m_RowMask;
    std::size_t m_NumberColumns;
    const core::CDataFrame& m_Frame;
    std::stringstream m_Encodings;
};

class API_EXPORT CRetrainableModelJsonDeserializer {
public:
    using TEncoderUPtr = maths::CBoostedTreeFactory::TEncoderUPtr;
    using TNodeVecVecUPtr = maths::CBoostedTreeFactory::TNodeVecVecUPtr;
    using TIStreamSPtr = std::shared_ptr<std::istream>;

public:
    //! \brief Retrieve data summarization from decompressed JSON stream.
    static TEncoderUPtr dataSummarizationFromJsonStream(const TIStreamSPtr& istream,
                                                        core::CDataFrame& frame);

    //! \brief Retrieve data summarization from compressed and chunked JSON blob.
    static TEncoderUPtr dataSummarizationFromDocumentCompressed(const TIStreamSPtr& istream,
                                                                core::CDataFrame& frame);

    //! \brief Retrieve best forest from decompressed JSON stream.
    static TNodeVecVecUPtr
    bestForestFromJsonStream(const core::CDataSearcher::TIStreamP& istream);

    //! \brief Retrieve best forest from compressed and chunked JSON blob.
    static TNodeVecVecUPtr
    bestForestFromDocumentCompressed(const core::CDataSearcher::TIStreamP& istream);
};
}
}
#endif //INCLUDED_ml_api_CDataSummarizationJsonSerializer_h
