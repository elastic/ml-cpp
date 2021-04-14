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
                                     std::stringstream encodings);

    CDataSummarizationJsonSerializer(const CDataSummarizationJsonSerializer&) = delete;
    CDataSummarizationJsonSerializer&
    operator=(const CDataSummarizationJsonSerializer&) = delete;

    void addToJsonStream(TGenericLineWriter& writer) const override;
    void addToDocumentCompressed(TRapidJsonWriter& writer) const override;
    std::string jsonString() const;

private:
    core::CPackedBitVector m_RowMask;
    const core::CDataFrame& m_Frame;
    std::stringstream m_Encodings;
};

class CRetrainableModelJsonDeserializer {
public:
    using TDataSummarization = maths::CBoostedTreeFactory::TDataSummarization;
    using TModelDefinition = maths::CBoostedTreeFactory::TModelDefinition;
    using TIStreamSPtr = std::shared_ptr<std::istream>;

public:
    //! \brief Retrieve data summarization from decompressed JSON stream.
    static TDataSummarization dataSummarizationFromJsonStream(const TIStreamSPtr& istream);

    //! \brief Retrieve data summarization from compressed and chunked JSON blob.
    static TDataSummarization
    dataSummarizationFromDocumentCompressed(const TIStreamSPtr& istream);

    static TModelDefinition forestFromJsonStream(const core::CDataSearcher::TIStreamP& istream);

    static TModelDefinition
    fromDocumentCompressed(const core::CDataSearcher::TIStreamP& istream);
};
}
}
#endif //INCLUDED_ml_api_CDataSummarizationJsonSerializer_h
