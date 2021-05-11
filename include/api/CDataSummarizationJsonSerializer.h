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

//! \brief Class which writes the data summarization as a compressed, base64 encoded
//! and chunked JSON blob.
//!
//! DESCRIPTION:\n
//! The data summarization contains data rows as well as the feature value encoding
//! information.
class API_EXPORT CDataSummarizationJsonWriter final : public CSerializableToJsonDocumentCompressed {
public:
    CDataSummarizationJsonWriter(const core::CDataFrame& frame,
                                 core::CPackedBitVector rowMask,
                                 std::size_t numberColumns,
                                 std::stringstream encodings);

    CDataSummarizationJsonWriter(const CDataSummarizationJsonWriter&) = delete;
    CDataSummarizationJsonWriter& operator=(const CDataSummarizationJsonWriter&) = delete;

    //! Write the JSON data summarisation.
    void addToJsonStream(TGenericLineWriter& writer) const override;

    //! Write a compressed and chunked JSON data summarisation.
    void addToDocumentCompressed(TRapidJsonWriter& writer) const override;

    //! \name Test Only
    //@{
    std::string jsonString() const;
    //@}

private:
    core::CPackedBitVector m_RowMask;
    std::size_t m_NumberColumns;
    const core::CDataFrame& m_Frame;
    std::stringstream m_Encodings;
};

//! \brief Reads a compressed, base64 encoded chunked JSON representation of a
//! data summarisation and an inference model which can be used to initialise
//! incremental training.
// TODO #1849 reading from chunked output is not supported yet.
class API_EXPORT CRetrainableModelJsonReader {
public:
    using TEncoderUPtr = maths::CBoostedTreeFactory::TEncoderUPtr;
    using TNodeVecVecUPtr = maths::CBoostedTreeFactory::TNodeVecVecUPtr;
    using TIStreamSPtr = core::CDataSearcher::TIStreamP;

public:
    //! Retrieve the data summarization from a JSON blob.
    static TEncoderUPtr dataSummarizationFromJsonStream(TIStreamSPtr istream,
                                                        core::CDataFrame& frame);

    //! Retrieve the data summarization from a compressed and chunked JSON blob.
    static TEncoderUPtr dataSummarizationFromDocumentCompressed(TIStreamSPtr istream,
                                                                core::CDataFrame& frame);

    //! Retrieve the best forest from a JSON blob.
    static TNodeVecVecUPtr bestForestFromJsonStream(TIStreamSPtr istream);

    //! Retrieve the best forest from a compressed and chunked JSON blob.
    static TNodeVecVecUPtr bestForestFromDocumentCompressed(TIStreamSPtr istream);

private:
    static TEncoderUPtr dataSummarizationFromJson(std::istream& istream,
                                                  core::CDataFrame& frame);
    static TNodeVecVecUPtr bestForestFromJson(std::istream& istream);
};
}
}
#endif //INCLUDED_ml_api_CDataSummarizationJsonSerializer_h
