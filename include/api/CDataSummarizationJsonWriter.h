/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#ifndef INCLUDED_ml_api_CDataSummarizationJsonWriter_h
#define INCLUDED_ml_api_CDataSummarizationJsonWriter_h

#include <core/CPackedBitVector.h>

#include <api/CDataSummarizationJsonTags.h>
#include <api/CSerializableToJson.h>
#include <api/ImportExport.h>

#include <cstddef>

namespace ml {
namespace core {
class CDataFrame;
}
namespace maths {
class CDataFrameCategoryEncoder;
}
namespace api {

//! \brief Class which writes the data summarization as a compressed, base64 encoded
//! and chunked JSON blob.
//!
//! DESCRIPTION:\n
//! The data summarization contains data rows as well as the feature value encoding
//! information.
class API_EXPORT CDataSummarizationJsonWriter final
    : public CSerializableToCompressedChunkedJson,
      public CDataSummarizationJsonTags {
public:
    CDataSummarizationJsonWriter(const core::CDataFrame& frame,
                                 core::CPackedBitVector rowMask,
                                 std::size_t numberColumns,
                                 const maths::CDataFrameCategoryEncoder& encodings);

    CDataSummarizationJsonWriter(const CDataSummarizationJsonWriter&) = delete;
    CDataSummarizationJsonWriter& operator=(const CDataSummarizationJsonWriter&) = delete;

    //! Write the JSON data summarisation.
    void addToJsonStream(TGenericLineWriter& writer) const override;

    //! Write a compressed and chunked JSON data summarisation.
    void addCompressedToJsonStream(TRapidJsonWriter& writer) const override;

private:
    core::CPackedBitVector m_RowMask;
    std::size_t m_NumberColumns;
    const core::CDataFrame& m_Frame;
    const maths::CDataFrameCategoryEncoder& m_Encodings;
};
}
}

#endif // INCLUDED_ml_api_CDataSummarizationJsonWriter_h
