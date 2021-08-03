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

#ifndef INCLUDED_ml_api_CRetrainableModelJsonReader_h
#define INCLUDED_ml_api_CRetrainableModelJsonReader_h

#include <maths/CBoostedTreeFactory.h>

#include <api/CDataSummarizationJsonTags.h>
#include <api/CSerializableToJson.h>
#include <api/ImportExport.h>

#include <boost/unordered/unordered_map.hpp>

#include <istream>
#include <string>

namespace ml {
namespace core {
class CDataFrame;
}
namespace maths {
class CDataFrameCategoryEncoder;
}
namespace api {

//! \brief Reads a compressed, base64 encoded chunked JSON representation of a
//! data summarisation and an inference model which can be used to initialise
//! incremental training.
class API_EXPORT CRetrainableModelJsonReader
    : private CSerializableFromCompressedChunkedJson,
      public CDataSummarizationJsonTags {
public:
    using TEncoderUPtr = maths::CBoostedTreeFactory::TEncoderUPtr;
    using TStrSizeUMap = boost::unordered_map<std::string, std::size_t>;
    using TEncoderUPtrStrSizeUMapPr = std::pair<TEncoderUPtr, TStrSizeUMap>;
    using TNodeVecVecUPtr = maths::CBoostedTreeFactory::TNodeVecVecUPtr;
    using TIStreamSPtr = core::CDataSearcher::TIStreamP;

public:
    //! Retrieve the data summarization from a JSON blob.
    static TEncoderUPtrStrSizeUMapPr
    dataSummarizationFromJsonStream(TIStreamSPtr istream, core::CDataFrame& frame);

    //! Retrieve the data summarization from a compressed and chunked JSON blob.
    static TEncoderUPtrStrSizeUMapPr
    dataSummarizationFromCompressedJsonStream(TIStreamSPtr istream, core::CDataFrame& frame);

    //! Retrieve the best forest from a JSON blob.
    static TNodeVecVecUPtr bestForestFromJsonStream(TIStreamSPtr istream,
                                                    const TStrSizeUMap& encodingIndices);

    //! Retrieve the best forest from a compressed and chunked JSON blob.
    static TNodeVecVecUPtr bestForestFromCompressedJsonStream(TIStreamSPtr istream,
                                                              const TStrSizeUMap& encodingIndices);

private:
    static TEncoderUPtrStrSizeUMapPr
    doDataSummarizationFromJsonStream(std::istream& istream, core::CDataFrame& frame);
    static TNodeVecVecUPtr doBestForestFromJsonStream(std::istream& istream,
                                                      const TStrSizeUMap& encodingIndices);
};
}
}

#endif //INCLUDED_ml_api_CRetrainableModelJsonReader_h
