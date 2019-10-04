/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_api_CDataFrameAnalyzer_h
#define INCLUDED_ml_api_CDataFrameAnalyzer_h

#include <api/ImportExport.h>

#include <core/CRapidJsonConcurrentLineWriter.h>

#include <boost/unordered_map.hpp>

#include <cinttypes>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ml {
namespace core {
class CDataFrame;
class CJsonOutputStreamWrapper;
class CTemporaryDirectory;
}
namespace api {
class CDataFrameAnalysisRunner;
class CDataFrameAnalysisSpecification;

//! \brief Handles input to the data_frame_analyzer command.
//! TODO implement functionality for restoring state from the index
//!
class API_EXPORT CDataFrameAnalyzer {
public:
    using TBoolVec = std::vector<bool>;
    using TStrVec = std::vector<std::string>;
    using TJsonOutputStreamWrapperUPtr = std::unique_ptr<core::CJsonOutputStreamWrapper>;
    using TJsonOutputStreamWrapperUPtrSupplier =
        std::function<TJsonOutputStreamWrapperUPtr()>;
    using TDataFrameAnalysisSpecificationUPtr = std::unique_ptr<CDataFrameAnalysisSpecification>;
    using TTemporaryDirectoryPtr = std::shared_ptr<core::CTemporaryDirectory>;

public:
    //! The maximum number of distinct categorical fields we can faithfully
    //! represent.
    static const std::size_t MAX_CATEGORICAL_CARDINALITY;

public:
    CDataFrameAnalyzer(TDataFrameAnalysisSpecificationUPtr analysisSpecification,
                       TJsonOutputStreamWrapperUPtrSupplier resultsStreamSupplier);
    ~CDataFrameAnalyzer();

    //! This is true if the analyzer is receiving control messages.
    bool usingControlMessages() const;

    //! Handle receiving a row of the data frame or a control message.
    bool handleRecord(const TStrVec& fieldNames, const TStrVec& fieldValues);

    //! Call when all row have been received.
    void receivedAllRows();

    //! Run the configured analysis.
    void run();

    //! Get the data frame's temporary directory if there is one.
    const TTemporaryDirectoryPtr& dataFrameDirectory() const;

    //! Get the data frame asserting there is one.
    const core::CDataFrame& dataFrame() const;

private:
    using TDataFrameUPtr = std::unique_ptr<core::CDataFrame>;
    using TStrSizeUMap = std::unordered_map<std::string, std::size_t>;
    using TStrSizeUMapVec = std::vector<TStrSizeUMap>;

private:
    static const std::ptrdiff_t FIELD_UNSET{-2};
    static const std::ptrdiff_t FIELD_MISSING{-1};

private:
    bool sufficientFieldValues(const TStrVec& fieldNames) const;
    bool readyToReceiveControlMessages() const;
    bool prepareToReceiveControlMessages(const TStrVec& fieldNames);
    bool isControlMessage(const TStrVec& fieldValues) const;
    bool handleControlMessage(const TStrVec& fieldValues);
    void captureFieldNames(const TStrVec& fieldNames);
    void addRowToDataFrame(const TStrVec& fieldValues);
    void monitorProgress(const CDataFrameAnalysisRunner& analysis,
                         core::CRapidJsonConcurrentLineWriter& writer) const;
    void writeProgress(int progress, core::CRapidJsonConcurrentLineWriter& writer) const;
    void writeResultsOf(const CDataFrameAnalysisRunner& analysis,
                        core::CRapidJsonConcurrentLineWriter& writer) const;

private:
    // This has values: -2 (unset), -1 (missing), >= 0 (control field index).
    std::ptrdiff_t m_ControlFieldIndex = FIELD_UNSET;
    std::ptrdiff_t m_BeginDataFieldValues = FIELD_UNSET;
    std::ptrdiff_t m_EndDataFieldValues = FIELD_UNSET;
    std::ptrdiff_t m_DocHashFieldIndex = FIELD_UNSET;
    std::uint64_t m_MissingValueCount = 0;
    std::uint64_t m_BadValueCount = 0;
    std::uint64_t m_BadDocHashCount = 0;
    TDataFrameAnalysisSpecificationUPtr m_AnalysisSpecification;
    TStrVec m_CategoricalFieldNames;
    TStrSizeUMapVec m_CategoricalFieldValues;
    TBoolVec m_EmptyAsMissing;
    TDataFrameUPtr m_DataFrame;
    TStrVec m_FieldNames;
    TTemporaryDirectoryPtr m_DataFrameDirectory;
    TJsonOutputStreamWrapperUPtrSupplier m_ResultsStreamSupplier;
};
}
}

#endif // INCLUDED_ml_api_CDataFrameAnalyzer_h
