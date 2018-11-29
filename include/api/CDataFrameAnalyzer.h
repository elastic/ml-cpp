/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CDataFrameAnalyzer_h
#define INCLUDED_ml_api_CDataFrameAnalyzer_h

#include <core/CDataFrame.h>

#include <api/CDataFrameAnalysisSpecification.h>
#include <api/ImportExport.h>

#include <cinttypes>
#include <string>
#include <vector>

namespace ml {
namespace api {

//! \brief Handles input to the data_frame_analyzer command.
class API_EXPORT CDataFrameAnalyzer {
public:
    using TStrVec = std::vector<std::string>;

public:
    explicit CDataFrameAnalyzer(CDataFrameAnalysisSpecification analysisSpecification);

    //! This is true if the analyzer is receiving control messages.
    bool usingControlMessages() const;

    //! Handle adding a row of the data frame or a control message.
    bool handleRecord(const TStrVec& fieldNames, const TStrVec& fieldValues);

    //! Run the configured analysis.
    void run();

private:
    static const std::ptrdiff_t CONTROL_FIELD_UNSET{-2};
    static const std::ptrdiff_t CONTROL_FIELD_MISSING{-1};

private:
    bool readyToReceiveControlMessages() const;
    bool prepareToReceiveControlMessages(const TStrVec& fieldNames);
    bool isControlMessage(const TStrVec& fieldValues) const;
    bool handleControlMessage(const TStrVec& fieldValues);
    void addRowToDataFrame(const TStrVec& fieldValues);

private:
    // This has values: -2 (unset), -1 (missing), >= 0 (control field index).
    std::ptrdiff_t m_ControlFieldValue = CONTROL_FIELD_UNSET;
    std::ptrdiff_t m_BeginDataFieldValues = -1;
    std::ptrdiff_t m_EndDataFieldValues = -1;
    std::uint64_t m_BadValueCount;
    CDataFrameAnalysisSpecification m_AnalysisSpecification;
    core::CDataFrame m_DataFrame;
};
}
}

#endif // INCLUDED_ml_api_CDataFrameAnalyzer_h
