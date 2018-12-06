/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CDataFrameAnalyzer.h>

#include <core/CContainerPrinter.h>
#include <core/CFloatStorage.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CLogger.h>
#include <core/CRapidJsonConcurrentLineWriter.h>

#include <maths/CTools.h>

#include <limits>

namespace ml {
namespace api {
namespace {
double truncateToFloatRange(double value) {
    double largest{static_cast<double>(std::numeric_limits<float>::max())};
    return maths::CTools::truncate(value, -largest, largest);
}

const std::string CONTROL_MESSAGE_FIELD_NAME{"."};
// Control message types:
const char RUN_ANALYSIS_CONTROL_MESSAGE_FIELD_VALUE{'r'};

const std::string ID_HASH{"id_hash"};
const std::string RESULTS{"results"};
const std::string OUTLIER_SCORE{"outlier_score"};
}

// TODO memory calculations plus choose data frame storage strategy.
CDataFrameAnalyzer::CDataFrameAnalyzer(CDataFrameAnalysisSpecification analysisSpecification,
                                       TJsonOutputStreamWrapperUPtrSupplier outStreamSupplier)
    : m_AnalysisSpecification{std::move(analysisSpecification)},
      m_DataFrame{core::makeMainStorageDataFrame(m_AnalysisSpecification.cols())},
      m_OutStreamSupplier{outStreamSupplier} {
}

bool CDataFrameAnalyzer::usingControlMessages() const {
    return m_ControlFieldValue >= 0;
}

bool CDataFrameAnalyzer::handleRecord(const TStrVec& fieldNames, const TStrVec& fieldValues) {

    // Control messages are signified by a dot in the field name. This supports:
    //   - using the last field for a control message,
    //   - missing.
    //
    // Note if the the control message field name is missing the analysis must
    // be triggered to run by calling run explicitly.

    bool result{true};
    if (this->readyToReceiveControlMessages() == false) {
        result = this->prepareToReceiveControlMessages(fieldNames);
    }

    if (this->isControlMessage(fieldValues)) {
        return this->handleControlMessage(fieldValues);
    }

    this->addRowToDataFrame(fieldValues);

    return result;
}

void CDataFrameAnalyzer::run() {
    // TODO

    // This is writing fake results just to enable testing
    // the parsing and merging of results in the java side
    auto outStream = m_OutStreamSupplier();
    core::CRapidJsonConcurrentLineWriter outputWriter{*outStream};
    for (std::size_t i = 0; i < m_AnalysisSpecification.rows(); ++i) {
        outputWriter.StartObject();
        outputWriter.String(ID_HASH);
        outputWriter.String(ID_HASH);
        outputWriter.String(RESULTS);
        outputWriter.StartObject();
        outputWriter.String(RESULTS);
        outputWriter.Double(i);
        outputWriter.EndObject();
        outputWriter.EndObject();
    }
    outputWriter.flush();
}

bool CDataFrameAnalyzer::readyToReceiveControlMessages() const {
    return m_ControlFieldValue != CONTROL_FIELD_UNSET;
}

bool CDataFrameAnalyzer::prepareToReceiveControlMessages(const TStrVec& fieldNames) {
    if (fieldNames.empty() || fieldNames.back() != CONTROL_MESSAGE_FIELD_NAME) {
        m_ControlFieldValue = CONTROL_FIELD_MISSING;
        m_BeginDataFieldValues = 0;
        m_EndDataFieldValues = static_cast<std::ptrdiff_t>(fieldNames.size());
    } else {
        m_ControlFieldValue = static_cast<std::ptrdiff_t>(fieldNames.size() - 1);
        m_BeginDataFieldValues = 0;
        m_EndDataFieldValues = m_ControlFieldValue;
        auto pos = std::find(fieldNames.begin(), fieldNames.end(), CONTROL_MESSAGE_FIELD_NAME);
        if (pos != fieldNames.end() - 1) {
            LOG_ERROR(<< "Unexpected possible control field: ignoring");
            return false;
        }
    }
    return true;
}

bool CDataFrameAnalyzer::isControlMessage(const TStrVec& fieldValues) const {
    return m_ControlFieldValue >= 0 && fieldValues[m_ControlFieldValue].size() > 0;
}

bool CDataFrameAnalyzer::handleControlMessage(const TStrVec& fieldValues) {

    bool unrecognised{false};
    switch (fieldValues[m_ControlFieldValue][0]) {
    case ' ':
        // Spaces are just used to fill the buffers and force prior messages
        // through the system - we don't need to do anything else.
        LOG_TRACE(<< "Received pad of length " << controlMessage.length());
        return true;
    case RUN_ANALYSIS_CONTROL_MESSAGE_FIELD_VALUE:
        this->run();
        break;
    default:
        unrecognised = true;
        break;
    }
    if (unrecognised || fieldValues[m_ControlFieldValue].size() > 1) {
        LOG_ERROR(<< "Invalid control message value '"
                  << fieldValues[m_ControlFieldValue] << "'");
        return false;
    }
    return true;
}

void CDataFrameAnalyzer::addRowToDataFrame(const TStrVec& fieldValues) {
    using TFloatVec = std::vector<core::CFloatStorage>;
    using TFloatVecItr = TFloatVec::iterator;

    m_DataFrame.writeRow([&](TFloatVecItr output) {
        for (std::ptrdiff_t i = m_BeginDataFieldValues;
             i != m_EndDataFieldValues; ++i, ++output) {
            double value;
            if (core::CStringUtils::stringToType(fieldValues[i], value) == false) {
                ++m_BadValueCount;

                // TODO this is a can of worms we can deal with later.
                // Use NaN to indicate missing in the data frame, but this needs
                // handling with care from an analysis perspective. If analyses
                // can deal with missing values we need to make sure they would
                // deal with NaNs correctly otherwise we need to impute or exit
                // with failure.
                *output = core::CFloatStorage{std::numeric_limits<float>::quiet_NaN()};
            } else {
                // Tuncation is very unlikely since the values will typically be
                // standardised.
                *output = truncateToFloatRange(value);
            }
        }
    });
}
}
}
