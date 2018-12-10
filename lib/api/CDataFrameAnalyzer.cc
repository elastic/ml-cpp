/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CDataFrameAnalyzer.h>

#include <core/CContainerPrinter.h>
#include <core/CDataFrame.h>
#include <core/CFloatStorage.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CLogger.h>
#include <core/CRapidJsonConcurrentLineWriter.h>

#include <maths/CTools.h>

#include <api/CDataFrameAnalysisSpecification.h>

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
const char FINISHED_DATA_CONTROL_MESSAGE_FIELD_VALUE{'$'};

const std::string ID_HASH{"id_hash"};
const std::string RESULTS{"results"};
}

CDataFrameAnalyzer::CDataFrameAnalyzer(TDataFrameAnalysisSpecificationUPtr analysisSpecification,
                                       TJsonOutputStreamWrapperUPtrSupplier outStreamSupplier)
    : m_AnalysisSpecification{std::move(analysisSpecification)}, m_OutStreamSupplier{outStreamSupplier} {

    // TODO memory calculations plus choose data frame storage strategy.
    if (m_AnalysisSpecification != nullptr && m_AnalysisSpecification->bad() == false) {
        m_DataFrame = core::makeMainStorageDataFrame(m_AnalysisSpecification->numberColumns());
        m_DataFrame->reserve(m_AnalysisSpecification->numberThreads(),
                             m_AnalysisSpecification->numberColumns() +
                                 m_AnalysisSpecification->numberExtraColumns());
    }
}

CDataFrameAnalyzer::~CDataFrameAnalyzer() = default;

bool CDataFrameAnalyzer::usingControlMessages() const {
    return m_ControlFieldIndex >= 0;
}

bool CDataFrameAnalyzer::handleRecord(const TStrVec& fieldNames, const TStrVec& fieldValues) {

    // Control messages are signified by a dot in the field name. This supports:
    //   - using the last field for a control message,
    //   - missing.
    //
    // Note if the the control message field name is missing the analysis must
    // be triggered to run by calling run explicitly.

    if (m_AnalysisSpecification == nullptr || m_AnalysisSpecification->bad()) {
        // TODO We need to communicate an error but don't want one for each row.
        // Revisit this when we have finalized our monitoring strategy.
        LOG_TRACE(<< "Specification is bad");
        return false;
    }

    if (this->readyToReceiveControlMessages() == false &&
        this->prepareToReceiveControlMessages(fieldNames) == false) {
        return false;
    }

    if (this->sufficientFieldValues(fieldValues) == false) {
        // TODO We need to communicate an error but don't want one for each row.
        // Revisit this when we have finalized our monitoring strategy.
        LOG_TRACE(<< "Expected " << m_AnalysisSpecification->numberColumns()
                  << " field values and got " << fieldValues.size());
        return false;
    }

    if (this->isControlMessage(fieldValues)) {
        return this->handleControlMessage(fieldValues);
    }

    this->addRowToDataFrame(fieldValues);
    return false;
}

void CDataFrameAnalyzer::receivedAllRows() {
    m_DataFrame->finishWritingRows();
    LOG_DEBUG(<< "Received " << m_DataFrame->numberRows() << " rows");
}

void CDataFrameAnalyzer::run() {

    if (m_AnalysisSpecification == nullptr || m_AnalysisSpecification->bad() ||
        m_DataFrame == nullptr) {
        return;
    }

    LOG_TRACE(<< "Running analysis...");

    CDataFrameAnalysisRunner* analysis{m_AnalysisSpecification->run(*m_DataFrame)};

    if (analysis == nullptr) {
        return;
    }

    // TODO progress monitoring, etc.

    analysis->waitToFinish();

    this->writeResultsOf(*analysis);
}

bool CDataFrameAnalyzer::readyToReceiveControlMessages() const {
    return m_ControlFieldIndex != CONTROL_FIELD_UNSET;
}

bool CDataFrameAnalyzer::prepareToReceiveControlMessages(const TStrVec& fieldNames) {
    if (fieldNames.empty() || fieldNames.back() != CONTROL_MESSAGE_FIELD_NAME) {
        m_ControlFieldIndex = CONTROL_FIELD_MISSING;
        m_BeginDataFieldValues = 0;
        m_EndDataFieldValues = static_cast<std::ptrdiff_t>(fieldNames.size());
    } else {
        m_ControlFieldIndex = static_cast<std::ptrdiff_t>(fieldNames.size() - 1);
        m_BeginDataFieldValues = 0;
        m_EndDataFieldValues = m_ControlFieldIndex;
        auto pos = std::find(fieldNames.begin(), fieldNames.end(), CONTROL_MESSAGE_FIELD_NAME);
        if (pos != fieldNames.end() - 1) {
            LOG_ERROR(<< "Unexpected possible control field: ignoring");
            return false;
        }
    }
    return true;
}

bool CDataFrameAnalyzer::isControlMessage(const TStrVec& fieldValues) const {
    return m_ControlFieldIndex >= 0 && fieldValues[m_ControlFieldIndex].size() > 0;
}

bool CDataFrameAnalyzer::sufficientFieldValues(const TStrVec& fieldValues) const {
    if (m_ControlFieldIndex >= 0) {
        return fieldValues.size() == m_AnalysisSpecification->numberColumns() + 1;
    }
    return fieldValues.size() == m_AnalysisSpecification->numberColumns();
}

bool CDataFrameAnalyzer::handleControlMessage(const TStrVec& fieldValues) {
    LOG_TRACE(<< "Control message: '" << fieldValues[m_ControlFieldIndex] << "'");

    bool unrecognised{false};
    switch (fieldValues[m_ControlFieldIndex][0]) {
    case ' ':
        // Spaces are just used to fill the buffers and force prior messages
        // through the system - we don't need to do anything else.
        LOG_TRACE(<< "Received pad of length " << controlMessage.length());
        return true;
    case FINISHED_DATA_CONTROL_MESSAGE_FIELD_VALUE:
        this->receivedAllRows();
        this->run();
        break;
    default:
        unrecognised = true;
        break;
    }
    if (unrecognised || fieldValues[m_ControlFieldIndex].size() > 1) {
        LOG_ERROR(<< "Invalid control message value '"
                  << fieldValues[m_ControlFieldIndex] << "'");
        return false;
    }
    return true;
}

void CDataFrameAnalyzer::addRowToDataFrame(const TStrVec& fieldValues) {
    if (m_DataFrame == nullptr) {
        return;
    }

    using TFloatVec = std::vector<core::CFloatStorage>;
    using TFloatVecItr = TFloatVec::iterator;

    m_DataFrame->writeRow([&](TFloatVecItr output) {
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

void CDataFrameAnalyzer::writeResultsOf(const CDataFrameAnalysisRunner& analysis) const {
    // TODO Revisit this can probably be core::CRapidJsonLineWriter.
    auto outStream = m_OutStreamSupplier();
    core::CRapidJsonConcurrentLineWriter outputWriter{*outStream};

    // We write results single threaded because we need to write the rows to
    // Java in the order they were written to the data_frame_analyzer so it
    // can join the extra columns with the original data frame.
    std::size_t numberThreads{1};

    using TRowItr = core::CDataFrame::TRowItr;
    m_DataFrame->readRows(numberThreads, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            outputWriter.StartObject();
            outputWriter.String(ID_HASH);
            outputWriter.String(ID_HASH); // TODO this should be the actual hash
            outputWriter.String(RESULTS);
            analysis.writeOneRow(*row, outputWriter);
            outputWriter.EndObject();
        }
    });

    outputWriter.flush();
}
}
}
