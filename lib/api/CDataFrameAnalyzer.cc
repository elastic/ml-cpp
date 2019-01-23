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
using TStrVec = std::vector<std::string>;

double truncateToFloatRange(double value) {
    double largest{static_cast<double>(std::numeric_limits<float>::max())};
    return maths::CTools::truncate(value, -largest, largest);
}

const std::string SPECIAL_COLUMN_FIELD_NAME{"."};

// Control message types:
const char FINISHED_DATA_CONTROL_MESSAGE_FIELD_VALUE{'$'};

const std::string CHECKSUM{"checksum"};
const std::string RESULTS{"results"};
}

CDataFrameAnalyzer::CDataFrameAnalyzer(TDataFrameAnalysisSpecificationUPtr analysisSpecification,
                                       TJsonOutputStreamWrapperUPtrSupplier outStreamSupplier)
    : m_AnalysisSpecification{std::move(analysisSpecification)}, m_OutStreamSupplier{outStreamSupplier} {

    if (m_AnalysisSpecification != nullptr) {
        auto frameAndDirectory = m_AnalysisSpecification->makeDataFrame();
        m_DataFrame = std::move(frameAndDirectory.first);
        m_DataFrameDirectory = frameAndDirectory.second;
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

    // Note that returning false from this function immediately causes us to stop
    // processing the input stream. Therefore, any error logged in this context is
    // emitted at most once.

    if (m_AnalysisSpecification == nullptr) {
        // Logging handled when the analysis specification is created.
        return false;
    }

    if (this->readyToReceiveControlMessages() == false &&
        this->prepareToReceiveControlMessages(fieldNames) == false) {
        // Logging handled in functions.
        return false;
    }

    if (this->sufficientFieldValues(fieldValues) == false) {
        // Logging handled in sufficientFieldValues.
        return false;
    }

    if (this->isControlMessage(fieldValues)) {
        return this->handleControlMessage(fieldValues);
    }

    this->addRowToDataFrame(fieldValues);
    return true;
}

void CDataFrameAnalyzer::receivedAllRows() {
    if (m_DataFrame != nullptr) {
        m_DataFrame->finishWritingRows();
        LOG_DEBUG(<< "Received " << m_DataFrame->numberRows() << " rows");
    }
}

void CDataFrameAnalyzer::run() {

    if (m_AnalysisSpecification == nullptr || m_DataFrame == nullptr) {
        return;
    }

    LOG_TRACE(<< "Running analysis...");

    CDataFrameAnalysisRunner* analysis{m_AnalysisSpecification->run(*m_DataFrame)};

    if (analysis == nullptr) {
        return;
    }

    // TODO report progress.

    analysis->waitToFinish();

    this->writeResultsOf(*analysis);
}

const CDataFrameAnalyzer::TTemporaryDirectoryPtr& CDataFrameAnalyzer::dataFrameDirectory() const {
    return m_DataFrameDirectory;
}

bool CDataFrameAnalyzer::readyToReceiveControlMessages() const {
    return m_ControlFieldIndex != FIELD_UNSET;
}

bool CDataFrameAnalyzer::prepareToReceiveControlMessages(const TStrVec& fieldNames) {
    // If this is being called by the Java API we'll use the last two columns for
    // special purposes:
    //   - penultimate contains a 32 bit hash of the document.
    //   - last contains the control message.
    //
    // These will both be called . to avoid collision with any real field name.

    auto posDocHash = std::find(fieldNames.begin(), fieldNames.end(), SPECIAL_COLUMN_FIELD_NAME);
    auto posControlMessage = posDocHash == fieldNames.end()
                                 ? fieldNames.end()
                                 : std::find(posDocHash + 1, fieldNames.end(),
                                             SPECIAL_COLUMN_FIELD_NAME);

    if (posDocHash == fieldNames.end() && posControlMessage == fieldNames.end()) {
        m_ControlFieldIndex = FIELD_MISSING;
        m_BeginDataFieldValues = 0;
        m_EndDataFieldValues = static_cast<std::ptrdiff_t>(fieldNames.size());
        m_DocHashFieldIndex = FIELD_MISSING;

    } else if (fieldNames.size() < 2 || posDocHash != fieldNames.end() - 2 ||
               posControlMessage != fieldNames.end() - 1) {

        HANDLE_FATAL(<< "Input error: expected exacly two special "
                     << "fields in last two positions, got '"
                     << core::CContainerPrinter::print(fieldNames)
                     << "'. Please report this problem.");
        return false;

    } else {
        m_ControlFieldIndex = posControlMessage - fieldNames.begin();
        m_BeginDataFieldValues = 0;
        m_EndDataFieldValues = posDocHash - fieldNames.begin();
        m_DocHashFieldIndex = m_ControlFieldIndex - 1;
    }

    return true;
}

bool CDataFrameAnalyzer::isControlMessage(const TStrVec& fieldValues) const {
    return m_ControlFieldIndex >= 0 && fieldValues[m_ControlFieldIndex].size() > 0;
}

bool CDataFrameAnalyzer::sufficientFieldValues(const TStrVec& fieldValues) const {
    std::size_t expectedNumberFieldValues{m_AnalysisSpecification->numberColumns() +
                                          (m_ControlFieldIndex >= 0 ? 2 : 0)};
    if (fieldValues.size() != expectedNumberFieldValues) {
        HANDLE_FATAL(<< "Input error: expected " << expectedNumberFieldValues << " field values and got "
                     << fieldValues.size() << ". Please report this problem.");
        return false;
    }
    return true;
}

bool CDataFrameAnalyzer::handleControlMessage(const TStrVec& fieldValues) {
    LOG_TRACE(<< "Control message: '" << fieldValues[m_ControlFieldIndex] << "'");

    bool unrecognised{false};
    switch (fieldValues[m_ControlFieldIndex][0]) {
    case ' ':
        // Spaces are just used to fill the buffers and force prior messages
        // through the system - we don't need to do anything else.
        LOG_TRACE(<< "Received pad of length "
                  << fieldValues[m_ControlFieldIndex].length());
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
        HANDLE_FATAL(<< "Input error: invalid control message value '"
                     << fieldValues[m_ControlFieldIndex] << "'. Please report this problem.");
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

    m_DataFrame->writeRow([&](TFloatVecItr columns, std::int32_t& docHash) {
        for (std::ptrdiff_t i = m_BeginDataFieldValues;
             i != m_EndDataFieldValues; ++i, ++columns) {
            double value;
            if (core::CStringUtils::stringToType(fieldValues[i], value) == false) {
                ++m_BadValueCount;

                // TODO this is a can of worms we can deal with later.
                // Use NaN to indicate missing in the data frame, but this needs
                // handling with care from an analysis perspective. If analyses
                // can deal with missing values they need to treat NaNs as missing
                // otherwise we must impute or exit with failure.
                *columns = core::CFloatStorage{std::numeric_limits<float>::quiet_NaN()};
            } else {
                // Tuncation is very unlikely since the values will typically be
                // standardised.
                *columns = truncateToFloatRange(value);
            }
        }
        docHash = 0;
        if (m_DocHashFieldIndex != FIELD_MISSING &&
            core::CStringUtils::stringToType(fieldValues[m_DocHashFieldIndex], docHash) == false) {
            ++m_BadDocHashCount;
        }
    });
}

void CDataFrameAnalyzer::writeResultsOf(const CDataFrameAnalysisRunner& analysis) const {
    // TODO Revisit this can probably be core::CRapidJsonLineWriter.
    // TODO This should probably be a member variable so it is only created once.
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
            outputWriter.Key(CHECKSUM);
            outputWriter.Int(row->docHash());
            outputWriter.Key(RESULTS);
            analysis.writeOneRow(*row, outputWriter);
            outputWriter.EndObject();
        }
    });

    outputWriter.flush();
}
}
}
