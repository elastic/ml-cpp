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
#include <core/CStopWatch.h>

#include <maths/CBasicStatistics.h>

#include <api/CDataFrameAnalysisInstrumentation.h>
#include <api/CDataFrameAnalysisSpecification.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace ml {
namespace api {
namespace {
using TStrVec = std::vector<std::string>;
using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

const std::string SPECIAL_COLUMN_FIELD_NAME{"."};

// Control message types:
const char FINISHED_DATA_CONTROL_MESSAGE_FIELD_VALUE{'$'};

// Result types
const std::string ROW_RESULTS{"row_results"};

// Row result fields
const std::string CHECKSUM{"checksum"};
const std::string RESULTS{"results"};
}

CDataFrameAnalyzer::CDataFrameAnalyzer(TDataFrameAnalysisSpecificationUPtr analysisSpecification,
                                       TJsonOutputStreamWrapperUPtrSupplier resultsStreamSupplier)
    : m_AnalysisSpecification{std::move(analysisSpecification)},
      m_ResultsStreamSupplier{std::move(resultsStreamSupplier)} {

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

    this->captureFieldNames(fieldNames);
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

    if (m_AnalysisSpecification->validate(*m_DataFrame) == false) {
        return;
    }

    LOG_TRACE(<< "Running analysis...");

    // We create the writer in run so that when it is finished destructors
    // get called and the wrapped stream does its job to close the array.

    auto analysisRunner = m_AnalysisSpecification->runner();
    if (analysisRunner != nullptr) {
        // We currently use a stream factory because the results are wrapped in
        // an array. This is managed by the CJsonOutputStreamWrapper constructor
        // and destructor. We should probably migrate to NDJSON format at which
        // point this would no longer be necessary.
        auto outStream = m_ResultsStreamSupplier();

        auto& instrumentation = analysisRunner->instrumentation();
        CDataFrameAnalysisInstrumentation::CScopeSetOutputStream setStream{
            instrumentation, *outStream};
        instrumentation.updateMemoryUsage(
            static_cast<std::int64_t>(m_DataFrame->memoryUsage()));
        instrumentation.flush();

        analysisRunner->run(*m_DataFrame);

        core::CRapidJsonConcurrentLineWriter outputWriter{*outStream};

        CDataFrameAnalysisInstrumentation::monitor(instrumentation, outputWriter);

        analysisRunner->waitToFinish();
        this->writeInferenceModel(*analysisRunner, outputWriter);
        this->writeResultsOf(*analysisRunner, outputWriter);
        this->writeInferenceModelMetadata(*analysisRunner, outputWriter);
    }
}

const CDataFrameAnalyzer::TTemporaryDirectoryPtr& CDataFrameAnalyzer::dataFrameDirectory() const {
    return m_DataFrameDirectory;
}

const core::CDataFrame& CDataFrameAnalyzer::dataFrame() const {
    if (m_DataFrame == nullptr) {
        HANDLE_FATAL(<< "Internal error: missing data frame");
    }
    return *m_DataFrame;
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

        HANDLE_FATAL(<< "Input error: expected exactly two special "
                     << "fields in last two positions but got '"
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
        HANDLE_FATAL(<< "Input error: expected " << expectedNumberFieldValues << " field"
                     << " values and got " << core::CContainerPrinter::print(fieldValues)
                     << ". Please report this problem.");
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

void CDataFrameAnalyzer::captureFieldNames(const TStrVec& fieldNames) {
    if (m_DataFrame == nullptr) {
        return;
    }
    if (m_DataFrame != nullptr && m_CapturedFieldNames == false) {
        TStrVec columnNames{fieldNames.begin() + m_BeginDataFieldValues,
                            fieldNames.begin() + m_EndDataFieldValues};
        m_DataFrame->columnNames(columnNames);
        m_DataFrame->categoricalColumns(m_AnalysisSpecification->categoricalFieldNames());
        m_CapturedFieldNames = true;
    }
}

void CDataFrameAnalyzer::addRowToDataFrame(const TStrVec& fieldValues) {
    if (m_DataFrame == nullptr) {
        return;
    }
    auto columnValues = core::make_range(fieldValues, m_BeginDataFieldValues,
                                         m_EndDataFieldValues);
    m_DataFrame->parseAndWriteRow(columnValues, m_DocHashFieldIndex != FIELD_MISSING
                                                    ? &fieldValues[m_DocHashFieldIndex]
                                                    : nullptr);
}

void CDataFrameAnalyzer::writeInferenceModel(const CDataFrameAnalysisRunner& analysis,
                                             core::CRapidJsonConcurrentLineWriter& writer) const {
    // Write the resulting model for inference.
    auto modelDefinition = analysis.inferenceModelDefinition(
        m_DataFrame->columnNames(), m_DataFrame->categoricalColumnValues());
    if (modelDefinition != nullptr) {
        auto modelDefinitionSizeInfo = modelDefinition->sizeInfo();
        rapidjson::Value sizeInfoObject{writer.makeObject()};
        modelDefinitionSizeInfo->addToJsonDocument(sizeInfoObject, writer);
        writer.StartObject();
        writer.Key(modelDefinitionSizeInfo->typeString());
        writer.write(sizeInfoObject);
        writer.EndObject();
        modelDefinition->addToDocumentCompressed(writer);
    }
    writer.flush();
}

void CDataFrameAnalyzer::writeInferenceModelMetadata(const CDataFrameAnalysisRunner& analysis,
                                                     core::CRapidJsonConcurrentLineWriter& writer) const {
    // Write model meta information
    auto modelMetadata = analysis.inferenceModelMetadata();
    if (modelMetadata) {
        writer.StartObject();
        writer.Key(modelMetadata->typeString());
        writer.StartObject();
        modelMetadata->write(writer);
        writer.EndObject();
        writer.EndObject();
    }
    writer.flush();
}

void CDataFrameAnalyzer::writeResultsOf(const CDataFrameAnalysisRunner& analysis,
                                        core::CRapidJsonConcurrentLineWriter& writer) const {

    // We write results single threaded because we need to write the rows to
    // Java in the order they were written to the data_frame_analyzer so it
    // can join the extra columns with the original data frame.
    std::size_t numberThreads{1};

    using TRowItr = core::CDataFrame::TRowItr;
    m_DataFrame->readRows(numberThreads, [&](TRowItr beginRows, TRowItr endRows) {
        TMeanVarAccumulator timeAccumulator;
        core::CStopWatch stopWatch;
        stopWatch.start();
        std::uint64_t lastLap{stopWatch.lap()};

        for (auto row = beginRows; row != endRows; ++row) {
            writer.StartObject();
            writer.Key(ROW_RESULTS);
            writer.StartObject();
            writer.Key(CHECKSUM);
            writer.Int(row->docHash());
            writer.Key(RESULTS);
            writer.StartObject();
            writer.Key(m_AnalysisSpecification->resultsField());
            analysis.writeOneRow(*m_DataFrame, *row, writer);
            writer.EndObject();
            writer.EndObject();
            writer.EndObject();

            std::uint64_t currentLap{stopWatch.lap()};
            std::uint64_t delta{currentLap - lastLap};

            timeAccumulator.add(static_cast<double>(delta));
            lastLap = currentLap;
        }
    });

    writer.flush();
}

const CDataFrameAnalysisRunner* CDataFrameAnalyzer::runner() const {
    return m_AnalysisSpecification->runner();
}
}
}
