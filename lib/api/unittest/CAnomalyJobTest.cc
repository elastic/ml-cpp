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

#include <core/CDataSearcher.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CLogger.h>
#include <core/COsFileFuncs.h>
#include <core/CRegex.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CDataGatherer.h>
#include <model/CLimits.h>

#include <api/CAnomalyJobConfig.h>
#include <api/CCsvInputParser.h>
#include <api/CHierarchicalResultsWriter.h>
#include <api/CNdJsonInputParser.h>
#include <api/CSingleStreamDataAdder.h>
#include <api/CSingleStreamSearcher.h>
#include <api/CStateRestoreStreamFilter.h>

#include "CTestAnomalyJob.h"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/json.hpp>
#include <boost/test/unit_test.hpp>

#include <cstdio>
#include <fstream>
#include <map>
#include <random>
#include <sstream>

BOOST_TEST_DONT_PRINT_LOG_VALUE(json::array::const_iterator)
BOOST_TEST_DONT_PRINT_LOG_VALUE(json::object::const_iterator)

BOOST_AUTO_TEST_SUITE(CAnomalyJobTest)

namespace {

void reportPersistComplete(ml::api::CModelSnapshotJsonWriter::SModelSnapshotReport modelSnapshotReport,
                           std::string& snapshotIdOut,
                           size_t& numDocsOut) {
    LOG_INFO(<< "Persist complete with description: " << modelSnapshotReport.s_Description);
    snapshotIdOut = modelSnapshotReport.s_SnapshotId;
    numDocsOut = modelSnapshotReport.s_NumDocs;
}

//! \brief
//! Mock object for state restore unit tests.
//!
//! DESCRIPTION:\n
//! CDataSearcher that returns an empty stream.
//!
class CEmptySearcher : public ml::core::CDataSearcher {
public:
    //! Do a search that results in an empty input stream.
    TIStreamP search(size_t /*currentDocNum*/, size_t /*limit*/) override {
        return TIStreamP(new std::istringstream());
    }
};

//! \brief
//! Mock object for unit tests
//!
//! DESCRIPTION:\n
//! Mock object for gathering anomaly results.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Only the minimal set of required functions are implemented.
//!

class CSingleResultVisitor : public ml::model::CHierarchicalResultsVisitor {
public:
    void visit(const ml::model::CHierarchicalResults& /*results*/,
               const TNode& node,
               bool /*pivot*/) override {
        if (!isSimpleCount(node) && isLeaf(node)) {
            if (node.s_AnnotatedProbability.s_AttributeProbabilities.empty()) {
                return;
            }
            if (node.s_Model == nullptr) {
                return;
            }
            const ml::model::SAttributeProbability& attribute =
                node.s_AnnotatedProbability.s_AttributeProbabilities[0];

            m_LastResult = node.s_Model->currentBucketValue(
                attribute.s_Feature, 0, 0, node.s_BucketStartTime)[0];
        }
    }

    double lastResults() const { return m_LastResult; }

private:
    double m_LastResult{0.0};
};

class CMultiResultVisitor : public ml::model::CHierarchicalResultsVisitor {
public:
    void visit(const ml::model::CHierarchicalResults& /*results*/,
               const TNode& node,
               bool /*pivot*/) override {
        if (!isSimpleCount(node) && isLeaf(node)) {
            if (node.s_AnnotatedProbability.s_AttributeProbabilities.empty()) {
                return;
            }
            if (node.s_Model == nullptr) {
                return;
            }
            std::size_t pid;
            const ml::model::CDataGatherer& gatherer = node.s_Model->dataGatherer();
            if (!gatherer.personId(node.s_Spec.s_PersonFieldValue.value_or(""), pid)) {
                LOG_ERROR(<< "No identifier for '"
                          << node.s_Spec.s_PersonFieldValue.value_or("") << "'");
                return;
            }
            for (std::size_t i = 0;
                 i < node.s_AnnotatedProbability.s_AttributeProbabilities.size(); ++i) {
                const ml::model::SAttributeProbability& attribute =
                    node.s_AnnotatedProbability.s_AttributeProbabilities[i];
                m_LastResult += node.s_Model->currentBucketValue(
                    attribute.s_Feature, pid, attribute.s_Cid, node.s_BucketStartTime)[0];
            }
        }
    }

    double lastResults() const { return m_LastResult; }

private:
    double m_LastResult{0.0};
};

class CResultsScoreVisitor : public ml::model::CHierarchicalResultsVisitor {
public:
    explicit CResultsScoreVisitor(int score) : m_Score(score) {}

    void visit(const ml::model::CHierarchicalResults& /*results*/,
               const TNode& node,
               bool /*pivot*/) override {
        if (isRoot(node)) {
            node.s_NormalizedAnomalyScore = m_Score;
        }
    }

private:
    int m_Score;
};

size_t countBuckets(const std::string& key, const std::string& output) {
    size_t count = 0;
    json::error_code ec;
    json::value results = json::parse(output, ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    BOOST_TEST_REQUIRE(results.is_array());

    const json::array& allRecords = results.as_array();
    for (const auto& r : allRecords) {
        BOOST_TEST_REQUIRE(r.is_object());
        json::object::const_iterator recordsIt = r.as_object().find(key);
        if (recordsIt != r.as_object().end()) {
            ++count;
        }
    }

    return count;
}

bool findLine(const std::string& regex, const ml::core::CRegex::TStrVec& lines) {
    ml::core::CRegex rx;
    rx.init(regex);
    std::size_t pos = 0;
    for (const auto& line : lines) {
        if (rx.search(line, pos)) {
            return true;
        }
    }
    return false;
}

const ml::core_t::TTime BUCKET_SIZE(3600);

using TStrStrPr = std::pair<std::string, std::string>;
using TStrStrPrVec = std::vector<TStrStrPr>;
}

using namespace ml;

BOOST_AUTO_TEST_CASE(testBadTimes) {
    {
        // Test with no time field
        model::CLimits limits;
        api::CAnomalyJobConfig jobConfig = CTestAnomalyJob::makeSimpleJobConfig(
            "metric", "value", "", "", "greenhouse");

        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);
        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        CTestAnomalyJob job("job", limits, jobConfig, modelConfig, wrappedOutputStream);

        CTestAnomalyJob::TStrStrUMap dataRows;
        dataRows["wibble"] = "12345678";
        dataRows["value"] = "1.0";
        dataRows["greenhouse"] = "rhubarb";

        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
        BOOST_REQUIRE_EQUAL(uint64_t(0), job.numRecordsHandled());
    }
    {
        // Test with bad time field
        model::CLimits limits;
        api::CAnomalyJobConfig jobConfig = CTestAnomalyJob::makeSimpleJobConfig(
            "metric", "value", "", "", "greenhouse");

        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);
        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        CTestAnomalyJob job("job", limits, jobConfig, modelConfig, wrappedOutputStream);

        CTestAnomalyJob::TStrStrUMap dataRows;
        dataRows["time"] = "hello";
        dataRows["value"] = "1.0";
        dataRows["greenhouse"] = "rhubarb";

        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
        BOOST_REQUIRE_EQUAL(uint64_t(0), job.numRecordsHandled());
    }
    {
        // Test with bad time field format
        model::CLimits limits;
        api::CAnomalyJobConfig jobConfig = CTestAnomalyJob::makeSimpleJobConfig(
            "metric", "value", "", "", "greenhouse");

        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);
        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        CTestAnomalyJob job("job", limits, jobConfig, modelConfig, wrappedOutputStream,
                            CTestAnomalyJob::TPersistCompleteFunc(), nullptr,
                            -1, "time", "%Y%m%m%H%M%S");

        CTestAnomalyJob::TStrStrUMap dataRows;
        dataRows["time"] = "hello world";
        dataRows["value"] = "1.0";
        dataRows["greenhouse"] = "rhubarb";

        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
        BOOST_REQUIRE_EQUAL(uint64_t(0), job.numRecordsHandled());
    }
}

BOOST_AUTO_TEST_CASE(testOutOfSequence) {
    {
        // Test out of sequence record
        model::CLimits limits;
        api::CAnomalyJobConfig jobConfig = CTestAnomalyJob::makeSimpleJobConfig(
            "metric", "value", "", "", "greenhouse");

        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);
        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        CTestAnomalyJob job("job", limits, jobConfig, modelConfig, wrappedOutputStream);

        job.description();
        job.descriptionAndDebugMemoryUsage();

        // add records which create partitions
        CTestAnomalyJob::TStrStrUMap dataRows;
        dataRows["time"] = "12345678";
        dataRows["value"] = "1.0";
        dataRows["greenhouse"] = "rhubarb";

        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
        BOOST_REQUIRE_EQUAL(uint64_t(1), job.numRecordsHandled());

        dataRows["time"] = "1234567";

        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
        BOOST_REQUIRE_EQUAL(uint64_t(1), job.numRecordsHandled());
        job.finalise();
    }
}

BOOST_AUTO_TEST_CASE(testOutputBucketResultsUntilGivenIncompleteInitialBucket) {
    const std::string inputFileName{"testfiles/incomplete_initial_bucket.txt"};
    const std::string configFileName{"testfiles/pop_sum_bytes_by_status_over_clientip.json"};

    const char* logFile{"test.log"};
    std::remove(logFile);
    BOOST_TEST_REQUIRE(ml::core::CLogger::instance().reconfigureFromFile(
        "testfiles/testLogErrors.boost.log.ini"));

    // Start by creating a detector with non-trivial state
    static const core_t::TTime BUCKET_SIZE{900};
    static const std::string JOB_ID{"pop_sum_bytes_by_status_over_clientip"};

    // Open the input and output files
    std::ifstream inputStrm{inputFileName.c_str()};
    BOOST_TEST_REQUIRE(inputStrm.is_open());

    std::ofstream outputStrm{core::COsFileFuncs::NULL_FILENAME};
    BOOST_TEST_REQUIRE(outputStrm.is_open());

    model::CLimits limits;
    api::CAnomalyJobConfig jobConfig;
    BOOST_TEST_REQUIRE(jobConfig.initFromFile(configFileName));

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE, model_t::E_None,
                                                          "", 0, false);

    core::CJsonOutputStreamWrapper wrappedOutputStream{outputStrm};

    std::string origSnapshotId;
    std::size_t numOrigDocs{0};

    CTestAnomalyJob origJob{JOB_ID,
                            limits,
                            jobConfig,
                            modelConfig,
                            wrappedOutputStream,
                            std::bind(&reportPersistComplete, std::placeholders::_1,
                                      std::ref(origSnapshotId), std::ref(numOrigDocs)),
                            nullptr,
                            -1,
                            api::CAnomalyJob::DEFAULT_TIME_FIELD_NAME,
                            api::CAnomalyJob::EMPTY_STRING};

    api::CDataProcessor* firstProcessor{&origJob};

    using TInputParserUPtr = std::unique_ptr<api::CInputParser>;
    const TInputParserUPtr parser{[&inputStrm]() -> TInputParserUPtr {
        return std::make_unique<api::CNdJsonInputParser>(inputStrm);
    }()};

    BOOST_TEST_REQUIRE(parser->readStreamIntoMaps(
        [firstProcessor](const api::CDataProcessor::TStrStrUMap& dataRowFields) {
            return firstProcessor->handleRecord(
                dataRowFields, api::CDataProcessor::TOptionalTime{});
        }));

    // Persist the detector state to a stringstream
    std::ostringstream* strm{nullptr};
    api::CSingleStreamDataAdder::TOStreamP ptr{strm = new std::ostringstream()};
    api::CSingleStreamDataAdder persister{ptr};
    BOOST_TEST_REQUIRE(firstProcessor->persistStateInForeground(persister, ""));
    const std::string origPersistedState{strm->str()};

    // restore the job and start the datafeed running in realtime

    std::string restoredSnapshotId;
    std::size_t numRestoredDocs{0};

    CTestAnomalyJob restoredJob{
        JOB_ID,
        limits,
        jobConfig,
        modelConfig,
        wrappedOutputStream,
        std::bind(&reportPersistComplete, std::placeholders::_1,
                  std::ref(restoredSnapshotId), std::ref(numRestoredDocs))};

    api::CDataProcessor* restoredFirstProcessor{&restoredJob};

    core_t::TTime completeToTime{0};

    auto restoredStrm = std::make_shared<boost::iostreams::filtering_istream>();
    restoredStrm->push(api::CStateRestoreStreamFilter());
    std::istringstream inputStream{origPersistedState};
    restoredStrm->push(inputStream);

    api::CSingleStreamSearcher retriever{restoredStrm};

    BOOST_TEST_REQUIRE(restoredFirstProcessor->restoreState(retriever, completeToTime));
    BOOST_TEST_REQUIRE(completeToTime > 0);

    restoredJob.outputBucketResultsUntil(1585701000);

    // Revert to the default logger settings
    ml::core::CLogger::instance().reset();

    std::ifstream log{logFile};
    // Boost.Log only creates files when the first message is logged,
    // and here we're asserting no messages logged
    if (log.is_open()) {
        char line[256];
        while (log.getline(line, 256)) {
            LOG_DEBUG(<< "Got '" << line << "'");
            BOOST_TEST_REQUIRE(false);
        }
        log.close();
        std::remove(logFile);
    }
}

BOOST_AUTO_TEST_CASE(testControlMessages) {
    {
        // Test control messages
        model::CLimits limits;
        api::CAnomalyJobConfig jobConfig = CTestAnomalyJob::makeSimpleJobConfig(
            "metric", "value", "", "", "greenhouse");

        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);
        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        CTestAnomalyJob job("job", limits, jobConfig, modelConfig, wrappedOutputStream);

        CTestAnomalyJob::TStrStrUMap dataRows;
        dataRows["."] = " ";
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
        BOOST_REQUIRE_EQUAL(uint64_t(0), job.numRecordsHandled());

        dataRows["."] = ".";
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
        BOOST_REQUIRE_EQUAL(uint64_t(0), job.numRecordsHandled());

        dataRows["."] = "f";
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
        BOOST_REQUIRE_EQUAL(uint64_t(0), job.numRecordsHandled());

        dataRows["."] = "f1";
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
        BOOST_REQUIRE_EQUAL(uint64_t(0), job.numRecordsHandled());
    }
    {
        // Test reset bucket
        model::CLimits limits;
        api::CAnomalyJobConfig jobConfig =
            CTestAnomalyJob::makeSimpleJobConfig("count", "", "", "", "greenhouse");

        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);

        CTestAnomalyJob::TStrStrUMap dataRows;
        dataRows["value"] = "2.0";
        dataRows["greenhouse"] = "rhubarb";

        std::stringstream outputStrm;
        {
            core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
            CTestAnomalyJob job("job", limits, jobConfig, modelConfig, wrappedOutputStream);

            core_t::TTime time = 12345678;
            for (std::size_t i = 0; i < 50; i++, time += (BUCKET_SIZE / 2)) {
                std::stringstream ss;
                ss << time;
                dataRows["time"] = ss.str();
                if (i == 40) {
                    for (std::size_t j = 0; j < 100; j++) {
                        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
                    }
                }
                BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
                if (i < 2) {
                    // We haven't processed one full bucket but it should be safe to flush.
                    dataRows["."] = "f1";
                    BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
                    dataRows.erase(".");
                }
            }
        }

        json::error_code ec;
        json::value results = json::parse(outputStrm.str(), ec);
        BOOST_TEST_REQUIRE(ec.failed() == false);
        BOOST_TEST_REQUIRE(results.is_array());

        const json::array& allRecords = results.as_array();
        bool foundRecord = false;
        for (auto& r : allRecords) {
            BOOST_TEST_REQUIRE(r.is_object());
            json::object::const_iterator recordsIt = r.as_object().find("records");
            if (recordsIt != r.as_object().end()) {
                auto& recordsArray = recordsIt->value().as_array().at(0);
                const json::value* actualIt = recordsArray.as_object().if_contains("actual");
                BOOST_TEST_REQUIRE(actualIt != nullptr);
                const json::array& values = actualIt->as_array();

                LOG_DEBUG(<< "values: " << values);
                BOOST_REQUIRE_EQUAL(102.0, values[0].to_number<double>());
                foundRecord = true;
            }
        }

        BOOST_TEST_REQUIRE(foundRecord);
        std::stringstream outputStrm2;
        {
            core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm2);
            CTestAnomalyJob job("job", limits, jobConfig, modelConfig, wrappedOutputStream);

            core_t::TTime time = 12345678;
            for (std::size_t i = 0; i < 50; i++, time += (BUCKET_SIZE / 2)) {
                std::stringstream ss;
                ss << time;
                dataRows["time"] = ss.str();
                if (i == 40) {
                    for (std::size_t j = 0; j < 100; j++) {
                        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
                    }
                }
                BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
                if (i == 40) {
                    CTestAnomalyJob::TStrStrUMap rows;
                    rows["."] = "r" + ss.str() + " " + ss.str();
                    BOOST_TEST_REQUIRE(job.handleRecord(rows));
                    for (std::size_t j = 0; j < 100; j++) {
                        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
                    }
                }
            }
        }

        json::value doc2_ = json::parse(outputStrm2.str(), ec);
        BOOST_TEST_REQUIRE(ec.failed() == false);
        BOOST_TEST_REQUIRE(doc2_.is_array());

        const json::value& allRecords2 = doc2_.as_array();
        foundRecord = false;
        for (auto& r : allRecords2.as_array()) {
            json::object::const_iterator recordsIt = r.as_object().find("records");
            if (recordsIt != r.as_object().end()) {
                auto& recordsArray = recordsIt->value().as_array().at(0);
                json::object::const_iterator actualIt =
                    recordsArray.as_object().find("actual");
                BOOST_TEST_REQUIRE(actualIt != recordsArray.as_object().end());
                const json::array& values = actualIt->value().as_array();

                BOOST_REQUIRE_EQUAL(101.0, values[0].to_number<double>());
                foundRecord = true;
            }
        }

        BOOST_TEST_REQUIRE(foundRecord);
    }
}

BOOST_AUTO_TEST_CASE(testSkipTimeControlMessage) {
    model::CLimits limits;
    api::CAnomalyJobConfig jobConfig =
        CTestAnomalyJob::makeSimpleJobConfig("count", "", "", "", "");

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);

    std::stringstream outputStrm;
    core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

    CTestAnomalyJob job("job", limits, jobConfig, modelConfig, wrappedOutputStream);

    CTestAnomalyJob::TStrStrUMap dataRows;

    core_t::TTime time = 3600;
    for (std::size_t i = 0; i < 10; ++i, time += BUCKET_SIZE) {
        std::ostringstream ss;
        ss << time;
        dataRows["time"] = ss.str();
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
    }

    wrappedOutputStream.syncFlush();
    BOOST_REQUIRE_EQUAL(9, countBuckets("bucket", outputStrm.str() + "]"));

    // Now let's skip time to Thursday, June 29, 2017 12:00:00 AM
    time = 1498694400;
    dataRows["."] = "s1498694400";
    BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
    dataRows.erase(".");

    // Check no new bucket results were written
    wrappedOutputStream.syncFlush();
    BOOST_REQUIRE_EQUAL(9, countBuckets("bucket", outputStrm.str() + "]"));

    // Let's send a few buckets after skip time
    for (std::size_t i = 0; i < 3; ++i, time += BUCKET_SIZE) {
        std::ostringstream ss;
        ss << time;
        dataRows["time"] = ss.str();
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
    }

    // Assert only 2 new buckets were written
    wrappedOutputStream.syncFlush();
    BOOST_REQUIRE_EQUAL(11, countBuckets("bucket", outputStrm.str() + "]"));
}

BOOST_AUTO_TEST_CASE(testIsPersistenceNeeded) {

    model::CLimits limits;
    api::CAnomalyJobConfig jobConfig =
        CTestAnomalyJob::makeSimpleJobConfig("count", "", "", "", "");

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);

    {
        // check that persistence is not needed if no input records have been handled
        // and the time has not been advanced

        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        CTestAnomalyJob job("job", limits, jobConfig, modelConfig, wrappedOutputStream);

        BOOST_REQUIRE_EQUAL(false, job.isPersistenceNeeded("test state"));

        job.finalise();
        wrappedOutputStream.syncFlush();

        std::string output = outputStrm.str();
        LOG_DEBUG(<< "Output has yielded: " << output);

        // check that no quantile state was persisted
        core::CRegex regex;
        regex.init("\n");
        core::CRegex::TStrVec lines;
        regex.split(output, lines);
        BOOST_REQUIRE_EQUAL(false, findLine("\"quantiles\":{\"job_id\":\"job\",\"quantile_state\".*",
                                            lines));
    }

    core_t::TTime time = 3600;
    {
        // check that persistence is needed if an input record has been handled

        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        CTestAnomalyJob job("job", limits, jobConfig, modelConfig, wrappedOutputStream);

        CTestAnomalyJob::TStrStrUMap dataRows;

        std::ostringstream ss;
        ss << time;
        dataRows["time"] = ss.str();
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));

        BOOST_REQUIRE_EQUAL(true, job.isPersistenceNeeded("test state"));

        job.finalise();
        wrappedOutputStream.syncFlush();

        std::string output = outputStrm.str();
        LOG_DEBUG(<< "Output has yielded: " << output);

        // check that the quantile state has actually been persisted
        core::CRegex regex;
        regex.init("\n");
        core::CRegex::TStrVec lines;
        regex.split(output, lines);
        BOOST_REQUIRE_EQUAL(true, findLine("\"quantiles\":{\"job_id\":\"job\",\"quantile_state\".*",
                                           lines));
    }

    {
        // check that persistence is needed if time has been advanced (via a control message)
        // even if no input data has been handled

        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        CTestAnomalyJob job("job", limits, jobConfig, modelConfig, wrappedOutputStream);

        CTestAnomalyJob::TStrStrUMap dataRows;

        time = 39600;
        dataRows["."] = "t39600";
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
        BOOST_TEST_REQUIRE(job.isPersistenceNeeded("test state"));

        BOOST_REQUIRE_EQUAL(true, job.isPersistenceNeeded("test state"));

        job.finalise();
        wrappedOutputStream.syncFlush();

        std::string output = outputStrm.str();
        LOG_DEBUG(<< "Output has yielded: " << output);

        // check that the quantile state has actually been persisted
        core::CRegex regex;
        regex.init("\n");
        core::CRegex::TStrVec lines;
        regex.split(output, lines);
        BOOST_REQUIRE_EQUAL(true, findLine("\"quantiles\":{\"job_id\":\"job\",\"quantile_state\".*",
                                           lines));
    }
}

BOOST_AUTO_TEST_CASE(testModelPlot) {
    core_t::TTime bucketSize = 10000;
    model::CLimits limits;

    ml::api::CAnomalyJobConfig jobConfig =
        CTestAnomalyJob::makeSimpleJobConfig("mean", "value", "animal", "", "");

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig(bucketSize, model_t::E_None,
                                                          "", 0, false);
    modelConfig.modelPlotBoundsPercentile(1.0);
    std::stringstream outputStrm;

    {
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        CTestAnomalyJob job("job", limits, jobConfig, modelConfig, wrappedOutputStream);

        CTestAnomalyJob::TStrStrUMap dataRows;
        dataRows["time"] = "10000000";
        dataRows["value"] = "2.0";
        dataRows["animal"] = "baboon";
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
        dataRows["value"] = "5.0";
        dataRows["animal"] = "shark";
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
        dataRows["time"] = "10010000";
        dataRows["value"] = "2.0";
        dataRows["animal"] = "baboon";
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
        dataRows["value"] = "5.0";
        dataRows["animal"] = "shark";
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
        dataRows["time"] = "10020000";
        dataRows["value"] = "2.0";
        dataRows["animal"] = "baboon";
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
        dataRows["value"] = "5.0";
        dataRows["animal"] = "shark";
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
        dataRows["time"] = "10030000";
        dataRows["value"] = "2.0";
        dataRows["animal"] = "baboon";
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
        dataRows["value"] = "5.0";
        dataRows["animal"] = "shark";
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
        dataRows["time"] = "10040000";
        dataRows["value"] = "3.0";
        dataRows["animal"] = "baboon";
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
        dataRows["value"] = "5.0";
        dataRows["animal"] = "shark";
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
        job.finalise();
    }

    std::string output = outputStrm.str();
    LOG_TRACE(<< "Output has yielded: " << output);
    core::CRegex regex;
    regex.init("\n");
    core::CRegex::TStrVec lines;
    regex.split(output, lines);
    BOOST_TEST_REQUIRE(findLine("model_feature.*timestamp.*10000000.*baboon", lines));
    BOOST_TEST_REQUIRE(findLine("model_feature.*timestamp.*10000000.*shark", lines));
    BOOST_TEST_REQUIRE(findLine("model_feature.*timestamp.*10010000.*baboon", lines));
    BOOST_TEST_REQUIRE(findLine("model_feature.*timestamp.*10010000.*shark", lines));
    BOOST_TEST_REQUIRE(findLine("model_feature.*timestamp.*10020000.*baboon", lines));
    BOOST_TEST_REQUIRE(findLine("model_feature.*timestamp.*10020000.*shark", lines));
    BOOST_TEST_REQUIRE(findLine("model_feature.*timestamp.*10030000.*baboon", lines));
    BOOST_TEST_REQUIRE(findLine("model_feature.*timestamp.*10030000.*shark", lines));
}

BOOST_AUTO_TEST_CASE(testInterimResultEdgeCases) {
    const char* logFile = "test.log";

    core_t::TTime bucketSize = 3600;
    model::CLimits limits;
    api::CAnomalyJobConfig jobConfig =
        CTestAnomalyJob::makeSimpleJobConfig("count", "", "error", "", "");

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig(bucketSize);

    std::stringstream outputStrm;

    core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

    CTestAnomalyJob job("job", limits, jobConfig, modelConfig, wrappedOutputStream);

    std::remove(logFile);
    BOOST_TEST_REQUIRE(ml::core::CLogger::instance().reconfigureFromFile(
        "testfiles/testLogErrors.boost.log.ini"));

    CTestAnomalyJob::TStrStrUMap dataRows;
    dataRows["time"] = "3610";
    dataRows["error"] = "e1";
    BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
    dataRows["time"] = "3670";
    dataRows["error"] = "e2";
    BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
    dataRows["time"] = "6820";
    dataRows["error"] = "e1";
    BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
    dataRows["time"] = "6820";
    dataRows["error"] = "e1";
    BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
    dataRows["time"] = "7850";
    dataRows["error"] = "e1";
    BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
    dataRows["time"] = "9310";
    dataRows["error"] = "e2";
    BOOST_TEST_REQUIRE(job.handleRecord(dataRows));

    dataRows["."] = "t7200";
    BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
    dataRows["."] = "i";
    BOOST_TEST_REQUIRE(job.handleRecord(dataRows));

    // Revert to the default logger settings
    ml::core::CLogger::instance().reset();

    std::ifstream log(logFile);
    // Boost.Log only creates files when the first message is logged,
    // and here we're asserting no messages logged
    if (log.is_open()) {
        char line[256];
        while (log.getline(line, 256)) {
            LOG_DEBUG(<< "Got '" << line << "'");
            BOOST_TEST_REQUIRE(false);
        }
        log.close();
        std::remove(logFile);
    }
}

BOOST_AUTO_TEST_CASE(testRestoreFailsWithEmptyStream) {
    model::CLimits limits;
    api::CAnomalyJobConfig jobConfig =
        CTestAnomalyJob::makeSimpleJobConfig("value", "", "", "", "greenhouse");

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);
    std::ostringstream outputStrm;
    core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

    CTestAnomalyJob job("job", limits, jobConfig, modelConfig, wrappedOutputStream);

    core_t::TTime completeToTime(0);
    CEmptySearcher restoreSearcher;
    BOOST_TEST_REQUIRE(job.restoreState(restoreSearcher, completeToTime) == false);
}

BOOST_AUTO_TEST_CASE(testConfigUpdate) {
    // This, in part, is essentially replicating the DetectionRulesIT/testScope Java REST test.
    // It proves useful to have the test here too, as it provides an entrypoint for investigating
    // any issues related to filters, especially when updating them when already referenced by anomaly detector models.
    // We simply want to see the job run to completion.
    ml::api::CAnomalyJobConfig jobConfig;
    BOOST_REQUIRE_EQUAL(true, jobConfig.initFromFiles("testfiles/count_over_ip_config.json",
                                                      "testfiles/filterConfig.json",
                                                      "testfiles/eventConfig.json"));

    const ml::api::CAnomalyJobConfig::CAnalysisConfig& analysisConfig =
        jobConfig.analysisConfig();

    model::CLimits limits;

    model::CAnomalyDetectorModelConfig modelConfig = analysisConfig.makeModelConfig();
    std::stringstream outputStrm;
    core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

    CTestAnomalyJob job("job", limits, jobConfig, modelConfig, wrappedOutputStream);

    auto generateRandomAlpha = [](int strLen) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 25);

        std::string str;
        for (int i = 0; i < strLen; ++i) {
            str += char('a' + dis(gen));
        }
        return str;
    };

    long timestamp = 1509062400000L;
    TStrStrPrVec data;

    for (int bucket = 0; bucket < 20; bucket++) {
        for (int i = 0; i < 5; i++) {
            data.emplace_back(core::CStringUtils::typeToString(timestamp),
                              generateRandomAlpha(10));
        }
        timestamp += 3600 * 1000;
    }

    // Now send anomalous counts for our filtered IPs plus 333.333.333.333
    auto namedIps = std::vector{"111.111.111.111", "222.222.222.222", "333.333.333.333"};
    for (int i = 0; i < 10; i++) {
        for (auto& ip : namedIps) {
            data.emplace_back(core::CStringUtils::typeToString(timestamp), ip);
        }
    }

    for (int bucket = 0; bucket < 3; bucket++) {
        for (int i = 0; i < 5; i++) {
            data.emplace_back(core::CStringUtils::typeToString(timestamp),
                              generateRandomAlpha(10));
        }
        timestamp += 3600 * 1000;
    }

    CTestAnomalyJob::TStrStrUMap dataRows;

    for (const auto & [ time, ip ] : data) {
        dataRows["time"] = time;
        dataRows["ip"] = ip;
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
    }

    BOOST_REQUIRE_EQUAL(145, job.numRecordsHandled());

    const std::string& detectorConfig1{R"(
        {
            "filters":[{"filter_id":"safe_ips", "items":["111.111.111.111","222.222.222.222"]}],
            "events":[{"description":"event_1", "rules":[{"actions":["skip_result","skip_model_update"],"conditions":[{"applies_to":"time","operator":"gte","value": 1.0},{"applies_to":"time","operator":"lt","value": 2.0}]}]}],
            "model_plot_config":{"enabled":true,"annotations_enabled":false},
            "detector_rules":{"detector_index":0,"custom_rules":[{"actions":["skip_result"],"conditions":[{"applies_to":"actual","operator":"gte","value":15.0},{"applies_to":"actual","operator":"lte","value":30.0}]}]}
        }
    )"};

    job.updateConfig(detectorConfig1);

    BOOST_REQUIRE_EQUAL(1, jobConfig.analysisConfig().detectionRules().size());
    auto itr = jobConfig.analysisConfig().detectionRules().find(0);
    BOOST_REQUIRE_EQUAL(1, itr->second.size());
    std::string rule{itr->second[0].print()};
    BOOST_REQUIRE_EQUAL(
        std::string("SKIP_RESULT IF ACTUAL >= 15.000000 AND ACTUAL <= 30.000000"), rule);

    api::CAnomalyJobConfig::CModelPlotConfig& modelPlotConfig = jobConfig.modelPlotConfig();
    BOOST_REQUIRE_EQUAL(false, modelPlotConfig.annotationsEnabled());
    BOOST_REQUIRE_EQUAL(true, modelPlotConfig.enabled());

    auto events = jobConfig.analysisConfig().scheduledEvents();
    BOOST_REQUIRE_EQUAL(1, events.size());
    BOOST_REQUIRE_EQUAL(std::string("event_1"), events[0].first);
    BOOST_REQUIRE_EQUAL(std::string("SKIP_RESULT AND SKIP_MODEL_UPDATE IF TIME >= 1.000000 AND TIME < 2.000000"),
                        events[0].second.print());

    auto ruleFilters = jobConfig.ruleFilters();
    BOOST_REQUIRE_EQUAL(1, ruleFilters.size());

    BOOST_REQUIRE_EQUAL(true, ruleFilters["safe_ips"].contains("111.111.111.111"));
    BOOST_REQUIRE_EQUAL(true, ruleFilters["safe_ips"].contains("222.222.222.222"));
    BOOST_REQUIRE_EQUAL(false, ruleFilters["safe_ips"].contains("333.333.333.333"));

    const std::string& detectorConfig2{R"(
        {
        "filters":[{"filter_id":"safe_ips", "items":["333.333.333.333"]}],
        "events":[{"description":"event_1", "rules":[{"actions":["skip_result","skip_model_update"],"conditions":[{"applies_to":"time","operator":"gte","value": 2.0},{"applies_to":"time","operator":"lt","value": 3.0}]}]}],
        "model_plot_config":{"enabled":false,"annotations_enabled":true},
        "detector_rules":{"detector_index":0,"custom_rules":[{"actions":["skip_result"],"conditions":[{"applies_to":"typical","operator":"gte","value":10.0},{"applies_to":"typical","operator":"lte","value":50.0}]}]}
        })"};

    job.updateConfig(detectorConfig2);

    data.clear();
    // Send another anomalous bucket
    for (int i = 0; i < 10; i++) {
        for (auto& ip : namedIps) {
            data.emplace_back(core::CStringUtils::typeToString(timestamp), ip);
        }
    }

    // Some more normal buckets
    for (int bucket = 0; bucket < 3; bucket++) {
        for (int i = 0; i < 5; i++) {
            data.emplace_back(core::CStringUtils::typeToString(timestamp),
                              generateRandomAlpha(10));
        }
        timestamp += 3600 * 1000;
    }

    dataRows.clear();
    for (const auto & [ time, ip ] : data) {
        dataRows["time"] = time;
        dataRows["ip"] = ip;
        BOOST_TEST_REQUIRE(job.handleRecord(dataRows));
    }

    BOOST_REQUIRE_EQUAL(190, job.numRecordsHandled());

    BOOST_REQUIRE_EQUAL(1, jobConfig.analysisConfig().detectionRules().size());
    itr = jobConfig.analysisConfig().detectionRules().find(0);
    BOOST_REQUIRE_EQUAL(1, itr->second.size());
    rule = itr->second[0].print();
    BOOST_REQUIRE_EQUAL(
        std::string("SKIP_RESULT IF TYPICAL >= 10.000000 AND TYPICAL <= 50.000000"), rule);

    modelPlotConfig = jobConfig.modelPlotConfig();
    BOOST_REQUIRE_EQUAL(true, modelPlotConfig.annotationsEnabled());
    BOOST_REQUIRE_EQUAL(false, modelPlotConfig.enabled());

    events = jobConfig.analysisConfig().scheduledEvents();
    BOOST_REQUIRE_EQUAL(1, events.size());
    BOOST_REQUIRE_EQUAL(std::string("event_1"), events[0].first);
    BOOST_REQUIRE_EQUAL(std::string("SKIP_RESULT AND SKIP_MODEL_UPDATE IF TIME >= 2.000000 AND TIME < 3.000000"),
                        events[0].second.print());

    ruleFilters = jobConfig.ruleFilters();
    BOOST_REQUIRE_EQUAL(1, ruleFilters.size());

    BOOST_REQUIRE_EQUAL(false, ruleFilters["safe_ips"].contains("111.111.111.111"));
    BOOST_REQUIRE_EQUAL(false, ruleFilters["safe_ips"].contains("222.222.222.222"));
    BOOST_REQUIRE_EQUAL(true, ruleFilters["safe_ips"].contains("333.333.333.333"));

    job.finalise();
    wrappedOutputStream.syncFlush();

    std::string output = outputStrm.str();
    LOG_TRACE(<< "Output has yielded: " << output);

    // check that the quantile state has actually been persisted
    core::CRegex regex;
    regex.init("\n");
    core::CRegex::TStrVec lines;
    regex.split(output, lines);
    BOOST_REQUIRE_EQUAL(
        true, findLine("\"quantiles\":{\"job_id\":\"job\",\"quantile_state\".*", lines));
}

BOOST_AUTO_TEST_CASE(testParsePersistControlMessageArgs) {
    {
        const ml::core_t::TTime expectedSnapshotTimestamp{1283524206};
        const std::string expectedSnapshotId{"my_special_snapshot"};
        const std::string expectedSnapshotDescription{
            "Supplied description for snapshot at " +
            ml::core::CTimeUtils::toIso8601(expectedSnapshotTimestamp)};

        std::ostringstream ostrm;
        ostrm << expectedSnapshotTimestamp << " " << expectedSnapshotId << " "
              << expectedSnapshotDescription;

        const std::string& validPersistControlMessage{ostrm.str()};

        ml::core_t::TTime snapshotTimestamp;
        std::string snapshotId;
        std::string snapshotDescription;
        BOOST_TEST_REQUIRE(ml::api::CAnomalyJob::parsePersistControlMessageArgs(
            validPersistControlMessage, snapshotTimestamp, snapshotId, snapshotDescription));

        BOOST_TEST_REQUIRE(expectedSnapshotTimestamp == snapshotTimestamp);
        BOOST_TEST_REQUIRE(expectedSnapshotId == snapshotId);
        BOOST_TEST_REQUIRE(expectedSnapshotDescription == snapshotDescription);
    }
    {
        const std::string invalidPersistControlMessage{
            "junk_timestamp snapshot_id invalid snapshot control message"};

        ml::core_t::TTime snapshotTimestamp;
        std::string snapshotId;
        std::string snapshotDescription;
        BOOST_TEST_REQUIRE(ml::api::CAnomalyJob::parsePersistControlMessageArgs(
                               invalidPersistControlMessage, snapshotTimestamp,
                               snapshotId, snapshotDescription) == false);
    }
    {
        const std::string invalidPersistControlMessage{" "};

        ml::core_t::TTime snapshotTimestamp;
        std::string snapshotId;
        std::string snapshotDescription;
        BOOST_TEST_REQUIRE(ml::api::CAnomalyJob::parsePersistControlMessageArgs(
                               invalidPersistControlMessage, snapshotTimestamp,
                               snapshotId, snapshotDescription) == false);
    }
    {
        const std::string invalidPersistControlMessage;

        ml::core_t::TTime snapshotTimestamp;
        std::string snapshotId;
        std::string snapshotDescription;
        BOOST_TEST_REQUIRE(ml::api::CAnomalyJob::parsePersistControlMessageArgs(
                               invalidPersistControlMessage, snapshotTimestamp,
                               snapshotId, snapshotDescription) == false);
    }
    {
        const ml::core_t::TTime expectedSnapshotTimestamp{1283524206};
        const std::string expectedSnapshotId{"my_special_snapshot"};

        // Empty description is valid.
        const std::string expectedSnapshotDescription;

        std::ostringstream ostrm;
        ostrm << expectedSnapshotTimestamp << " " << expectedSnapshotId << " "
              << expectedSnapshotDescription;

        const std::string& validPersistControlMessage{ostrm.str()};

        ml::core_t::TTime snapshotTimestamp;
        std::string snapshotId;
        std::string snapshotDescription;
        BOOST_TEST_REQUIRE(ml::api::CAnomalyJob::parsePersistControlMessageArgs(
                               validPersistControlMessage, snapshotTimestamp,
                               snapshotId, snapshotDescription) == true);
    }
    {
        const ml::core_t::TTime expectedSnapshotTimestamp{1283524206};
        const std::string expectedSnapshotId;
        const std::string expectedSnapshotDescription;

        std::ostringstream ostrm;
        ostrm << expectedSnapshotTimestamp << " " << expectedSnapshotId << " "
              << expectedSnapshotDescription;

        const std::string& invalidPersistControlMessage{ostrm.str()};

        ml::core_t::TTime snapshotTimestamp;
        std::string snapshotId;
        std::string snapshotDescription;
        BOOST_TEST_REQUIRE(ml::api::CAnomalyJob::parsePersistControlMessageArgs(
                               invalidPersistControlMessage, snapshotTimestamp,
                               snapshotId, snapshotDescription) == false);
    }
    {
        ml::core_t::TTime snapshotTimestamp;
        std::string snapshotId;
        std::string snapshotDescription;
        const std::string invalidPersistControlMessage{"invalid_control_message"};
        BOOST_TEST_REQUIRE(ml::api::CAnomalyJob::parsePersistControlMessageArgs(
                               invalidPersistControlMessage, snapshotTimestamp,
                               snapshotId, snapshotDescription) == false);
    }
}

BOOST_AUTO_TEST_CASE(testRestoreFromBadState) {
    using TStrIntMap = std::map<std::string, int>;
    // map of names of state files to the number of times the fatal error message
    // "Failed to restore time series decomposition." occurs in the output
    TStrIntMap stateFiles{{"testfiles/badState1.json", 0},
                          {"testfiles/badState2.json", 0},
                          {"testfiles/badState3.json", 2},
                          {"testfiles/badState4.json", 1}};
    for (const auto& stateFile : stateFiles) {

        // Open the input state file
        std::ifstream inputStrm(stateFile.first.c_str());
        BOOST_TEST_REQUIRE(inputStrm.is_open());
        std::string persistedState(std::istreambuf_iterator<char>{inputStrm},
                                   std::istreambuf_iterator<char>{});

        model::CLimits limits;
        api::CAnomalyJobConfig jobConfig = CTestAnomalyJob::makeSimpleJobConfig(
            "high_sum", "responsetime", "airline", "", "");

        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);

        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        CTestAnomalyJob job("job", limits, jobConfig, modelConfig, wrappedOutputStream);

        ml::core_t::TTime completeToTime(0);

        std::stringstream* output = new std::stringstream();
        ml::api::CSingleStreamSearcher::TIStreamP strm(output);
        boost::iostreams::filtering_ostream in;
        in.push(ml::api::CStateRestoreStreamFilter());
        in.push(*output);
        in << persistedState;
        in.flush();

        ml::api::CSingleStreamSearcher restoreSearcher(strm);

        BOOST_TEST_REQUIRE(job.restoreState(restoreSearcher, completeToTime) == false);
    }
}

BOOST_AUTO_TEST_SUITE_END()
