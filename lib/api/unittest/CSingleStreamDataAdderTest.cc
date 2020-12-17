/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CJsonOutputStreamWrapper.h>
#include <core/COsFileFuncs.h>
#include <core/CStringUtils.h>
#include <core/CoreTypes.h>

#include <maths/CModelWeight.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CLimits.h>

#include <api/CAnomalyJobConfig.h>
#include <api/CCsvInputParser.h>
#include <api/CNdJsonInputParser.h>
#include <api/CSingleStreamDataAdder.h>
#include <api/CSingleStreamSearcher.h>
#include <api/CStateRestoreStreamFilter.h>

#include "CTestAnomalyJob.h"
#include "CTestFieldDataCategorizer.h"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/test/unit_test.hpp>

#include <fstream>
#include <memory>
#include <sstream>
#include <string>

BOOST_AUTO_TEST_SUITE(CSingleStreamDataAdderTest)

namespace {

void reportPersistComplete(ml::api::CModelSnapshotJsonWriter::SModelSnapshotReport modelSnapshotReport,
                           std::string& snapshotIdOut,
                           size_t& numDocsOut) {
    LOG_INFO(<< "Persist complete with description: " << modelSnapshotReport.s_Description);
    snapshotIdOut = modelSnapshotReport.s_SnapshotId;
    numDocsOut = modelSnapshotReport.s_NumDocs;
}

void detectorPersistHelper(const std::string& configFileName,
                           const std::string& inputFilename,
                           int latencyBuckets,
                           const std::string& timeFormat = std::string()) {
    // Start by creating a detector with non-trivial state
    static const ml::core_t::TTime BUCKET_SIZE(3600);
    static const std::string JOB_ID("job");

    // Open the input and output files
    std::ifstream inputStrm(inputFilename.c_str());
    BOOST_TEST_REQUIRE(inputStrm.is_open());

    std::ofstream outputStrm(ml::core::COsFileFuncs::NULL_FILENAME);
    BOOST_TEST_REQUIRE(outputStrm.is_open());

    ml::model::CLimits limits;
    ml::api::CAnomalyJobConfig jobConfig;
    BOOST_TEST_REQUIRE(jobConfig.initFromFile(configFileName));

    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig(
            BUCKET_SIZE, ml::model_t::E_None, "", BUCKET_SIZE * latencyBuckets, false);

    ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

    std::string origSnapshotId;
    std::size_t numOrigDocs(0);
    std::string origPersistedState;

    {
        CTestAnomalyJob origJob(
            JOB_ID, limits, jobConfig, modelConfig, wrappedOutputStream,
            std::bind(&reportPersistComplete, std::placeholders::_1,
                      std::ref(origSnapshotId), std::ref(numOrigDocs)),
            nullptr, -1, "time", timeFormat);

        // The categorizer knows how to assign categories to records
        CTestFieldDataCategorizer categorizer(JOB_ID, jobConfig.analysisConfig(),
                                              limits, &origJob, wrappedOutputStream);

        ml::api::CDataProcessor* firstProcessor{nullptr};
        if (jobConfig.analysisConfig().categorizationFieldName().empty() == false) {
            LOG_DEBUG(<< "Applying the categorization categorizer for anomaly detection");
            firstProcessor = &categorizer;
        } else {
            firstProcessor = &origJob;
        }

        using TInputParserUPtr = std::unique_ptr<ml::api::CInputParser>;
        const TInputParserUPtr parser{[&inputFilename, &inputStrm]() -> TInputParserUPtr {
            ml::api::CInputParser::TStrVec mutableFields{CTestFieldDataCategorizer::MLCATEGORY_NAME};
            if (inputFilename.rfind(".csv") == inputFilename.length() - 4) {
                return std::make_unique<ml::api::CCsvInputParser>(
                    std::move(mutableFields), inputStrm);
            }
            return std::make_unique<ml::api::CNdJsonInputParser>(
                std::move(mutableFields), inputStrm);
        }()};

        BOOST_TEST_REQUIRE(parser->readStreamIntoMaps(
            [firstProcessor](const ml::api::CDataProcessor::TStrStrUMap& dataRowFields) {
                return firstProcessor->handleRecord(
                    dataRowFields, ml::api::CDataProcessor::TOptionalTime{});
            }));

        // Persist the detector state to a stringstream
        std::ostringstream* strm(nullptr);
        ml::api::CSingleStreamDataAdder::TOStreamP ptr(strm = new std::ostringstream());
        ml::api::CSingleStreamDataAdder persister(ptr);
        BOOST_TEST_REQUIRE(firstProcessor->persistStateInForeground(persister, ""));
        origPersistedState = strm->str();
    }

    // Now restore the state into a different detector

    std::string restoredSnapshotId;
    std::size_t numRestoredDocs(0);
    std::string newPersistedState;

    {
        CTestAnomalyJob restoredJob(
            JOB_ID, limits, jobConfig, modelConfig, wrappedOutputStream,
            std::bind(&reportPersistComplete, std::placeholders::_1,
                      std::ref(restoredSnapshotId), std::ref(numRestoredDocs)));

        // The categorizer knows how to assign categories to records
        CTestFieldDataCategorizer restoredCategorizer(
            JOB_ID, jobConfig.analysisConfig(), limits, &restoredJob, wrappedOutputStream);

        size_t numCategorizerDocs(0);

        ml::api::CDataProcessor* restoredFirstProcessor{nullptr};
        if (jobConfig.analysisConfig().categorizationFieldName().empty() == false) {
            LOG_DEBUG(<< "Applying the categorization categorizer for anomaly detection");
            numCategorizerDocs = 1;
            restoredFirstProcessor = &restoredCategorizer;
        } else {
            restoredFirstProcessor = &restoredJob;
        }

        {
            ml::core_t::TTime completeToTime(0);

            auto strm = std::make_shared<boost::iostreams::filtering_istream>();
            strm->push(ml::api::CStateRestoreStreamFilter());
            std::istringstream inputStream(origPersistedState);
            strm->push(inputStream);

            ml::api::CSingleStreamSearcher retriever(strm);

            BOOST_TEST_REQUIRE(restoredFirstProcessor->restoreState(retriever, completeToTime));
            BOOST_TEST_REQUIRE(completeToTime > 0);
            BOOST_REQUIRE_EQUAL(
                numOrigDocs + numCategorizerDocs,
                strm->component<ml::api::CStateRestoreStreamFilter>(0)->getDocCount());
        }

        // Finally, persist the new detector state and compare the result
        std::ostringstream* strm(nullptr);
        ml::api::CSingleStreamDataAdder::TOStreamP ptr(strm = new std::ostringstream());
        ml::api::CSingleStreamDataAdder persister(ptr);
        BOOST_TEST_REQUIRE(restoredFirstProcessor->persistStateInForeground(persister, ""));
        newPersistedState = strm->str();
    }

    BOOST_REQUIRE_EQUAL(numOrigDocs, numRestoredDocs);

    // The snapshot ID can be different between the two persists, so replace the
    // first occurrence of it (which is in the bulk metadata)
    BOOST_REQUIRE_EQUAL(size_t(1), ml::core::CStringUtils::replaceFirst(
                                       origSnapshotId, "snap", origPersistedState));
    BOOST_REQUIRE_EQUAL(size_t(1), ml::core::CStringUtils::replaceFirst(
                                       restoredSnapshotId, "snap", newPersistedState));

    // Replace the zero byte separators to avoid '\0's in the output if the
    // test fails
    std::replace(origPersistedState.begin(), origPersistedState.end(), '\0', ',');
    std::replace(newPersistedState.begin(), newPersistedState.end(), '\0', ',');

    BOOST_REQUIRE_EQUAL(origPersistedState, newPersistedState);
}
}

BOOST_AUTO_TEST_CASE(testDetectorPersistBy) {
    detectorPersistHelper("testfiles/new_mlfields.json",
                          "testfiles/big_ascending.txt", 0, "%d/%b/%Y:%T %z");
}

BOOST_AUTO_TEST_CASE(testDetectorPersistOver) {
    detectorPersistHelper("testfiles/new_mlfields_over.json",
                          "testfiles/big_ascending.txt", 0, "%d/%b/%Y:%T %z");
}

BOOST_AUTO_TEST_CASE(testDetectorPersistPartition) {
    detectorPersistHelper("testfiles/new_mlfields_partition.json",
                          "testfiles/big_ascending.txt", 0, "%d/%b/%Y:%T %z");
}

BOOST_AUTO_TEST_CASE(testDetectorPersistDc) {
    detectorPersistHelper("testfiles/new_persist_dc.json",
                          "testfiles/files_users_programs.csv", 5);
}

BOOST_AUTO_TEST_CASE(testDetectorPersistCount) {
    detectorPersistHelper("testfiles/new_persist_count.json",
                          "testfiles/files_users_programs.csv", 5);
}

BOOST_AUTO_TEST_CASE(testDetectorPersistCategorization) {
    detectorPersistHelper("testfiles/new_persist_categorization.json",
                          "testfiles/time_messages.csv", 0);
}

BOOST_AUTO_TEST_SUITE_END()
