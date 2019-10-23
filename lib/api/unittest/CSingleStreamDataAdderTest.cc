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

#include <api/CAnomalyJob.h>
#include <api/CCsvInputParser.h>
#include <api/CFieldConfig.h>
#include <api/CFieldDataTyper.h>
#include <api/CJsonOutputWriter.h>
#include <api/CNdJsonInputParser.h>
#include <api/COutputChainer.h>
#include <api/CSingleStreamDataAdder.h>
#include <api/CSingleStreamSearcher.h>
#include <api/CStateRestoreStreamFilter.h>

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
}

BOOST_AUTO_TEST_CASE(testDetectorPersistBy) {
    this->detectorPersistHelper("testfiles/new_mlfields.conf",
                                "testfiles/big_ascending.txt", 0, "%d/%b/%Y:%T %z");
}

BOOST_AUTO_TEST_CASE(testDetectorPersistOver) {
    this->detectorPersistHelper("testfiles/new_mlfields_over.conf",
                                "testfiles/big_ascending.txt", 0, "%d/%b/%Y:%T %z");
}

BOOST_AUTO_TEST_CASE(testDetectorPersistPartition) {
    this->detectorPersistHelper("testfiles/new_mlfields_partition.conf",
                                "testfiles/big_ascending.txt", 0, "%d/%b/%Y:%T %z");
}

BOOST_AUTO_TEST_CASE(testDetectorPersistDc) {
    this->detectorPersistHelper("testfiles/new_persist_dc.conf",
                                "testfiles/files_users_programs.csv", 5);
}

BOOST_AUTO_TEST_CASE(testDetectorPersistCount) {
    this->detectorPersistHelper("testfiles/new_persist_count.conf",
                                "testfiles/files_users_programs.csv", 5);
}

BOOST_AUTO_TEST_CASE(testDetectorPersistCategorization) {
    this->detectorPersistHelper("testfiles/new_persist_categorization.conf",
                                "testfiles/time_messages.csv", 0);
}

void CSingleStreamDataAdderTest::detectorPersistHelper(const std::string& configFileName,
                                                       const std::string& inputFilename,
                                                       int latencyBuckets,
                                                       const std::string& timeFormat) {
    // Start by creating a detector with non-trivial state
    static const ml::core_t::TTime BUCKET_SIZE(3600);
    static const std::string JOB_ID("job");

    // Open the input and output files
    std::ifstream inputStrm(inputFilename.c_str());
    BOOST_TEST_REQUIRE(inputStrm.is_open());

    std::ofstream outputStrm(ml::core::COsFileFuncs::NULL_FILENAME);
    BOOST_TEST_REQUIRE(outputStrm.is_open());

    ml::model::CLimits limits;
    ml::api::CFieldConfig fieldConfig;
    BOOST_TEST_REQUIRE(fieldConfig.initFromFile(configFileName));

    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig(
            BUCKET_SIZE, ml::model_t::E_None, "", BUCKET_SIZE * latencyBuckets, false);

    ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
    ml::api::CJsonOutputWriter outputWriter(JOB_ID, wrappedOutputStream);

    std::string origSnapshotId;
    std::size_t numOrigDocs(0);
    ml::api::CAnomalyJob origJob(
        JOB_ID, limits, fieldConfig, modelConfig, wrappedOutputStream,
        std::bind(&reportPersistComplete, std::placeholders::_1,
                  std::ref(origSnapshotId), std::ref(numOrigDocs)),
        nullptr, -1, "time", timeFormat);

    ml::api::CDataProcessor* firstProcessor(&origJob);

    // Chain the detector's input
    ml::api::COutputChainer outputChainer(origJob);

    // The typer knows how to assign categories to records
    ml::api::CFieldDataTyper typer(JOB_ID, fieldConfig, limits, outputChainer, outputWriter);

    if (fieldConfig.fieldNameSuperset().count(ml::api::CFieldDataTyper::MLCATEGORY_NAME) > 0) {
        LOG_DEBUG(<< "Applying the categorization typer for anomaly detection");
        firstProcessor = &typer;
    }

    using TInputParserUPtr = std::unique_ptr<ml::api::CInputParser>;
    const TInputParserUPtr parser{[&inputFilename, &inputStrm]() -> TInputParserUPtr {
        if (inputFilename.rfind(".csv") == inputFilename.length() - 4) {
            return std::make_unique<ml::api::CCsvInputParser>(inputStrm);
        }
        return std::make_unique<ml::api::CNdJsonInputParser>(inputStrm);
    }()};

    BOOST_TEST_REQUIRE(parser->readStreamIntoMaps(std::bind(
        &ml::api::CDataProcessor::handleRecord, firstProcessor, std::placeholders::_1)));

    // Persist the detector state to a stringstream

    std::string origPersistedState;
    {
        std::ostringstream* strm(nullptr);
        ml::api::CSingleStreamDataAdder::TOStreamP ptr(strm = new std::ostringstream());
        ml::api::CSingleStreamDataAdder persister(ptr);
        BOOST_TEST_REQUIRE(firstProcessor->persistState(persister, ""));
        origPersistedState = strm->str();
    }

    // Now restore the state into a different detector

    std::string restoredSnapshotId;
    std::size_t numRestoredDocs(0);
    ml::api::CAnomalyJob restoredJob(
        JOB_ID, limits, fieldConfig, modelConfig, wrappedOutputStream,
        std::bind(&reportPersistComplete, std::placeholders::_1,
                  std::ref(restoredSnapshotId), std::ref(numRestoredDocs)));

    ml::api::CDataProcessor* restoredFirstProcessor(&restoredJob);

    // Chain the detector's input
    ml::api::COutputChainer restoredOutputChainer(restoredJob);

    // The typer knows how to assign categories to records
    ml::api::CFieldDataTyper restoredTyper(JOB_ID, fieldConfig, limits,
                                           restoredOutputChainer, outputWriter);

    size_t numCategorizerDocs(0);

    if (fieldConfig.fieldNameSuperset().count(ml::api::CFieldDataTyper::MLCATEGORY_NAME) > 0) {
        LOG_DEBUG(<< "Applying the categorization typer for anomaly detection");
        numCategorizerDocs = 1;
        restoredFirstProcessor = &restoredTyper;
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
    std::string newPersistedState;
    {
        std::ostringstream* strm(nullptr);
        ml::api::CSingleStreamDataAdder::TOStreamP ptr(strm = new std::ostringstream());
        ml::api::CSingleStreamDataAdder persister(ptr);
        BOOST_TEST_REQUIRE(restoredFirstProcessor->persistState(persister, ""));
        newPersistedState = strm->str();
    }

    BOOST_REQUIRE_EQUAL(numOrigDocs, numRestoredDocs);

    // The snapshot ID can be different between the two persists, so replace the
    // first occurrence of it (which is in the bulk metadata)
    BOOST_REQUIRE_EQUAL(size_t(1), ml::core::CStringUtils::replaceFirst(
                                       origSnapshotId, "snap", origPersistedState));
    BOOST_REQUIRE_EQUAL(size_t(1), ml::core::CStringUtils::replaceFirst(
                                       restoredSnapshotId, "snap", newPersistedState));

    BOOST_REQUIRE_EQUAL(origPersistedState, newPersistedState);
}

BOOST_AUTO_TEST_SUITE_END()
