/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CLogger.h>
#include <core/COsFileFuncs.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CLimits.h>

#include <api/CAnomalyJob.h>
#include <api/CCsvOutputWriter.h>
#include <api/CFieldConfig.h>
#include <api/CFieldDataCategorizer.h>
#include <api/CJsonOutputWriter.h>
#include <api/CResultNormalizer.h>
#include <api/CSingleStreamDataAdder.h>
#include <api/CSingleStreamSearcher.h>
#include <api/CStateRestoreStreamFilter.h>

#include <boost/test/unit_test.hpp>

#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

BOOST_AUTO_TEST_SUITE(CRestorePreviousStateTest)

namespace {

void reportPersistComplete(ml::api::CModelSnapshotJsonWriter::SModelSnapshotReport modelSnapshotReport,
                           std::string& snapshotIdOut,
                           size_t& numDocsOut) {
    LOG_DEBUG(<< "Persist complete with description: " << modelSnapshotReport.s_Description);
    snapshotIdOut = modelSnapshotReport.s_SnapshotId;
    numDocsOut = modelSnapshotReport.s_NumDocs;
}

struct SRestoreTestConfig {
    std::string s_Version;
    bool s_DetectorRestoreIsSymmetric;
    bool s_CategorizerRestoreIsSymmetric;
};

const std::vector<SRestoreTestConfig> BWC_VERSIONS{
    SRestoreTestConfig{"5.6.0", false, false}, SRestoreTestConfig{"6.0.0", false, false},
    SRestoreTestConfig{"6.1.0", false, false}};

std::string stripDocIds(const std::string& persistedState) {
    // State is persisted in the Elasticsearch bulk format.
    // This is an index action followed by the document source:
    // { "index": { "id": "foo" ... }}\n
    // { "field1" : "value1", ... }\n
    //
    // Only the doc IDs should be different so strip out the lines with index operations
    std::istringstream input(persistedState);
    std::ostringstream output;

    std::string line;
    while (std::getline(input, line)) {
        // Remove lines with the document IDs
        if (line.compare(0, 16, "{\"index\":{\"_id\":") != 0) {
            output << line;
        }
    }

    std::string strippedText = output.str();
    LOG_TRACE(<< "Stripped:" << strippedText << ml::core_t::LINE_ENDING);
    return strippedText;
}

void categorizerRestoreHelper(const std::string& stateFile, bool isSymmetric) {
    ml::model::CLimits limits;
    ml::api::CFieldConfig config("count", "mlcategory");

    std::ostringstream outputStrm;
    ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
    ml::api::CJsonOutputWriter writer("job", wrappedOutputStream);
    ml::api::CFieldDataCategorizer restoredCategorizer("job", config, limits, writer, writer);

    std::ifstream inputStrm(stateFile.c_str());
    BOOST_TEST_REQUIRE(inputStrm.is_open());
    std::string origPersistedState(std::istreambuf_iterator<char>{inputStrm},
                                   std::istreambuf_iterator<char>{});

    {
        ml::core_t::TTime completeToTime(0);
        auto strm = std::make_shared<boost::iostreams::filtering_istream>();

        strm->push(ml::api::CStateRestoreStreamFilter());

        // rewind the stream so we read from the beginning
        inputStrm.seekg(0);
        strm->push(inputStrm);
        ml::api::CSingleStreamSearcher retriever(strm);
        BOOST_TEST_REQUIRE(restoredCategorizer.restoreState(retriever, completeToTime));
    }

    if (isSymmetric) {
        // Test the persisted state of the restored detector is the
        // same as the original
        std::string newPersistedState;
        {
            std::ostringstream* strm(nullptr);
            ml::api::CSingleStreamDataAdder::TOStreamP ptr(strm = new std::ostringstream());
            ml::api::CSingleStreamDataAdder persister(ptr);
            BOOST_TEST_REQUIRE(restoredCategorizer.persistState(persister, ""));
            newPersistedState = strm->str();
        }
        BOOST_REQUIRE_EQUAL(stripDocIds(origPersistedState), stripDocIds(newPersistedState));
    }
}

void anomalyDetectorRestoreHelper(const std::string& stateFile,
                                  const std::string& configFileName,
                                  bool isSymmetric,
                                  int latencyBuckets) {
    // Open the input state file
    std::ifstream inputStrm(stateFile.c_str());
    BOOST_TEST_REQUIRE(inputStrm.is_open());
    std::string origPersistedState(std::istreambuf_iterator<char>{inputStrm},
                                   std::istreambuf_iterator<char>{});

    // Start by creating a detector with non-trivial state
    static const ml::core_t::TTime BUCKET_SIZE(3600);
    static const std::string JOB_ID("job");

    ml::model::CLimits limits;
    ml::api::CFieldConfig fieldConfig;
    BOOST_TEST_REQUIRE(fieldConfig.initFromFile(configFileName));

    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig(
            BUCKET_SIZE, ml::model_t::E_None, "", BUCKET_SIZE * latencyBuckets, false);

    std::ofstream outputStrm(ml::core::COsFileFuncs::NULL_FILENAME);
    BOOST_TEST_REQUIRE(outputStrm.is_open());

    ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

    std::string restoredSnapshotId;
    std::size_t numRestoredDocs(0);
    ml::api::CAnomalyJob restoredJob(
        JOB_ID, limits, fieldConfig, modelConfig, wrappedOutputStream,
        std::bind(&reportPersistComplete, std::placeholders::_1,
                  std::ref(restoredSnapshotId), std::ref(numRestoredDocs)));

    std::size_t numDocsInStateFile(0);
    {
        ml::core_t::TTime completeToTime(0);

        std::stringstream* output = new std::stringstream();
        ml::api::CSingleStreamSearcher::TIStreamP strm(output);
        boost::iostreams::filtering_ostream in;
        in.push(ml::api::CStateRestoreStreamFilter());
        in.push(*output);
        in << origPersistedState;
        in.flush();

        numDocsInStateFile =
            in.component<ml::api::CStateRestoreStreamFilter>(0)->getDocCount();

        ml::api::CSingleStreamSearcher retriever(strm);
        BOOST_TEST_REQUIRE(restoredJob.restoreState(retriever, completeToTime));
        BOOST_TEST_REQUIRE(completeToTime > 0);
    }

    if (isSymmetric) {
        // Test the persisted state of the restored detector is the
        // same as the original
        std::string newPersistedState;
        {
            std::ostringstream* strm(nullptr);
            ml::api::CSingleStreamDataAdder::TOStreamP ptr(strm = new std::ostringstream());
            ml::api::CSingleStreamDataAdder persister(ptr);
            BOOST_TEST_REQUIRE(restoredJob.persistState(persister, ""));
            newPersistedState = strm->str();
        }

        BOOST_REQUIRE_EQUAL(numRestoredDocs, numDocsInStateFile);
        BOOST_REQUIRE_EQUAL(stripDocIds(origPersistedState), stripDocIds(newPersistedState));
    }
}
}

BOOST_AUTO_TEST_CASE(testRestoreDetectorBy) {
    for (const auto& version : BWC_VERSIONS) {
        LOG_INFO(<< "Test restoring state from version " << version.s_Version);
        anomalyDetectorRestoreHelper(
            "testfiles/state/" + version.s_Version + "/by_detector_state.json",
            "testfiles/new_mlfields.conf", version.s_DetectorRestoreIsSymmetric, 0);
    }
}

BOOST_AUTO_TEST_CASE(testRestoreDetectorOver) {
    for (const auto& version : BWC_VERSIONS) {
        LOG_INFO(<< "Test restoring state from version " << version.s_Version);
        anomalyDetectorRestoreHelper("testfiles/state/" + version.s_Version + "/over_detector_state.json",
                                     "testfiles/new_mlfields_over.conf",
                                     version.s_DetectorRestoreIsSymmetric, 0);
    }
}

BOOST_AUTO_TEST_CASE(testRestoreDetectorPartition) {
    for (const auto& version : BWC_VERSIONS) {
        LOG_INFO(<< "Test restoring state from version " << version.s_Version);
        anomalyDetectorRestoreHelper("testfiles/state/" + version.s_Version + "/partition_detector_state.json",
                                     "testfiles/new_mlfields_partition.conf",
                                     version.s_DetectorRestoreIsSymmetric, 0);
    }
}

BOOST_AUTO_TEST_CASE(testRestoreDetectorDc) {
    for (const auto& version : BWC_VERSIONS) {
        LOG_INFO(<< "Test restoring state from version " << version.s_Version);
        anomalyDetectorRestoreHelper(
            "testfiles/state/" + version.s_Version + "/dc_detector_state.json",
            "testfiles/new_persist_dc.conf", version.s_DetectorRestoreIsSymmetric, 5);
    }
}

BOOST_AUTO_TEST_CASE(testRestoreDetectorCount) {
    for (const auto& version : BWC_VERSIONS) {
        LOG_INFO(<< "Test restoring state from version " << version.s_Version);
        anomalyDetectorRestoreHelper("testfiles/state/" + version.s_Version + "/count_detector_state.json",
                                     "testfiles/new_persist_count.conf",
                                     version.s_DetectorRestoreIsSymmetric, 5);
    }
}

BOOST_AUTO_TEST_CASE(testRestoreNormalizer) {
    for (const auto& version : BWC_VERSIONS) {
        ml::model::CAnomalyDetectorModelConfig modelConfig =
            ml::model::CAnomalyDetectorModelConfig::defaultConfig(3600);
        ml::api::CCsvOutputWriter outputWriter;
        ml::api::CResultNormalizer normalizer(modelConfig, outputWriter);
        BOOST_TEST_REQUIRE(normalizer.initNormalizer(
            "testfiles/state/" + version.s_Version + "/normalizer_state.json"));
    }
}

BOOST_AUTO_TEST_CASE(testRestoreCategorizer) {
    for (const auto& version : BWC_VERSIONS) {
        LOG_INFO(<< "Test restoring state from version " << version.s_Version);
        categorizerRestoreHelper("testfiles/state/" + version.s_Version + "/categorizer_state.json",
                                 version.s_CategorizerRestoreIsSymmetric);
    }
}

BOOST_AUTO_TEST_SUITE_END()
