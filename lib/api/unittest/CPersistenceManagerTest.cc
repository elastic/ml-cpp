/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CJsonOutputStreamWrapper.h>
#include <core/COsFileFuncs.h>
#include <core/CProgramCounters.h>
#include <core/CStringUtils.h>
#include <core/CoreTypes.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CLimits.h>

#include <api/CAnomalyJobConfig.h>
#include <api/CDataProcessor.h>
#include <api/CModelSnapshotJsonWriter.h>
#include <api/CNdJsonInputParser.h>
#include <api/CPersistenceManager.h>
#include <api/CSingleStreamDataAdder.h>

#include "CTestAnomalyJob.h"
#include "CTestFieldDataCategorizer.h"

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

BOOST_AUTO_TEST_SUITE(CPersistenceManagerTest)

namespace {

void reportPersistComplete(ml::api::CModelSnapshotJsonWriter::SModelSnapshotReport modelSnapshotReport,
                           ml::core_t::TTime& snapshotTimestamp,
                           std::string& description,
                           std::string& snapshotIdOut,
                           std::size_t& numDocsOut) {
    LOG_DEBUG(<< "Persist complete with description: " << modelSnapshotReport.s_Description);
    snapshotTimestamp = modelSnapshotReport.s_SnapshotTimestamp;
    description = modelSnapshotReport.s_Description;
    snapshotIdOut = modelSnapshotReport.s_SnapshotId;
    numDocsOut = modelSnapshotReport.s_NumDocs;
}
}

class CTestFixture {
protected:
    void foregroundBackgroundCompCategorizationAndAnomalyDetection(const std::string& configFileName) {
        // Start by creating processors with non-trivial state

        static const ml::core_t::TTime BUCKET_SIZE{3600};
        static const std::string JOB_ID{"job"};

        std::string inputFilename{"testfiles/big_ascending.txt"};

        // Open the input and output files
        std::ifstream inputStrm{inputFilename};
        BOOST_TEST_REQUIRE(inputStrm.is_open());

        std::ofstream outputStrm{ml::core::COsFileFuncs::NULL_FILENAME};
        BOOST_TEST_REQUIRE(outputStrm.is_open());

        ml::model::CLimits limits;
        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_TEST_REQUIRE(jobConfig.initFromFile(configFileName));

        ml::model::CAnomalyDetectorModelConfig modelConfig{
            ml::model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE)};

        std::ostringstream* backgroundStream{nullptr};
        ml::api::CSingleStreamDataAdder::TOStreamP backgroundStreamPtr{
            backgroundStream = new std::ostringstream()};
        ml::api::CSingleStreamDataAdder backgroundDataAdder{backgroundStreamPtr};

        std::ostringstream* foregroundStream{nullptr};
        ml::api::CSingleStreamDataAdder::TOStreamP foregroundStreamPtr{
            foregroundStream = new std::ostringstream()};
        ml::api::CSingleStreamDataAdder foregroundDataAdder{foregroundStreamPtr};

        // The 30000 second persist interval is set large enough that the timer
        // will not trigger during the test - we bypass the timer in this test
        // and kick off the background persistence chain explicitly
        ml::api::CPersistenceManager persistenceManager{
            30000, false, backgroundDataAdder, foregroundDataAdder};

        ml::core_t::TTime snapshotTimestamp;
        std::string description;
        std::string snapshotId;
        std::size_t numDocs{0};

        std::string backgroundSnapshotId;
        std::string foregroundSnapshotId;
        std::string foregroundSnapshotId2;

        std::ostringstream* foregroundStream2{nullptr};
        ml::api::CSingleStreamDataAdder::TOStreamP foregroundStreamPtr2{
            foregroundStream2 = new std::ostringstream()};
        {
            ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

            CTestAnomalyJob job{JOB_ID,
                                limits,
                                jobConfig,
                                modelConfig,
                                wrappedOutputStream,
                                std::bind(&reportPersistComplete, std::placeholders::_1,
                                          std::ref(snapshotTimestamp), std::ref(description),
                                          std::ref(snapshotId), std::ref(numDocs)),
                                &persistenceManager,
                                -1,
                                "time",
                                "%d/%b/%Y:%T %z"};

            // The categorizer knows how to assign categories to records
            CTestFieldDataCategorizer categorizer{
                JOB_ID, jobConfig.analysisConfig(), limits,
                &job,   wrappedOutputStream,        &persistenceManager};

            ml::api::CDataProcessor* firstProcessor{nullptr};
            if (jobConfig.analysisConfig().categorizationFieldName().empty() == false) {
                LOG_DEBUG(<< "Applying the categorization categorizer for anomaly detection");
                firstProcessor = &categorizer;
            } else {
                firstProcessor = &job;
            }

            ml::api::CNdJsonInputParser parser{
                {CTestFieldDataCategorizer::MLCATEGORY_NAME}, inputStrm};

            BOOST_TEST_REQUIRE(parser.readStreamIntoMaps(
                [firstProcessor](const ml::api::CDataProcessor::TStrStrUMap& dataRowFields) {
                    return firstProcessor->handleRecord(
                        dataRowFields, ml::api::CDataProcessor::TOptionalTime{});
                }));

            // Persist the processors' state in the background
            BOOST_TEST_REQUIRE(firstProcessor->periodicPersistStateInBackground());
            BOOST_TEST_REQUIRE(persistenceManager.startPersistInBackground());

            LOG_DEBUG(<< "Before waiting for the background persister to be idle");
            BOOST_TEST_REQUIRE(persistenceManager.waitForIdle());
            LOG_DEBUG(<< "After waiting for the background persister to be idle");
            backgroundSnapshotId = snapshotId;

            // Now persist the processors' state in the foreground

            BOOST_TEST_REQUIRE(firstProcessor->periodicPersistStateInForeground());
            persistenceManager.startPersist();
            foregroundSnapshotId = snapshotId;

            // ... persist in foreground again by directly calling persistStateInForeground
            ml::api::CSingleStreamDataAdder foregroundDataAdder2(foregroundStreamPtr2);
            BOOST_TEST_REQUIRE(firstProcessor->persistStateInForeground(
                foregroundDataAdder2, "Periodic foreground persistence at "));
            foregroundSnapshotId2 = snapshotId;
        }

        std::string backgroundState{backgroundStream->str()};
        std::string foregroundState{foregroundStream->str()};
        std::string foregroundState2{foregroundStream2->str()};

        // The snapshot ID can be different between the two persists, so replace the
        // first occurrence of it (which is in the bulk metadata)
        BOOST_REQUIRE_EQUAL(1, ml::core::CStringUtils::replaceFirst(
                                   backgroundSnapshotId, "snap", backgroundState));
        BOOST_REQUIRE_EQUAL(1, ml::core::CStringUtils::replaceFirst(
                                   foregroundSnapshotId, "snap", foregroundState));
        BOOST_REQUIRE_EQUAL(1, ml::core::CStringUtils::replaceFirst(
                                   foregroundSnapshotId2, "snap", foregroundState2));

        // Replace the zero byte separators to avoid '\0's in the output if the
        // test fails
        std::replace(backgroundState.begin(), backgroundState.end(), '\0', ',');
        std::replace(foregroundState.begin(), foregroundState.end(), '\0', ',');
        std::replace(foregroundState2.begin(), foregroundState2.end(), '\0', ',');

        BOOST_REQUIRE_EQUAL(foregroundState, foregroundState2);
        BOOST_REQUIRE_EQUAL(backgroundState, foregroundState2);
        BOOST_REQUIRE_EQUAL(backgroundState, foregroundState);
    }

    void foregroundBackgroundCompAnomalyDetectionAfterStaticsUpdate(const std::string& configFileName) {
        // Start by creating processors with non-trivial state

        static const ml::core_t::TTime BUCKET_SIZE{3600};
        static const std::string JOB_ID{"job"};

        std::string inputFilename{"testfiles/big_ascending.txt"};

        // Open the input and output files
        std::ifstream inputStrm{inputFilename};
        BOOST_TEST_REQUIRE(inputStrm.is_open());

        std::ofstream outputStrm{ml::core::COsFileFuncs::NULL_FILENAME};
        BOOST_TEST_REQUIRE(outputStrm.is_open());

        ml::model::CLimits limits;
        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_TEST_REQUIRE(jobConfig.initFromFile(configFileName));

        ml::model::CAnomalyDetectorModelConfig modelConfig{
            ml::model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE)};

        std::ostringstream* backgroundStream{nullptr};
        ml::api::CSingleStreamDataAdder::TOStreamP backgroundStreamPtr{
            backgroundStream = new std::ostringstream()};
        ml::api::CSingleStreamDataAdder backgroundDataAdder(backgroundStreamPtr);

        std::ostringstream* foregroundStream{nullptr};
        ml::api::CSingleStreamDataAdder::TOStreamP foregroundStreamPtr{
            foregroundStream = new std::ostringstream()};

        // Persist the processors' state in the foreground
        ml::api::CSingleStreamDataAdder foregroundDataAdder{foregroundStreamPtr};

        // The 30000 second persist interval is set large enough that the timer
        // will not trigger during the test - we bypass the timer in this test
        // and kick off the background persistence chain explicitly
        ml::api::CPersistenceManager persistenceManager{
            30000, false, backgroundDataAdder, foregroundDataAdder};

        ml::core_t::TTime snapshotTimestamp;
        std::string description;
        std::string snapshotId;
        std::size_t numDocs{0};

        std::string backgroundSnapshotId;
        std::string foregroundSnapshotId;

        {
            ml::core::CJsonOutputStreamWrapper wrappedOutputStream{outputStrm};

            CTestAnomalyJob job{JOB_ID,
                                limits,
                                jobConfig,
                                modelConfig,
                                wrappedOutputStream,
                                std::bind(&reportPersistComplete, std::placeholders::_1,
                                          std::ref(snapshotTimestamp), std::ref(description),
                                          std::ref(snapshotId), std::ref(numDocs)),
                                &persistenceManager,
                                -1,
                                "time",
                                "%d/%b/%Y:%T %z"};

            ml::api::CDataProcessor* firstProcessor{&job};

            ml::api::CNdJsonInputParser parser{
                {CTestFieldDataCategorizer::MLCATEGORY_NAME}, inputStrm};

            BOOST_TEST_REQUIRE(parser.readStreamIntoMaps(
                [firstProcessor](const ml::api::CDataProcessor::TStrStrUMap& dataRowFields) {
                    return firstProcessor->handleRecord(
                        dataRowFields, ml::api::CDataProcessor::TOptionalTime{});
                }));

            // Ensure the model size stats are up to date
            job.finalise();

            BOOST_TEST_REQUIRE(firstProcessor->periodicPersistStateInForeground());
            persistenceManager.startPersist();

            foregroundSnapshotId = snapshotId;

            // Now persist the processors' state in the background
            BOOST_TEST_REQUIRE(firstProcessor->periodicPersistStateInBackground());
            BOOST_TEST_REQUIRE(persistenceManager.startPersistInBackground());

            //Increment one of the counter values
            ++ml::core::CProgramCounters::counter(ml::counter_t::E_TSADMemoryUsage);

            LOG_DEBUG(<< "Before waiting for the background persister to be idle");
            BOOST_TEST_REQUIRE(persistenceManager.waitForIdle());
            LOG_DEBUG(<< "After waiting for the background persister to be idle");
            backgroundSnapshotId = snapshotId;
        }

        std::string backgroundState{backgroundStream->str()};
        std::string foregroundState{foregroundStream->str()};

        // The snapshot ID can be different between the two persists, so replace the
        // first occurrence of it (which is in the bulk metadata)
        BOOST_REQUIRE_EQUAL(1, ml::core::CStringUtils::replaceFirst(
                                   backgroundSnapshotId, "snap", backgroundState));
        BOOST_REQUIRE_EQUAL(1, ml::core::CStringUtils::replaceFirst(
                                   foregroundSnapshotId, "snap", foregroundState));

        // Replace the zero byte separators to avoid '\0's in the output if the
        // test fails
        std::replace(backgroundState.begin(), backgroundState.end(), '\0', ',');
        std::replace(foregroundState.begin(), foregroundState.end(), '\0', ',');

        BOOST_REQUIRE_EQUAL(backgroundState, foregroundState);
    }

    void foregroundPersistWithGivenSnapshotDescriptors(const std::string& configFileName) {
        // Start by creating processors with non-trivial state

        static const ml::core_t::TTime BUCKET_SIZE{3600};
        static const std::string JOB_ID{"job"};

        std::string inputFilename{"testfiles/big_ascending.txt"};

        // Open the input and output files
        std::ifstream inputStrm{inputFilename};
        BOOST_TEST_REQUIRE(inputStrm.is_open());

        std::ofstream outputStrm{ml::core::COsFileFuncs::NULL_FILENAME};
        BOOST_TEST_REQUIRE(outputStrm.is_open());

        ml::model::CLimits limits;
        ml::api::CAnomalyJobConfig jobConfig;
        BOOST_TEST_REQUIRE(jobConfig.initFromFile(configFileName));

        ml::model::CAnomalyDetectorModelConfig modelConfig{
            ml::model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE)};

        std::ostringstream* dataStream{nullptr};
        ml::api::CSingleStreamDataAdder::TOStreamP dataStreamPtr{
            dataStream = new std::ostringstream()};

        // Persist the processors' state
        ml::api::CSingleStreamDataAdder dataAdder{dataStreamPtr};

        // The 30000 second persist interval is set large enough that the timer
        // will not trigger during the test - we bypass the timer in this test
        // and kick off the background persistence chain explicitly
        ml::api::CPersistenceManager persistenceManager{30000, false, dataAdder};

        ml::core_t::TTime snapshotTimestamp_;
        std::string description_;
        std::string snapshotId_;
        std::size_t numDocs_{0};

        std::string backgroundSnapshotId;
        std::string foregroundSnapshotId;

        {
            ml::core::CJsonOutputStreamWrapper wrappedOutputStream{outputStrm};

            CTestAnomalyJob job{
                JOB_ID,
                limits,
                jobConfig,
                modelConfig,
                wrappedOutputStream,
                std::bind(&reportPersistComplete, std::placeholders::_1,
                          std::ref(snapshotTimestamp_), std::ref(description_),
                          std::ref(snapshotId_), std::ref(numDocs_)),
                &persistenceManager,
                -1,
                "time",
                "%d/%b/%Y:%T %z"};

            ml::api::CDataProcessor* firstProcessor{&job};

            ml::api::CNdJsonInputParser parser{
                {CTestFieldDataCategorizer::MLCATEGORY_NAME}, inputStrm};

            BOOST_TEST_REQUIRE(parser.readStreamIntoMaps(
                [firstProcessor](const ml::api::CDataProcessor::TStrStrUMap& dataRowFields) {
                    return firstProcessor->handleRecord(
                        dataRowFields, ml::api::CDataProcessor::TOptionalTime{});
                }));

            // Ensure the model size stats are up to date
            job.finalise();

            ml::core_t::TTime snapshotTimestamp{1283524206};
            const std::string snapshotId{"my_special_snapshot"};
            const std::string description{"Supplied description for snapshot at " +
                                          ml::core::CTimeUtils::toIso8601(snapshotTimestamp)};
            BOOST_TEST_REQUIRE(job.doPersistStateInForeground(
                dataAdder, description, snapshotId, snapshotTimestamp));

            // Check that the snapshot description and Id reported by the "persist complete"
            // handler match those supplied to the persist function
            BOOST_REQUIRE_EQUAL(snapshotTimestamp, snapshotTimestamp_);
            BOOST_REQUIRE_EQUAL(description, description_);
            BOOST_REQUIRE_EQUAL(snapshotId, snapshotId_);

            std::string state{dataStream->str()};

            // Compare snapshot ID embedded in the state string with the supplied value.
            const std::string expectedId{"job_model_state_" + snapshotId + "#1"};
            BOOST_TEST_REQUIRE(state.find(expectedId) != std::string::npos);
        }
    }
};

BOOST_FIXTURE_TEST_CASE(testDetectorPersistByWithGivenSnapshotDescriptors, CTestFixture) {
    this->foregroundPersistWithGivenSnapshotDescriptors("testfiles/new_mlfields.json");
}

BOOST_FIXTURE_TEST_CASE(testDetectorPersistBy, CTestFixture) {
    this->foregroundBackgroundCompCategorizationAndAnomalyDetection("testfiles/new_mlfields.json");
}

BOOST_FIXTURE_TEST_CASE(testDetectorPersistOver, CTestFixture) {
    this->foregroundBackgroundCompCategorizationAndAnomalyDetection("testfiles/new_mlfields_over.json");
}

BOOST_FIXTURE_TEST_CASE(testDetectorPersistPartition, CTestFixture) {
    this->foregroundBackgroundCompCategorizationAndAnomalyDetection(
        "testfiles/new_mlfields_partition.json");
}

BOOST_FIXTURE_TEST_CASE(testDetectorBackgroundPersistStaticsConsistency, CTestFixture) {
    this->foregroundBackgroundCompAnomalyDetectionAfterStaticsUpdate("testfiles/new_mlfields_over.json");
}

BOOST_FIXTURE_TEST_CASE(testBackgroundPersistCategorizationConsistency, CTestFixture) {

    static const std::string JOB_ID{"job"};
    static const std::string FIRST_INPUT{
        "{ \"message\": \"the quick brown fox jumped over the lazy dog\" }\n"};
    static const std::string SECOND_INPUT{"{ \"message\": \"she sells sea shells on the seashore\" }\n"};

    std::ofstream outputStrm{ml::core::COsFileFuncs::NULL_FILENAME};
    BOOST_TEST_REQUIRE(outputStrm.is_open());

    ml::model::CLimits limits;
    ml::api::CAnomalyJobConfig::CAnalysisConfig fieldConfig{"message"};

    std::ostringstream* backgroundStream{nullptr};
    ml::api::CSingleStreamDataAdder::TOStreamP backgroundStreamPtr(
        backgroundStream = new std::ostringstream());
    ml::api::CSingleStreamDataAdder backgroundDataAdder(backgroundStreamPtr);

    // The 30000 second persist interval is set large enough that the timer
    // will not trigger during the test - we bypass the timer in this test
    // and kick off the background persistence chain explicitly
    ml::api::CPersistenceManager persistenceManager{30000, false, backgroundDataAdder};

    std::string backgroundState1;
    std::string backgroundState2;

    {
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream{outputStrm};

        CTestFieldDataCategorizer categorizer{
            JOB_ID,  fieldConfig,         limits,
            nullptr, wrappedOutputStream, &persistenceManager};

        std::istringstream inputStrm1{FIRST_INPUT};
        ml::api::CNdJsonInputParser parser1{
            {CTestFieldDataCategorizer::MLCATEGORY_NAME}, inputStrm1};

        BOOST_TEST_REQUIRE(parser1.readStreamIntoMaps(
            [&categorizer](const CTestFieldDataCategorizer::TStrStrUMap& dataRowFields) {
                return categorizer.handleRecord(dataRowFields);
            }));

        // Now persist the categorizer's state in the background
        BOOST_TEST_REQUIRE(categorizer.periodicPersistStateInBackground());
        BOOST_TEST_REQUIRE(persistenceManager.startPersistInBackground());
        BOOST_TEST_REQUIRE(persistenceManager.waitForIdle());

        backgroundState1 = backgroundStream->str();

        std::istringstream inputStrm2{SECOND_INPUT};
        ml::api::CNdJsonInputParser parser2{
            {CTestFieldDataCategorizer::MLCATEGORY_NAME}, inputStrm2};

        BOOST_TEST_REQUIRE(categorizer.periodicPersistStateInBackground());
        BOOST_TEST_REQUIRE(persistenceManager.startPersistInBackground());

        // This time handle another record during background persistence.
        // Due to thread scheduling, sometimes this will happen before the
        // persistence is complete.  This shouldn't be a problem if the
        // background persistence copied the state, but if it didn't then
        // it should fail the test.
        BOOST_TEST_REQUIRE(parser2.readStreamIntoMaps(
            [&categorizer](const CTestFieldDataCategorizer::TStrStrUMap& dataRowFields) {
                return categorizer.handleRecord(dataRowFields);
            }));

        BOOST_TEST_REQUIRE(persistenceManager.waitForIdle());

        backgroundState2 = backgroundStream->str();
        backgroundState2.erase(0, backgroundState1.length());
    }

    // Replace the zero byte separators to avoid '\0's in the output if the
    // test fails
    std::replace(backgroundState1.begin(), backgroundState1.end(), '\0', ',');
    std::replace(backgroundState2.begin(), backgroundState2.end(), '\0', ',');

    BOOST_REQUIRE_EQUAL(backgroundState1, backgroundState2);
}

BOOST_FIXTURE_TEST_CASE(testCategorizationOnlyPersist, CTestFixture) {
    // Start by creating a categorizer with non-trivial state

    static const std::string JOB_ID{"job"};

    std::string inputFilename{"testfiles/big_ascending.txt"};

    // Open the input and output files
    std::ifstream inputStrm{inputFilename};
    BOOST_TEST_REQUIRE(inputStrm.is_open());

    std::ofstream outputStrm{ml::core::COsFileFuncs::NULL_FILENAME};
    BOOST_TEST_REQUIRE(outputStrm.is_open());

    ml::model::CLimits limits;
    ml::api::CAnomalyJobConfig::CAnalysisConfig fieldConfig{"agent"};

    std::ostringstream* backgroundStream{nullptr};
    ml::api::CSingleStreamDataAdder::TOStreamP backgroundStreamPtr{
        backgroundStream = new std::ostringstream()};
    ml::api::CSingleStreamDataAdder backgroundDataAdder{backgroundStreamPtr};

    std::ostringstream* foregroundStream{nullptr};
    ml::api::CSingleStreamDataAdder::TOStreamP foregroundStreamPtr{
        foregroundStream = new std::ostringstream()};
    ml::api::CSingleStreamDataAdder foregroundDataAdder{foregroundStreamPtr};

    // The 30000 second persist interval is set large enough that the timer will
    // not trigger during the test - we bypass the timer in this test and kick
    // off the background persistence chain explicitly
    ml::api::CPersistenceManager persistenceManager{30000, false, backgroundDataAdder,
                                                    foregroundDataAdder};

    {
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream{outputStrm};

        // The categorizer knows how to assign categories to records
        CTestFieldDataCategorizer categorizer{
            JOB_ID,  fieldConfig,         limits,
            nullptr, wrappedOutputStream, &persistenceManager};

        ml::api::CNdJsonInputParser parser{
            {CTestFieldDataCategorizer::MLCATEGORY_NAME}, inputStrm};

        BOOST_TEST_REQUIRE(parser.readStreamIntoMaps(
            [&categorizer](const CTestFieldDataCategorizer::TStrStrUMap& dataRowFields) {
                return categorizer.handleRecord(dataRowFields);
            }));

        // Persist the processors' state in the background
        BOOST_TEST_REQUIRE(categorizer.periodicPersistStateInBackground());
        BOOST_TEST_REQUIRE(persistenceManager.startPersistInBackground());

        LOG_DEBUG(<< "Before waiting for the background persister to be idle");
        BOOST_TEST_REQUIRE(persistenceManager.waitForIdle());
        LOG_DEBUG(<< "After waiting for the background persister to be idle");

        // Now persist the processors' state in the foreground
        BOOST_TEST_REQUIRE(categorizer.periodicPersistStateInForeground());
        persistenceManager.startPersist();
    }

    std::string backgroundState{backgroundStream->str()};
    std::string foregroundState{foregroundStream->str()};

    // Replace the zero byte separators to avoid '\0's in the output if the test
    // fails
    std::replace(backgroundState.begin(), backgroundState.end(), '\0', ',');
    std::replace(foregroundState.begin(), foregroundState.end(), '\0', ',');

    BOOST_REQUIRE_EQUAL(backgroundState, foregroundState);
}

BOOST_AUTO_TEST_SUITE_END()
