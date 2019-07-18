/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CPersistenceManagerTest.h"

#include <core/CJsonOutputStreamWrapper.h>
#include <core/COsFileFuncs.h>
#include <core/CProgramCounters.h>
#include <core/CStringUtils.h>
#include <core/CoreTypes.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CLimits.h>

#include <api/CAnomalyJob.h>
#include <api/CDataProcessor.h>
#include <api/CFieldConfig.h>
#include <api/CFieldDataTyper.h>
#include <api/CJsonOutputWriter.h>
#include <api/CModelSnapshotJsonWriter.h>
#include <api/CNdJsonInputParser.h>
#include <api/CNullOutput.h>
#include <api/COutputChainer.h>
#include <api/CPersistenceManager.h>
#include <api/CSingleStreamDataAdder.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

namespace {

void reportPersistComplete(ml::api::CModelSnapshotJsonWriter::SModelSnapshotReport modelSnapshotReport,
                           std::string& snapshotIdOut,
                           size_t& numDocsOut) {
    LOG_DEBUG(<< "Persist complete with description: " << modelSnapshotReport.s_Description);
    snapshotIdOut = modelSnapshotReport.s_SnapshotId;
    numDocsOut = modelSnapshotReport.s_NumDocs;
}
}

CppUnit::Test* CPersistenceManagerTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CPersistenceManagerTest");
    suiteOfTests->addTest(new CppUnit::TestCaller<CPersistenceManagerTest>(
        "CPersistenceManagerTest::testDetectorPersistBy",
        &CPersistenceManagerTest::testDetectorPersistBy));
    suiteOfTests->addTest(new CppUnit::TestCaller<CPersistenceManagerTest>(
        "CPersistenceManagerTest::testDetectorPersistOver",
        &CPersistenceManagerTest::testDetectorPersistOver));
    suiteOfTests->addTest(new CppUnit::TestCaller<CPersistenceManagerTest>(
        "CPersistenceManagerTest::testDetectorPersistPartition",
        &CPersistenceManagerTest::testDetectorPersistPartition));
    suiteOfTests->addTest(new CppUnit::TestCaller<CPersistenceManagerTest>(
        "CPersistenceManagerTest::testCategorizationOnlyPersist",
        &CPersistenceManagerTest::testCategorizationOnlyPersist));
    suiteOfTests->addTest(new CppUnit::TestCaller<CPersistenceManagerTest>(
        "CPersistenceManagerTest::testDetectorBackgroundPersistStaticsConsistency",
        &CPersistenceManagerTest::testDetectorBackgroundPersistStaticsConsistency));

    return suiteOfTests;
}

void CPersistenceManagerTest::testDetectorPersistBy() {
    this->foregroundBackgroundCompCategorizationAndAnomalyDetection("testfiles/new_mlfields.conf");
}

void CPersistenceManagerTest::testDetectorPersistOver() {
    this->foregroundBackgroundCompCategorizationAndAnomalyDetection("testfiles/new_mlfields_over.conf");
}

void CPersistenceManagerTest::testDetectorPersistPartition() {
    this->foregroundBackgroundCompCategorizationAndAnomalyDetection(
        "testfiles/new_mlfields_partition.conf");
}

void CPersistenceManagerTest::testDetectorBackgroundPersistStaticsConsistency() {
    this->foregroundBackgroundCompAnomalyDetectionAfterStaticsUpdate("testfiles/new_mlfields_over.conf");
}

void CPersistenceManagerTest::testCategorizationOnlyPersist() {
    // Start by creating a categorizer with non-trivial state

    static const std::string JOB_ID("job");

    std::string inputFilename("testfiles/big_ascending.txt");

    // Open the input and output files
    std::ifstream inputStrm(inputFilename);
    CPPUNIT_ASSERT(inputStrm.is_open());

    std::ofstream outputStrm(ml::core::COsFileFuncs::NULL_FILENAME);
    CPPUNIT_ASSERT(outputStrm.is_open());

    ml::model::CLimits limits;
    ml::api::CFieldConfig fieldConfig("agent");

    std::ostringstream* backgroundStream(nullptr);
    ml::api::CSingleStreamDataAdder::TOStreamP backgroundStreamPtr(
        backgroundStream = new std::ostringstream());
    ml::api::CSingleStreamDataAdder backgroundDataAdder(backgroundStreamPtr);

    std::ostringstream* foregroundStream(nullptr);
    ml::api::CSingleStreamDataAdder::TOStreamP foregroundStreamPtr(
        foregroundStream = new std::ostringstream());
    ml::api::CSingleStreamDataAdder foregroundDataAdder(foregroundStreamPtr);

    // The 300 second persist interval is irrelevant here - we bypass the timer
    // in this test and kick off the background persistence chain explicitly
    ml::api::CPersistenceManager persistenceManager(300, false, backgroundDataAdder,
                                                    foregroundDataAdder);

    {
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
        ml::api::CJsonOutputWriter outputWriter(JOB_ID, wrappedOutputStream);

        // All output we're interested in goes via the JSON output writer, so
        // output of the categorised input data can be dropped
        ml::api::CNullOutput nullOutput;

        // The typer knows how to assign categories to records
        ml::api::CFieldDataTyper typer(JOB_ID, fieldConfig, limits, nullOutput,
                                       outputWriter, &persistenceManager);

        ml::api::CNdJsonInputParser parser(inputStrm);

        CPPUNIT_ASSERT(parser.readStreamIntoMaps(std::bind(
            &ml::api::CDataProcessor::handleRecord, &typer, std::placeholders::_1)));

        // Persist the processors' state in the background
        CPPUNIT_ASSERT(typer.periodicPersistStateInBackground());
        CPPUNIT_ASSERT(persistenceManager.startPersistInBackground());

        LOG_DEBUG(<< "Before waiting for the background persister to be idle");
        CPPUNIT_ASSERT(persistenceManager.waitForIdle());
        LOG_DEBUG(<< "After waiting for the background persister to be idle");

        // Now persist the processors' state in the foreground
        CPPUNIT_ASSERT(typer.periodicPersistStateInForeground());
        persistenceManager.startPersist();
    }

    std::string backgroundState = backgroundStream->str();
    std::string foregroundState = foregroundStream->str();

    // Replace the zero byte separators so the expected/actual strings don't get
    // truncated by CppUnit if the test fails
    std::replace(backgroundState.begin(), backgroundState.end(), '\0', ',');
    std::replace(foregroundState.begin(), foregroundState.end(), '\0', ',');

    CPPUNIT_ASSERT_EQUAL(backgroundState, foregroundState);
}

void CPersistenceManagerTest::foregroundBackgroundCompCategorizationAndAnomalyDetection(
    const std::string& configFileName) {
    // Start by creating processors with non-trivial state

    static const ml::core_t::TTime BUCKET_SIZE(3600);
    static const std::string JOB_ID("job");

    std::string inputFilename("testfiles/big_ascending.txt");

    // Open the input and output files
    std::ifstream inputStrm(inputFilename.c_str());
    CPPUNIT_ASSERT(inputStrm.is_open());

    std::ofstream outputStrm(ml::core::COsFileFuncs::NULL_FILENAME);
    CPPUNIT_ASSERT(outputStrm.is_open());

    ml::model::CLimits limits;
    ml::api::CFieldConfig fieldConfig;
    CPPUNIT_ASSERT(fieldConfig.initFromFile(configFileName));

    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);

    std::ostringstream* backgroundStream(nullptr);
    ml::api::CSingleStreamDataAdder::TOStreamP backgroundStreamPtr(
        backgroundStream = new std::ostringstream());
    ml::api::CSingleStreamDataAdder backgroundDataAdder(backgroundStreamPtr);
    std::ostringstream* foregroundStream(nullptr);
    ml::api::CSingleStreamDataAdder::TOStreamP foregroundStreamPtr(
        foregroundStream = new std::ostringstream());

    ml::api::CSingleStreamDataAdder foregroundDataAdder(foregroundStreamPtr);

    // The 300 second persist interval is irrelevant here - we bypass the timer
    // in this test and kick off the background persistence chain explicitly
    ml::api::CPersistenceManager persistenceManager(300, false, backgroundDataAdder,
                                                    foregroundDataAdder);

    std::string snapshotId;
    std::size_t numDocs(0);

    std::string backgroundSnapshotId;
    std::string foregroundSnapshotId;
    std::string foregroundSnapshotId2;

    std::ostringstream* foregroundStream2(nullptr);
    ml::api::CSingleStreamDataAdder::TOStreamP foregroundStreamPtr2(
        foregroundStream2 = new std::ostringstream());
    {
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
        ml::api::CJsonOutputWriter outputWriter(JOB_ID, wrappedOutputStream);

        ml::api::CAnomalyJob job(
            JOB_ID, limits, fieldConfig, modelConfig, wrappedOutputStream,
            std::bind(&reportPersistComplete, std::placeholders::_1,
                      std::ref(snapshotId), std::ref(numDocs)),
            &persistenceManager, -1, "time", "%d/%b/%Y:%T %z");

        ml::api::CDataProcessor* firstProcessor(&job);

        // Chain the detector's input
        ml::api::COutputChainer outputChainer(job);

        // The typer knows how to assign categories to records
        ml::api::CFieldDataTyper typer(JOB_ID, fieldConfig, limits, outputChainer,
                                       outputWriter, &persistenceManager);

        if (fieldConfig.fieldNameSuperset().count(ml::api::CFieldDataTyper::MLCATEGORY_NAME) > 0) {
            LOG_DEBUG(<< "Applying the categorization typer for anomaly detection");
            firstProcessor = &typer;
        }

        ml::api::CNdJsonInputParser parser(inputStrm);

        CPPUNIT_ASSERT(parser.readStreamIntoMaps(std::bind(
            &ml::api::CDataProcessor::handleRecord, firstProcessor, std::placeholders::_1)));

        // Persist the processors' state in the background
        CPPUNIT_ASSERT(firstProcessor->periodicPersistStateInBackground());
        CPPUNIT_ASSERT(persistenceManager.startPersistInBackground());

        LOG_DEBUG(<< "Before waiting for the background persister to be idle");
        CPPUNIT_ASSERT(persistenceManager.waitForIdle());
        LOG_DEBUG(<< "After waiting for the background persister to be idle");
        backgroundSnapshotId = snapshotId;

        // Now persist the processors' state in the foreground

        CPPUNIT_ASSERT(firstProcessor->periodicPersistStateInForeground());
        persistenceManager.startPersist();
        foregroundSnapshotId = snapshotId;

        // ... persist in foreground againi by directly calling persistState
        ml::api::CSingleStreamDataAdder foregroundDataAdder2(foregroundStreamPtr2);
        CPPUNIT_ASSERT(firstProcessor->persistState(
            foregroundDataAdder2, "Periodic foreground persistence at "));
        foregroundSnapshotId2 = snapshotId;
    }

    std::string backgroundState = backgroundStream->str();
    std::string foregroundState = foregroundStream->str();
    std::string foregroundState2 = foregroundStream2->str();

    // The snapshot ID can be different between the two persists, so replace the
    // first occurrence of it (which is in the bulk metadata)
    CPPUNIT_ASSERT_EQUAL(size_t(1), ml::core::CStringUtils::replaceFirst(
                                        backgroundSnapshotId, "snap", backgroundState));
    CPPUNIT_ASSERT_EQUAL(size_t(1), ml::core::CStringUtils::replaceFirst(
                                        foregroundSnapshotId, "snap", foregroundState));
    CPPUNIT_ASSERT_EQUAL(size_t(1), ml::core::CStringUtils::replaceFirst(
                                        foregroundSnapshotId2, "snap", foregroundState2));

    // Replace the zero byte separators so the expected/actual strings don't get
    // truncated by CppUnit if the test fails
    std::replace(backgroundState.begin(), backgroundState.end(), '\0', ',');
    std::replace(foregroundState.begin(), foregroundState.end(), '\0', ',');
    std::replace(foregroundState2.begin(), foregroundState2.end(), '\0', ',');

    CPPUNIT_ASSERT_EQUAL(foregroundState, foregroundState2);
    CPPUNIT_ASSERT_EQUAL(backgroundState, foregroundState2);
    CPPUNIT_ASSERT_EQUAL(backgroundState, foregroundState);
}

void CPersistenceManagerTest::foregroundBackgroundCompAnomalyDetectionAfterStaticsUpdate(
    const std::string& configFileName) {
    // Start by creating processors with non-trivial state

    static const ml::core_t::TTime BUCKET_SIZE(3600);
    static const std::string JOB_ID("job");

    std::string inputFilename("testfiles/big_ascending.txt");

    // Open the input and output files
    std::ifstream inputStrm(inputFilename.c_str());
    CPPUNIT_ASSERT(inputStrm.is_open());

    std::ofstream outputStrm(ml::core::COsFileFuncs::NULL_FILENAME);
    CPPUNIT_ASSERT(outputStrm.is_open());

    ml::model::CLimits limits;
    ml::api::CFieldConfig fieldConfig;
    CPPUNIT_ASSERT(fieldConfig.initFromFile(configFileName));

    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);

    std::ostringstream* backgroundStream(nullptr);
    ml::api::CSingleStreamDataAdder::TOStreamP backgroundStreamPtr(
        backgroundStream = new std::ostringstream());
    ml::api::CSingleStreamDataAdder backgroundDataAdder(backgroundStreamPtr);

    std::ostringstream* foregroundStream(nullptr);
    ml::api::CSingleStreamDataAdder::TOStreamP foregroundStreamPtr(
        foregroundStream = new std::ostringstream());

    // Persist the processors' state in the foreground
    ml::api::CSingleStreamDataAdder foregroundDataAdder(foregroundStreamPtr);

    // The 300 second persist interval is irrelevant here - we bypass the timer
    // in this test and kick off the background persistence chain explicitly
    ml::api::CPersistenceManager persistenceManager(300, false, backgroundDataAdder,
                                                    foregroundDataAdder);

    std::string snapshotId;
    std::size_t numDocs(0);

    std::string backgroundSnapshotId;
    std::string foregroundSnapshotId;

    {
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
        ml::api::CJsonOutputWriter outputWriter(JOB_ID, wrappedOutputStream);

        ml::api::CAnomalyJob job(
            JOB_ID, limits, fieldConfig, modelConfig, wrappedOutputStream,
            std::bind(&reportPersistComplete, std::placeholders::_1,
                      std::ref(snapshotId), std::ref(numDocs)),
            &persistenceManager, -1, "time", "%d/%b/%Y:%T %z");

        ml::api::CDataProcessor* firstProcessor(&job);

        ml::api::CNdJsonInputParser parser(inputStrm);

        CPPUNIT_ASSERT(parser.readStreamIntoMaps(std::bind(
            &ml::api::CDataProcessor::handleRecord, firstProcessor, std::placeholders::_1)));

        // Ensure the model size stats are up to date
        job.finalise();

        CPPUNIT_ASSERT(firstProcessor->periodicPersistStateInForeground());
        persistenceManager.startPersist();

        foregroundSnapshotId = snapshotId;

        // Now persist the processors' state in the background
        CPPUNIT_ASSERT(firstProcessor->periodicPersistStateInBackground());
        CPPUNIT_ASSERT(persistenceManager.startPersistInBackground());

        //Increment one of the counter values
        ++ml::core::CProgramCounters::counter(ml::counter_t::E_TSADMemoryUsage);

        LOG_DEBUG(<< "Before waiting for the background persister to be idle");
        CPPUNIT_ASSERT(persistenceManager.waitForIdle());
        LOG_DEBUG(<< "After waiting for the background persister to be idle");
        backgroundSnapshotId = snapshotId;
    }

    std::string backgroundState = backgroundStream->str();
    std::string foregroundState = foregroundStream->str();

    // The snapshot ID can be different between the two persists, so replace the
    // first occurrence of it (which is in the bulk metadata)
    CPPUNIT_ASSERT_EQUAL(size_t(1), ml::core::CStringUtils::replaceFirst(
                                        backgroundSnapshotId, "snap", backgroundState));
    CPPUNIT_ASSERT_EQUAL(size_t(1), ml::core::CStringUtils::replaceFirst(
                                        foregroundSnapshotId, "snap", foregroundState));

    // Replace the zero byte separators so the expected/actual strings don't get
    // truncated by CppUnit if the test fails
    std::replace(backgroundState.begin(), backgroundState.end(), '\0', ',');
    std::replace(foregroundState.begin(), foregroundState.end(), '\0', ',');

    CPPUNIT_ASSERT_EQUAL(backgroundState, foregroundState);
}
