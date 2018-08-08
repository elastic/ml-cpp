/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CEventRateAnomalyDetectorTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CStringUtils.h>

#include <maths/CIntegerTools.h>
#include <maths/CModelWeight.h>

#include <model/CAnomalyDetector.h>
#include <model/CHierarchicalResults.h>
#include <model/CHierarchicalResultsAggregator.h>
#include <model/CHierarchicalResultsProbabilityFinalizer.h>
#include <model/CLimits.h>
#include <model/CSearchKey.h>
#include <model/FunctionTypes.h>

#include <boost/bind.hpp>
#include <boost/ref.hpp>

#include <fstream>

namespace {

using TTimeVec = std::vector<ml::core_t::TTime>;
using TStrVec = std::vector<std::string>;
using TTimeDoubleMap = std::map<ml::core_t::TTime, double>;
using TTimeDoubleMapCItr = TTimeDoubleMap::const_iterator;
using TTimeStrPr = std::pair<ml::core_t::TTime, std::string>;
using TTimeStrPrSet = std::set<TTimeStrPr>;

const std::string EMPTY_STRING;

class CResultWriter : public ml::model::CHierarchicalResultsVisitor {
public:
    CResultWriter(const ml::model::CAnomalyDetectorModelConfig& modelConfig,
                  const ml::model::CLimits& limits)
        : m_ModelConfig(modelConfig), m_Limits(limits), m_Calls(0) {}

    void operator()(ml::model::CAnomalyDetector& detector,
                    ml::core_t::TTime start,
                    ml::core_t::TTime end) {
        ml::model::CHierarchicalResults results;
        detector.buildResults(start, end, results);
        results.buildHierarchy();
        ml::model::CHierarchicalResultsAggregator aggregator(m_ModelConfig);
        results.bottomUpBreadthFirst(aggregator);
        ml::model::CHierarchicalResultsProbabilityFinalizer finalizer;
        results.bottomUpBreadthFirst(finalizer);
        results.bottomUpBreadthFirst(*this);
    }

    virtual void visit(const ml::model::CHierarchicalResults& results,
                       const ml::model::CHierarchicalResults::TNode& node,
                       bool pivot) {
        if (pivot) {
            return;
        }

        if (!this->shouldWriteResult(m_Limits, results, node, pivot)) {
            return;
        }

        if (this->isSimpleCount(node)) {
            return;
        }
        if (!this->isLeaf(node)) {
            return;
        }

        const std::string analysisFieldValue = *node.s_Spec.s_PersonFieldValue;
        ml::core_t::TTime bucketTime = node.s_BucketStartTime;
        double anomalyFactor = node.s_RawAnomalyScore;
        LOG_DEBUG(<< analysisFieldValue << " bucket time " << bucketTime
                  << " anomalyFactor " << anomalyFactor);
        ++m_Calls;
        m_AllAnomalies.insert(TTimeStrPr(bucketTime, analysisFieldValue));
        m_AnomalyScores[bucketTime] += anomalyFactor;
    }

    bool operator()(ml::core_t::TTime time,
                    const ml::model::CHierarchicalResults::TNode& node,
                    bool isBucketInfluencer) {
        LOG_DEBUG(<< (isBucketInfluencer ? "BucketInfluencer" : "Influencer ")
                  << node.s_Spec.print() << " initial score "
                  << node.probability() << ", time:  " << time);

        return true;
    }

    size_t calls() const { return m_Calls; }

    size_t numDistinctTimes() const { return m_AllAnomalies.size(); }

    const TTimeDoubleMap& anomalyScores() const { return m_AnomalyScores; }

    const TTimeStrPrSet& allAnomalies() const { return m_AllAnomalies; }

private:
    const ml::model::CAnomalyDetectorModelConfig& m_ModelConfig;
    ml::model::CLimits m_Limits;
    std::size_t m_Calls;
    TTimeStrPrSet m_AllAnomalies;
    TTimeDoubleMap m_AnomalyScores;
};

void importData(ml::core_t::TTime firstTime,
                ml::core_t::TTime lastTime,
                ml::core_t::TTime bucketLength,
                CResultWriter& outputResults,
                const TStrVec& fileNames,
                ml::model::CAnomalyDetector& detector) {
    using TifstreamPtr = std::shared_ptr<std::ifstream>;
    using TifstreamPtrVec = std::vector<TifstreamPtr>;

    TifstreamPtrVec ifss;
    for (std::size_t i = 0u; i < fileNames.size(); ++i) {
        TifstreamPtr ifs(new std::ifstream(fileNames[i].c_str()));
        CPPUNIT_ASSERT(ifs->is_open());
        ifss.push_back(ifs);
    }

    ml::core_t::TTime lastBucketTime = ml::maths::CIntegerTools::ceil(firstTime, bucketLength);

    TTimeVec times(ifss.size());
    for (std::size_t i = 0u; i < ifss.size(); ++i) {
        std::string line;
        std::getline(*ifss[i], line);
        CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType(line, times[i]));
    }

    ml::core_t::TTime time(0);
    for (;;) {
        std::size_t file(std::min_element(times.begin(), times.end()) - times.begin());

        std::string attributeFieldValue = fileNames[file];
        time = times[file];

        if (time == std::numeric_limits<ml::core_t::TTime>::max()) {
            break;
        }

        for (/**/; lastBucketTime + bucketLength <= time; lastBucketTime += bucketLength) {
            outputResults(detector, lastBucketTime, lastBucketTime + bucketLength);
        }

        ml::model::CAnomalyDetector::TStrCPtrVec fieldValues;
        fieldValues.push_back(&attributeFieldValue);
        detector.addRecord(time, fieldValues);

        std::string line;
        if (!std::getline(*ifss[file], line)) {
            times[file] = std::numeric_limits<ml::core_t::TTime>::max();
            ifss[file].reset();
        } else {
            CPPUNIT_ASSERT(ml::core::CStringUtils::stringToType(line, times[file]));
        }
    }

    for (/**/; lastBucketTime + bucketLength <= lastTime; lastBucketTime += bucketLength) {
        outputResults(detector, lastBucketTime, lastBucketTime + bucketLength);
    }
}
}

void CEventRateAnomalyDetectorTest::testAnomalies() {
    // We have 10 instances of correlated 503s and rare SQL statements
    // and one extended drop in status 200s, which are the principal
    // anomalies to find in this data set.
    static const double HIGH_ANOMALY_SCORE(0.0019);
    static const size_t EXPECTED_ANOMALOUS_HOURS(11);

    static const ml::core_t::TTime FIRST_TIME(1346713620);
    static const ml::core_t::TTime LAST_TIME(1347317974);
    static const ml::core_t::TTime BUCKET_SIZE(1800);

    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);
    ml::model::CLimits limits;

    ml::model::CSearchKey key(1, // identifier
                              ml::model::function_t::E_IndividualRareCount, false,
                              ml::model_t::E_XF_None, EMPTY_STRING, "status");
    ml::model::CAnomalyDetector detector(1, // identifier
                                         limits, modelConfig, EMPTY_STRING,
                                         FIRST_TIME, modelConfig.factory(key));
    CResultWriter writer(modelConfig, limits);
    TStrVec files;
    files.push_back("testfiles/status200.txt");
    files.push_back("testfiles/status503.txt");
    files.push_back("testfiles/mysqlabort.txt");

    importData(FIRST_TIME, LAST_TIME, BUCKET_SIZE, writer, files, detector);

    LOG_DEBUG(<< "visitor.calls() = " << writer.calls());

    const TTimeDoubleMap& anomalyScores = writer.anomalyScores();
    TTimeVec peaks;
    for (const auto& score : anomalyScores) {
        if (score.second > HIGH_ANOMALY_SCORE) {
            peaks.push_back(score.first);
        }
    }

    CPPUNIT_ASSERT_EQUAL(EXPECTED_ANOMALOUS_HOURS, peaks.size());

    std::size_t detected503 = 0u;
    std::size_t detectedMySQL = 0u;
    for (std::size_t i = 0u; i < peaks.size(); ++i) {
        LOG_DEBUG(<< "Checking for status 503 anomaly at " << peaks[i]);
        if (writer.allAnomalies().count(TTimeStrPr(peaks[i], "testfiles/status503.txt"))) {
            ++detected503;
        }
        LOG_DEBUG(<< "Checking for MySQL anomaly at " << peaks[i]);
        if (writer.allAnomalies().count(TTimeStrPr(peaks[i], "testfiles/mysqlabort.txt"))) {
            ++detectedMySQL;
        }
    }
    LOG_DEBUG(<< "# 503 = " << detected503 << ", # My SQL = " << detectedMySQL);
    CPPUNIT_ASSERT_EQUAL(std::size_t(10), detected503);
    CPPUNIT_ASSERT_EQUAL(std::size_t(10), detectedMySQL);
}

void CEventRateAnomalyDetectorTest::testPersist() {
    static const ml::core_t::TTime FIRST_TIME(1346713620);
    static const ml::core_t::TTime LAST_TIME(1347317974);
    static const ml::core_t::TTime BUCKET_SIZE(3600);

    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);
    ml::model::CLimits limits;

    ml::model::CSearchKey key(1, // identifier
                              ml::model::function_t::E_IndividualCount, false,
                              ml::model_t::E_XF_None, EMPTY_STRING, "status");

    ml::model::CAnomalyDetector origDetector(1, // identifier
                                             limits, modelConfig, EMPTY_STRING,
                                             FIRST_TIME, modelConfig.factory(key));
    CResultWriter writer(modelConfig, limits);
    TStrVec files;
    files.push_back("testfiles/status503.txt");

    importData(FIRST_TIME, LAST_TIME, BUCKET_SIZE, writer, files, origDetector);

    std::string origXml;
    {
        ml::core::CRapidXmlStatePersistInserter inserter("root");
        origDetector.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_TRACE(<< "Event rate detector XML representation:\n" << origXml);

    // Restore the XML into a new detector
    ml::model::CAnomalyDetector restoredDetector(1, // identifier
                                                 limits, modelConfig, "", 0,
                                                 modelConfig.factory(key));
    {
        ml::core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
        ml::core::CRapidXmlStateRestoreTraverser traverser(parser);
        CPPUNIT_ASSERT(traverser.traverseSubLevel(
            boost::bind(&ml::model::CAnomalyDetector::acceptRestoreTraverser,
                        &restoredDetector, EMPTY_STRING, _1)));
    }

    // The XML representation of the new typer should be the same as the original
    std::string newXml;
    {
        ml::core::CRapidXmlStatePersistInserter inserter("root");
        restoredDetector.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    CPPUNIT_ASSERT_EQUAL(origXml, newXml);
}

CppUnit::Test* CEventRateAnomalyDetectorTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CEventRateAnomalyDetectorTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateAnomalyDetectorTest>(
        "CEventRateAnomalyDetectorTest::testAnomalies",
        &CEventRateAnomalyDetectorTest::testAnomalies));
    suiteOfTests->addTest(new CppUnit::TestCaller<CEventRateAnomalyDetectorTest>(
        "CEventRateAnomalyDetectorTest::testPersist",
        &CEventRateAnomalyDetectorTest::testPersist));

    return suiteOfTests;
}
