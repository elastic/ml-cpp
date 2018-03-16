/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#include "CAnomalyJobTest.h"

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CLogger.h>
#include <core/CRegex.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CDataGatherer.h>
#include <model/CLimits.h>

#include <api/CAnomalyJob.h>
#include <api/CCsvInputParser.h>
#include <api/CFieldConfig.h>
#include <api/CHierarchicalResultsWriter.h>
#include <api/CJsonOutputWriter.h>

#include <rapidjson/document.h>

#include <boost/tuple/tuple.hpp>

#include <cstdio>
#include <fstream>
#include <sstream>

namespace {

//! \brief
//! Mock object for state restore unit tests.
//!
//! DESCRIPTION:\n
//! CDataSearcher that returns an empty stream.
//!
class CEmptySearcher : public ml::core::CDataSearcher {
public:
    //! Do a search that results in an empty input stream.
    virtual TIStreamP search(size_t /*currentDocNum*/, size_t /*limit*/) { return TIStreamP(new std::istringstream()); }
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
    CSingleResultVisitor(void) : m_LastResult(0.0) {}

    virtual ~CSingleResultVisitor(void) {}

    virtual void visit(const ml::model::CHierarchicalResults& /*results*/, const TNode& node, bool /*pivot*/) {
        if (!this->isSimpleCount(node) && this->isLeaf(node)) {
            if (node.s_AnnotatedProbability.s_AttributeProbabilities.size() == 0) {
                return;
            }
            if (!node.s_Model) {
                return;
            }
            const ml::model::SAttributeProbability& attribute = node.s_AnnotatedProbability.s_AttributeProbabilities[0];

            m_LastResult = node.s_Model->currentBucketValue(attribute.s_Feature, 0, 0, node.s_BucketStartTime)[0];
        }
    }

    double lastResults(void) const { return m_LastResult; }

private:
    double m_LastResult;
};

class CMultiResultVisitor : public ml::model::CHierarchicalResultsVisitor {
public:
    CMultiResultVisitor(void) : m_LastResult(0.0) {}

    virtual ~CMultiResultVisitor(void) {}

    virtual void visit(const ml::model::CHierarchicalResults& /*results*/, const TNode& node, bool /*pivot*/) {
        if (!this->isSimpleCount(node) && this->isLeaf(node)) {
            if (node.s_AnnotatedProbability.s_AttributeProbabilities.size() == 0) {
                return;
            }
            if (!node.s_Model) {
                return;
            }
            std::size_t pid;
            const ml::model::CDataGatherer& gatherer = node.s_Model->dataGatherer();
            if (!gatherer.personId(*node.s_Spec.s_PersonFieldValue, pid)) {
                LOG_ERROR("No identifier for '" << *node.s_Spec.s_PersonFieldValue << "'");
                return;
            }
            for (std::size_t i = 0; i < node.s_AnnotatedProbability.s_AttributeProbabilities.size(); ++i) {
                const ml::model::SAttributeProbability& attribute =
                    node.s_AnnotatedProbability.s_AttributeProbabilities[i];
                m_LastResult += node.s_Model->currentBucketValue(
                    attribute.s_Feature, pid, attribute.s_Cid, node.s_BucketStartTime)[0];
            }
        }
    }

    double lastResults(void) const { return m_LastResult; }

private:
    double m_LastResult;
};

class CResultsScoreVisitor : public ml::model::CHierarchicalResultsVisitor {
public:
    CResultsScoreVisitor(int score) : m_Score(score) {}

    virtual ~CResultsScoreVisitor(void) {}

    virtual void visit(const ml::model::CHierarchicalResults& /*results*/, const TNode& node, bool /*pivot*/) {
        if (this->isRoot(node)) {
            node.s_NormalizedAnomalyScore = m_Score;
        }
    }

private:
    int m_Score;
};

size_t countBuckets(const std::string& key, const std::string& output) {
    size_t count = 0;
    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseDefaultFlags>(output);
    CPPUNIT_ASSERT(!doc.HasParseError());
    CPPUNIT_ASSERT(doc.IsArray());

    const rapidjson::Value& allRecords = doc.GetArray();
    for (auto& r : allRecords.GetArray()) {
        rapidjson::Value::ConstMemberIterator recordsIt = r.GetObject().FindMember(key);
        if (recordsIt != r.GetObject().MemberEnd()) {
            ++count;
        }
    }

    return count;
}

bool findLine(const std::string& regex, const ml::core::CRegex::TStrVec& lines) {
    ml::core::CRegex rx;
    rx.init(regex);
    std::size_t pos = 0;
    for (ml::core::CRegex::TStrVecCItr i = lines.begin(); i != lines.end(); ++i) {
        if (rx.search(*i, pos)) {
            return true;
        }
    }
    return false;
}

const ml::core_t::TTime BUCKET_SIZE(3600);
}

using namespace ml;

void CAnomalyJobTest::testBadTimes(void) {
    {
        // Test with no time field
        model::CLimits limits;
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clauses;
        clauses.push_back("value");
        clauses.push_back("partitionfield=greenhouse");
        fieldConfig.initFromClause(clauses);
        model::CAnomalyDetectorModelConfig modelConfig = model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);
        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

        api::CAnomalyJob::TStrStrUMap dataRows;
        dataRows["wibble"] = "12345678";
        dataRows["value"] = "1.0";
        dataRows["greenhouse"] = "rhubarb";

        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(uint64_t(0), job.numRecordsHandled());
    }
    {
        // Test with bad time field
        model::CLimits limits;
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clauses;
        clauses.push_back("value");
        clauses.push_back("partitionfield=greenhouse");
        fieldConfig.initFromClause(clauses);
        model::CAnomalyDetectorModelConfig modelConfig = model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);
        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

        api::CAnomalyJob::TStrStrUMap dataRows;
        dataRows["time"] = "hello";
        dataRows["value"] = "1.0";
        dataRows["greenhouse"] = "rhubarb";

        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(uint64_t(0), job.numRecordsHandled());
    }
    {
        // Test with bad time field format
        model::CLimits limits;
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clauses;
        clauses.push_back("value");
        clauses.push_back("partitionfield=greenhouse");
        fieldConfig.initFromClause(clauses);
        model::CAnomalyDetectorModelConfig modelConfig = model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);
        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job",
                             limits,
                             fieldConfig,
                             modelConfig,
                             wrappedOutputStream,
                             api::CAnomalyJob::TPersistCompleteFunc(),
                             nullptr,
                             -1,
                             "time",
                             "%Y%m%m%H%M%S");

        api::CAnomalyJob::TStrStrUMap dataRows;
        dataRows["time"] = "hello world";
        dataRows["value"] = "1.0";
        dataRows["greenhouse"] = "rhubarb";

        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(uint64_t(0), job.numRecordsHandled());
    }
}

void CAnomalyJobTest::testOutOfSequence(void) {
    {
        // Test out of sequence record
        model::CLimits limits;
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clauses;
        clauses.push_back("value");
        clauses.push_back("partitionfield=greenhouse");
        fieldConfig.initFromClause(clauses);
        model::CAnomalyDetectorModelConfig modelConfig = model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);
        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

        job.description();
        job.descriptionAndDebugMemoryUsage();

        // add records which create partitions
        api::CAnomalyJob::TStrStrUMap dataRows;
        dataRows["time"] = "12345678";
        dataRows["value"] = "1.0";
        dataRows["greenhouse"] = "rhubarb";

        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(uint64_t(1), job.numRecordsHandled());

        dataRows["time"] = "1234567";

        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(uint64_t(1), job.numRecordsHandled());
        job.finalise();
    }
}

void CAnomalyJobTest::testControlMessages(void) {
    {
        // Test control messages
        model::CLimits limits;
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clauses;
        clauses.push_back("value");
        clauses.push_back("partitionfield=greenhouse");
        fieldConfig.initFromClause(clauses);
        model::CAnomalyDetectorModelConfig modelConfig = model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);
        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

        api::CAnomalyJob::TStrStrUMap dataRows;
        dataRows["."] = " ";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(uint64_t(0), job.numRecordsHandled());

        dataRows["."] = ".";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(uint64_t(0), job.numRecordsHandled());

        dataRows["."] = "f";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(uint64_t(0), job.numRecordsHandled());

        dataRows["."] = "f1";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(uint64_t(0), job.numRecordsHandled());
    }
    {
        // Test reset bucket
        model::CLimits limits;
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clauses;
        clauses.push_back("count");
        clauses.push_back("partitionfield=greenhouse");
        fieldConfig.initFromClause(clauses);
        model::CAnomalyDetectorModelConfig modelConfig = model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);

        api::CAnomalyJob::TStrStrUMap dataRows;
        dataRows["value"] = "2.0";
        dataRows["greenhouse"] = "rhubarb";

        std::stringstream outputStrm;
        {
            core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
            api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

            core_t::TTime time = 12345678;
            for (std::size_t i = 0; i < 50; i++, time += (BUCKET_SIZE / 2)) {
                std::stringstream ss;
                ss << time;
                dataRows["time"] = ss.str();
                if (i == 40) {
                    for (std::size_t j = 0; j < 100; j++) {
                        CPPUNIT_ASSERT(job.handleRecord(dataRows));
                    }
                }
                CPPUNIT_ASSERT(job.handleRecord(dataRows));
                if (i < 2) {
                    // We haven't processed one full bucket but it should be safe to flush.
                    dataRows["."] = "f1";
                    CPPUNIT_ASSERT(job.handleRecord(dataRows));
                    dataRows.erase(".");
                }
            }
        }

        rapidjson::Document doc;
        doc.Parse<rapidjson::kParseDefaultFlags>(outputStrm.str());
        CPPUNIT_ASSERT(!doc.HasParseError());
        CPPUNIT_ASSERT(doc.IsArray());

        const rapidjson::Value& allRecords = doc.GetArray();
        bool foundRecord = false;
        for (auto& r : allRecords.GetArray()) {
            rapidjson::Value::ConstMemberIterator recordsIt = r.GetObject().FindMember("records");
            if (recordsIt != r.GetObject().MemberEnd()) {
                auto& recordsArray = recordsIt->value.GetArray()[0];
                rapidjson::Value::ConstMemberIterator actualIt = recordsArray.FindMember("actual");
                CPPUNIT_ASSERT(actualIt != recordsArray.MemberEnd());
                const rapidjson::Value::ConstArray& values = actualIt->value.GetArray();

                CPPUNIT_ASSERT_EQUAL(102.0, values[0].GetDouble());
                foundRecord = true;
            }
        }

        CPPUNIT_ASSERT(foundRecord);
        std::stringstream outputStrm2;
        {
            core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm2);
            api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

            core_t::TTime time = 12345678;
            for (std::size_t i = 0; i < 50; i++, time += (BUCKET_SIZE / 2)) {
                std::stringstream ss;
                ss << time;
                dataRows["time"] = ss.str();
                if (i == 40) {
                    for (std::size_t j = 0; j < 100; j++) {
                        CPPUNIT_ASSERT(job.handleRecord(dataRows));
                    }
                }
                CPPUNIT_ASSERT(job.handleRecord(dataRows));
                if (i == 40) {
                    api::CAnomalyJob::TStrStrUMap rows;
                    rows["."] = "r" + ss.str() + " " + ss.str();
                    CPPUNIT_ASSERT(job.handleRecord(rows));
                    for (std::size_t j = 0; j < 100; j++) {
                        CPPUNIT_ASSERT(job.handleRecord(dataRows));
                    }
                }
            }
        }

        rapidjson::Document doc2;
        doc2.Parse<rapidjson::kParseDefaultFlags>(outputStrm2.str());
        CPPUNIT_ASSERT(!doc2.HasParseError());
        CPPUNIT_ASSERT(doc2.IsArray());

        const rapidjson::Value& allRecords2 = doc2.GetArray();
        foundRecord = false;
        for (auto& r : allRecords2.GetArray()) {
            rapidjson::Value::ConstMemberIterator recordsIt = r.GetObject().FindMember("records");
            if (recordsIt != r.GetObject().MemberEnd()) {
                auto& recordsArray = recordsIt->value.GetArray()[0];
                rapidjson::Value::ConstMemberIterator actualIt = recordsArray.FindMember("actual");
                CPPUNIT_ASSERT(actualIt != recordsArray.MemberEnd());
                const rapidjson::Value::ConstArray& values = actualIt->value.GetArray();

                CPPUNIT_ASSERT_EQUAL(101.0, values[0].GetDouble());
                foundRecord = true;
            }
        }

        CPPUNIT_ASSERT(foundRecord);
    }
}

void CAnomalyJobTest::testSkipTimeControlMessage(void) {
    model::CLimits limits;
    api::CFieldConfig fieldConfig;
    api::CFieldConfig::TStrVec clauses;
    clauses.push_back("count");
    fieldConfig.initFromClause(clauses);
    model::CAnomalyDetectorModelConfig modelConfig = model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);

    std::stringstream outputStrm;
    core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

    api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

    api::CAnomalyJob::TStrStrUMap dataRows;

    core_t::TTime time = 3600;
    for (std::size_t i = 0; i < 10; ++i, time += BUCKET_SIZE) {
        std::ostringstream ss;
        ss << time;
        dataRows["time"] = ss.str();
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
    }

    wrappedOutputStream.syncFlush();
    CPPUNIT_ASSERT_EQUAL(std::size_t(9), countBuckets("bucket", outputStrm.str() + "]"));

    // Now let's skip time to Thursday, June 29, 2017 12:00:00 AM
    time = 1498694400;
    dataRows["."] = "s1498694400";
    CPPUNIT_ASSERT(job.handleRecord(dataRows));
    dataRows.erase(".");

    // Check no new bucket results were written
    wrappedOutputStream.syncFlush();
    CPPUNIT_ASSERT_EQUAL(std::size_t(9), countBuckets("bucket", outputStrm.str() + "]"));

    // Let's send a few buckets after skip time
    for (std::size_t i = 0; i < 3; ++i, time += BUCKET_SIZE) {
        std::ostringstream ss;
        ss << time;
        dataRows["time"] = ss.str();
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
    }

    // Assert only 2 new buckets were written
    wrappedOutputStream.syncFlush();
    CPPUNIT_ASSERT_EQUAL(std::size_t(11), countBuckets("bucket", outputStrm.str() + "]"));
}

void CAnomalyJobTest::testOutOfPhase(void) {
    // Ensure the right data ends up in the right buckets
    // First we test that it works as expected for non-out-of-phase,
    // then we crank in the out-of-phase

    // Ensure that gaps in a bucket's data do not cause errors or problems

    // Ensure that we can start at a variety of times,
    // and finish at a variety of times, and get the
    // right output always

    // The code is pretty verbose here, but executes quickly
    {
        LOG_DEBUG("*** testing non-out-of-phase metric ***");
        core_t::TTime bucketSize = 100;
        model::CLimits limits;
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clauses;
        clauses.push_back("mean(value)");
        fieldConfig.initFromClause(clauses);
        model::CAnomalyDetectorModelConfig modelConfig = model::CAnomalyDetectorModelConfig::defaultConfig(bucketSize);
        std::stringstream outputStrm;

        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

        api::CAnomalyJob::TStrStrUMap dataRows;

        CPPUNIT_ASSERT_EQUAL(core_t::TTime(99), job.m_ResultsQueue.latestBucketEnd());
        dataRows["time"] = "10000";
        dataRows["value"] = "1.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(99), job.m_ResultsQueue.latestBucketEnd());

        dataRows["time"] = "10050";
        dataRows["value"] = "3.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(99), job.m_ResultsQueue.latestBucketEnd());

        dataRows["time"] = "10100";
        dataRows["value"] = "1.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(10099), job.m_ResultsQueue.latestBucketEnd());
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10200";
        dataRows["value"] = "0.0005";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10300";
        dataRows["value"] = "5.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0005, visitor.lastResults(), 0.000005);
        }

        dataRows["time"] = "10400";
        dataRows["value"] = "5.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10500";
        dataRows["value"] = "5.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0, visitor.lastResults(), 0.0005);
        }

        // Bucket at 10600 not present

        dataRows["time"] = "10700";
        dataRows["value"] = "50";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10700";
        dataRows["value"] = "80";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));

        dataRows["time"] = "10700";
        dataRows["value"] = "20";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));

        dataRows["time"] = "10800";
        dataRows["value"] = "5.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(50.0, visitor.lastResults(), 0.005);
        }

        dataRows["time"] = "10800";
        dataRows["value"] = "6.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(10799), job.m_ResultsQueue.latestBucketEnd());
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(50.0, visitor.lastResults(), 0.005);
        }
        job.finalise();
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(10799), job.m_ResultsQueue.latestBucketEnd());
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(50.0, visitor.lastResults(), 0.005);
        }
    }
    {
        LOG_DEBUG("*** testing non-out-of-phase metric ***");
        // Same as previous test but starting not on a bucket boundary
        core_t::TTime bucketSize = 100;
        model::CLimits limits;
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clauses;
        clauses.push_back("mean(value)");
        fieldConfig.initFromClause(clauses);
        model::CAnomalyDetectorModelConfig modelConfig = model::CAnomalyDetectorModelConfig::defaultConfig(bucketSize);
        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

        api::CAnomalyJob::TStrStrUMap dataRows;

        // The first two values are in an incomplete bucket and should be ignored
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(99), job.m_ResultsQueue.latestBucketEnd());
        dataRows["time"] = "10001";
        dataRows["value"] = "1.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(99), job.m_ResultsQueue.latestBucketEnd());

        dataRows["time"] = "10051";
        dataRows["value"] = "3.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT(job.m_ResultsQueue.latest().empty());

        // This next bucket should be the first valid one
        dataRows["time"] = "10101";
        dataRows["value"] = "1.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT(job.m_ResultsQueue.latest().empty());

        dataRows["time"] = "10201";
        dataRows["value"] = "0.0005";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(10199), job.m_ResultsQueue.latestBucketEnd());
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10301";
        dataRows["value"] = "5.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0005, visitor.lastResults(), 0.000005);
        }

        dataRows["time"] = "10401";
        dataRows["value"] = "5.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10501";
        dataRows["value"] = "5.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0, visitor.lastResults(), 0.0005);
        }

        // Bucket at 10600 not present

        dataRows["time"] = "10701";
        dataRows["value"] = "50";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10701";
        dataRows["value"] = "80";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));

        dataRows["time"] = "10701";
        dataRows["value"] = "20";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));

        dataRows["time"] = "10801";
        dataRows["value"] = "5.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(50.0, visitor.lastResults(), 0.005);
        }

        dataRows["time"] = "10895";
        dataRows["value"] = "6.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));

        job.finalise();
    }
    {
        LOG_DEBUG("*** testing non-out-of-phase count ***");
        core_t::TTime bucketSize = 100;
        model::CLimits limits;
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clauses;
        clauses.push_back("count");
        fieldConfig.initFromClause(clauses);
        model::CAnomalyDetectorModelConfig modelConfig = model::CAnomalyDetectorModelConfig::defaultConfig(bucketSize);
        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

        api::CAnomalyJob::TStrStrUMap dataRows;

        CPPUNIT_ASSERT_EQUAL(core_t::TTime(99), job.m_ResultsQueue.latestBucketEnd());
        dataRows["time"] = "10000";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(99), job.m_ResultsQueue.latestBucketEnd());

        dataRows["time"] = "10050";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(99), job.m_ResultsQueue.latestBucketEnd());

        dataRows["time"] = "10100";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10110";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10120";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(10099), job.m_ResultsQueue.latestBucketEnd());
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10200";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10300";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10300";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, visitor.lastResults(), 0.000005);
        }

        dataRows["time"] = "10400";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10401";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10402";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10403";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10500";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, visitor.lastResults(), 0.0005);
        }

        // Bucket at 10600 not present

        dataRows["time"] = "10700";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10700";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));

        dataRows["time"] = "10700";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));

        dataRows["time"] = "10800";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, visitor.lastResults(), 0.005);
        }

        dataRows["time"] = "10895";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(10799), job.m_ResultsQueue.latestBucketEnd());
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, visitor.lastResults(), 0.005);
        }
        job.finalise();
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(10799), job.m_ResultsQueue.latestBucketEnd());
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, visitor.lastResults(), 0.005);
        }
    }
    {
        LOG_DEBUG("*** testing non-out-of-phase count ***");
        core_t::TTime bucketSize = 100;
        model::CLimits limits;
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clauses;
        clauses.push_back("count");
        fieldConfig.initFromClause(clauses);
        model::CAnomalyDetectorModelConfig modelConfig = model::CAnomalyDetectorModelConfig::defaultConfig(bucketSize);
        std::stringstream outputStrm;

        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

        api::CAnomalyJob::TStrStrUMap dataRows;

        CPPUNIT_ASSERT_EQUAL(core_t::TTime(99), job.m_ResultsQueue.latestBucketEnd());
        dataRows["time"] = "10088";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(99), job.m_ResultsQueue.latestBucketEnd());

        dataRows["time"] = "10097";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(99), job.m_ResultsQueue.latestBucketEnd());

        dataRows["time"] = "10100";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10110";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10120";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT(job.m_ResultsQueue.latest().empty());

        dataRows["time"] = "10200";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10300";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10300";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, visitor.lastResults(), 0.000005);
        }

        dataRows["time"] = "10400";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10401";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10402";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10403";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10500";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, visitor.lastResults(), 0.0005);
        }

        // Bucket at 10600 not present

        dataRows["time"] = "10700";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10700";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));

        dataRows["time"] = "10700";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));

        dataRows["time"] = "10800";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, visitor.lastResults(), 0.005);
        }

        dataRows["time"] = "10805";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(10799), job.m_ResultsQueue.latestBucketEnd());
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, visitor.lastResults(), 0.005);
        }
        job.finalise();
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(10799), job.m_ResultsQueue.latestBucketEnd());
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, visitor.lastResults(), 0.005);
        }
    }
    // Now we come to the real meat and potatoes of the test, the out-of-phase buckets
    {
        LOG_DEBUG("*** testing out-of-phase metric ***");
        core_t::TTime bucketSize = 100;
        model::CLimits limits;
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clauses;
        clauses.push_back("mean(value)");
        fieldConfig.initFromClause(clauses);

        // 2 delay buckets
        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(bucketSize, model_t::E_None, "", 0, 2, false, "");
        std::stringstream outputStrm;

        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

        api::CAnomalyJob::TStrStrUMap dataRows;

        // main bucket should start at 10000 -> 10100
        // out-of-phase bucket start at 10050 -> 10150
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(49), job.m_ResultsQueue.latestBucketEnd());
        dataRows["time"] = "10000";
        dataRows["value"] = "5.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(49), job.m_ResultsQueue.latestBucketEnd());

        dataRows["time"] = "10050";
        dataRows["value"] = "3.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));

        dataRows["time"] = "10100";
        dataRows["value"] = "1.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, visitor.lastResults(), 0.0005);
        }
        dataRows["time"] = "10150";
        dataRows["value"] = "4.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(10099), job.m_ResultsQueue.latestBucketEnd());
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10200";
        dataRows["value"] = "0.0005";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(10149), job.m_ResultsQueue.latestBucketEnd());
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.5, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10300";
        dataRows["value"] = "5.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0005, visitor.lastResults(), 0.000005);
        }

        dataRows["time"] = "10499";
        dataRows["value"] = "5.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        LOG_DEBUG("Result time is " << (job.m_ResultsQueue.latestBucketEnd() - 49));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, visitor.lastResults(), 0.0000005);
        }

        dataRows["time"] = "10500";
        dataRows["value"] = "5.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0, visitor.lastResults(), 0.0005);
        }

        // Bucket at 10600 not present

        dataRows["time"] = "10700";
        dataRows["value"] = "50";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10720";
        dataRows["value"] = "80";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));

        dataRows["time"] = "10760";
        dataRows["value"] = "20";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(65.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10780";
        dataRows["value"] = "80";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));

        dataRows["time"] = "10800";
        dataRows["value"] = "5.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(57.5, visitor.lastResults(), 0.005);
        }

        // 10895, triggers bucket  10750->10850
        dataRows["time"] = "10895";
        dataRows["value"] = "6.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        LOG_DEBUG("Result time is " << (job.m_ResultsQueue.latestBucketEnd()));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(10799), job.m_ResultsQueue.latestBucketEnd());
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(35.0, visitor.lastResults(), 0.005);
        }
        LOG_DEBUG("Finalising job");
        job.finalise();
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(10799), job.m_ResultsQueue.latestBucketEnd());
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(35.0, visitor.lastResults(), 0.005);
        }
    }
    {
        LOG_DEBUG("*** testing out-of-phase metric ***");
        core_t::TTime bucketSize = 100;
        model::CLimits limits;
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clauses;
        clauses.push_back("mean(value)");
        fieldConfig.initFromClause(clauses);

        // 2 delay buckets
        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(bucketSize, model_t::E_None, "", 0, 2, false, "");
        std::stringstream outputStrm;

        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

        api::CAnomalyJob::TStrStrUMap dataRows;

        CPPUNIT_ASSERT_EQUAL(core_t::TTime(49), job.m_ResultsQueue.latestBucketEnd());
        dataRows["time"] = "10045";
        dataRows["value"] = "5.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(49), job.m_ResultsQueue.latestBucketEnd());

        dataRows["time"] = "10050";
        dataRows["value"] = "3.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));

        // This is the first complete bucket
        dataRows["time"] = "10100";
        dataRows["value"] = "1.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, visitor.lastResults(), 0.0005);
        }
        dataRows["time"] = "10150";
        dataRows["value"] = "4.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(10099), job.m_ResultsQueue.latestBucketEnd());
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10200";
        dataRows["value"] = "0.0005";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(10149), job.m_ResultsQueue.latestBucketEnd());
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.5, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10300";
        dataRows["value"] = "5.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0005, visitor.lastResults(), 0.000005);
        }

        dataRows["time"] = "10499";
        dataRows["value"] = "5.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        LOG_DEBUG("Result time is " << (job.m_ResultsQueue.latestBucketEnd() - 49));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, visitor.lastResults(), 0.0000005);
        }

        dataRows["time"] = "10500";
        dataRows["value"] = "5.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0, visitor.lastResults(), 0.0005);
        }

        // Bucket at 10600 not present

        dataRows["time"] = "10700";
        dataRows["value"] = "50";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10720";
        dataRows["value"] = "80";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));

        dataRows["time"] = "10760";
        dataRows["value"] = "20";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(65.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10780";
        dataRows["value"] = "80";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));

        dataRows["time"] = "10800";
        dataRows["value"] = "5.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(57.5, visitor.lastResults(), 0.005);
        }

        // 10895, triggers bucket  10750->10850
        dataRows["time"] = "10895";
        dataRows["value"] = "6.0";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        LOG_DEBUG("Result time is " << (job.m_ResultsQueue.latestBucketEnd()));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(10799), job.m_ResultsQueue.latestBucketEnd());
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(35.0, visitor.lastResults(), 0.005);
        }
        LOG_DEBUG("Finalising job");
        job.finalise();
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(10799), job.m_ResultsQueue.latestBucketEnd());
        {
            CSingleResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(35.0, visitor.lastResults(), 0.005);
        }
    }
    {
        LOG_DEBUG("*** testing out-of-phase eventrate ***");
        core_t::TTime bucketSize = 100;
        model::CLimits limits;
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clauses;
        clauses.push_back("high_count");
        clauses.push_back("by");
        clauses.push_back("person");
        fieldConfig.initFromClause(clauses);

        // 2 delay buckets
        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(bucketSize, model_t::E_None, "", 0, 2, false, "");
        std::stringstream outputStrm;

        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

        api::CAnomalyJob::TStrStrUMap dataRows;

        // main bucket should start at 10000 -> 10100
        // out-of-phase bucket start at 10050 -> 10150
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(49), job.m_ResultsQueue.latestBucketEnd());
        dataRows["time"] = "10000";
        dataRows["person"] = "Candice";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10001";
        dataRows["person"] = "Behati";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10002";
        dataRows["person"] = "Cara";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10003";
        dataRows["person"] = "Kate";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10004";
        dataRows["person"] = "Gisele";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(49), job.m_ResultsQueue.latestBucketEnd());

        dataRows["time"] = "10050";
        dataRows["person"] = "Behati";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));

        dataRows["time"] = "10070";
        dataRows["person"] = "Candice";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));

        dataRows["time"] = "10101";
        dataRows["person"] = "Cara";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CMultiResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(7.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10110";
        dataRows["person"] = "Kate";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));

        dataRows["time"] = "10150";
        dataRows["person"] = "Gisele";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(10099), job.m_ResultsQueue.latestBucketEnd());
        {
            CMultiResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10201";
        dataRows["person"] = "Behati";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10201";
        dataRows["person"] = "Candice";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10201";
        dataRows["person"] = "Gisele";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(10149), job.m_ResultsQueue.latestBucketEnd());
        {
            CMultiResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10300";
        dataRows["person"] = "Cara";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10300";
        dataRows["person"] = "Kate";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10300";
        dataRows["person"] = "Gisele the imposter";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10301";
        dataRows["person"] = "Cara";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CMultiResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10490";
        dataRows["person"] = "Gisele";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10492";
        dataRows["person"] = "Kate";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10494";
        dataRows["person"] = "Behati";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        dataRows["time"] = "10499";
        dataRows["person"] = "Cara";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        LOG_DEBUG("Result time is " << (job.m_ResultsQueue.latestBucketEnd() - 49));
        {
            CMultiResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10500";
        dataRows["person"] = "Cara";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CMultiResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, visitor.lastResults(), 0.0005);
        }

        // Bucket at 10600 not present

        dataRows["time"] = "10700";
        dataRows["person"] = "Behati";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CMultiResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10720";
        dataRows["person"] = "Kate";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));

        dataRows["time"] = "10760";
        dataRows["person"] = "Behati";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CMultiResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, visitor.lastResults(), 0.0005);
        }

        dataRows["time"] = "10780";
        dataRows["person"] = "Cara";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));

        dataRows["time"] = "10800";
        dataRows["person"] = "Candice";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        {
            CMultiResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, visitor.lastResults(), 0.005);
        }

        // 10895, triggers bucket  10750->10850
        dataRows["time"] = "10895";
        dataRows["person"] = "Cara";
        CPPUNIT_ASSERT(job.handleRecord(dataRows));
        LOG_DEBUG("Result time is " << (job.m_ResultsQueue.latestBucketEnd()));
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(10799), job.m_ResultsQueue.latestBucketEnd());
        {
            CMultiResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, visitor.lastResults(), 0.005);
        }
        LOG_DEBUG("Finalising job");
        job.finalise();
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(10799), job.m_ResultsQueue.latestBucketEnd());
        {
            CMultiResultVisitor visitor;
            job.m_ResultsQueue.latest().topDownBreadthFirst(visitor);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, visitor.lastResults(), 0.005);
        }
    }
}

void CAnomalyJobTest::testBucketSelection(void) {
    LOG_DEBUG("*** testBucketSelection ***");
    core_t::TTime bucketSize = 100;
    model::CLimits limits;
    api::CFieldConfig fieldConfig;
    api::CFieldConfig::TStrVec clauses;
    clauses.push_back("mean(value)");
    fieldConfig.initFromClause(clauses);

    // 2 delay buckets
    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig(bucketSize, model_t::E_None, "", 0, 2, false, "");
    std::stringstream outputStrm;

    core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

    api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

    job.m_ResultsQueue.reset(950);
    {
        model::SAnnotatedProbability prob(1.0);

        model::CHierarchicalResults results;
        results.addModelResult(
            0, false, "mean", model::function_t::E_IndividualMetricMean, "", "", "", "", "value", prob, 0, 1000);
        CResultsScoreVisitor visitor(10);
        results.topDownBreadthFirst(visitor);
        job.m_ResultsQueue.push(results, 1000);
        LOG_DEBUG("Adding 10 at 1000");
    }
    {
        model::SAnnotatedProbability prob(1.0);

        model::CHierarchicalResults results;
        results.addModelResult(
            0, false, "mean", model::function_t::E_IndividualMetricMean, "", "", "", "", "value", prob, 0, 1000);
        CResultsScoreVisitor visitor(20);
        results.topDownBreadthFirst(visitor);
        job.m_ResultsQueue.push(results, 1050);
        LOG_DEBUG("Adding 20 at 1050");
    }
    {
        model::SAnnotatedProbability prob(1.0);

        model::CHierarchicalResults results;
        results.addModelResult(
            0, false, "mean", model::function_t::E_IndividualMetricMean, "", "", "", "", "value", prob, 0, 1000);
        CResultsScoreVisitor visitor(15);
        results.topDownBreadthFirst(visitor);
        job.m_ResultsQueue.push(results, 1100);
        LOG_DEBUG("Adding 15 at 1100");
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(0), job.m_ResultsQueue.chooseResultTime(1100, bucketSize, results));
    }
    {
        model::SAnnotatedProbability prob(1.0);

        model::CHierarchicalResults results;
        results.addModelResult(
            0, false, "mean", model::function_t::E_IndividualMetricMean, "", "", "", "", "value", prob, 0, 1000);
        CResultsScoreVisitor visitor(20);
        results.topDownBreadthFirst(visitor);
        job.m_ResultsQueue.push(results, 1150);
        LOG_DEBUG("Adding 20 at 1150");
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(0), job.m_ResultsQueue.chooseResultTime(1150, bucketSize, results));
    }
    {
        model::SAnnotatedProbability prob(1.0);

        model::CHierarchicalResults results;
        results.addModelResult(
            0, false, "mean", model::function_t::E_IndividualMetricMean, "", "", "", "", "value", prob, 0, 1000);
        CResultsScoreVisitor visitor(25);
        results.topDownBreadthFirst(visitor);
        job.m_ResultsQueue.push(results, 1200);
        LOG_DEBUG("Adding 25 at 1200");
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(1100), job.m_ResultsQueue.chooseResultTime(1200, bucketSize, results));
    }
    {
        model::SAnnotatedProbability prob(1.0);

        model::CHierarchicalResults results;
        results.addModelResult(
            0, false, "mean", model::function_t::E_IndividualMetricMean, "", "", "", "", "value", prob, 0, 1000);
        CResultsScoreVisitor visitor(0);
        results.topDownBreadthFirst(visitor);
        job.m_ResultsQueue.push(results, 1250);
        LOG_DEBUG("Adding 0 at 1250");
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(0), job.m_ResultsQueue.chooseResultTime(1250, bucketSize, results));
    }
    {
        model::SAnnotatedProbability prob(1.0);

        model::CHierarchicalResults results;
        results.addModelResult(
            0, false, "mean", model::function_t::E_IndividualMetricMean, "", "", "", "", "value", prob, 0, 1000);
        CResultsScoreVisitor visitor(5);
        results.topDownBreadthFirst(visitor);
        job.m_ResultsQueue.push(results, 1300);
        LOG_DEBUG("Adding 5 at 1300");
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(1200), job.m_ResultsQueue.chooseResultTime(1300, bucketSize, results));
    }
    {
        model::SAnnotatedProbability prob(1.0);

        model::CHierarchicalResults results;
        results.addModelResult(
            0, false, "mean", model::function_t::E_IndividualMetricMean, "", "", "", "", "value", prob, 0, 1000);
        CResultsScoreVisitor visitor(5);
        results.topDownBreadthFirst(visitor);
        job.m_ResultsQueue.push(results, 1350);
        LOG_DEBUG("Adding 5 at 1350");
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(0), job.m_ResultsQueue.chooseResultTime(1350, bucketSize, results));
    }
    {
        model::SAnnotatedProbability prob(1.0);

        model::CHierarchicalResults results;
        results.addModelResult(
            0, false, "mean", model::function_t::E_IndividualMetricMean, "", "", "", "", "value", prob, 0, 1000);
        CResultsScoreVisitor visitor(1);
        results.topDownBreadthFirst(visitor);
        job.m_ResultsQueue.push(results, 1400);
        LOG_DEBUG("Adding 1 at 1400");
        CPPUNIT_ASSERT_EQUAL(core_t::TTime(1300), job.m_ResultsQueue.chooseResultTime(1400, bucketSize, results));
    }
}

void CAnomalyJobTest::testModelPlot(void) {
    LOG_DEBUG("*** testModelPlot ***");
    {
        // Test non-overlapping buckets
        core_t::TTime bucketSize = 10000;
        model::CLimits limits;
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clauses;
        clauses.push_back("mean(value)");
        clauses.push_back("by");
        clauses.push_back("animal");
        fieldConfig.initFromClause(clauses);

        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(bucketSize, model_t::E_None, "", 0, 0, false, "");
        modelConfig.modelPlotBoundsPercentile(1.0);
        std::stringstream outputStrm;

        {
            core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

            api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

            api::CAnomalyJob::TStrStrUMap dataRows;
            dataRows["time"] = "10000000";
            dataRows["value"] = "2.0";
            dataRows["animal"] = "baboon";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["value"] = "5.0";
            dataRows["animal"] = "shark";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10010000";
            dataRows["value"] = "2.0";
            dataRows["animal"] = "baboon";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["value"] = "5.0";
            dataRows["animal"] = "shark";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10020000";
            dataRows["value"] = "2.0";
            dataRows["animal"] = "baboon";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["value"] = "5.0";
            dataRows["animal"] = "shark";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10030000";
            dataRows["value"] = "2.0";
            dataRows["animal"] = "baboon";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["value"] = "5.0";
            dataRows["animal"] = "shark";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10040000";
            dataRows["value"] = "3.0";
            dataRows["animal"] = "baboon";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["value"] = "5.0";
            dataRows["animal"] = "shark";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            job.finalise();
        }

        std::string output = outputStrm.str();
        LOG_TRACE("Output has yielded: " << output);
        core::CRegex regex;
        regex.init("\n");
        core::CRegex::TStrVec lines;
        regex.split(output, lines);
        CPPUNIT_ASSERT(findLine("model_feature.*timestamp.*10000000.*baboon", lines));
        CPPUNIT_ASSERT(findLine("model_feature.*timestamp.*10000000.*shark", lines));
        CPPUNIT_ASSERT(findLine("model_feature.*timestamp.*10010000.*baboon", lines));
        CPPUNIT_ASSERT(findLine("model_feature.*timestamp.*10010000.*shark", lines));
        CPPUNIT_ASSERT(findLine("model_feature.*timestamp.*10020000.*baboon", lines));
        CPPUNIT_ASSERT(findLine("model_feature.*timestamp.*10020000.*shark", lines));
        CPPUNIT_ASSERT(findLine("model_feature.*timestamp.*10030000.*baboon", lines));
        CPPUNIT_ASSERT(findLine("model_feature.*timestamp.*10030000.*shark", lines));
    }
    {
        // Test overlapping buckets
        core_t::TTime bucketSize = 10000;
        model::CLimits limits;
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clauses;
        clauses.push_back("max(value)");
        fieldConfig.initFromClause(clauses);

        // 2 delay buckets
        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(bucketSize, model_t::E_None, "", 0, 2, false, "");
        modelConfig.modelPlotBoundsPercentile(1.0);

        std::stringstream outputStrm;
        {
            core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

            api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

            api::CAnomalyJob::TStrStrUMap dataRows;

            // Data contains 3 anomalies
            dataRows["time"] = "10000000";
            dataRows["value"] = "2.0";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10010000";
            dataRows["value"] = "2.1";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10020000";
            dataRows["value"] = "2.0";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10030000";
            dataRows["value"] = "2.3";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10040000";
            dataRows["value"] = "2.2";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10055500";
            dataRows["value"] = "2.0";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10060000";
            dataRows["value"] = "2.0";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10077700";
            dataRows["value"] = "2.1";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10080000";
            dataRows["value"] = "2.4";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10090000";
            dataRows["value"] = "2.1";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10094400";
            dataRows["value"] = "2.0003";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10110000";
            dataRows["value"] = "2.01";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10120000";
            dataRows["value"] = "2.03";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10140000";
            dataRows["value"] = "2.001";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10150000";
            dataRows["value"] = "2.1";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10167000";
            dataRows["value"] = "200.0";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10170000";
            dataRows["value"] = "2.0";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10183000";
            dataRows["value"] = "400.0";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10190000";
            dataRows["value"] = "2.0";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10200000";
            dataRows["value"] = "2.0";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10210000";
            dataRows["value"] = "2.0";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));
            dataRows["time"] = "10230000";
            dataRows["value"] = "2.0";
            CPPUNIT_ASSERT(job.handleRecord(dataRows));

            job.finalise();
        }

        std::string output = outputStrm.str();
        LOG_TRACE("Output has yielded: " << output);
        core::CRegex regex;
        regex.init("\n");
        core::CRegex::TStrVec lines;
        regex.split(output, lines);
        CPPUNIT_ASSERT(findLine("model_feature.*timestamp.*10000000000", lines));
        CPPUNIT_ASSERT(findLine("model_feature.*timestamp.*10010000000", lines));
        CPPUNIT_ASSERT(findLine("model_feature.*timestamp.*10020000000", lines));
        CPPUNIT_ASSERT(findLine("model_feature.*timestamp.*10075000000.*actual..2\\.4", lines));
        CPPUNIT_ASSERT(findLine("model_feature.*timestamp.*10165000000.*actual..200", lines));
        CPPUNIT_ASSERT(findLine("model_feature.*timestamp.*10175000000.*actual..400", lines));
    }
}

void CAnomalyJobTest::testInterimResultEdgeCases(void) {
    LOG_DEBUG("*** testInterimResultEdgeCases ***");

    const char* logFile = "test.log";

    core_t::TTime bucketSize = 3600;
    model::CLimits limits;
    api::CFieldConfig fieldConfig;
    api::CFieldConfig::TStrVec clauses{"count", "by", "error"};
    fieldConfig.initFromClause(clauses);

    model::CAnomalyDetectorModelConfig modelConfig = model::CAnomalyDetectorModelConfig::defaultConfig(bucketSize);

    std::stringstream outputStrm;

    core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

    api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

    std::remove(logFile);
    CPPUNIT_ASSERT(ml::core::CLogger::instance().reconfigureFromFile("testfiles/testLogErrorsLog4cxx.properties"));

    api::CAnomalyJob::TStrStrUMap dataRows;
    dataRows["time"] = "3610";
    dataRows["error"] = "e1";
    CPPUNIT_ASSERT(job.handleRecord(dataRows));
    dataRows["time"] = "3670";
    dataRows["error"] = "e2";
    CPPUNIT_ASSERT(job.handleRecord(dataRows));
    dataRows["time"] = "6820";
    dataRows["error"] = "e1";
    CPPUNIT_ASSERT(job.handleRecord(dataRows));
    dataRows["time"] = "6820";
    dataRows["error"] = "e1";
    CPPUNIT_ASSERT(job.handleRecord(dataRows));
    dataRows["time"] = "7850";
    dataRows["error"] = "e1";
    CPPUNIT_ASSERT(job.handleRecord(dataRows));
    dataRows["time"] = "9310";
    dataRows["error"] = "e2";
    CPPUNIT_ASSERT(job.handleRecord(dataRows));

    dataRows["."] = "t7200";
    CPPUNIT_ASSERT(job.handleRecord(dataRows));
    dataRows["."] = "i";
    CPPUNIT_ASSERT(job.handleRecord(dataRows));

    // The test log4cxx.properties is very similar to the hardcoded default.
    CPPUNIT_ASSERT(ml::core::CLogger::instance().reconfigureFromFile("testfiles/log4cxx.properties"));

    std::ifstream log(logFile);
    CPPUNIT_ASSERT(log.is_open());
    char line[256];
    while (log.getline(line, 256)) {
        LOG_DEBUG("Got '" << line << "'");
        CPPUNIT_ASSERT(false);
    }
    log.close();
    std::remove(logFile);
}

void CAnomalyJobTest::testRestoreFailsWithEmptyStream(void) {
    model::CLimits limits;
    api::CFieldConfig fieldConfig;
    api::CFieldConfig::TStrVec clauses;
    clauses.push_back("value");
    clauses.push_back("partitionfield=greenhouse");
    fieldConfig.initFromClause(clauses);
    model::CAnomalyDetectorModelConfig modelConfig = model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);
    std::ostringstream outputStrm;
    core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

    api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

    core_t::TTime completeToTime(0);
    CEmptySearcher restoreSearcher;
    CPPUNIT_ASSERT(job.restoreState(restoreSearcher, completeToTime) == false);
}

CppUnit::Test* CAnomalyJobTest::suite(void) {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CAnomalyJobTest");

    suiteOfTests->addTest(
        new CppUnit::TestCaller<CAnomalyJobTest>("CAnomalyJobTest::testBadTimes", &CAnomalyJobTest::testBadTimes));
    suiteOfTests->addTest(new CppUnit::TestCaller<CAnomalyJobTest>("CAnomalyJobTest::testOutOfSequence",
                                                                   &CAnomalyJobTest::testOutOfSequence));
    suiteOfTests->addTest(new CppUnit::TestCaller<CAnomalyJobTest>("CAnomalyJobTest::testControlMessages",
                                                                   &CAnomalyJobTest::testControlMessages));
    suiteOfTests->addTest(new CppUnit::TestCaller<CAnomalyJobTest>("CAnomalyJobTest::testSkipTimeControlMessage",
                                                                   &CAnomalyJobTest::testSkipTimeControlMessage));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CAnomalyJobTest>("CAnomalyJobTest::testOutOfPhase", &CAnomalyJobTest::testOutOfPhase));
    suiteOfTests->addTest(new CppUnit::TestCaller<CAnomalyJobTest>("CAnomalyJobTest::testBucketSelection",
                                                                   &CAnomalyJobTest::testBucketSelection));
    suiteOfTests->addTest(
        new CppUnit::TestCaller<CAnomalyJobTest>("CAnomalyJobTest::testModelPlot", &CAnomalyJobTest::testModelPlot));
    suiteOfTests->addTest(new CppUnit::TestCaller<CAnomalyJobTest>("CAnomalyJobTest::testInterimResultEdgeCases",
                                                                   &CAnomalyJobTest::testInterimResultEdgeCases));
    suiteOfTests->addTest(new CppUnit::TestCaller<CAnomalyJobTest>("CAnomalyJobTest::testRestoreFailsWithEmptyStream",
                                                                   &CAnomalyJobTest::testRestoreFailsWithEmptyStream));
    return suiteOfTests;
}
