/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

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

#include <boost/test/unit_test.hpp>
#include <boost/tuple/tuple.hpp>

#include <cstdio>
#include <fstream>
#include <sstream>

BOOST_AUTO_TEST_SUITE(CAnomalyJobTest)

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
    virtual TIStreamP search(size_t /*currentDocNum*/, size_t /*limit*/) {
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
    CSingleResultVisitor() : m_LastResult(0.0) {}

    virtual ~CSingleResultVisitor() {}

    virtual void visit(const ml::model::CHierarchicalResults& /*results*/,
                       const TNode& node,
                       bool /*pivot*/) {
        if (!this->isSimpleCount(node) && this->isLeaf(node)) {
            if (node.s_AnnotatedProbability.s_AttributeProbabilities.size() == 0) {
                return;
            }
            if (!node.s_Model) {
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
    double m_LastResult;
};

class CMultiResultVisitor : public ml::model::CHierarchicalResultsVisitor {
public:
    CMultiResultVisitor() : m_LastResult(0.0) {}

    virtual ~CMultiResultVisitor() {}

    virtual void visit(const ml::model::CHierarchicalResults& /*results*/,
                       const TNode& node,
                       bool /*pivot*/) {
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
                LOG_ERROR(<< "No identifier for '" << *node.s_Spec.s_PersonFieldValue << "'");
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
    double m_LastResult;
};

class CResultsScoreVisitor : public ml::model::CHierarchicalResultsVisitor {
public:
    CResultsScoreVisitor(int score) : m_Score(score) {}

    virtual ~CResultsScoreVisitor() {}

    virtual void visit(const ml::model::CHierarchicalResults& /*results*/,
                       const TNode& node,
                       bool /*pivot*/) {
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
    BOOST_TEST(!doc.HasParseError());
    BOOST_TEST(doc.IsArray());

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

BOOST_AUTO_TEST_CASE(testBadTimes) {
    {
        // Test with no time field
        model::CLimits limits;
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clauses;
        clauses.push_back("value");
        clauses.push_back("partitionfield=greenhouse");
        fieldConfig.initFromClause(clauses);
        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);
        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

        api::CAnomalyJob::TStrStrUMap dataRows;
        dataRows["wibble"] = "12345678";
        dataRows["value"] = "1.0";
        dataRows["greenhouse"] = "rhubarb";

        BOOST_TEST(job.handleRecord(dataRows));
        BOOST_CHECK_EQUAL(uint64_t(0), job.numRecordsHandled());
    }
    {
        // Test with bad time field
        model::CLimits limits;
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clauses;
        clauses.push_back("value");
        clauses.push_back("partitionfield=greenhouse");
        fieldConfig.initFromClause(clauses);
        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);
        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

        api::CAnomalyJob::TStrStrUMap dataRows;
        dataRows["time"] = "hello";
        dataRows["value"] = "1.0";
        dataRows["greenhouse"] = "rhubarb";

        BOOST_TEST(job.handleRecord(dataRows));
        BOOST_CHECK_EQUAL(uint64_t(0), job.numRecordsHandled());
    }
    {
        // Test with bad time field format
        model::CLimits limits;
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clauses;
        clauses.push_back("value");
        clauses.push_back("partitionfield=greenhouse");
        fieldConfig.initFromClause(clauses);
        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);
        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream,
                             api::CAnomalyJob::TPersistCompleteFunc(), nullptr,
                             -1, "time", "%Y%m%m%H%M%S");

        api::CAnomalyJob::TStrStrUMap dataRows;
        dataRows["time"] = "hello world";
        dataRows["value"] = "1.0";
        dataRows["greenhouse"] = "rhubarb";

        BOOST_TEST(job.handleRecord(dataRows));
        BOOST_CHECK_EQUAL(uint64_t(0), job.numRecordsHandled());
    }
}

BOOST_AUTO_TEST_CASE(testOutOfSequence) {
    {
        // Test out of sequence record
        model::CLimits limits;
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clauses;
        clauses.push_back("value");
        clauses.push_back("partitionfield=greenhouse");
        fieldConfig.initFromClause(clauses);
        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);
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

        BOOST_TEST(job.handleRecord(dataRows));
        BOOST_CHECK_EQUAL(uint64_t(1), job.numRecordsHandled());

        dataRows["time"] = "1234567";

        BOOST_TEST(job.handleRecord(dataRows));
        BOOST_CHECK_EQUAL(uint64_t(1), job.numRecordsHandled());
        job.finalise();
    }
}

BOOST_AUTO_TEST_CASE(testControlMessages) {
    {
        // Test control messages
        model::CLimits limits;
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clauses;
        clauses.push_back("value");
        clauses.push_back("partitionfield=greenhouse");
        fieldConfig.initFromClause(clauses);
        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);
        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

        api::CAnomalyJob::TStrStrUMap dataRows;
        dataRows["."] = " ";
        BOOST_TEST(job.handleRecord(dataRows));
        BOOST_CHECK_EQUAL(uint64_t(0), job.numRecordsHandled());

        dataRows["."] = ".";
        BOOST_TEST(job.handleRecord(dataRows));
        BOOST_CHECK_EQUAL(uint64_t(0), job.numRecordsHandled());

        dataRows["."] = "f";
        BOOST_TEST(job.handleRecord(dataRows));
        BOOST_CHECK_EQUAL(uint64_t(0), job.numRecordsHandled());

        dataRows["."] = "f1";
        BOOST_TEST(job.handleRecord(dataRows));
        BOOST_CHECK_EQUAL(uint64_t(0), job.numRecordsHandled());
    }
    {
        // Test reset bucket
        model::CLimits limits;
        api::CFieldConfig fieldConfig;
        api::CFieldConfig::TStrVec clauses;
        clauses.push_back("count");
        clauses.push_back("partitionfield=greenhouse");
        fieldConfig.initFromClause(clauses);
        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);

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
                        BOOST_TEST(job.handleRecord(dataRows));
                    }
                }
                BOOST_TEST(job.handleRecord(dataRows));
                if (i < 2) {
                    // We haven't processed one full bucket but it should be safe to flush.
                    dataRows["."] = "f1";
                    BOOST_TEST(job.handleRecord(dataRows));
                    dataRows.erase(".");
                }
            }
        }

        rapidjson::Document doc;
        doc.Parse<rapidjson::kParseDefaultFlags>(outputStrm.str());
        BOOST_TEST(!doc.HasParseError());
        BOOST_TEST(doc.IsArray());

        const rapidjson::Value& allRecords = doc.GetArray();
        bool foundRecord = false;
        for (auto& r : allRecords.GetArray()) {
            rapidjson::Value::ConstMemberIterator recordsIt =
                r.GetObject().FindMember("records");
            if (recordsIt != r.GetObject().MemberEnd()) {
                auto& recordsArray = recordsIt->value.GetArray()[0];
                rapidjson::Value::ConstMemberIterator actualIt =
                    recordsArray.FindMember("actual");
                BOOST_TEST(actualIt != recordsArray.MemberEnd());
                const rapidjson::Value::ConstArray& values = actualIt->value.GetArray();

                BOOST_CHECK_EQUAL(102.0, values[0].GetDouble());
                foundRecord = true;
            }
        }

        BOOST_TEST(foundRecord);
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
                        BOOST_TEST(job.handleRecord(dataRows));
                    }
                }
                BOOST_TEST(job.handleRecord(dataRows));
                if (i == 40) {
                    api::CAnomalyJob::TStrStrUMap rows;
                    rows["."] = "r" + ss.str() + " " + ss.str();
                    BOOST_TEST(job.handleRecord(rows));
                    for (std::size_t j = 0; j < 100; j++) {
                        BOOST_TEST(job.handleRecord(dataRows));
                    }
                }
            }
        }

        rapidjson::Document doc2;
        doc2.Parse<rapidjson::kParseDefaultFlags>(outputStrm2.str());
        BOOST_TEST(!doc2.HasParseError());
        BOOST_TEST(doc2.IsArray());

        const rapidjson::Value& allRecords2 = doc2.GetArray();
        foundRecord = false;
        for (auto& r : allRecords2.GetArray()) {
            rapidjson::Value::ConstMemberIterator recordsIt =
                r.GetObject().FindMember("records");
            if (recordsIt != r.GetObject().MemberEnd()) {
                auto& recordsArray = recordsIt->value.GetArray()[0];
                rapidjson::Value::ConstMemberIterator actualIt =
                    recordsArray.FindMember("actual");
                BOOST_TEST(actualIt != recordsArray.MemberEnd());
                const rapidjson::Value::ConstArray& values = actualIt->value.GetArray();

                BOOST_CHECK_EQUAL(101.0, values[0].GetDouble());
                foundRecord = true;
            }
        }

        BOOST_TEST(foundRecord);
    }
}

BOOST_AUTO_TEST_CASE(testSkipTimeControlMessage) {
    model::CLimits limits;
    api::CFieldConfig fieldConfig;
    api::CFieldConfig::TStrVec clauses;
    clauses.push_back("count");
    fieldConfig.initFromClause(clauses);
    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);

    std::stringstream outputStrm;
    core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

    api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

    api::CAnomalyJob::TStrStrUMap dataRows;

    core_t::TTime time = 3600;
    for (std::size_t i = 0; i < 10; ++i, time += BUCKET_SIZE) {
        std::ostringstream ss;
        ss << time;
        dataRows["time"] = ss.str();
        BOOST_TEST(job.handleRecord(dataRows));
    }

    wrappedOutputStream.syncFlush();
    BOOST_CHECK_EQUAL(std::size_t(9), countBuckets("bucket", outputStrm.str() + "]"));

    // Now let's skip time to Thursday, June 29, 2017 12:00:00 AM
    time = 1498694400;
    dataRows["."] = "s1498694400";
    BOOST_TEST(job.handleRecord(dataRows));
    dataRows.erase(".");

    // Check no new bucket results were written
    wrappedOutputStream.syncFlush();
    BOOST_CHECK_EQUAL(std::size_t(9), countBuckets("bucket", outputStrm.str() + "]"));

    // Let's send a few buckets after skip time
    for (std::size_t i = 0; i < 3; ++i, time += BUCKET_SIZE) {
        std::ostringstream ss;
        ss << time;
        dataRows["time"] = ss.str();
        BOOST_TEST(job.handleRecord(dataRows));
    }

    // Assert only 2 new buckets were written
    wrappedOutputStream.syncFlush();
    BOOST_CHECK_EQUAL(std::size_t(11), countBuckets("bucket", outputStrm.str() + "]"));
}

BOOST_AUTO_TEST_CASE(testIsPersistenceNeeded) {

    model::CLimits limits;
    api::CFieldConfig fieldConfig;
    api::CFieldConfig::TStrVec clauses;
    clauses.push_back("count");
    fieldConfig.initFromClause(clauses);
    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);

    {
        // check that persistence is not needed if no input records have been handled
        // and the time has not been advanced

        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

        BOOST_CHECK_EQUAL(false, job.isPersistenceNeeded("test state"));

        job.finalise();
        wrappedOutputStream.syncFlush();

        std::string output = outputStrm.str();
        LOG_DEBUG(<< "Output has yielded: " << output);

        // check that no quantile state was persisted
        core::CRegex regex;
        regex.init("\n");
        core::CRegex::TStrVec lines;
        regex.split(output, lines);
        BOOST_CHECK_EQUAL(false, findLine("\"quantiles\":{\"job_id\":\"job\",\"quantile_state\".*",
                                          lines));
    }

    core_t::TTime time = 3600;
    {
        // check that persistence is needed if an input record has been handled

        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

        api::CAnomalyJob::TStrStrUMap dataRows;

        std::ostringstream ss;
        ss << time;
        dataRows["time"] = ss.str();
        BOOST_TEST(job.handleRecord(dataRows));

        BOOST_CHECK_EQUAL(true, job.isPersistenceNeeded("test state"));

        job.finalise();
        wrappedOutputStream.syncFlush();

        std::string output = outputStrm.str();
        LOG_DEBUG(<< "Output has yielded: " << output);

        // check that the quantile state has actually been persisted
        core::CRegex regex;
        regex.init("\n");
        core::CRegex::TStrVec lines;
        regex.split(output, lines);
        BOOST_CHECK_EQUAL(true, findLine("\"quantiles\":{\"job_id\":\"job\",\"quantile_state\".*",
                                         lines));
    }

    {
        // check that persistence is needed if time has been advanced (via a control message)
        // even if no input data has been handled

        std::stringstream outputStrm;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

        api::CAnomalyJob::TStrStrUMap dataRows;

        time = 39600;
        dataRows["."] = "t39600";
        BOOST_TEST(job.handleRecord(dataRows));
        BOOST_TEST(job.isPersistenceNeeded("test state"));

        BOOST_CHECK_EQUAL(true, job.isPersistenceNeeded("test state"));

        job.finalise();
        wrappedOutputStream.syncFlush();

        std::string output = outputStrm.str();
        LOG_DEBUG(<< "Output has yielded: " << output);

        // check that the quantile state has actually been persisted
        core::CRegex regex;
        regex.init("\n");
        core::CRegex::TStrVec lines;
        regex.split(output, lines);
        BOOST_CHECK_EQUAL(true, findLine("\"quantiles\":{\"job_id\":\"job\",\"quantile_state\".*",
                                         lines));
    }
}

BOOST_AUTO_TEST_CASE(testModelPlot) {
    core_t::TTime bucketSize = 10000;
    model::CLimits limits;
    api::CFieldConfig fieldConfig;
    api::CFieldConfig::TStrVec clauses;
    clauses.push_back("mean(value)");
    clauses.push_back("by");
    clauses.push_back("animal");
    fieldConfig.initFromClause(clauses);

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig(bucketSize, model_t::E_None,
                                                          "", 0, false);
    modelConfig.modelPlotBoundsPercentile(1.0);
    std::stringstream outputStrm;

    {
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

        api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

        api::CAnomalyJob::TStrStrUMap dataRows;
        dataRows["time"] = "10000000";
        dataRows["value"] = "2.0";
        dataRows["animal"] = "baboon";
        BOOST_TEST(job.handleRecord(dataRows));
        dataRows["value"] = "5.0";
        dataRows["animal"] = "shark";
        BOOST_TEST(job.handleRecord(dataRows));
        dataRows["time"] = "10010000";
        dataRows["value"] = "2.0";
        dataRows["animal"] = "baboon";
        BOOST_TEST(job.handleRecord(dataRows));
        dataRows["value"] = "5.0";
        dataRows["animal"] = "shark";
        BOOST_TEST(job.handleRecord(dataRows));
        dataRows["time"] = "10020000";
        dataRows["value"] = "2.0";
        dataRows["animal"] = "baboon";
        BOOST_TEST(job.handleRecord(dataRows));
        dataRows["value"] = "5.0";
        dataRows["animal"] = "shark";
        BOOST_TEST(job.handleRecord(dataRows));
        dataRows["time"] = "10030000";
        dataRows["value"] = "2.0";
        dataRows["animal"] = "baboon";
        BOOST_TEST(job.handleRecord(dataRows));
        dataRows["value"] = "5.0";
        dataRows["animal"] = "shark";
        BOOST_TEST(job.handleRecord(dataRows));
        dataRows["time"] = "10040000";
        dataRows["value"] = "3.0";
        dataRows["animal"] = "baboon";
        BOOST_TEST(job.handleRecord(dataRows));
        dataRows["value"] = "5.0";
        dataRows["animal"] = "shark";
        BOOST_TEST(job.handleRecord(dataRows));
        job.finalise();
    }

    std::string output = outputStrm.str();
    LOG_TRACE(<< "Output has yielded: " << output);
    core::CRegex regex;
    regex.init("\n");
    core::CRegex::TStrVec lines;
    regex.split(output, lines);
    BOOST_TEST(findLine("model_feature.*timestamp.*10000000.*baboon", lines));
    BOOST_TEST(findLine("model_feature.*timestamp.*10000000.*shark", lines));
    BOOST_TEST(findLine("model_feature.*timestamp.*10010000.*baboon", lines));
    BOOST_TEST(findLine("model_feature.*timestamp.*10010000.*shark", lines));
    BOOST_TEST(findLine("model_feature.*timestamp.*10020000.*baboon", lines));
    BOOST_TEST(findLine("model_feature.*timestamp.*10020000.*shark", lines));
    BOOST_TEST(findLine("model_feature.*timestamp.*10030000.*baboon", lines));
    BOOST_TEST(findLine("model_feature.*timestamp.*10030000.*shark", lines));
}

BOOST_AUTO_TEST_CASE(testInterimResultEdgeCases) {
    const char* logFile = "test.log";

    core_t::TTime bucketSize = 3600;
    model::CLimits limits;
    api::CFieldConfig fieldConfig;
    api::CFieldConfig::TStrVec clauses{"count", "by", "error"};
    fieldConfig.initFromClause(clauses);

    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig(bucketSize);

    std::stringstream outputStrm;

    core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

    api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

    std::remove(logFile);
    BOOST_TEST(ml::core::CLogger::instance().reconfigureFromFile(
        "testfiles/testLogErrors.boost.log.ini"));

    api::CAnomalyJob::TStrStrUMap dataRows;
    dataRows["time"] = "3610";
    dataRows["error"] = "e1";
    BOOST_TEST(job.handleRecord(dataRows));
    dataRows["time"] = "3670";
    dataRows["error"] = "e2";
    BOOST_TEST(job.handleRecord(dataRows));
    dataRows["time"] = "6820";
    dataRows["error"] = "e1";
    BOOST_TEST(job.handleRecord(dataRows));
    dataRows["time"] = "6820";
    dataRows["error"] = "e1";
    BOOST_TEST(job.handleRecord(dataRows));
    dataRows["time"] = "7850";
    dataRows["error"] = "e1";
    BOOST_TEST(job.handleRecord(dataRows));
    dataRows["time"] = "9310";
    dataRows["error"] = "e2";
    BOOST_TEST(job.handleRecord(dataRows));

    dataRows["."] = "t7200";
    BOOST_TEST(job.handleRecord(dataRows));
    dataRows["."] = "i";
    BOOST_TEST(job.handleRecord(dataRows));

    // Revert to the default logger settings
    ml::core::CLogger::instance().reset();

    std::ifstream log(logFile);
    // Boost.Log only creates files when the first message is logged,
    // and here we're asserting no messages logged
    if (log.is_open()) {
        char line[256];
        while (log.getline(line, 256)) {
            LOG_DEBUG(<< "Got '" << line << "'");
            BOOST_TEST(false);
        }
        log.close();
        std::remove(logFile);
    }
}

BOOST_AUTO_TEST_CASE(testRestoreFailsWithEmptyStream) {
    model::CLimits limits;
    api::CFieldConfig fieldConfig;
    api::CFieldConfig::TStrVec clauses;
    clauses.push_back("value");
    clauses.push_back("partitionfield=greenhouse");
    fieldConfig.initFromClause(clauses);
    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);
    std::ostringstream outputStrm;
    core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

    api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream);

    core_t::TTime completeToTime(0);
    CEmptySearcher restoreSearcher;
    BOOST_TEST(job.restoreState(restoreSearcher, completeToTime) == false);
}

BOOST_AUTO_TEST_SUITE_END()
