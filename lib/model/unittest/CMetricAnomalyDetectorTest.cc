/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CRegex.h>

#include <maths/CIntegerTools.h>
#include <maths/CModelWeight.h>

#include <model/CAnomalyDetector.h>
#include <model/CHierarchicalResults.h>
#include <model/CHierarchicalResultsAggregator.h>
#include <model/CHierarchicalResultsPopulator.h>
#include <model/CHierarchicalResultsProbabilityFinalizer.h>
#include <model/CSearchKey.h>
#include <model/FunctionTypes.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CTimeSeriesTestData.h>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

BOOST_AUTO_TEST_SUITE(CMetricAnomalyDetectorTest)

using namespace ml;

namespace {

using TTimeTimePr = std::pair<core_t::TTime, core_t::TTime>;
using TTimeTimePrVec = std::vector<TTimeTimePr>;
using TDoubleVec = std::vector<double>;

bool doIntersect(const TTimeTimePr& i1, const TTimeTimePr& i2) {
    return !(i2.second <= i1.first || i1.second <= i2.first);
}

class CResultWriter : public ml::model::CHierarchicalResultsVisitor {
public:
    static const double HIGH_ANOMALY_SCORE;

public:
    CResultWriter(const model::CAnomalyDetectorModelConfig& modelConfig,
                  const model::CLimits& limits,
                  core_t::TTime bucketLength)
        : m_ModelConfig(modelConfig), m_Limits(limits), m_BucketLength(bucketLength) {}

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
        ml::model::CHierarchicalResultsPopulator populator(m_Limits);
        results.bottomUpBreadthFirst(populator);
        results.bottomUpBreadthFirst(*this);
    }

    //! Visit a node.
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

        core_t::TTime bucketTime = node.s_BucketStartTime;
        double anomalyFactor = node.s_RawAnomalyScore;
        if (anomalyFactor > HIGH_ANOMALY_SCORE) {
            m_HighAnomalyTimes.push_back(TTimeTimePr(bucketTime, bucketTime + m_BucketLength));
            m_HighAnomalyFactors.push_back(anomalyFactor);
        } else if (anomalyFactor > 0.0) {
            m_AnomalyFactors.push_back(anomalyFactor);
            uint64_t currentRate(0);
            if (node.s_AnnotatedProbability.s_CurrentBucketCount) {
                currentRate = *node.s_AnnotatedProbability.s_CurrentBucketCount;
            }
            m_AnomalyRates.push_back(static_cast<double>(currentRate));
        }
    }

    bool operator()(ml::core_t::TTime time,
                    const ml::model::CHierarchicalResults::TNode& node,
                    bool isBucketInfluencer) {
        LOG_DEBUG(<< (isBucketInfluencer ? "BucketInfluencer" : "Influencer ")
                  << node.s_Spec.print() << " initial score "
                  << node.probability() << ", time:  " << time);

        return true;
    }

    const TTimeTimePrVec& highAnomalyTimes() const {
        return m_HighAnomalyTimes;
    }

    const TDoubleVec& highAnomalyFactors() const {
        return m_HighAnomalyFactors;
    }

    const TDoubleVec& anomalyFactors() const { return m_AnomalyFactors; }

    const TDoubleVec& anomalyRates() const { return m_AnomalyRates; }

private:
    const model::CAnomalyDetectorModelConfig& m_ModelConfig;
    const model::CLimits& m_Limits;
    core_t::TTime m_BucketLength;
    TTimeTimePrVec m_HighAnomalyTimes;
    TDoubleVec m_HighAnomalyFactors;
    TDoubleVec m_AnomalyFactors;
    TDoubleVec m_AnomalyRates;
};

const double CResultWriter::HIGH_ANOMALY_SCORE(0.35);

void importData(core_t::TTime firstTime,
                core_t::TTime lastTime,
                core_t::TTime bucketLength,
                CResultWriter& outputResults,
                const std::string& fileName,
                model::CAnomalyDetector& detector) {
    test::CTimeSeriesTestData::TTimeDoublePrVec timeData;
    BOOST_TEST_REQUIRE(test::CTimeSeriesTestData::parse(fileName, timeData));

    core_t::TTime lastBucketTime = maths::CIntegerTools::ceil(firstTime, bucketLength);

    for (std::size_t i = 0; i < timeData.size(); ++i) {
        core_t::TTime time = timeData[i].first;

        for (/**/; lastBucketTime + bucketLength <= time; lastBucketTime += bucketLength) {
            outputResults(detector, lastBucketTime, lastBucketTime + bucketLength);
        }

        std::string value = core::CStringUtils::typeToString(timeData[i].second);

        model::CAnomalyDetector::TStrCPtrVec fieldValues;
        fieldValues.push_back(&fileName);
        fieldValues.push_back(&value);
        detector.addRecord(time, fieldValues);
    }

    for (/**/; lastBucketTime + bucketLength <= lastTime; lastBucketTime += bucketLength) {
        outputResults(detector, lastBucketTime, lastBucketTime + bucketLength);
    }
}

void importCsvData(core_t::TTime firstTime,
                   core_t::TTime bucketLength,
                   CResultWriter& outputResults,
                   const std::string& fileName,
                   model::CAnomalyDetector& detector) {
    using TifstreamPtr = std::shared_ptr<std::ifstream>;
    TifstreamPtr ifs(new std::ifstream(fileName.c_str()));
    BOOST_TEST_REQUIRE(ifs->is_open());

    core::CRegex regex;
    BOOST_TEST_REQUIRE(regex.init(","));

    std::string line;
    // read the header
    BOOST_TEST_REQUIRE(std::getline(*ifs, line).good());

    core_t::TTime lastBucketTime = firstTime;

    while (std::getline(*ifs, line)) {
        LOG_TRACE(<< "Got string: " << line);
        core::CRegex::TStrVec tokens;
        regex.split(line, tokens);

        core_t::TTime time{0};
        BOOST_TEST_REQUIRE(core::CStringUtils::stringToType(tokens[0], time));

        for (/**/; lastBucketTime + bucketLength <= time; lastBucketTime += bucketLength) {
            outputResults(detector, lastBucketTime, lastBucketTime + bucketLength);
        }

        model::CAnomalyDetector::TStrCPtrVec fieldValues;
        fieldValues.push_back(&tokens[2]);
        fieldValues.push_back(&tokens[1]);

        detector.addRecord(time, fieldValues);
    }

    outputResults(detector, lastBucketTime, lastBucketTime + bucketLength);

    ifs.reset();
}

const std::string EMPTY_STRING;
}

BOOST_AUTO_TEST_CASE(testAnomalies) {
    // The test data has one genuine anomaly in the interval
    // [1360617335, 1360617481]. The rest of the samples are
    // Gaussian with mean 30 and standard deviation 5. The
    // arrival rate it Poisson distributed with constant mean
    // in each of the 24 hour periods. However, the rate varies
    // from hour to hour. In particular, the mean rates are:
    //
    //   Interval        |  Mean
    // ------------------+--------
    //  [ 0hr - 1hr ]    |  1000
    //  [ 1hr - 2hr ]    |  1000
    //  [ 2hr - 3hr ]    |  1200
    //  [ 3hr - 4hr ]    |  1400
    //  [ 4hr - 5hr ]    |  1600
    //  [ 5hr - 6hr ]    |  1800
    //  [ 6hr - 7hr ]    |  500
    //  [ 7hr - 8hr ]    |  200
    //  [ 8hr - 9hr ]    |  100
    //  [ 9hr - 10hr ]   |  50
    //  [ 10hr - 11hr ]  |  50
    //  [ 11hr - 12hr ]  |  50
    //  [ 12hr - 13hr ]  |  50
    //  [ 13hr - 14hr ]  |  50
    //  [ 14hr - 15hr ]  |  50
    //  [ 15hr - 16hr ]  |  50
    //  [ 16hr - 17hr ]  |  50
    //  [ 17hr - 18hr ]  |  100
    //  [ 18hr - 19hr ]  |  200
    //  [ 19hr - 20hr ]  |  500
    //  [ 20hr - 21hr ]  |  1500
    //  [ 21hr - 22hr ]  |  1400
    //  [ 22hr - 23hr ]  |  1000
    //  [ 23hr - 24hr ]  |  1000

    static const core_t::TTime FIRST_TIME(1360540800);
    static const core_t::TTime LAST_TIME(FIRST_TIME + 86400);
    static const core_t::TTime BUCKET_LENGTHS[] = {120, 150, 180, 210, 240,
                                                   300, 450, 600, 900, 1200};
    static const TTimeTimePr ANOMALOUS_INTERVALS[] = {
        TTimeTimePr(1360576852, 1360578629), TTimeTimePr(1360617335, 1360617481)};

    double highRateNoise = 0.0;
    double lowRateNoise = 0.0;

    for (auto bucketLength : BUCKET_LENGTHS) {
        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(bucketLength);
        modelConfig.useMultibucketFeatures(false);
        model::CLimits limits;
        model::CSearchKey key(1, // detectorIndex
                              model::function_t::E_IndividualMetric, false,
                              model_t::E_XF_None, "n/a", "n/a");
        model::CAnomalyDetector detector(limits, modelConfig, "", FIRST_TIME,
                                         modelConfig.factory(key));
        CResultWriter writer(modelConfig, limits, bucketLength);

        importData(FIRST_TIME, LAST_TIME, bucketLength, writer,
                   "testfiles/variable_rate_metric.data", detector);

        TTimeTimePrVec highAnomalyTimes(writer.highAnomalyTimes());
        TDoubleVec highAnomalyFactors(writer.highAnomalyFactors());
        TDoubleVec anomalyFactors(writer.anomalyFactors());
        TDoubleVec anomalyRates(writer.anomalyRates());

        LOG_DEBUG(<< "bucket length = " << bucketLength);
        LOG_DEBUG(<< "high anomalies in = "
                  << core::CContainerPrinter::print(highAnomalyTimes));
        LOG_DEBUG(<< "high anomaly factors = "
                  << core::CContainerPrinter::print(highAnomalyFactors));
        LOG_DEBUG(<< "anomaly factors = " << core::CContainerPrinter::print(anomalyFactors));
        LOG_DEBUG(<< "anomaly rates = " << core::CContainerPrinter::print(anomalyRates));

        for (std::size_t j = 0; j < highAnomalyTimes.size(); ++j) {
            LOG_DEBUG(<< "Testing " << core::CContainerPrinter::print(highAnomalyTimes[j])
                      << ' ' << highAnomalyFactors[j]);
            BOOST_TEST_REQUIRE(
                (doIntersect(highAnomalyTimes[j], ANOMALOUS_INTERVALS[0]) ||
                 doIntersect(highAnomalyTimes[j], ANOMALOUS_INTERVALS[1])));
        }

        if (!anomalyFactors.empty()) {
            double signal = std::accumulate(highAnomalyFactors.begin(),
                                            highAnomalyFactors.end(), 0.0);
            double noise = std::accumulate(anomalyFactors.begin(),
                                           anomalyFactors.end(), 0.0);
            LOG_DEBUG(<< "S/N = " << (signal / noise));
            BOOST_TEST_REQUIRE(signal / noise > 33.0);
        }

        // Find the high/low rate partition point.
        TDoubleVec orderedAnomalyRates(anomalyRates);
        std::sort(orderedAnomalyRates.begin(), orderedAnomalyRates.end());
        std::size_t maxStep = 1;
        for (std::size_t j = 2; j < orderedAnomalyRates.size(); ++j) {
            if (orderedAnomalyRates[j] - orderedAnomalyRates[j - 1] >
                orderedAnomalyRates[maxStep] - orderedAnomalyRates[maxStep - 1]) {
                maxStep = j;
            }
        }
        double partitionRate = 0.0;
        if (maxStep < orderedAnomalyRates.size()) {
            partitionRate = 0.5 * (orderedAnomalyRates[maxStep] +
                                   orderedAnomalyRates[maxStep - 1]);
        }
        LOG_DEBUG(<< "partition rate = " << partitionRate);

        // Compute the ratio of noise in the two rate channels.
        for (std::size_t j = 0; j < anomalyFactors.size(); ++j) {
            (anomalyRates[j] > partitionRate ? highRateNoise : lowRateNoise) +=
                anomalyFactors[j];
        }
    }

    LOG_DEBUG(<< "high rate noise = " << highRateNoise << ", low rate noise = " << lowRateNoise);

    // We don't have significantly more noise in the low rate channel.
    BOOST_TEST_REQUIRE(lowRateNoise / highRateNoise < 1.5);
}

BOOST_AUTO_TEST_CASE(testPersist) {
    static const core_t::TTime FIRST_TIME(1360540800);
    static const core_t::TTime LAST_TIME(FIRST_TIME + 86400);
    static const core_t::TTime BUCKET_LENGTH(300);

    core::CProgramCounters& counters = core::CProgramCounters::instance();
    model::CAnomalyDetectorModelConfig modelConfig =
        model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_LENGTH);
    model::CLimits limits;
    model::CSearchKey key(1, // detectorIndex
                          model::function_t::E_IndividualMetric, false,
                          model_t::E_XF_None, "responsetime", "Airline");
    model::CAnomalyDetector origDetector(limits, modelConfig, EMPTY_STRING,
                                         FIRST_TIME, modelConfig.factory(key));
    CResultWriter writer(modelConfig, limits, BUCKET_LENGTH);

    importData(FIRST_TIME, LAST_TIME, BUCKET_LENGTH, writer,
               "testfiles/variable_rate_metric.data", origDetector);

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origDetector.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    std::string origStaticsXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        core::CProgramCounters::staticsAcceptPersistInserter(inserter);
        inserter.toXml(origStaticsXml);
    }

    LOG_TRACE(<< "Event rate detector XML representation:\n" << origXml);

    uint64_t peakMemoryUsageBeforeRestoring = counters.counter(counter_t::E_TSADPeakMemoryUsage);

    // Clear the counter to verify that restoring detector also restores counter's value.
    counters.counter(counter_t::E_TSADPeakMemoryUsage) = 0;
    BOOST_REQUIRE_EQUAL(0, counters.counter(counter_t::E_TSADPeakMemoryUsage));

    // Restore the XML into a new detector
    model::CAnomalyDetector restoredDetector(limits, modelConfig, EMPTY_STRING,
                                             0, modelConfig.factory(key));
    {
        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        BOOST_TEST_REQUIRE(traverser.traverseSubLevel(
            std::bind(&model::CAnomalyDetector::acceptRestoreTraverser,
                      &restoredDetector, EMPTY_STRING, std::placeholders::_1)));
    }

    {
        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origStaticsXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        BOOST_TEST_REQUIRE(traverser.traverseSubLevel(
            &core::CProgramCounters::staticsAcceptRestoreTraverser));
    }

    uint64_t peakMemoryUsageAfterRestoring = counters.counter(counter_t::E_TSADPeakMemoryUsage);
    BOOST_REQUIRE_EQUAL(peakMemoryUsageBeforeRestoring, peakMemoryUsageAfterRestoring);

    // The XML representation of the new detector should be the same as the original
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredDetector.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    BOOST_REQUIRE_EQUAL(origXml, newXml);
}

BOOST_AUTO_TEST_CASE(testExcludeFrequent) {
    static const core_t::TTime FIRST_TIME(1406916000);
    static const core_t::TTime BUCKET_LENGTH(3600);

    {
        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_LENGTH);
        modelConfig.useMultibucketFeatures(false);
        model::CLimits limits;
        model::CSearchKey key(1, // detectorIndex
                              model::function_t::E_IndividualMetric, false,
                              model_t::E_XF_None, "bytes", "host");
        model::CAnomalyDetector detector(limits, modelConfig, "", FIRST_TIME,
                                         modelConfig.factory(key));
        CResultWriter writer(modelConfig, limits, BUCKET_LENGTH);

        importCsvData(FIRST_TIME, BUCKET_LENGTH, writer,
                      "testfiles/excludefrequent_two_series.txt", detector);

        TTimeTimePrVec highAnomalyTimes(writer.highAnomalyTimes());
        TDoubleVec highAnomalyFactors(writer.highAnomalyFactors());

        LOG_DEBUG(<< "high anomalies in = "
                  << core::CContainerPrinter::print(highAnomalyTimes));
        LOG_DEBUG(<< "high anomaly factors = "
                  << core::CContainerPrinter::print(highAnomalyFactors));

        // expect there to be 2 anomalies
        BOOST_REQUIRE_EQUAL(std::size_t(2), highAnomalyTimes.size());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(99.0, highAnomalyFactors[1], 2.0);
    }
    {
        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_LENGTH);
        modelConfig.useMultibucketFeatures(false);
        model::CLimits limits;
        model::CSearchKey key(1, // detectorIndex
                              model::function_t::E_IndividualMetric, false,
                              model_t::E_XF_By, "bytes", "host");
        model::CAnomalyDetector detector(limits, modelConfig, "", FIRST_TIME,
                                         modelConfig.factory(key));
        CResultWriter writer(modelConfig, limits, BUCKET_LENGTH);

        importCsvData(FIRST_TIME, BUCKET_LENGTH, writer,
                      "testfiles/excludefrequent_two_series.txt", detector);

        TTimeTimePrVec highAnomalyTimes(writer.highAnomalyTimes());
        TDoubleVec highAnomalyFactors(writer.highAnomalyFactors());

        LOG_DEBUG(<< "high anomalies in = "
                  << core::CContainerPrinter::print(highAnomalyTimes));
        LOG_DEBUG(<< "high anomaly factors = "
                  << core::CContainerPrinter::print(highAnomalyFactors));

        // expect there to be 1 anomaly
        BOOST_REQUIRE_EQUAL(std::size_t(1), highAnomalyTimes.size());
        BOOST_REQUIRE_CLOSE_ABSOLUTE(12.0, highAnomalyFactors[0], 2.0);
    }
}

BOOST_AUTO_TEST_SUITE_END()
