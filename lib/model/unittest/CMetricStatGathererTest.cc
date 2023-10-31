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

#include <core/CLogger.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CoreTypes.h>

#include <maths/common/COrderings.h>

#include <model/CMetricStatGatherer.h>
#include <model/CStringStore.h>

#include <test/BoostTestCloseAbsolute.h>

#include <boost/test/unit_test.hpp>

#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

BOOST_AUTO_TEST_SUITE(CMetricStatGathererTest)

using namespace ml;
using namespace model;
using TStrVec = std::vector<std::string>;

template<typename DATA>
std::string printInfluencers(const DATA& featureData) {
    auto influencers = featureData.s_InfluenceValues;
    for (auto& influencer : influencers) {
        std::sort(influencer.begin(), influencer.end(),
                  maths::common::COrderings::SFirstLess{});
    }
    std::ostringstream result;
    result << ml::core::CScopePrintContainers{} << influencers;
    return result.str();
}

BOOST_AUTO_TEST_CASE(testSumGatherer) {

    core_t::TTime time{1672531200};
    TStrVec empty;

    auto assertFirstBucketStats = [&](const TSumGatherer& gatherer) {
        auto data = gatherer.featureData(time + 599);
        BOOST_REQUIRE_EQUAL(true, data.s_IsNonNegative);
        BOOST_REQUIRE_EQUAL(false, data.s_IsInteger);
        BOOST_REQUIRE_EQUAL(1, data.s_Samples.size());
        BOOST_REQUIRE_EQUAL(true, data.s_BucketValue != std::nullopt);
        BOOST_REQUIRE_EQUAL(1672531500, data.s_BucketValue->time());
        BOOST_REQUIRE_EQUAL(1.0, data.s_BucketValue->varianceScale());
        BOOST_REQUIRE_EQUAL(5.4, data.s_BucketValue->value()[0]);
        BOOST_REQUIRE_EQUAL(4, data.s_BucketValue->count());
        BOOST_REQUIRE_EQUAL(1672531500, data.s_Samples[0].time());
        BOOST_REQUIRE_EQUAL(1.0, data.s_Samples[0].varianceScale());
        BOOST_REQUIRE_EQUAL(5.4, data.s_Samples[0].value()[0]);
        BOOST_REQUIRE_EQUAL(4, data.s_Samples[0].count());
    };

    auto assertSecondBucketStats = [&](const TSumGatherer& gatherer) {
        auto data = gatherer.featureData(time + 1199);
        BOOST_REQUIRE_EQUAL(true, data.s_IsNonNegative);
        BOOST_REQUIRE_EQUAL(false, data.s_IsInteger);
        BOOST_REQUIRE_EQUAL(1, data.s_Samples.size());
        BOOST_REQUIRE_EQUAL(true, data.s_BucketValue != std::nullopt);
        BOOST_REQUIRE_EQUAL(1672532100, data.s_BucketValue->time());
        BOOST_REQUIRE_EQUAL(1.0, data.s_BucketValue->varianceScale());
        BOOST_REQUIRE_EQUAL(4.4, data.s_BucketValue->value()[0]);
        BOOST_REQUIRE_EQUAL(3, data.s_BucketValue->count());
        BOOST_REQUIRE_EQUAL(1672532100, data.s_Samples[0].time());
        BOOST_REQUIRE_EQUAL(1.0, data.s_Samples[0].varianceScale());
        BOOST_REQUIRE_EQUAL(4.4, data.s_Samples[0].value()[0]);
        BOOST_REQUIRE_EQUAL(3, data.s_Samples[0].count());
    };

    for (std::size_t i = 1; i <= 2; ++i) {
        TSumGatherer gatherer{i, 1, time, 600, empty.begin(), empty.end()};

        gatherer.add(time + 30, {1.3}, 1, {});
        gatherer.add(time + 70, {3.1}, 2, {});
        gatherer.add(time + 122, {1.0}, 1, {});

        assertFirstBucketStats(gatherer);
        BOOST_REQUIRE_EQUAL(
            true, gatherer.featureData(time + 599).s_InfluenceValues.empty());

        gatherer.startNewBucket(time + 600);
        gatherer.add(time + 630, {0.3}, 1, {});
        gatherer.add(time + 670, {2.1}, 1, {});
        gatherer.add(time + 722, {2.0}, 1, {});

        assertSecondBucketStats(gatherer);
        BOOST_REQUIRE_EQUAL(
            true, gatherer.featureData(time + 1199).s_InfluenceValues.empty());
        if (i == 2) {
            assertFirstBucketStats(gatherer);
            BOOST_REQUIRE_EQUAL(
                true, gatherer.featureData(time + 599).s_InfluenceValues.empty());
        }
    }

    for (std::size_t i = 1; i <= 2; ++i) {
        TStrVec influencers{"i1", "i2"};
        TSumGatherer gatherer{
            i, 1, time, 600, influencers.begin(), influencers.end()};

        gatherer.add(time + 30, {1.3}, 1,
                     {model::CStringStore::influencers().get("i11"),
                      model::CStringStore::influencers().get("i21")});
        gatherer.add(time + 70, {3.1}, 2,
                     {model::CStringStore::influencers().get("i11"),
                      model::CStringStore::influencers().get("i22")});
        gatherer.add(time + 122, {1.0}, 1,
                     {model::CStringStore::influencers().get("i12"),
                      model::CStringStore::influencers().get("i21")});

        assertFirstBucketStats(gatherer);
        BOOST_REQUIRE_EQUAL("[[(i11, ([4.4], 3)), (i12, ([1], 1))], [(i21, ([2.3], 2)), (i22, ([3.1], 2))]]",
                            printInfluencers(gatherer.featureData(time + 599)));

        gatherer.startNewBucket(time + 600);
        gatherer.add(time + 630, {0.3}, 1,
                     {model::CStringStore::influencers().get("i11"),
                      model::CStringStore::influencers().get("i21")});
        gatherer.add(time + 670, {2.1}, 1,
                     {model::CStringStore::influencers().get("i11"),
                      model::CStringStore::influencers().get("i22")});
        gatherer.add(time + 722, {2.0}, 1,
                     {model::CStringStore::influencers().get("i12"),
                      model::CStringStore::influencers().get("i21")});

        assertSecondBucketStats(gatherer);
        BOOST_REQUIRE_EQUAL("[[(i11, ([2.4], 2)), (i12, ([2], 1))], [(i21, ([2.3], 2)), (i22, ([2.1], 1))]]",
                            printInfluencers(gatherer.featureData(time + 1199)));
        if (i == 2) {
            assertFirstBucketStats(gatherer);
            BOOST_REQUIRE_EQUAL("[[(i11, ([4.4], 3)), (i12, ([1], 1))], [(i21, ([2.3], 2)), (i22, ([3.1], 2))]]",
                                printInfluencers(gatherer.featureData(time + 599)));
        }

        std::string xml;
        core::CRapidXmlStatePersistInserter inserter("root");
        gatherer.acceptPersistInserter(inserter);
        inserter.toXml(xml);
        LOG_TRACE(<< xml);
        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(xml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        TSumGatherer restoredGatherer{
            i, 1, 1672531200, 600, influencers.begin(), influencers.end()};
        BOOST_TEST_REQUIRE(traverser.traverseSubLevel([&](auto& traverser_) {
            return restoredGatherer.acceptRestoreTraverser(traverser_);
        }));
        BOOST_REQUIRE_EQUAL(gatherer.checksum(), restoredGatherer.checksum());
    }
}

BOOST_AUTO_TEST_CASE(testMeanGatherer) {

    core_t::TTime time{1672531200};

    auto assertFirstBucketStats = [&](const TMeanGatherer& gatherer) {
        auto data = gatherer.featureData(time + 599);
        BOOST_REQUIRE_EQUAL(true, data.s_IsNonNegative);
        BOOST_REQUIRE_EQUAL(false, data.s_IsInteger);
        BOOST_REQUIRE_EQUAL(1, data.s_Samples.size());
        BOOST_REQUIRE_EQUAL(true, data.s_BucketValue != std::nullopt);
        BOOST_REQUIRE_EQUAL(1672531273, data.s_BucketValue->time());
        BOOST_REQUIRE_EQUAL(1.0, data.s_BucketValue->varianceScale());
        BOOST_REQUIRE_CLOSE(2.125, data.s_BucketValue->value()[0], 1e-6);
        BOOST_REQUIRE_EQUAL(4, data.s_BucketValue->count());
        BOOST_REQUIRE_EQUAL(1672531273, data.s_Samples[0].time());
        BOOST_REQUIRE_EQUAL(1.0, data.s_Samples[0].varianceScale());
        BOOST_REQUIRE_CLOSE(2.125, data.s_Samples[0].value()[0], 1e-6);
        BOOST_REQUIRE_EQUAL(4, data.s_Samples[0].count());
        BOOST_REQUIRE_EQUAL("[[(i11, ([2.5], 3)), (i12, ([1], 1))], [(i21, ([1.15], 2)), (i22, ([3.1], 2))]]",
                            printInfluencers(gatherer.featureData(time + 599)));
    };

    auto assertSecondBucketStats = [&](const TMeanGatherer& gatherer) {
        auto data = gatherer.featureData(time + 1199);
        BOOST_REQUIRE_EQUAL(true, data.s_IsNonNegative);
        BOOST_REQUIRE_EQUAL(false, data.s_IsInteger);
        BOOST_REQUIRE_EQUAL(1, data.s_Samples.size());
        BOOST_REQUIRE_EQUAL(true, data.s_BucketValue != std::nullopt);
        BOOST_REQUIRE_EQUAL(1672531874, data.s_BucketValue->time());
        BOOST_REQUIRE_EQUAL(1.0, data.s_BucketValue->varianceScale());
        BOOST_REQUIRE_EQUAL(1.1, data.s_BucketValue->value()[0]);
        BOOST_REQUIRE_EQUAL(3, data.s_BucketValue->count());
        BOOST_REQUIRE_EQUAL(1672531874, data.s_Samples[0].time());
        BOOST_REQUIRE_EQUAL(1.0, data.s_Samples[0].varianceScale());
        BOOST_REQUIRE_EQUAL(1.1, data.s_Samples[0].value()[0]);
        BOOST_REQUIRE_EQUAL(3, data.s_Samples[0].count());
        BOOST_REQUIRE_EQUAL("[[(i11, ([0.65], 2)), (i12, ([2], 1))], [(i21, ([1.15], 2)), (i22, ([1], 1))]]",
                            printInfluencers(gatherer.featureData(time + 1199)));
    };

    for (std::size_t i = 1; i <= 2; ++i) {
        TStrVec influencers{"i1", "i2"};
        TMeanGatherer gatherer{
            i, 1, time, 600, influencers.begin(), influencers.end()};

        gatherer.add(time + 30, {1.3}, 1,
                     {model::CStringStore::influencers().get("i11"),
                      model::CStringStore::influencers().get("i21")});
        gatherer.add(time + 70, {3.1}, 2,
                     {model::CStringStore::influencers().get("i11"),
                      model::CStringStore::influencers().get("i22")});
        gatherer.add(time + 122, {1.0}, 1,
                     {model::CStringStore::influencers().get("i12"),
                      model::CStringStore::influencers().get("i21")});

        assertFirstBucketStats(gatherer);

        gatherer.startNewBucket(time + 600);
        gatherer.add(time + 630, {0.3}, 1,
                     {model::CStringStore::influencers().get("i11"),
                      model::CStringStore::influencers().get("i21")});
        gatherer.add(time + 670, {1.0}, 1,
                     {model::CStringStore::influencers().get("i11"),
                      model::CStringStore::influencers().get("i22")});
        gatherer.add(time + 722, {2.0}, 1,
                     {model::CStringStore::influencers().get("i12"),
                      model::CStringStore::influencers().get("i21")});

        assertSecondBucketStats(gatherer);
        if (i == 2) {
            assertFirstBucketStats(gatherer);
        }

        std::string xml;
        core::CRapidXmlStatePersistInserter inserter("root");
        gatherer.acceptPersistInserter(inserter);
        inserter.toXml(xml);
        LOG_TRACE(<< xml);
        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(xml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        TMeanGatherer restoredGatherer{
            i, 1, 1672531200, 600, influencers.begin(), influencers.end()};
        BOOST_TEST_REQUIRE(traverser.traverseSubLevel([&](auto& traverser_) {
            return restoredGatherer.acceptRestoreTraverser(traverser_);
        }));
        BOOST_REQUIRE_EQUAL(gatherer.checksum(), restoredGatherer.checksum());
    }
}

BOOST_AUTO_TEST_CASE(testMinGatherer) {

    core_t::TTime time{1672531200};

    auto assertFirstBucketStats = [&](const TMinGatherer& gatherer) {
        auto data = gatherer.featureData(time + 599);
        BOOST_REQUIRE_EQUAL(true, data.s_IsNonNegative);
        BOOST_REQUIRE_EQUAL(false, data.s_IsInteger);
        BOOST_REQUIRE_EQUAL(1, data.s_Samples.size());
        BOOST_REQUIRE_EQUAL(true, data.s_BucketValue != std::nullopt);
        BOOST_REQUIRE_EQUAL(1672531322, data.s_BucketValue->time());
        BOOST_REQUIRE_EQUAL(1.0, data.s_BucketValue->varianceScale());
        BOOST_REQUIRE_EQUAL(1.0, data.s_BucketValue->value()[0]);
        BOOST_REQUIRE_EQUAL(1, data.s_BucketValue->count());
        BOOST_REQUIRE_EQUAL(1672531322, data.s_Samples[0].time());
        BOOST_REQUIRE_EQUAL(1.0, data.s_Samples[0].varianceScale());
        BOOST_REQUIRE_EQUAL(1.0, data.s_Samples[0].value()[0]);
        BOOST_REQUIRE_EQUAL(1, data.s_Samples[0].count());
        BOOST_REQUIRE_EQUAL("[[(i11, ([1.3], 1)), (i12, ([1], 1))], [(i21, ([1], 1)), (i22, ([3.1], 1))]]",
                            printInfluencers(gatherer.featureData(time + 599)));
    };

    auto assertSecondBucketStats = [&](const TMinGatherer& gatherer) {
        auto data = gatherer.featureData(time + 1199);
        BOOST_REQUIRE_EQUAL(true, data.s_IsNonNegative);
        BOOST_REQUIRE_EQUAL(false, data.s_IsInteger);
        BOOST_REQUIRE_EQUAL(1, data.s_Samples.size());
        BOOST_REQUIRE_EQUAL(true, data.s_BucketValue != std::nullopt);
        BOOST_REQUIRE_EQUAL(1672531830, data.s_BucketValue->time());
        BOOST_REQUIRE_EQUAL(1.0, data.s_BucketValue->varianceScale());
        BOOST_REQUIRE_EQUAL(0.3, data.s_BucketValue->value()[0]);
        BOOST_REQUIRE_EQUAL(1, data.s_BucketValue->count());
        BOOST_REQUIRE_EQUAL(1672531830, data.s_Samples[0].time());
        BOOST_REQUIRE_EQUAL(1.0, data.s_Samples[0].varianceScale());
        BOOST_REQUIRE_EQUAL(0.3, data.s_Samples[0].value()[0]);
        BOOST_REQUIRE_EQUAL(1, data.s_Samples[0].count());
        BOOST_REQUIRE_EQUAL("[[(i11, ([0.3], 1)), (i12, ([2], 1))], [(i21, ([0.3], 1)), (i22, ([1], 1))]]",
                            printInfluencers(gatherer.featureData(time + 1199)));
    };

    for (std::size_t i = 1; i <= 2; ++i) {
        TStrVec influencers{"i1", "i2"};
        TMinGatherer gatherer{
            i, 1, time, 600, influencers.begin(), influencers.end()};

        gatherer.add(time + 30, {1.3}, 1,
                     {model::CStringStore::influencers().get("i11"),
                      model::CStringStore::influencers().get("i21")});
        gatherer.add(time + 70, {3.1}, 2,
                     {model::CStringStore::influencers().get("i11"),
                      model::CStringStore::influencers().get("i22")});
        gatherer.add(time + 122, {1.0}, 1,
                     {model::CStringStore::influencers().get("i12"),
                      model::CStringStore::influencers().get("i21")});

        assertFirstBucketStats(gatherer);

        gatherer.startNewBucket(time + 600);
        gatherer.add(time + 630, {0.3}, 1,
                     {model::CStringStore::influencers().get("i11"),
                      model::CStringStore::influencers().get("i21")});
        gatherer.add(time + 670, {1.0}, 1,
                     {model::CStringStore::influencers().get("i11"),
                      model::CStringStore::influencers().get("i22")});
        gatherer.add(time + 722, {2.0}, 1,
                     {model::CStringStore::influencers().get("i12"),
                      model::CStringStore::influencers().get("i21")});

        assertSecondBucketStats(gatherer);
        if (i == 2) {
            assertFirstBucketStats(gatherer);
        }

        std::string xml;
        core::CRapidXmlStatePersistInserter inserter("root");
        gatherer.acceptPersistInserter(inserter);
        inserter.toXml(xml);
        LOG_TRACE(<< xml);
        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(xml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        TMinGatherer restoredGatherer{
            i, 1, 1672531200, 600, influencers.begin(), influencers.end()};
        BOOST_TEST_REQUIRE(traverser.traverseSubLevel([&](auto& traverser_) {
            return restoredGatherer.acceptRestoreTraverser(traverser_);
        }));
        BOOST_REQUIRE_EQUAL(gatherer.checksum(), restoredGatherer.checksum());
    }
}

BOOST_AUTO_TEST_CASE(testMaxGatherer) {

    core_t::TTime time{1672531200};

    auto assertFirstBucketStats = [&](const TMaxGatherer& gatherer) {
        auto data = gatherer.featureData(time + 599);
        BOOST_REQUIRE_EQUAL(true, data.s_IsNonNegative);
        BOOST_REQUIRE_EQUAL(false, data.s_IsInteger);
        BOOST_REQUIRE_EQUAL(1, data.s_Samples.size());
        BOOST_REQUIRE_EQUAL(true, data.s_BucketValue != std::nullopt);
        BOOST_REQUIRE_EQUAL(1672531270, data.s_BucketValue->time());
        BOOST_REQUIRE_EQUAL(1.0, data.s_BucketValue->varianceScale());
        BOOST_REQUIRE_EQUAL(3.1, data.s_BucketValue->value()[0]);
        BOOST_REQUIRE_EQUAL(1, data.s_BucketValue->count());
        BOOST_REQUIRE_EQUAL(1672531270, data.s_Samples[0].time());
        BOOST_REQUIRE_EQUAL(1.0, data.s_Samples[0].varianceScale());
        BOOST_REQUIRE_EQUAL(3.1, data.s_Samples[0].value()[0]);
        BOOST_REQUIRE_EQUAL(1, data.s_Samples[0].count());
        BOOST_REQUIRE_EQUAL("[[(i11, ([3.1], 1)), (i12, ([1], 1))], [(i21, ([1.3], 1)), (i22, ([3.1], 1))]]",
                            printInfluencers(gatherer.featureData(time + 599)));
    };

    auto assertSecondBucketStats = [&](const TMaxGatherer& gatherer) {
        auto data = gatherer.featureData(time + 1199);
        BOOST_REQUIRE_EQUAL(true, data.s_IsNonNegative);
        BOOST_REQUIRE_EQUAL(false, data.s_IsInteger);
        BOOST_REQUIRE_EQUAL(1, data.s_Samples.size());
        BOOST_REQUIRE_EQUAL(true, data.s_BucketValue != std::nullopt);
        BOOST_REQUIRE_EQUAL(1672531922, data.s_BucketValue->time());
        BOOST_REQUIRE_EQUAL(1.0, data.s_BucketValue->varianceScale());
        BOOST_REQUIRE_EQUAL(2.0, data.s_BucketValue->value()[0]);
        BOOST_REQUIRE_EQUAL(1, data.s_BucketValue->count());
        BOOST_REQUIRE_EQUAL(1672531922, data.s_Samples[0].time());
        BOOST_REQUIRE_EQUAL(1.0, data.s_Samples[0].varianceScale());
        BOOST_REQUIRE_EQUAL(2.0, data.s_Samples[0].value()[0]);
        BOOST_REQUIRE_EQUAL(1, data.s_Samples[0].count());
        BOOST_REQUIRE_EQUAL("[[(i11, ([1], 1)), (i12, ([2], 1))], [(i21, ([2], 1)), (i22, ([1], 1))]]",
                            printInfluencers(gatherer.featureData(time + 1199)));
    };

    for (std::size_t i = 1; i <= 2; ++i) {
        TStrVec influencers{"i1", "i2"};
        TMaxGatherer gatherer{
            i, 1, time, 600, influencers.begin(), influencers.end()};

        gatherer.add(time + 30, {1.3}, 1,
                     {model::CStringStore::influencers().get("i11"),
                      model::CStringStore::influencers().get("i21")});
        gatherer.add(time + 70, {3.1}, 2,
                     {model::CStringStore::influencers().get("i11"),
                      model::CStringStore::influencers().get("i22")});
        gatherer.add(time + 122, {1.0}, 1,
                     {model::CStringStore::influencers().get("i12"),
                      model::CStringStore::influencers().get("i21")});

        assertFirstBucketStats(gatherer);

        gatherer.startNewBucket(time + 600);
        gatherer.add(time + 630, {0.3}, 1,
                     {model::CStringStore::influencers().get("i11"),
                      model::CStringStore::influencers().get("i21")});
        gatherer.add(time + 670, {1.0}, 1,
                     {model::CStringStore::influencers().get("i11"),
                      model::CStringStore::influencers().get("i22")});
        gatherer.add(time + 722, {2.0}, 1,
                     {model::CStringStore::influencers().get("i12"),
                      model::CStringStore::influencers().get("i21")});

        assertSecondBucketStats(gatherer);
        if (i == 2) {
            assertFirstBucketStats(gatherer);
        }

        std::string xml;
        core::CRapidXmlStatePersistInserter inserter("root");
        gatherer.acceptPersistInserter(inserter);
        inserter.toXml(xml);
        LOG_TRACE(<< xml);
        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(xml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        TMaxGatherer restoredGatherer{
            i, 1, 1672531200, 600, influencers.begin(), influencers.end()};
        BOOST_TEST_REQUIRE(traverser.traverseSubLevel([&](auto& traverser_) {
            return restoredGatherer.acceptRestoreTraverser(traverser_);
        }));
        BOOST_REQUIRE_EQUAL(gatherer.checksum(), restoredGatherer.checksum());
    }
}

/*BOOST_AUTO_TEST_CASE(testMedianGatherer) {
}

BOOST_AUTO_TEST_CASE(testVarianceGatherer) {
}

BOOST_AUTO_TEST_CASE(testMultivariateMeanGatherer) {
}*/

BOOST_AUTO_TEST_SUITE_END()
}
