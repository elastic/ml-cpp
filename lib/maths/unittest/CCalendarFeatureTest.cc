/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CTimeUtils.h>
#include <core/CTimezone.h>
#include <core/CoreTypes.h>

#include <maths/CCalendarFeature.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <vector>

BOOST_AUTO_TEST_SUITE(CCalendarFeatureTest)

using namespace ml;

using TSizeVec = std::vector<std::size_t>;
using TTimeVec = std::vector<core_t::TTime>;
using TCalendarFeatureVec = std::vector<maths::CCalendarFeature>;

namespace {
const core_t::TTime DAY = 86400;

//! \brief Sets the timezone to GMT for the lifetime of the objct.
class CScopeGMT {
public:
    CScopeGMT() {
        m_Timezone = core::CTimezone::instance().timezoneName();
        core::CTimezone::instance().timezoneName("GMT");
    }
    ~CScopeGMT() { core::CTimezone::instance().timezoneName(m_Timezone); }

private:
    std::string m_Timezone;
};
}

BOOST_AUTO_TEST_CASE(testInitialize) {
    // Check we get the expected features.

    test::CRandomNumbers rng;

    TSizeVec times;
    rng.generateUniformSamples(100000, 10000000, 10, times);

    for (std::size_t i = 0; i < times.size(); ++i) {
        core_t::TTime time{static_cast<core_t::TTime>(times[i])};

        maths::CCalendarFeature::TCalendarFeature4Ary expected;
        expected[0] = maths::CCalendarFeature(
            maths::CCalendarFeature::DAYS_SINCE_START_OF_MONTH, time);
        expected[1] = maths::CCalendarFeature(
            maths::CCalendarFeature::DAYS_BEFORE_END_OF_MONTH, time);
        expected[2] = maths::CCalendarFeature(
            maths::CCalendarFeature::DAY_OF_WEEK_AND_WEEKS_SINCE_START_OF_MONTH, time);
        expected[3] = maths::CCalendarFeature(
            maths::CCalendarFeature::DAY_OF_WEEK_AND_WEEKS_BEFORE_END_OF_MONTH, time);

        maths::CCalendarFeature::TCalendarFeature4Ary actual =
            maths::CCalendarFeature::features(time);

        BOOST_TEST_REQUIRE(expected == actual);
    }
}

BOOST_AUTO_TEST_CASE(testComparison) {
    // Check some comparison invariants.

    test::CRandomNumbers rng;

    TSizeVec times;
    for (core_t::TTime time = 1; time < 31 * DAY; time += DAY) {
        times.push_back(time);
    }

    LOG_DEBUG(<< "times = " << core::CContainerPrinter::print(times));

    TCalendarFeatureVec features;

    for (std::size_t i = 0; i < times.size(); ++i) {
        core_t::TTime time{static_cast<core_t::TTime>(times[i])};
        maths::CCalendarFeature::TCalendarFeature4Ary fi =
            maths::CCalendarFeature::features(time);
        features.insert(features.end(), fi.begin(), fi.end());
    }

    std::sort(features.begin(), features.end());
    features.erase(std::unique(features.begin(), features.end()), features.end());

    for (std::size_t i = 0; i < features.size(); ++i) {
        BOOST_TEST_REQUIRE(features[i] == features[i]);
        BOOST_REQUIRE(!(features[i] < features[i] || features[i] > features[i]));
        for (std::size_t j = i + 1; j < features.size(); ++j) {
            BOOST_TEST_REQUIRE(features[i] != features[j]);
            BOOST_REQUIRE(features[i] < features[j] || features[i] > features[j]);
        }
    }
}

BOOST_AUTO_TEST_CASE(testOffset) {
    // Check some properties of offset. Specifically,
    //    - offset(time + delta) = offset(time) + delta provided
    //      times are in same month except when the delta crosses
    //      daylight saving.
    //    - we get the expected offset for some known dates.

    test::CRandomNumbers rng;

    core_t::TTime start = core::CTimeUtils::now() - 30000000;
    LOG_DEBUG(<< "start = " << start);

    TSizeVec times;
    rng.generateUniformSamples(0, 30000000, 1000, times);

    std::size_t tests = 0;

    for (const auto& time_ : times) {
        core_t::TTime time{start + static_cast<core_t::TTime>(time_)};
        maths::CCalendarFeature::TCalendarFeature4Ary features =
            maths::CCalendarFeature::features(time);

        int dummy;
        int month;
        core::CTimezone::instance().dateFields(time, dummy, dummy, dummy, month, dummy, dummy);

        TTimeVec offsets{-86400, -43400, -12800, -3600, 0,
                         3600,   12800,  43400,  86400};

        for (const auto& offset : offsets) {
            core_t::TTime offsetTime = time + offset;
            int offsetMonth;
            core::CTimezone::instance().dateFields(offsetTime, dummy, dummy, dummy,
                                                   offsetMonth, dummy, dummy);

            if (month == offsetMonth) {
                for (const auto& feature : features) {
                    BOOST_TEST_REQUIRE(
                        (feature.offset(time) + offset == feature.offset(offsetTime) ||
                         feature.offset(time) + offset == feature.offset(offsetTime) - 3600 ||
                         feature.offset(time) + offset == feature.offset(offsetTime) + 3600));
                    ++tests;
                }
            }
        }
    }

    CScopeGMT gmt;

    LOG_DEBUG(<< "# tests = " << tests);
    BOOST_TEST_REQUIRE(tests > 30000);

    core_t::TTime feb1st = 31 * DAY;
    core_t::TTime march1st = feb1st + 28 * DAY;
    core_t::TTime april1st = march1st + 31 * DAY;
    core_t::TTime may1st = april1st + 30 * DAY;

    LOG_DEBUG(<< "Test days since start of month");
    {
        maths::CCalendarFeature feature(
            maths::CCalendarFeature::DAYS_SINCE_START_OF_MONTH, feb1st);
        for (core_t::TTime time = march1st; time < april1st; time += DAY) {
            BOOST_REQUIRE_EQUAL(time - march1st, feature.offset(time));
            BOOST_REQUIRE_EQUAL(time - march1st + 4800, feature.offset(time + 4800));
        }
    }
    {
        maths::CCalendarFeature feature(
            maths::CCalendarFeature::DAYS_SINCE_START_OF_MONTH, feb1st + 12 * DAY);
        for (core_t::TTime time = march1st; time < april1st; time += DAY) {
            BOOST_REQUIRE_EQUAL(time - march1st - 12 * DAY, feature.offset(time));
            BOOST_REQUIRE_EQUAL(time - march1st - 12 * DAY + 43400,
                                feature.offset(time + 43400));
        }
    }

    LOG_DEBUG(<< "Test days before end of month");
    {
        maths::CCalendarFeature feature(
            maths::CCalendarFeature::DAYS_BEFORE_END_OF_MONTH, feb1st);
        for (core_t::TTime time = march1st; time < april1st; time += DAY) {
            BOOST_REQUIRE_EQUAL(time - march1st - 3 * DAY, feature.offset(time));
            BOOST_REQUIRE_EQUAL(time - march1st - 3 * DAY + 7200,
                                feature.offset(time + 7200));
        }
    }
    {
        maths::CCalendarFeature feature(
            maths::CCalendarFeature::DAYS_BEFORE_END_OF_MONTH, feb1st + 10 * DAY);
        for (core_t::TTime time = march1st; time < april1st; time += DAY) {
            BOOST_REQUIRE_EQUAL(time - march1st - 13 * DAY, feature.offset(time));
            BOOST_REQUIRE_EQUAL(time - march1st - 13 * DAY + 86399,
                                feature.offset(time + 86399));
        }
    }

    LOG_DEBUG(<< "Test day of week and week of month");
    {
        // Feb 1 1970 is a Sunday and April 1st 1970 is a Wednesday.
        maths::CCalendarFeature feature(
            maths::CCalendarFeature::DAY_OF_WEEK_AND_WEEKS_SINCE_START_OF_MONTH, feb1st);
        for (core_t::TTime time = april1st; time < may1st; time += DAY) {
            BOOST_REQUIRE_EQUAL(time - april1st - 4 * DAY, feature.offset(time));
            BOOST_REQUIRE_EQUAL(time - april1st - 4 * DAY + 7200,
                                feature.offset(time + 7200));
        }
    }
    {
        // Feb 13 1970 is a Friday and April 1st 1970 is a Wednesday.
        maths::CCalendarFeature feature(maths::CCalendarFeature::DAY_OF_WEEK_AND_WEEKS_SINCE_START_OF_MONTH,
                                        feb1st + 12 * DAY);
        for (core_t::TTime time = april1st; time < may1st; time += DAY) {
            BOOST_REQUIRE_EQUAL(time - april1st - 9 * DAY, feature.offset(time));
            BOOST_REQUIRE_EQUAL(time - april1st - 9 * DAY + 73000,
                                feature.offset(time + 73000));
        }
    }

    LOG_DEBUG(<< "Test day of week and week until end of month");
    {
        // Feb 1 1970 is a Sunday and April 31st 1970 is a Friday.
        maths::CCalendarFeature feature(
            maths::CCalendarFeature::DAY_OF_WEEK_AND_WEEKS_BEFORE_END_OF_MONTH, feb1st);
        for (core_t::TTime time = april1st; time < may1st; time += DAY) {
            BOOST_REQUIRE_EQUAL(time - april1st - 4 * DAY, feature.offset(time));
            BOOST_REQUIRE_EQUAL(time - april1st - 4 * DAY + 7200,
                                feature.offset(time + 7200));
        }
    }
    {
        // Feb 13 1970 is a Friday and April 1st 1970 is a Wednesday.
        maths::CCalendarFeature feature(maths::CCalendarFeature::DAY_OF_WEEK_AND_WEEKS_SINCE_START_OF_MONTH,
                                        feb1st + 12 * DAY);
        for (core_t::TTime time = april1st; time < may1st; time += DAY) {
            BOOST_REQUIRE_EQUAL(time - april1st - 9 * DAY, feature.offset(time));
            BOOST_REQUIRE_EQUAL(time - april1st - 9 * DAY + 73000,
                                feature.offset(time + 73000));
        }
    }
}

BOOST_AUTO_TEST_CASE(testPersist) {
    maths::CCalendarFeature::TCalendarFeature4Ary features =
        maths::CCalendarFeature::features(core::CTimeUtils::now());

    for (std::size_t i = 0; i < 4; ++i) {
        std::string state = features[i].toDelimited();
        LOG_DEBUG(<< "state = " << state);

        maths::CCalendarFeature restored;
        restored.fromDelimited(state);

        BOOST_REQUIRE_EQUAL(features[i].checksum(), restored.checksum());
        BOOST_REQUIRE_EQUAL(state, restored.toDelimited());
    }
}

BOOST_AUTO_TEST_SUITE_END()
