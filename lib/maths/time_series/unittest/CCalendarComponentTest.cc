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
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CTimezone.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CTools.h>
#include <maths/common/MathsTypes.h>

#include <maths/time_series/CCalendarComponent.h>
#include <maths/time_series/CCalendarFeature.h>

#include <test/CRandomNumbers.h>

#include "TestUtils.h"

#include <boost/test/unit_test.hpp>

#include <vector>

BOOST_AUTO_TEST_SUITE(CCalendarComponentTest)

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;
using TTimeVec = std::vector<core_t::TTime>;
using TMeanAccumulator = maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator;

class CTestFixture {
public:
    CTestFixture()
        : m_OrigTimezone{core::CTimezone::instance().timezoneName()} {
        core::CTimezone::instance().setTimezone("GMT");
    }

    ~CTestFixture() { core::CTimezone::instance().setTimezone(m_OrigTimezone); }

private:
    std::string m_OrigTimezone;
};

class CTestCalendarComponent : public maths::time_series::CCalendarComponent {
public:
    // Bring base class method hidden by the signature above into scope.
    using maths::time_series::CCalendarComponent::initialize;

public:
    CTestCalendarComponent(const maths::time_series::CCalendarFeature& feature,
                           core_t::TTime timeZoneOffset,
                           std::size_t maxSize,
                           double decayRate = 0.0,
                           double minBucketLength = 0.0)
        : CCalendarComponent{feature, timeZoneOffset, maxSize, decayRate, minBucketLength} {
        this->initialize();
    }

    void addPoint(core_t::TTime time, double value, double weight = 1.0) {
        if (this->shouldInterpolate(time)) {
            this->interpolate(time);
        }
        if (this->feature().inWindow(time)) {
            this->add(time, value, weight);
        }
    }
};

const core_t::TTime HOUR{core::constants::HOUR};
const core_t::TTime DAY{core::constants::DAY};

auto calendarCyclic(core_t::TTime timeZoneOffset) {

    TTimeVec months{
        86400,    // 2nd Jan
        2764800,  // 2nd Feb
        5184000,  // 2nd Mar
        7862400,  // 2nd Apr
        10454400, // 2nd May
        13132800, // 2nd June
        15724800  // 2nd July
    };

    TTimeVec knots{-1,        2 * HOUR,  4 * HOUR,  6 * HOUR,  8 * HOUR,
                   10 * HOUR, 12 * HOUR, 14 * HOUR, 16 * HOUR, 18 * HOUR,
                   20 * HOUR, 22 * HOUR, 24 * HOUR};
    TDoubleVec values{10, 10, 10, 30, 80, 90, 80, 70, 50, 90, 30, 20, 10};

    return [=](core_t::TTime time) {
        auto month = std::upper_bound(months.begin(), months.end(),
                                      time + timeZoneOffset - DAY);
        if (time + timeZoneOffset >= *month && time + timeZoneOffset < *month + DAY) {
            time -= *month - timeZoneOffset;
            auto knot = std::upper_bound(knots.begin(), knots.end(), time);
            double ta{static_cast<double>(*knot - 2 * HOUR)};
            double tb{static_cast<double>(*knot)};
            double va{values[(knot - knots.begin() - 1)]};
            double vb{values[(knot - knots.begin())]};
            return maths::common::CTools::linearlyInterpolate(
                ta, tb, va, vb, static_cast<double>(time));
        }
        return 0.0;
    };
}
}

BOOST_FIXTURE_TEST_CASE(testFit, CTestFixture) {

    // Test the quality of fit to a supplied calendar feature.

    for (core_t::TTime timeZoneOffset :
         {0 * core::constants::HOUR, -6 * core::constants::HOUR, 6 * core::constants::HOUR}) {
        LOG_DEBUG(<< "time zone offset = " << timeZoneOffset);

        auto trend = calendarCyclic(timeZoneOffset);

        test::CRandomNumbers rng;

        CTestCalendarComponent component{
            maths::time_series::CCalendarFeature{maths::time_series::CCalendarFeature::DAYS_SINCE_START_OF_MONTH,
                                                 DAY + 12 * HOUR},
            timeZoneOffset, 24};

        TDoubleVec noise;
        for (core_t::TTime time = 0; time < 15724800; time += HOUR) {
            rng.generateNormalSamples(0.0, 9.0, 1, noise);
            component.addPoint(time, trend(time) + noise[0]);
        }
        TMeanAccumulator mae;
        for (core_t::TTime time = 15724800 - timeZoneOffset;
             time < 15724800 - timeZoneOffset + DAY; time += HOUR) {
            mae.add(std::fabs(
                (maths::common::CBasicStatistics::mean(component.value(time, 0.0)) - trend(time)) /
                trend(time)));
        }
        BOOST_TEST_REQUIRE(maths::common::CBasicStatistics::mean(mae) < 0.06);
    }
}

BOOST_FIXTURE_TEST_CASE(testPersist, CTestFixture) {
    // Check that persistence is idempotent.

    for (core_t::TTime timeZoneOffset :
         {0 * core::constants::HOUR, -6 * core::constants::HOUR, 6 * core::constants::HOUR}) {
        LOG_DEBUG(<< "time zone offset = " << timeZoneOffset);

        auto trend = calendarCyclic(timeZoneOffset);

        test::CRandomNumbers rng;

        CTestCalendarComponent origComponent{
            maths::time_series::CCalendarFeature{maths::time_series::CCalendarFeature::DAYS_SINCE_START_OF_MONTH,
                                                 DAY + 12 * HOUR},
            timeZoneOffset, 24};

        TDoubleVec noise;
        for (core_t::TTime time = 0; time < 15724800; time += HOUR) {
            rng.generateNormalSamples(0.0, 9.0, 1, noise);
            origComponent.addPoint(time, trend(time) + noise[0]);
        }

        std::string origXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            origComponent.acceptPersistInserter(inserter);
            inserter.toXml(origXml);
        }

        LOG_DEBUG(<< "seasonal component XML representation:\n" << origXml);

        // Restore the XML into a new component.
        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        maths::time_series::CCalendarComponent restoredComponent{0.0, 0.0, traverser};

        std::string newXml;
        {
            core::CRapidXmlStatePersistInserter inserter("root");
            restoredComponent.acceptPersistInserter(inserter);
            inserter.toXml(newXml);
        }
        BOOST_REQUIRE_EQUAL(origXml, newXml);
        BOOST_REQUIRE_EQUAL(origComponent.checksum(), restoredComponent.checksum());
    }
}

BOOST_AUTO_TEST_SUITE_END()
