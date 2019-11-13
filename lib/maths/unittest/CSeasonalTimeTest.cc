/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>

#include <maths/CSeasonalTime.h>

#include <boost/test/unit_test.hpp>

#include <string>

using namespace ml;

BOOST_AUTO_TEST_SUITE(CSeasonalTimeTest)

BOOST_AUTO_TEST_CASE(testPersist) {

    // Check that persistence preserves checksums.

    maths::CDiurnalTime origDiurnal{3600, 0, 86400, 86400, 1.5};

    LOG_DEBUG(<< "Test persist/restore");

    std::string diurnalRepresentation{origDiurnal.toString()};
    LOG_DEBUG(<< "diurnal time representation: " << diurnalRepresentation);
    maths::CDiurnalTime restoredDiurnal;
    BOOST_TEST_REQUIRE(restoredDiurnal.fromString(diurnalRepresentation));
    BOOST_REQUIRE_EQUAL(origDiurnal.checksum(), restoredDiurnal.checksum());

    maths::CGeneralPeriodTime origPeriodic{7200, 2.1};
    std::string periodicRepresentation{origPeriodic.toString()};
    LOG_DEBUG(<< "periodic time representation: " << periodicRepresentation);
    maths::CGeneralPeriodTime restoredPeriodic;
    BOOST_TEST_REQUIRE(restoredPeriodic.fromString(periodicRepresentation));
    BOOST_REQUIRE_EQUAL(origPeriodic.checksum(), restoredPeriodic.checksum());

    LOG_DEBUG(<< "Test upgrade");

    restoredDiurnal = maths::CDiurnalTime{0, 3600, 43200, 43200, 1.5};
    BOOST_TEST_REQUIRE(restoredDiurnal.fromString("3600:0:86400:86400:0"));
    BOOST_REQUIRE_EQUAL(origDiurnal.checksum(), restoredDiurnal.checksum());

    restoredPeriodic = maths::CGeneralPeriodTime{16800, 2.1};
    BOOST_TEST_REQUIRE(restoredPeriodic.fromString("7200:0"));
    BOOST_REQUIRE_EQUAL(origPeriodic.checksum(), restoredPeriodic.checksum());
}

BOOST_AUTO_TEST_SUITE_END()
