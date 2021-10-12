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

#include <maths/time_series/CSeasonalTime.h>

#include <boost/test/unit_test.hpp>

#include <string>

using namespace ml;

BOOST_AUTO_TEST_SUITE(CSeasonalTimeTest)

BOOST_AUTO_TEST_CASE(testPersist) {

    // Check that persistence preserves checksums.

    maths::time_series::CDiurnalTime origDiurnal{3600, 0, 86400, 86400};

    LOG_DEBUG(<< "Test persist/restore");

    std::string diurnalRepresentation{origDiurnal.toString()};
    LOG_DEBUG(<< "diurnal time representation: " << diurnalRepresentation);
    maths::time_series::CDiurnalTime restoredDiurnal;
    BOOST_TEST_REQUIRE(restoredDiurnal.fromString(diurnalRepresentation));
    BOOST_REQUIRE_EQUAL(origDiurnal.checksum(), restoredDiurnal.checksum());

    maths::time_series::CGeneralPeriodTime origPeriodic{7200};
    std::string periodicRepresentation{origPeriodic.toString()};
    LOG_DEBUG(<< "periodic time representation: " << periodicRepresentation);
    maths::time_series::CGeneralPeriodTime restoredPeriodic;
    BOOST_TEST_REQUIRE(restoredPeriodic.fromString(periodicRepresentation));
    BOOST_REQUIRE_EQUAL(origPeriodic.checksum(), restoredPeriodic.checksum());

    LOG_DEBUG(<< "Test upgrade");

    restoredDiurnal = maths::time_series::CDiurnalTime{0, 3600, 43200, 43200};
    BOOST_TEST_REQUIRE(restoredDiurnal.fromString("3600:0:86400:86400:0"));
    BOOST_REQUIRE_EQUAL(origDiurnal.checksum(), restoredDiurnal.checksum());

    restoredPeriodic = maths::time_series::CGeneralPeriodTime{16800};
    BOOST_TEST_REQUIRE(restoredPeriodic.fromString("7200:0"));
    BOOST_REQUIRE_EQUAL(origPeriodic.checksum(), restoredPeriodic.checksum());
}

BOOST_AUTO_TEST_SUITE_END()
