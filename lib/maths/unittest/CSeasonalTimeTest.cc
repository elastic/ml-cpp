/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CSeasonalTimeTest.h"

#include <core/CLogger.h>

#include <maths/CSeasonalTime.h>

using namespace ml;

void CSeasonalTimeTest::testPersist() {
    // Check that persistence preserves checksums.

    maths::CDiurnalTime origDiurnal{3600, 0, 86400, 86400, 1.5};

    LOG_DEBUG(<< "Test persist/restore");

    std::string diurnalRepresentation{origDiurnal.toString()};
    LOG_DEBUG(<< "diurnal time representation: " << diurnalRepresentation);
    maths::CDiurnalTime restoredDiurnal;
    CPPUNIT_ASSERT(restoredDiurnal.fromString(diurnalRepresentation));
    CPPUNIT_ASSERT_EQUAL(origDiurnal.checksum(), restoredDiurnal.checksum());

    maths::CGeneralPeriodTime origPeriodic{7200, 2.1};
    std::string periodicRepresentation{origPeriodic.toString()};
    LOG_DEBUG(<< "periodic time representation: " << periodicRepresentation);
    maths::CGeneralPeriodTime restoredPeriodic;
    CPPUNIT_ASSERT(restoredPeriodic.fromString(periodicRepresentation));
    CPPUNIT_ASSERT_EQUAL(origPeriodic.checksum(), restoredPeriodic.checksum());

    LOG_DEBUG(<< "Test upgrade");

    restoredDiurnal = maths::CDiurnalTime{0, 3600, 43200, 43200, 1.5};
    CPPUNIT_ASSERT(restoredDiurnal.fromString("3600:0:86400:86400:0"));
    CPPUNIT_ASSERT_EQUAL(origDiurnal.checksum(), restoredDiurnal.checksum());

    restoredPeriodic = maths::CGeneralPeriodTime{16800, 2.1};
    CPPUNIT_ASSERT(restoredPeriodic.fromString("7200:0"));
    CPPUNIT_ASSERT_EQUAL(origPeriodic.checksum(), restoredPeriodic.checksum());
}

CppUnit::Test* CSeasonalTimeTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CSeasonalTimeTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CSeasonalTimeTest>(
        "CSeasonalTimeTest::testPersist", &CSeasonalTimeTest::testPersist));

    return suiteOfTests;
}
