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

#include "CBoundingBoxTest.h"

#include <core/CLogger.h>

#include <maths/CBoundingBox.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraTools.h>

#include <test/CRandomNumbers.h>

using namespace ml;

typedef std::vector<double>           TDoubleVec;
typedef maths::CVectorNx1<double, 2>  TVector2;
typedef maths::CVectorNx1<double, 4>  TVector4;
typedef maths::CBoundingBox<TVector2> TBoundingBox2;
typedef maths::CBoundingBox<TVector4> TBoundingBox4;

namespace {

bool closerToX(const TBoundingBox2 &bb,
               const TVector2 &x,
               const TVector2 &y) {
    TVector2 cc[] = { bb.blc(), bb.trc() };
    for (std::size_t c = 0u; c < 4; ++c) {
        double p[] = { cc[c / 2](0), cc[c % 2](1) };
        TVector2 corner(p, p + 2);
        if ((x - corner).euclidean() > (y - corner).euclidean()) {
            return false;
        }
    }
    return true;
}

bool closerToX(const TBoundingBox4 &bb,
               const TVector4 &x,
               const TVector4 &y) {
    TVector4 cc[] = { bb.blc(), bb.trc() };
    for (std::size_t c = 0u; c < 16; ++c) {
        double p[] = { cc[c / 8](0), cc[(c / 4) % 2](1), cc[(c / 2) % 2](2), cc[c % 2](3) };
        TVector4 corner(p, p + 4);
        if ((x - corner).euclidean() > (y - corner).euclidean()) {
            return false;
        }
    }
    return true;
}

}

void CBoundingBoxTest::testAdd(void) {
    LOG_DEBUG("+-----------------------------+");
    LOG_DEBUG("|  CBoundingBoxTest::testAdd  |");
    LOG_DEBUG("+-----------------------------+");

    double points[][2] =
    {
        {  -1.0,  5.0 },
        {   2.0, 20.0 },
        {  10.0,  4.0 },
        { -10.0, -3.0 },
        { 200.0, 50.0 }
    };

    TBoundingBox2 bb(TVector2(&points[0][0], &points[0][0] + 2));
    CPPUNIT_ASSERT_EQUAL(-1.0, bb.blc()(0));
    CPPUNIT_ASSERT_EQUAL(5.0, bb.blc()(1));
    CPPUNIT_ASSERT_EQUAL(-1.0, bb.trc()(0));
    CPPUNIT_ASSERT_EQUAL(5.0, bb.trc()(1));
    CPPUNIT_ASSERT_EQUAL((-1.0 + -1.0) / 2.0, bb.centre()(0));
    CPPUNIT_ASSERT_EQUAL((5.0 + 5.0) / 2.0, bb.centre()(1));

    bb.add(TVector2(&points[1][0], &points[1][0] + 2));

    CPPUNIT_ASSERT_EQUAL(-1.0, bb.blc()(0));
    CPPUNIT_ASSERT_EQUAL(5.0, bb.blc()(1));
    CPPUNIT_ASSERT_EQUAL(2.0, bb.trc()(0));
    CPPUNIT_ASSERT_EQUAL(20.0, bb.trc()(1));
    CPPUNIT_ASSERT_EQUAL((-1.0 + 2.0) / 2.0, bb.centre()(0));
    CPPUNIT_ASSERT_EQUAL((5.0 + 20.0) / 2.0, bb.centre()(1));

    bb.add(TVector2(&points[2][0], &points[2][0] + 2));
    CPPUNIT_ASSERT_EQUAL(-1.0, bb.blc()(0));
    CPPUNIT_ASSERT_EQUAL(4.0, bb.blc()(1));
    CPPUNIT_ASSERT_EQUAL(10.0, bb.trc()(0));
    CPPUNIT_ASSERT_EQUAL(20.0, bb.trc()(1));
    CPPUNIT_ASSERT_EQUAL((-1.0 + 10.0) / 2.0, bb.centre()(0));
    CPPUNIT_ASSERT_EQUAL((4.0 + 20.0) / 2.0, bb.centre()(1));

    bb.add(TVector2(&points[3][0], &points[3][0] + 2));
    CPPUNIT_ASSERT_EQUAL(-10.0, bb.blc()(0));
    CPPUNIT_ASSERT_EQUAL(-3.0, bb.blc()(1));
    CPPUNIT_ASSERT_EQUAL(10.0, bb.trc()(0));
    CPPUNIT_ASSERT_EQUAL(20.0, bb.trc()(1));
    CPPUNIT_ASSERT_EQUAL((-10.0 + 10.0) / 2.0, bb.centre()(0));
    CPPUNIT_ASSERT_EQUAL((-3.0 + 20.0) / 2.0, bb.centre()(1));

    bb.add(TVector2(&points[4][0], &points[4][0] + 2));
    CPPUNIT_ASSERT_EQUAL(-10.0, bb.blc()(0));
    CPPUNIT_ASSERT_EQUAL(-3.0, bb.blc()(1));
    CPPUNIT_ASSERT_EQUAL(200.0, bb.trc()(0));
    CPPUNIT_ASSERT_EQUAL(50.0, bb.trc()(1));
    CPPUNIT_ASSERT_EQUAL((-10.0 + 200.0) / 2.0, bb.centre()(0));
    CPPUNIT_ASSERT_EQUAL((-3.0 + 50.0) / 2.0, bb.centre()(1));
}

void CBoundingBoxTest::testCloserTo(void) {
    LOG_DEBUG("+----------------------------------+");
    LOG_DEBUG("|  CBoundingBoxTest::testCloserTo  |");
    LOG_DEBUG("+----------------------------------+");

    const std::size_t n = 1000;

    test::CRandomNumbers rng;

    TDoubleVec points;
    rng.generateUniformSamples(-400.0, 400.0, 4 * n, points);

    TDoubleVec probes;
    rng.generateUniformSamples(-1000.0, 1000.0, 160, probes);

    for (std::size_t i = 0u; i < n; i += 4) {
        TVector2 x1(&points[i    ], &points[i + 2]);
        TVector2 x2(&points[i + 2], &points[i + 4]);

        TBoundingBox2 bb(x1);
        bb.add(x2);

        for (std::size_t j = 0u; j < probes.size(); j += 4) {
            TVector2 y1(&probes[j    ], &probes[j + 2]);
            TVector2 y2(&probes[j + 2], &probes[j + 4]);
            bool closer = closerToX(bb, y1, y2);
            if (closer) {
                LOG_DEBUG("bb = " << bb.print()
                          << " is closer to " << y1 << " than " << y2);
            }
            CPPUNIT_ASSERT_EQUAL(closer, bb.closerToX(y1, y2));
            closer = closerToX(bb, y2, y1);
            if (closer) {
                LOG_DEBUG("bb = " << bb.print()
                          << " is closer to " << y2 << " than " << y1);
            }
            CPPUNIT_ASSERT_EQUAL(closer, bb.closerToX(y2, y1));
        }
    }

    for (std::size_t i = 0u; i < n; i += 8) {
        TVector4 x1(&points[i    ], &points[i + 4]);
        TVector4 x2(&points[i + 4], &points[i + 8]);

        TBoundingBox4 bb(x1);
        bb.add(x2);

        for (std::size_t j = 0u; j < probes.size(); j += 4) {
            TVector4 y1(&probes[j    ], &probes[j + 4]);
            TVector4 y2(&probes[j + 4], &probes[j + 8]);
            bool closer = closerToX(bb, y1, y2);
            if (closer) {
                LOG_DEBUG("bb = " << bb.print()
                          << " is closer to " << y1 << " than " << y2);
            }
            CPPUNIT_ASSERT_EQUAL(closer, bb.closerToX(y1, y2));
            closer = closerToX(bb, y2, y1);
            if (closer) {
                LOG_DEBUG("bb = " << bb.print()
                          << " is closer to " << y2 << " than " << y1);
            }
            CPPUNIT_ASSERT_EQUAL(closer, bb.closerToX(y2, y1));
        }
    }
}

CppUnit::Test *CBoundingBoxTest::suite(void) {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CBoundingBoxTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CBoundingBoxTest>(
                               "CBoundingBoxTest::testAdd",
                               &CBoundingBoxTest::testAdd) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CBoundingBoxTest>(
                               "CBoundingBoxTest::testCloserTo",
                               &CBoundingBoxTest::testCloserTo) );

    return suiteOfTests;
}
