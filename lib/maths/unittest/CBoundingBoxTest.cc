/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>

#include <maths/CBoundingBox.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraTools.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CBoundingBoxTest)

using namespace ml;

using TDoubleVec = std::vector<double>;
using TVector2 = maths::CVectorNx1<double, 2>;
using TVector4 = maths::CVectorNx1<double, 4>;
using TBoundingBox2 = maths::CBoundingBox<TVector2>;
using TBoundingBox4 = maths::CBoundingBox<TVector4>;

namespace {

bool closerToX(const TBoundingBox2& bb, const TVector2& x, const TVector2& y) {
    TVector2 cc[] = {bb.blc(), bb.trc()};
    for (std::size_t c = 0u; c < 4; ++c) {
        double p[] = {cc[c / 2](0), cc[c % 2](1)};
        TVector2 corner(p, p + 2);
        if ((x - corner).euclidean() > (y - corner).euclidean()) {
            return false;
        }
    }
    return true;
}

bool closerToX(const TBoundingBox4& bb, const TVector4& x, const TVector4& y) {
    TVector4 cc[] = {bb.blc(), bb.trc()};
    for (std::size_t c = 0u; c < 16; ++c) {
        double p[] = {cc[c / 8](0), cc[(c / 4) % 2](1), cc[(c / 2) % 2](2), cc[c % 2](3)};
        TVector4 corner(p, p + 4);
        if ((x - corner).euclidean() > (y - corner).euclidean()) {
            return false;
        }
    }
    return true;
}
}

BOOST_AUTO_TEST_CASE(testAdd) {
    double points[][2] = {{-1.0, 5.0}, {2.0, 20.0}, {10.0, 4.0}, {-10.0, -3.0}, {200.0, 50.0}};

    TBoundingBox2 bb(TVector2(&points[0][0], &points[0][0] + 2));
    BOOST_REQUIRE_EQUAL(-1.0, bb.blc()(0));
    BOOST_REQUIRE_EQUAL(5.0, bb.blc()(1));
    BOOST_REQUIRE_EQUAL(-1.0, bb.trc()(0));
    BOOST_REQUIRE_EQUAL(5.0, bb.trc()(1));
    BOOST_REQUIRE_EQUAL((-1.0 + -1.0) / 2.0, bb.centre()(0));
    BOOST_REQUIRE_EQUAL((5.0 + 5.0) / 2.0, bb.centre()(1));

    bb.add(TVector2(&points[1][0], &points[1][0] + 2));

    BOOST_REQUIRE_EQUAL(-1.0, bb.blc()(0));
    BOOST_REQUIRE_EQUAL(5.0, bb.blc()(1));
    BOOST_REQUIRE_EQUAL(2.0, bb.trc()(0));
    BOOST_REQUIRE_EQUAL(20.0, bb.trc()(1));
    BOOST_REQUIRE_EQUAL((-1.0 + 2.0) / 2.0, bb.centre()(0));
    BOOST_REQUIRE_EQUAL((5.0 + 20.0) / 2.0, bb.centre()(1));

    bb.add(TVector2(&points[2][0], &points[2][0] + 2));
    BOOST_REQUIRE_EQUAL(-1.0, bb.blc()(0));
    BOOST_REQUIRE_EQUAL(4.0, bb.blc()(1));
    BOOST_REQUIRE_EQUAL(10.0, bb.trc()(0));
    BOOST_REQUIRE_EQUAL(20.0, bb.trc()(1));
    BOOST_REQUIRE_EQUAL((-1.0 + 10.0) / 2.0, bb.centre()(0));
    BOOST_REQUIRE_EQUAL((4.0 + 20.0) / 2.0, bb.centre()(1));

    bb.add(TVector2(&points[3][0], &points[3][0] + 2));
    BOOST_REQUIRE_EQUAL(-10.0, bb.blc()(0));
    BOOST_REQUIRE_EQUAL(-3.0, bb.blc()(1));
    BOOST_REQUIRE_EQUAL(10.0, bb.trc()(0));
    BOOST_REQUIRE_EQUAL(20.0, bb.trc()(1));
    BOOST_REQUIRE_EQUAL((-10.0 + 10.0) / 2.0, bb.centre()(0));
    BOOST_REQUIRE_EQUAL((-3.0 + 20.0) / 2.0, bb.centre()(1));

    bb.add(TVector2(&points[4][0], &points[4][0] + 2));
    BOOST_REQUIRE_EQUAL(-10.0, bb.blc()(0));
    BOOST_REQUIRE_EQUAL(-3.0, bb.blc()(1));
    BOOST_REQUIRE_EQUAL(200.0, bb.trc()(0));
    BOOST_REQUIRE_EQUAL(50.0, bb.trc()(1));
    BOOST_REQUIRE_EQUAL((-10.0 + 200.0) / 2.0, bb.centre()(0));
    BOOST_REQUIRE_EQUAL((-3.0 + 50.0) / 2.0, bb.centre()(1));
}

BOOST_AUTO_TEST_CASE(testCloserTo) {
    const std::size_t n = 1000;

    test::CRandomNumbers rng;

    TDoubleVec points;
    rng.generateUniformSamples(-400.0, 400.0, 4 * n, points);

    TDoubleVec probes;
    rng.generateUniformSamples(-1000.0, 1000.0, 160, probes);

    for (std::size_t i = 0u; i < n; i += 4) {
        TVector2 x1(&points[i], &points[i + 2]);
        TVector2 x2(&points[i + 2], &points[i + 4]);

        TBoundingBox2 bb(x1);
        bb.add(x2);

        for (std::size_t j = 0u; j + 4 <= probes.size(); j += 4) {
            TVector2 y1(&probes[j], &probes[j + 2]);
            TVector2 y2(&probes[j + 2], &probes[j + 4]);
            bool closer = closerToX(bb, y1, y2);
            if (closer) {
                LOG_DEBUG(<< "bb = " << bb.print() << " is closer to " << y1
                          << " than " << y2);
            }
            BOOST_REQUIRE_EQUAL(closer, bb.closerToX(y1, y2));
            closer = closerToX(bb, y2, y1);
            if (closer) {
                LOG_DEBUG(<< "bb = " << bb.print() << " is closer to " << y2
                          << " than " << y1);
            }
            BOOST_REQUIRE_EQUAL(closer, bb.closerToX(y2, y1));
        }
    }

    for (std::size_t i = 0u; i < n; i += 8) {
        TVector4 x1(&points[i], &points[i + 4]);
        TVector4 x2(&points[i + 4], &points[i + 8]);

        TBoundingBox4 bb(x1);
        bb.add(x2);

        for (std::size_t j = 0u; j + 8 <= probes.size(); j += 4) {
            TVector4 y1(&probes[j], &probes[j + 4]);
            TVector4 y2(&probes[j + 4], &probes[j + 8]);
            bool closer = closerToX(bb, y1, y2);
            if (closer) {
                LOG_DEBUG(<< "bb = " << bb.print() << " is closer to " << y1
                          << " than " << y2);
            }
            BOOST_REQUIRE_EQUAL(closer, bb.closerToX(y1, y2));
            closer = closerToX(bb, y2, y1);
            if (closer) {
                LOG_DEBUG(<< "bb = " << bb.print() << " is closer to " << y2
                          << " than " << y1);
            }
            BOOST_REQUIRE_EQUAL(closer, bb.closerToX(y2, y1));
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
