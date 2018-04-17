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

#include "CSplineTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CIntegration.h>
#include <maths/CSpline.h>

#include <boost/math/constants/constants.hpp>
#include <boost/range.hpp>

#include <test/CRandomNumbers.h>

using namespace ml;

namespace {

using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;

class CSplineFunctor {
public:
    CSplineFunctor(const maths::CSpline<>& spline) : m_Spline(&spline) {}

    bool operator()(double x, double& fx) const {
        fx = m_Spline->value(x);
        return true;
    }

private:
    const maths::CSpline<>* m_Spline;
};

std::string print(maths::CSplineTypes::EType type) {
    switch (type) {
    case maths::CSplineTypes::E_Linear:
        return "linear";
    case maths::CSplineTypes::E_Cubic:
        return "cubic";
    }
    return std::string();
}
}

void CSplineTest::testNatural() {
    // Test cubic spline with the natural boundary condition,
    // i.e. the case that the curvature vanishes at the interval
    // end points.

    {
        double x_[] = {0.0, 20.0, 21.0, 30.0, 56.0, 100.0, 102.0};
        double y_[] = {1.0, 5.0, 4.0, 13.0, 20.0, 12.0, 17.0};

        TDoubleVec x(boost::begin(x_), boost::end(x_));
        TDoubleVec y(boost::begin(y_), boost::end(y_));

        maths::CSpline<> spline(maths::CSplineTypes::E_Cubic);
        spline.interpolate(x, y, maths::CSplineTypes::E_Natural);

        for (std::size_t i = 0u; i < x.size(); ++i) {
            double yy = spline.value(x[i]);
            LOG_DEBUG(<< "f(x[" << i << "]) = " << yy);
            CPPUNIT_ASSERT_EQUAL(y[i], yy);

            double ym = spline.value(x[i] - 1e-3);
            CPPUNIT_ASSERT(std::fabs(yy - ym) < 1e-2);
            LOG_DEBUG(<< "f(x[" << i << " - eps]) = " << ym);

            double yp = spline.value(x[i] + 1e-3);
            LOG_DEBUG(<< "f(x[" << i << " + eps]) = " << yp);
            CPPUNIT_ASSERT(std::fabs(yp - yy) < 1e-2);
        }

        const TDoubleVec& curvatures = spline.curvatures();
        std::size_t n = curvatures.size();
        LOG_DEBUG(<< "curvatures[0] = " << curvatures[0]
                  << ", curvatures[n] = " << curvatures[n - 1]);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, curvatures[0], 1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, curvatures[n - 1], 1e-10);
    }

    {
        double x_[] = {0.0, 0.1, 0.3, 0.33, 0.5, 0.75, 0.8, 1.0};
        TDoubleVec x(boost::begin(x_), boost::end(x_));

        TDoubleVec y;
        y.reserve(x.size());
        for (std::size_t i = 0u; i < x.size(); ++i) {
            x[i] *= boost::math::double_constants::two_pi;
            y.push_back(std::sin(x[i]));
        }

        maths::CSpline<> spline(maths::CSplineTypes::E_Cubic);
        spline.interpolate(x, y, maths::CSplineTypes::E_Natural);

        for (std::size_t i = 0u; i < 21; ++i) {
            double xx = boost::math::double_constants::two_pi *
                        static_cast<double>(i) / 20.0;
            double yy = spline.value(xx);
            LOG_DEBUG(<< "spline(" << xx << ") = " << yy << ", f(" << xx
                      << ") = " << std::sin(xx));
            CPPUNIT_ASSERT(std::fabs(std::sin(xx) - yy) < 0.02);
        }

        const TDoubleVec& curvatures = spline.curvatures();
        std::size_t n = curvatures.size();
        LOG_DEBUG(<< "curvatures[0] = " << curvatures[0]
                  << ", curvatures[n] = " << curvatures[n - 1]);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, curvatures[0], 1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, curvatures[n - 1], 1e-10);
    }
}

void CSplineTest::testParabolicRunout() {
    {
        double x_[] = {0.0, 20.0, 21.0, 30.0, 56.0, 100.0, 102.0};
        double y_[] = {1.0, 5.0, 4.0, 13.0, 20.0, 12.0, 17.0};

        TDoubleVec x(boost::begin(x_), boost::end(x_));
        TDoubleVec y(boost::begin(y_), boost::end(y_));

        maths::CSpline<> spline(maths::CSplineTypes::E_Cubic);
        spline.interpolate(x, y, maths::CSplineTypes::E_ParabolicRunout);

        for (std::size_t i = 0u; i < x.size(); ++i) {
            double yy = spline.value(x[i]);
            LOG_DEBUG(<< "f(x[" << i << "]) = " << yy);
            CPPUNIT_ASSERT_EQUAL(y[i], yy);

            double ym = spline.value(x[i] - 1e-3);
            CPPUNIT_ASSERT(std::fabs(yy - ym) < 1e-2);
            LOG_DEBUG(<< "f(x[" << i << " - eps]) = " << ym);

            double yp = spline.value(x[i] + 1e-3);
            LOG_DEBUG(<< "f(x[" << i << " + eps]) = " << yp);
            CPPUNIT_ASSERT(std::fabs(yp - yy) < 1e-2);
        }

        const TDoubleVec& curvatures = spline.curvatures();
        std::size_t n = curvatures.size();
        LOG_DEBUG(<< "curvatures[0] = " << curvatures[0]
                  << ", curvatures[1] = " << curvatures[1]);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(curvatures[0], curvatures[1], 1e-10);
        LOG_DEBUG(<< "curvatures[n-1] = " << curvatures[n - 2]
                  << ", curvatures[n] = " << curvatures[n - 1]);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(curvatures[n - 2], curvatures[n - 1], 1e-10);
    }

    {
        double x_[] = {0.0, 0.1, 0.3, 0.33, 0.5, 0.75, 0.8, 1.0};
        TDoubleVec x(boost::begin(x_), boost::end(x_));

        TDoubleVec y;
        y.reserve(x.size());
        for (std::size_t i = 0u; i < x.size(); ++i) {
            x[i] *= boost::math::double_constants::two_pi;
            y.push_back(std::sin(x[i]));
        }

        maths::CSpline<> spline(maths::CSplineTypes::E_Cubic);
        spline.interpolate(x, y, maths::CSplineTypes::E_ParabolicRunout);

        for (std::size_t i = 0u; i < 21; ++i) {
            double xx = boost::math::double_constants::two_pi *
                        static_cast<double>(i) / 20.0;
            double yy = spline.value(xx);
            LOG_DEBUG(<< "spline(" << xx << ") = " << yy << ", f(" << xx
                      << ") = " << std::sin(xx));
            CPPUNIT_ASSERT(std::fabs(std::sin(xx) - yy) < 0.04);
        }

        const TDoubleVec& curvatures = spline.curvatures();
        std::size_t n = curvatures.size();
        LOG_DEBUG(<< "curvatures[0] = " << curvatures[0]
                  << ", curvatures[1] = " << curvatures[1]);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(curvatures[0], curvatures[1], 1e-10);
        LOG_DEBUG(<< "curvatures[n-1] = " << curvatures[n - 2]
                  << ", curvatures[n] = " << curvatures[n - 1]);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(curvatures[n - 2], curvatures[n - 1], 1e-10);
    }
}

void CSplineTest::testPeriodic() {
    {
        double x_[] = {0.0, 0.1, 0.3, 0.33, 0.5, 0.75, 0.8, 1.0};
        TDoubleVec x(boost::begin(x_), boost::end(x_));

        TDoubleVec y;
        y.reserve(x.size());
        for (std::size_t i = 0u; i < x.size(); ++i) {
            x[i] *= boost::math::double_constants::two_pi;
            y.push_back(std::cos(x[i]));
        }

        maths::CSpline<> spline(maths::CSplineTypes::E_Cubic);
        spline.interpolate(x, y, maths::CSplineTypes::E_Periodic);

        for (std::size_t i = 0u; i < 21; ++i) {
            double xx = boost::math::double_constants::two_pi *
                        static_cast<double>(i) / 20.0;
            double yy = spline.value(xx);
            LOG_DEBUG(<< "spline(" << xx << ") = " << yy << ", f(" << xx
                      << ") = " << std::cos(xx));
            CPPUNIT_ASSERT(std::fabs(std::cos(xx) - yy) < 0.02);
        }
    }

    {
        TDoubleVec x;
        for (std::size_t i = 0u; i < 40; ++i) {
            x.push_back(static_cast<double>(i) * 5.0);
        }
        double y_[] = {10.0, 7.0,  5.0,  3.0,  1.5,  3.5,  7.5,
                       15.5, 15.6, 15.5, 15.0, 14.0, 13.0, 12.0,
                       10.0, 8.0,  4.0,  4.1,  10.0, 10.0};
        TDoubleVec y(boost::begin(y_), boost::end(y_));
        y.insert(y.end(), boost::begin(y_), boost::end(y_));

        LOG_DEBUG(<< "x = " << core::CContainerPrinter::print(x));
        LOG_DEBUG(<< "y = " << core::CContainerPrinter::print(y));

        maths::CSpline<> spline(maths::CSplineTypes::E_Cubic);
        spline.interpolate(x, y, maths::CSplineTypes::E_Periodic);

        for (std::size_t i = 0; i < 200; ++i) {
            double xx = static_cast<double>(i);
            double yy = spline.value(static_cast<double>(i));
            if (i % 5 == 0) {
                LOG_DEBUG(<< "t = " << xx << ", y = " << y[i / 5] << ", ySpline= " << yy);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(y[i / 5], yy, 1e-6);
            }
        }
    }
}

void CSplineTest::testMean() {
    // Test that the mean of the cubic spline agrees with its
    // (numerical) integral and the expected mean of the cosine
    // over a whole number of periods.

    maths::CSplineTypes::EType types[] = {maths::CSplineTypes::E_Linear,
                                          maths::CSplineTypes::E_Cubic};

    {
        double x_[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
        TDoubleVec x(boost::begin(x_), boost::end(x_));
        std::size_t n = x.size() - 1;
        double y_[] = {0.0, 3.0, 4.0, 1.0, 6.0, 6.0, 5.0, 2.0, 2.5, 3.0, 5.0};
        TDoubleVec y(boost::begin(y_), boost::end(y_));

        for (std::size_t t = 0u; t < boost::size(types); ++t) {
            LOG_DEBUG(<< "*** Interpolation '" << print(types[t]) << "' ***");

            maths::CSpline<> spline(types[t]);
            spline.interpolate(x, y, maths::CSplineTypes::E_Natural);

            double expectedMean = 0.0;
            CSplineFunctor f(spline);
            for (std::size_t i = 1; i < x.size(); ++i) {
                double a = x[i - 1];
                double b = x[i];
                double integral;
                maths::CIntegration::gaussLegendre<maths::CIntegration::OrderThree>(
                    f, a, b, integral);
                expectedMean += integral;
            }
            expectedMean /= (x[n] - x[0]);

            LOG_DEBUG(<< "expectedMean = " << expectedMean
                      << ", mean = " << spline.mean());
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMean, spline.mean(),
                                         std::numeric_limits<double>::epsilon() * expectedMean);
        }
    }

    {
        test::CRandomNumbers rng;
        for (std::size_t i = 0u; i < 100; ++i) {
            TSizeVec n;
            rng.generateUniformSamples(10, 20, 1, n);
            TDoubleVec x;
            rng.generateUniformSamples(0.0, 10.0, n[0], x);
            std::sort(x.begin(), x.end());
            TDoubleVec y;
            rng.generateUniformSamples(5.0, 15.0, n[0], y);

            LOG_DEBUG(<< "n = " << n[0]);
            LOG_DEBUG(<< "x = " << core::CContainerPrinter::print(x));
            LOG_DEBUG(<< "y = " << core::CContainerPrinter::print(y));
            for (std::size_t t = 0; t < boost::size(types); ++t) {
                LOG_DEBUG(<< "*** Interpolation '" << print(types[t]) << "' ***");

                maths::CSpline<> spline(types[t]);
                spline.interpolate(x, y, maths::CSplineTypes::E_Natural);

                double expectedMean = 0.0;
                CSplineFunctor f(spline);
                for (std::size_t j = 1; j < x.size(); ++j) {
                    double a = x[j - 1];
                    double b = x[j];
                    double integral;
                    maths::CIntegration::gaussLegendre<maths::CIntegration::OrderThree>(
                        f, a, b, integral);
                    expectedMean += integral;
                }
                expectedMean /= (x[n[0] - 1] - x[0]);

                LOG_DEBUG(<< "expectedMean = " << expectedMean
                          << ", mean = " << spline.mean());

                CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedMean, spline.mean(), 1e-4);
            }
        }
    }

    {
        TDoubleVec x;
        TDoubleVec y;
        for (std::size_t i = 0u; i < 21; ++i) {
            x.push_back(static_cast<double>(20 * i));
            y.push_back(std::cos(boost::math::double_constants::two_pi *
                                 static_cast<double>(i) / 10.0));
        }

        for (std::size_t t = 0u; t < boost::size(types); ++t) {
            LOG_DEBUG(<< "*** Interpolation '" << print(types[t]) << "' ***");

            maths::CSpline<> spline(types[t]);
            spline.interpolate(x, y, maths::CSplineTypes::E_Periodic);

            LOG_DEBUG(<< "mean = " << spline.mean());
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, spline.mean(), 1e-10);
        }
    }
}

void CSplineTest::testIllposed() {
    // Test a case where some of the knot points are colocated.

    double x_[] = {0.0,  0.0,  10.0, 10.0, 15.0, 15.5,
                   20.0, 20.0, 20.0, 28.0, 30.0, 30.0};
    TDoubleVec x(boost::begin(x_), boost::end(x_));
    double y_[] = {0.0, 0.0, 1.9, 2.1, 3.0, 3.1, 4.0, 4.0, 4.0, 5.6, 5.9, 6.1};
    TDoubleVec y(boost::begin(y_), boost::end(y_));

    maths::CSplineTypes::EType types[] = {maths::CSplineTypes::E_Linear,
                                          maths::CSplineTypes::E_Cubic};

    for (std::size_t t = 0u; t < boost::size(types); ++t) {
        LOG_DEBUG(<< "*** Interpolation '" << print(types[t]) << "' ***");

        maths::CSpline<> spline(types[t]);
        spline.interpolate(x, y, maths::CSplineTypes::E_Natural);

        // The values lie on a straight line so the curvatures should
        // be zero (to working precision).

        TDoubleVec curvatures = spline.curvatures();
        LOG_DEBUG(<< "curvatures = " << core::CContainerPrinter::print(curvatures));
        for (std::size_t i = 0u; i < curvatures.size(); ++i) {
            CPPUNIT_ASSERT(std::fabs(curvatures[i]) < 2e-7);
        }

        for (std::size_t i = 0u; i <= 30; ++i) {
            LOG_DEBUG(<< "expected = " << 0.2 * static_cast<double>(i)
                      << ", actual = " << spline.value(static_cast<double>(i)));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.2 * static_cast<double>(i),
                                         spline.value(static_cast<double>(i)), 5e-7);
        }
    }
}

void CSplineTest::testSlope() {
    // Test that the slope and absolute slope agree with the
    // numerical derivatives of the value.

    maths::CSplineTypes::EType types[] = {maths::CSplineTypes::E_Linear,
                                          maths::CSplineTypes::E_Cubic};
    double eps = 1e-4;

    {
        double x_[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
        TDoubleVec x(boost::begin(x_), boost::end(x_));
        std::size_t n = x.size() - 1;
        double y_[] = {0.0, 3.0, 4.0, 1.0, 6.0, 6.0, 5.0, 2.0, 2.5, 3.0, 5.0};
        TDoubleVec y(boost::begin(y_), boost::end(y_));
        double range = x[n] - x[0];

        for (std::size_t t = 0u; t < boost::size(types); ++t) {
            LOG_DEBUG(<< "*** Interpolation '" << print(types[t]) << "' ***");

            maths::CSpline<> spline(types[t]);
            spline.interpolate(x, y, maths::CSplineTypes::E_Natural);

            CSplineFunctor f(spline);
            for (std::size_t i = 1u; i < 20; ++i) {
                double xi = x[0] + range * static_cast<double>(i) / 20.0 + eps;
                double xiPlusEps = xi + eps;
                double xiMinusEps = xi - eps;
                double slope = spline.slope(xi);
                double numericalSlope =
                    (spline.value(xiPlusEps) - spline.value(xiMinusEps)) / (2 * eps);
                LOG_DEBUG(<< "x = " << xi << ", slope = " << slope
                          << ", numerical slope = " << numericalSlope);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(numericalSlope, slope, 6.0 * eps * eps);
            }
        }
    }

    {
        test::CRandomNumbers rng;
        for (std::size_t i = 0u; i < 100; ++i) {
            TSizeVec n;
            rng.generateUniformSamples(10, 20, 1, n);
            TDoubleVec x;
            rng.generateUniformSamples(0.0, 100.0, n[0], x);
            std::sort(x.begin(), x.end());
            TDoubleVec y;
            rng.generateUniformSamples(5.0, 15.0, n[0], y);

            LOG_DEBUG(<< "n = " << n[0]);
            LOG_DEBUG(<< "x = " << core::CContainerPrinter::print(x));
            LOG_DEBUG(<< "y = " << core::CContainerPrinter::print(y));
            for (std::size_t t = 0; t < boost::size(types); ++t) {
                if (i % 10 == 0) {
                    LOG_DEBUG(<< "*** Interpolation '" << print(types[t]) << "' ***");
                }

                maths::CSpline<> spline(types[t]);
                spline.interpolate(x, y, maths::CSplineTypes::E_Natural);

                CSplineFunctor f(spline);
                for (std::size_t j = 1; j < n[0]; ++j) {
                    double xj = (x[j] + x[j - 1]) / 2.0;
                    double xiPlusEps = xj + eps;
                    double xiMinusEps = xj - eps;
                    double slope = spline.slope(xj);
                    double numericalSlope =
                        (spline.value(xiPlusEps) - spline.value(xiMinusEps)) / (2 * eps);
                    if (i % 10 == 0) {
                        LOG_DEBUG(<< "x = " << xj << ", slope = " << slope
                                  << ", numerical slope = " << numericalSlope);
                    }
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(numericalSlope, slope,
                                                 1e-3 * std::fabs(numericalSlope));
                }
            }
        }
    }

    {
        TDoubleVec x;
        TDoubleVec y;
        for (std::size_t i = 0u; i < 21; ++i) {
            x.push_back(static_cast<double>(20 * i));
            y.push_back(std::cos(boost::math::double_constants::two_pi *
                                 static_cast<double>(i) / 10.0));
        }
        double range = x[x.size() - 1] - x[0];

        for (std::size_t t = 0u; t < boost::size(types); ++t) {
            LOG_DEBUG(<< "*** Interpolation '" << print(types[t]) << "' ***");

            maths::CSpline<> spline(types[t]);
            spline.interpolate(x, y, maths::CSplineTypes::E_Periodic);

            for (std::size_t i = 1u; i < 20; ++i) {
                double xi = x[0] + range * static_cast<double>(i) / 20.0 + eps;
                double xiPlusEps = xi + eps;
                double xiMinusEps = xi - eps;
                double slope = spline.slope(xi);
                double numericalSlope =
                    (spline.value(xiPlusEps) - spline.value(xiMinusEps)) / (2 * eps);
                LOG_DEBUG(<< "x = " << xi << ", slope = " << slope
                          << ", numerical slope = " << numericalSlope);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(numericalSlope, slope, eps * eps);
            }
        }
    }
}

void CSplineTest::testSplineReference() {
    using TFloatVec = std::vector<maths::CFloatStorage>;
    using TFloatVecRef = boost::reference_wrapper<TFloatVec>;
    using TDoubleVecRef = boost::reference_wrapper<TDoubleVec>;
    using TSplineRef = maths::CSpline<TFloatVecRef, TFloatVecRef, TDoubleVecRef>;

    double x_[] = {0.0, 0.1, 0.3, 0.33, 0.5, 0.75, 0.8, 1.0};
    TDoubleVec x(boost::begin(x_), boost::end(x_));

    TDoubleVec y;
    y.reserve(x.size());
    for (std::size_t i = 0u; i < x.size(); ++i) {
        x[i] *= boost::math::double_constants::two_pi;
        y.push_back(std::sin(x[i]));
    }

    maths::CSpline<> spline(maths::CSplineTypes::E_Cubic);
    spline.interpolate(x, y, maths::CSplineTypes::E_Natural);

    TFloatVec knotsStorage;
    TFloatVec valuesStorage;
    TDoubleVec curvaturesStorage;
    TSplineRef splineRef(maths::CSplineTypes::E_Cubic, boost::ref(knotsStorage),
                         boost::ref(valuesStorage), boost::ref(curvaturesStorage));
    splineRef.interpolate(x, y, maths::CSplineTypes::E_Natural);

    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(spline.knots()),
                         core::CContainerPrinter::print(splineRef.knots()));
    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(spline.values()),
                         core::CContainerPrinter::print(splineRef.values()));
    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(spline.curvatures()),
                         core::CContainerPrinter::print(splineRef.curvatures()));

    for (std::size_t i = 0u; i < 21; ++i) {
        double xx = boost::math::double_constants::two_pi * static_cast<double>(i) / 20.0;

        LOG_DEBUG(<< "spline.value(" << xx << ") = " << spline.value(xx)
                  << ", splineRef.value(" << xx << ") = " << splineRef.value(xx));
        CPPUNIT_ASSERT_EQUAL(spline.value(xx), splineRef.value(xx));

        LOG_DEBUG(<< "spline.slope(" << xx << ") = " << spline.slope(xx)
                  << ", splineRef.slope(" << xx << ") = " << splineRef.slope(xx));
        CPPUNIT_ASSERT_EQUAL(spline.slope(xx), splineRef.slope(xx));
    }

    LOG_DEBUG(<< "spline.mean() = " << spline.mean()
              << ", splineRef.mean() = " << splineRef.mean());
    CPPUNIT_ASSERT_EQUAL(spline.mean(), splineRef.mean());

    LOG_DEBUG(<< "spline.absSlope() = " << spline.absSlope()
              << ", splineRef.absSlope() = " << splineRef.absSlope());
    CPPUNIT_ASSERT_EQUAL(spline.absSlope(), splineRef.absSlope());

    LOG_DEBUG(<< "splineRef.memoryUsage = " << splineRef.memoryUsage());
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), splineRef.memoryUsage());
}

CppUnit::Test* CSplineTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CSplineTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CSplineTest>(
        "CSplineTest::testNatural", &CSplineTest::testNatural));
    suiteOfTests->addTest(new CppUnit::TestCaller<CSplineTest>(
        "CSplineTest::testParabolicRunout", &CSplineTest::testParabolicRunout));
    suiteOfTests->addTest(new CppUnit::TestCaller<CSplineTest>(
        "CSplineTest::testPeriodic", &CSplineTest::testPeriodic));
    suiteOfTests->addTest(new CppUnit::TestCaller<CSplineTest>(
        "CSplineTest::testMean", &CSplineTest::testMean));
    suiteOfTests->addTest(new CppUnit::TestCaller<CSplineTest>(
        "CSplineTest::testIllposed", &CSplineTest::testIllposed));
    suiteOfTests->addTest(new CppUnit::TestCaller<CSplineTest>(
        "CSplineTest::testSlope", &CSplineTest::testSlope));
    suiteOfTests->addTest(new CppUnit::TestCaller<CSplineTest>(
        "CSplineTest::testSplineReference", &CSplineTest::testSplineReference));

    return suiteOfTests;
}
