/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>

#include <maths/CBasicStatistics.h>
#include <maths/CLowess.h>
#include <maths/CLowessDetail.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/math/constants/constants.hpp>
#include <boost/test/unit_test.hpp>

#include <cmath>

BOOST_AUTO_TEST_SUITE(CLowessTest)

using namespace ml;

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

BOOST_AUTO_TEST_CASE(testInvariants) {

    // Test invariants are satisfied on random input.

    // We check:
    //   1. Minimum is a local minimum.
    //   2. The sublevel set contains the minimum.
    //   3. The minimum is within 10% of the training data interval.
    //   4. The ends of the sublevel set is within 10% of the training data interval.
    //   5. The variance is greater than or equal to the variance of the residuals at
    //      the training data.

    test::CRandomNumbers rng;

    TDoubleVec scale;
    TDoubleVec offset;
    TDoubleVec noise;
    maths::CLowess<2>::TDoubleDoublePrVec data;

    std::function<double(double)> trends[]{
        [&](double x) {
            return scale[0] * std::sin(boost::math::double_constants::two_pi /
                                       20.0 * (x + offset[0]));
        },
        [&](double x) { return scale[0] * x / 10.0; },
        [&](double x) {
            return scale[0] * (x - offset[0]) * (x - offset[0]) / 100.0;
        }};

    for (std::size_t i = 0; i < 100; ++i) {

        for (const auto& trend : trends) {
            rng.generateUniformSamples(0.0, 10.0, 1, scale);
            rng.generateUniformSamples(0.0, 20.0, 1, offset);
            rng.generateNormalSamples(0.0, 4.0, 20, noise);

            data.clear();
            for (std::size_t j = 0; j < noise.size(); ++j) {
                double x{static_cast<double>(j)};
                data.emplace_back(x, trend(x) + noise[j]);
            }

            maths::CLowess<2> lowess;
            lowess.fit(data, 5);

            double xea, xeb;
            std::tie(xea, xeb) = lowess.extrapolationInterval();

            double xmin, fmin;
            std::tie(xmin, fmin) = lowess.minimum();
            BOOST_REQUIRE_EQUAL(fmin, lowess.predict(xmin));
            BOOST_TEST_REQUIRE(fmin <= lowess.predict(std::max(xmin - 0.1, xea)));
            BOOST_TEST_REQUIRE(fmin <= lowess.predict(std::min(xmin + 0.1, xeb)));

            double xa, xb;
            std::tie(xa, xb) = lowess.sublevelSet(xmin, fmin, fmin + 0.1);
            BOOST_TEST_REQUIRE(xa <= xmin);
            BOOST_TEST_REQUIRE(xb >= xmin);

            BOOST_TEST_REQUIRE(xmin >= xea);
            BOOST_TEST_REQUIRE(xmin <= xeb);
            BOOST_TEST_REQUIRE(xa >= xea);
            BOOST_TEST_REQUIRE(xb <= xeb);
            BOOST_TEST_REQUIRE(xa >= xea);
            BOOST_TEST_REQUIRE(xb <= xeb);

            TMeanVarAccumulator residualMoments;
            for (const auto& x : data) {
                residualMoments.add(x.second - lowess.predict(x.first));
            }
            BOOST_TEST_REQUIRE(maths::CBasicStatistics::variance(residualMoments) <
                               lowess.residualVariance());
        }
    }
}

BOOST_AUTO_TEST_CASE(testSmooth) {

    // Test the prediction errors on a smooth function.

    test::CRandomNumbers rng;

    auto trend = [](double x) {
        return 8.0 * std::sin(boost::math::double_constants::two_pi / 20.0 * x);
    };

    maths::CLowess<2>::TDoubleDoublePrVec data;
    for (std::size_t i = 0; i < 20; ++i) {
        double x{static_cast<double>(i)};
        data.emplace_back(x, trend(x));
    }

    maths::CLowess<2> lowess;
    lowess.fit(data, 5);

    TMeanVarAccumulator errorMoments;
    for (std::size_t i = 0; i < 20; ++i) {
        double x{static_cast<double>(i)};
        errorMoments.add(std::fabs(lowess.predict(x) - trend(x)));
    }
    LOG_DEBUG(<< "mean error = " << maths::CBasicStatistics::mean(errorMoments));

    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(errorMoments) < 0.1);
}

BOOST_AUTO_TEST_CASE(testSmoothPlusNoise) {

    // Test the prediction errors on a smooth function plus noise.

    test::CRandomNumbers rng;

    TDoubleVec noise;
    rng.generateNormalSamples(0.0, 4.0, 20, noise);

    auto trend = [](double x) {
        return 8.0 * std::sin(boost::math::double_constants::two_pi / 20.0 * x);
    };

    maths::CLowess<2>::TDoubleDoublePrVec data;
    for (std::size_t i = 0; i < noise.size(); ++i) {
        double x{static_cast<double>(i)};
        data.emplace_back(x, trend(x) + noise[i]);
    }

    maths::CLowess<2> lowess;
    lowess.fit(data, 5);

    TMeanVarAccumulator errorMoments;
    for (std::size_t i = 0; i < 20; ++i) {
        double x{static_cast<double>(i)};
        errorMoments.add(std::fabs(lowess.predict(x) - trend(x)));
    }
    LOG_DEBUG(<< "mean error = " << maths::CBasicStatistics::mean(errorMoments));

    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(errorMoments) < 0.8);
    BOOST_TEST_REQUIRE(std::fabs(std::sqrt(lowess.residualVariance()) - 2.0) < 0.6);
}

BOOST_AUTO_TEST_CASE(testMinimum) {

    // Check that the minimum and the predicted value at the minimum is close to
    // what we'd expect.

    test::CRandomNumbers rng;

    auto trend = [](double x) {
        return 8.0 * std::sin(boost::math::double_constants::two_pi / 20.0 * x);
    };

    maths::CLowess<2>::TDoubleDoublePrVec data;
    for (std::size_t i = 0; i < 20; ++i) {
        double x{static_cast<double>(i)};
        data.emplace_back(x, trend(x));
    }

    maths::CLowess<2> lowess;
    lowess.fit(data, 5);

    double x, fx;
    std::tie(x, fx) = lowess.minimum();

    // Expect minimum at ((3 / 2) * pi) / (2 pi / 20) = 15 and a value of around -8.0;

    LOG_DEBUG(<< "xmin = " << x << ", f(xmin) = " << fx);
    BOOST_REQUIRE_CLOSE(15.0, x, 1.0);  // 1%
    BOOST_REQUIRE_CLOSE(-8.0, fx, 5.0); // 5%
}

BOOST_AUTO_TEST_CASE(testTrainingLossCurves) {

    // Test minimization of some training loss curves from boosted tree hyperparameter
    // line searches for:
    //   1. Miniboone
    //   2. Car-parts
    //   3. Boston

    using TDoubleDoublePrVecVec = std::vector<maths::CLowess<2>::TDoubleDoublePrVec>;

    // clang-format off
    TDoubleDoublePrVecVec curves{
        {{2.0, 0.1767327}, {6.080264, 0.1659147}, {9.615924, 0.1607294}, {10.16053, 0.1614871}, {14.24079, 0.1633198}},
        {{-2.561376, 0.1672884}, {-1.085517, 0.1647196}, {0.3903422, 0.1639279}, {1.474411, 0.1662013}, {1.866201, 0.1628465}},
        {{-2.561376, 0.162188}, {-1.085517, 0.1600827}, {-0.5958557, 0.1598617}, {0.3903422, 0.1642588}, {1.866201, 0.1778405}},
        {{-1.600108, 0.1588888}, {0.342874, 0.1574784}, {2.285856, 0.1569175}, {3.825301, 0.1527161}, {4.228838, 0.1555854}},
        {{-4.969813, 0.5935475}, {-3.313209, 0.2387051}, {-1.656604, 0.1552702}, {-0.7187975, 0.1507938}, {0, 0.1494794}},
        {{-2.302585, 0.1651654}, {-1.609438, 0.1712131}, {-0.9162907, 0.1550724}, {-0.4452244, 0.1491943}, {-0.2231436, 0.1489314}},
        {{2.0, 0.01361971}, {5.811543, 0.002268836}, {6.648845, 0.001762906}, {6.731061, 0.001930386}, {8.76648, 0.001210521},
         {9.58383, 0.002405683}, {9.623085, 0.002132054}, {9.787585, 0.002502508}, {10.42778, 0.001915853}, {13.43463, 0.001321818}},
        {{1.71972, 0.003296972}, {3.890569, 0.002917327}, {3.939936, 0.00103488}, {3.97139, 0.003646344}, {6.022504, 0.002943863},
         {6.061419, 0.001830975}, {7.801588, 0.003221994}, {7.930129, 0.003912988}, {8.232269, 0.004673212}},
        {{1.71972, 0.003408918}, {2.043608, 0.003519984}, {3.890569, 0.01988785}, {6.061419, 0.0764257}, {8.232269, 0.1406254}},
        {{-0.05942321, 0.003394985}, {0.6689442, 0.003665651}, {1.924394, 0.004942474}, {3.908212, 0.006659611}, {5.892029, 0.0157031}},
        {{-4.969813, 0.1798482}, {-3.313209, 0.1798566}, {-1.656604, 0.01256459}, {-1.154333, 0.004852421}, {-0.8191196, 0.003527397},
         {-0.2381196, 0.001983409}, {0, 0.002551422}},
        {{-2.302585, 0.001822712}, {-1.609438, 0.00345773}, {-0.9162907, 0.003139631}, {-0.2855592, 0.003175851}, {-0.2231436, 0.002630656}},
        {{-3.800451, 0.002890249}, {-2.801874, 0.002432233}, {-2.446324, 0.002333384}, {-2.291018, 0.001627785}, {-2.190441, 0.001669799},
         {-1.999605, 0.002137923}, {-1.803296, 0.001832592}, {-1.628174, 0.003295475}, {-0.8946376, 0.001722856}, {-0.804719, 0.001301327}},
        {{2.0, 10.71672}, {4.827566, 9.507881}, {4.830618, 8.36871}, {7.661235, 9.822492}, {10.49185, 10.09627}},
        {{-5.991457, 9.803939}, {-2.538955, 9.975635}, {0.9135475, 9.298096}, {3.543894, 8.223675}, {4.36605, 8.962077}},
        {{-5.991457, 9.35017}, {-3.357034, 9.962562}, {-2.538955, 9.027685}, {-1.97598, 8.668243}, {0.9135475, 10.19129}, {4.36605, 11.89721}},
        {{0.6931472, 9.422628}, {1.610698, 9.089348}, {1.691725, 8.93955}, {2.158699, 10.18192}, {2.545694, 9.212234}, {2.690302, 9.148424},
         {2.943044, 10.4056}, {3.688879, 11.13337}},
        {{-1.279388, 11.9904}, {-0.8885609, 9.800607}, {-0.6476757, 8.581057}, {-0.5692195, 7.907454}, {-0.4977335, 8.514873},
         {-0.1069061, 9.885219}},
        {{-3.800451, 8.317797}, {-3.738576, 8.053429}, {-3.403612, 8.338234}, {-2.801874, 8.890816}, {-2.333564, 8.705093},
         {-2.208987, 10.69139}, {-1.803296, 9.234116}, {-1.002829, 10.67219}, {-0.9090844, 12.46085}, {-0.804719, 13.98731}}};
    // clang-format on

    for (const auto& curve : curves) {
        maths::CLowess<2> lowess;
        lowess.fit(curve, curve.size());
        double xmin, fmin;
        std::tie(xmin, fmin) = lowess.minimum();
        double variance{lowess.residualVariance()};

        double xa, xb;
        double ftarget{fmin + std::sqrt(variance)};
        std::tie(xa, xb) = lowess.sublevelSet(xmin, fmin, ftarget);

        if (xa <= curve.front().first) {
            BOOST_TEST_REQUIRE(lowess.predict(xa) <= 1.01 * ftarget);
        } else {
            BOOST_REQUIRE_CLOSE(lowess.predict(xa), ftarget, 1.0); // 1.0%
        }
        if (xb >= curve.back().first) {
            BOOST_TEST_REQUIRE(lowess.predict(xb) <= 1.01 * ftarget);
        } else {
            BOOST_REQUIRE_CLOSE(lowess.predict(xb), ftarget, 1.0); // 1.0%
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
