/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CLogNormalMeanPrecConjugate.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/CProbabilityCalibrator.h>

#include <test/CRandomNumbers.h>

#include "TestUtils.h"

#include <boost/test/unit_test.hpp>
#include <boost/range.hpp>

#include <vector>

BOOST_AUTO_TEST_SUITE(CProbabilityCalibratorTest)

using namespace ml;

BOOST_AUTO_TEST_CASE(testCalibration) {
    using TDoubleVec = std::vector<double>;
    using CLogNormalMeanPrecConjugate =
        CPriorTestInterfaceMixin<maths::CLogNormalMeanPrecConjugate>;
    using CNormalMeanPrecConjugate = CPriorTestInterfaceMixin<maths::CNormalMeanPrecConjugate>;

    // Test some things which we know will give poorly calibrated
    // probabilities, i.e. fitting a normal a log-normal and multi-
    // modal distributions.

    maths::CProbabilityCalibrator::EStyle styles[] = {
        maths::CProbabilityCalibrator::E_PartialCalibration,
        maths::CProbabilityCalibrator::E_FullCalibration};

    test::CRandomNumbers rng;

    {
        LOG_DEBUG(<< "*** log-normal ***");
        TDoubleVec samples;
        rng.generateLogNormalSamples(2.0, 0.9, 5000u, samples);

        double improvements[] = {0.03, 0.07};
        double maxImprovements[] = {0.01, 0.9};

        for (std::size_t i = 0u; i < boost::size(styles); ++i) {
            maths::CProbabilityCalibrator calibrator(styles[i], 0.99);

            CNormalMeanPrecConjugate normal =
                CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);
            CLogNormalMeanPrecConjugate lognormal =
                CLogNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);

            double rawError = 0.0;
            double calibratedError = 0.0;
            double maxRawError = 0.0;
            double maxCalibratedError = 0.0;

            for (std::size_t j = 0u; j < samples.size(); ++j) {
                TDoubleVec sample(1u, samples[j]);
                normal.addSamples(sample);
                lognormal.addSamples(sample);

                double lowerBound;
                double upperBound;

                double rawProbability = 1.0;
                if (normal.probabilityOfLessLikelySamples(
                        maths_t::E_TwoSided, sample, lowerBound, upperBound)) {
                    rawProbability = (lowerBound + upperBound) / 2.0;
                }

                calibrator.add(rawProbability);
                double calibratedProbability = calibrator.calibrate(rawProbability);

                double trueProbability = 1.0;
                if (lognormal.probabilityOfLessLikelySamples(
                        maths_t::E_TwoSided, sample, lowerBound, upperBound)) {
                    trueProbability = (lowerBound + upperBound) / 2.0;
                }

                double raw = std::fabs(std::log(rawProbability) - std::log(trueProbability));
                double calibrated = std::fabs(std::log(calibratedProbability) -
                                              std::log(trueProbability));

                rawError += raw;
                calibratedError += calibrated;
                maxRawError = std::max(maxRawError, raw);
                maxCalibratedError = std::max(maxCalibratedError, calibrated);
            }

            LOG_DEBUG(<< "totalRawError =        " << rawError
                      << ", maxRawError =        " << maxRawError);
            LOG_DEBUG(<< "totalCalibratedError = " << calibratedError
                      << ", maxCalibratedError = " << maxCalibratedError);
            BOOST_TEST((rawError - calibratedError) / rawError > improvements[i]);
            BOOST_TEST((maxRawError - maxCalibratedError) / maxRawError >
                           maxImprovements[i]);
        }
    }

    {
        LOG_DEBUG(<< "*** multimode ***");

        TDoubleVec samples1;
        rng.generateNormalSamples(5.0, 1.0, 4500u, samples1);
        TDoubleVec samples2;
        rng.generateNormalSamples(15.0, 1.0, 500u, samples2);
        TDoubleVec samples(samples1);
        samples.insert(samples.end(), samples2.begin(), samples2.end());

        rng.random_shuffle(samples.begin(), samples.end());

        double improvements[] = {0.18, 0.19};
        double maxImprovements[] = {0.0, -0.04};

        for (std::size_t i = 0u; i < boost::size(styles); ++i) {
            maths::CProbabilityCalibrator calibrator(styles[i], 0.99);

            CNormalMeanPrecConjugate normal =
                CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);
            CNormalMeanPrecConjugate normal1 =
                CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);
            CNormalMeanPrecConjugate normal2 =
                CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);

            double rawError = 0.0;
            double calibratedError = 0.0;
            double maxRawError = 0.0;
            double maxCalibratedError = 0.0;

            for (std::size_t j = 0u; j < samples.size(); ++j) {
                TDoubleVec sample(1u, samples[j]);
                normal.addSamples(sample);
                CNormalMeanPrecConjugate& mode = samples[j] < 10.0 ? normal1 : normal2;
                mode.addSamples(sample);

                double lowerBound;
                double upperBound;

                double rawProbability = 1.0;
                if (normal.probabilityOfLessLikelySamples(
                        maths_t::E_TwoSided, sample, lowerBound, upperBound)) {
                    rawProbability = (lowerBound + upperBound) / 2.0;
                }

                calibrator.add(rawProbability);
                double calibratedProbability = calibrator.calibrate(rawProbability);

                double trueProbability = 1.0;
                if (mode.probabilityOfLessLikelySamples(maths_t::E_TwoSided, sample,
                                                        lowerBound, upperBound)) {
                    trueProbability = (lowerBound + upperBound) / 2.0;
                }

                double raw = std::fabs(std::log(rawProbability) - std::log(trueProbability));
                double calibrated = std::fabs(std::log(calibratedProbability) -
                                              std::log(trueProbability));

                rawError += raw;
                calibratedError += calibrated;
                maxRawError = std::max(maxRawError, raw);
                maxCalibratedError = std::max(maxCalibratedError, calibrated);
            }

            LOG_DEBUG(<< "totalRawError =        " << rawError
                      << ", maxRawError =        " << maxRawError);
            LOG_DEBUG(<< "totalCalibratedError = " << calibratedError
                      << ", maxCalibratedError = " << maxCalibratedError);
            BOOST_TEST((rawError - calibratedError) / rawError >= improvements[i]);
            BOOST_TEST((maxRawError - maxCalibratedError) / maxRawError >=
                           maxImprovements[i]);
        }
    }
}


BOOST_AUTO_TEST_SUITE_END()
