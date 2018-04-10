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

#include "CProbabilityCalibratorTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CLogNormalMeanPrecConjugate.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/CProbabilityCalibrator.h>

#include <test/CRandomNumbers.h>
#include <test/CRandomNumbersDetail.h>

#include "TestUtils.h"

#include <boost/range.hpp>

#include <vector>

using namespace ml;

void CProbabilityCalibratorTest::testCalibration() {
    LOG_DEBUG("+-----------------------------------------------+");
    LOG_DEBUG("|  CProbabilityCalibratorTest::testCalibration  |");
    LOG_DEBUG("+-----------------------------------------------+");

    using TDoubleVec = std::vector<double>;
    using CLogNormalMeanPrecConjugate = CPriorTestInterfaceMixin<maths::CLogNormalMeanPrecConjugate>;
    using CNormalMeanPrecConjugate = CPriorTestInterfaceMixin<maths::CNormalMeanPrecConjugate>;

    // Test some things which we know will give poorly calibrated
    // probabilities, i.e. fitting a normal a log-normal and multi-
    // modal distributions.

    maths::CProbabilityCalibrator::EStyle styles[] = {maths::CProbabilityCalibrator::E_PartialCalibration,
                                                      maths::CProbabilityCalibrator::E_FullCalibration};

    test::CRandomNumbers rng;

    {
        LOG_DEBUG("*** log-normal ***");
        TDoubleVec samples;
        rng.generateLogNormalSamples(2.0, 0.9, 5000u, samples);

        double improvements[] = {0.03, 0.07};
        double maxImprovements[] = {0.01, 0.9};

        for (std::size_t i = 0u; i < boost::size(styles); ++i) {
            maths::CProbabilityCalibrator calibrator(styles[i], 0.99);

            CNormalMeanPrecConjugate normal = CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);
            CLogNormalMeanPrecConjugate lognormal = CLogNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);

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
                if (normal.probabilityOfLessLikelySamples(maths_t::E_TwoSided, sample, lowerBound, upperBound)) {
                    rawProbability = (lowerBound + upperBound) / 2.0;
                }

                calibrator.add(rawProbability);
                double calibratedProbability = calibrator.calibrate(rawProbability);

                double trueProbability = 1.0;
                if (lognormal.probabilityOfLessLikelySamples(maths_t::E_TwoSided, sample, lowerBound, upperBound)) {
                    trueProbability = (lowerBound + upperBound) / 2.0;
                }

                double raw = std::fabs(std::log(rawProbability) - std::log(trueProbability));
                double calibrated = std::fabs(std::log(calibratedProbability) - std::log(trueProbability));

                rawError += raw;
                calibratedError += calibrated;
                maxRawError = std::max(maxRawError, raw);
                maxCalibratedError = std::max(maxCalibratedError, calibrated);
            }

            LOG_DEBUG("totalRawError =        " << rawError << ", maxRawError =        " << maxRawError);
            LOG_DEBUG("totalCalibratedError = " << calibratedError << ", maxCalibratedError = " << maxCalibratedError);
            CPPUNIT_ASSERT((rawError - calibratedError) / rawError > improvements[i]);
            CPPUNIT_ASSERT((maxRawError - maxCalibratedError) / maxRawError > maxImprovements[i]);
        }
    }

    {
        LOG_DEBUG("*** multimode ***");

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

            CNormalMeanPrecConjugate normal = CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);
            CNormalMeanPrecConjugate normal1 = CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);
            CNormalMeanPrecConjugate normal2 = CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData);

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
                if (normal.probabilityOfLessLikelySamples(maths_t::E_TwoSided, sample, lowerBound, upperBound)) {
                    rawProbability = (lowerBound + upperBound) / 2.0;
                }

                calibrator.add(rawProbability);
                double calibratedProbability = calibrator.calibrate(rawProbability);

                double trueProbability = 1.0;
                if (mode.probabilityOfLessLikelySamples(maths_t::E_TwoSided, sample, lowerBound, upperBound)) {
                    trueProbability = (lowerBound + upperBound) / 2.0;
                }

                double raw = std::fabs(std::log(rawProbability) - std::log(trueProbability));
                double calibrated = std::fabs(std::log(calibratedProbability) - std::log(trueProbability));

                rawError += raw;
                calibratedError += calibrated;
                maxRawError = std::max(maxRawError, raw);
                maxCalibratedError = std::max(maxCalibratedError, calibrated);
            }

            LOG_DEBUG("totalRawError =        " << rawError << ", maxRawError =        " << maxRawError);
            LOG_DEBUG("totalCalibratedError = " << calibratedError << ", maxCalibratedError = " << maxCalibratedError);
            CPPUNIT_ASSERT((rawError - calibratedError) / rawError >= improvements[i]);
            CPPUNIT_ASSERT((maxRawError - maxCalibratedError) / maxRawError >= maxImprovements[i]);
        }
    }
}

CppUnit::Test* CProbabilityCalibratorTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CProbabilityCalibratorTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CProbabilityCalibratorTest>("CProbabilityCalibratorTest::testCalibration",
                                                                              &CProbabilityCalibratorTest::testCalibration));

    return suiteOfTests;
}
