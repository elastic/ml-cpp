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

#include "CDetectorEqualizerTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CStatisticalTests.h>

#include <model/CDetectorEqualizer.h>

#include <test/CRandomNumbers.h>

#include <boost/range.hpp>

using namespace ml;

using TDoubleVec = std::vector<double>;

namespace {

using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
const double THRESHOLD = std::log(0.05);
}

void CDetectorEqualizerTest::testCorrect() {
    LOG_DEBUG(<< "*** CDetectorEqualizerTest::testCorrect ***");

    // Test that the distribution of scores are more similar after correcting.

    double scales[] = {1.0, 2.1, 3.2};

    model::CDetectorEqualizer equalizer;

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < boost::size(scales); ++i) {
        TDoubleVec logp;
        rng.generateGammaSamples(1.0, scales[i], 1000, logp);

        for (std::size_t j = 0u; j < logp.size(); ++j) {
            if (-logp[j] <= THRESHOLD) {
                double p = std::exp(-logp[j]);
                equalizer.add(static_cast<int>(i), p);
            }
        }
    }

    TDoubleVec raw[3];
    TDoubleVec corrected[3];
    for (std::size_t i = 0u; i < boost::size(scales); ++i) {
        TDoubleVec logp;
        rng.generateGammaSamples(1.0, scales[i], 1000, logp);

        for (std::size_t j = 0u; j < logp.size(); ++j) {
            if (-logp[j] <= THRESHOLD) {
                double p = std::exp(-logp[j]);
                raw[i].push_back(p);
                corrected[i].push_back(equalizer.correct(static_cast<int>(i), p));
            }
        }
    }

    TMeanAccumulator similarityIncrease;
    for (std::size_t i = 1u, k = 0u; i < 3; ++i) {
        for (std::size_t j = 0u; j < i; ++j, ++k) {
            double increase =
                maths::CStatisticalTests::twoSampleKS(corrected[i], corrected[j]) /
                maths::CStatisticalTests::twoSampleKS(raw[i], raw[j]);
            similarityIncrease.add(std::log(increase));
            LOG_DEBUG(<< "similarity increase = " << increase);
            CPPUNIT_ASSERT(increase > 3.0);
        }
    }
    LOG_DEBUG(<< "mean similarity increase = "
              << std::exp(maths::CBasicStatistics::mean(similarityIncrease)));
    CPPUNIT_ASSERT(std::exp(maths::CBasicStatistics::mean(similarityIncrease)) > 40.0);
}

void CDetectorEqualizerTest::testAge() {
    LOG_DEBUG(<< "*** CDetectorEqualizerTest::testAge ***");

    // Test that propagation doesn't introduce a bias into the corrections.

    double scales[] = {1.0, 2.1, 3.2};

    model::CDetectorEqualizer equalizer;
    model::CDetectorEqualizer equalizerAged;

    test::CRandomNumbers rng;

    for (std::size_t i = 0u; i < boost::size(scales); ++i) {
        TDoubleVec logp;
        rng.generateGammaSamples(1.0, scales[i], 1000, logp);

        for (std::size_t j = 0u; j < logp.size(); ++j) {
            if (-logp[j] <= THRESHOLD) {
                double p = std::exp(-logp[j]);
                equalizer.add(static_cast<int>(i), p);
                equalizerAged.add(static_cast<int>(i), p);
                equalizerAged.age(0.995);
            }
        }
    }

    for (int i = 0; i < 3; ++i) {
        TMeanAccumulator meanBias;
        TMeanAccumulator meanError;
        double logp = THRESHOLD;
        for (std::size_t j = 0u; j < 150; ++j, logp += std::log(0.9)) {
            double p = std::exp(logp);
            double pc = equalizer.correct(i, p);
            double pca = equalizerAged.correct(i, p);
            double error = std::fabs((std::log(pca) - std::log(pc)) / std::log(pc));
            meanError.add(error);
            meanBias.add((std::log(pca) - std::log(pc)) / std::log(pc));
            CPPUNIT_ASSERT(error < 0.18);
        }
        LOG_DEBUG(<< "mean bias  = " << maths::CBasicStatistics::mean(meanBias));
        LOG_DEBUG(<< "mean error = " << maths::CBasicStatistics::mean(meanError));
        CPPUNIT_ASSERT(std::fabs(maths::CBasicStatistics::mean(meanBias)) < 0.053);
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError) < 0.053);
    }
}

void CDetectorEqualizerTest::testPersist() {
    LOG_DEBUG(<< "*** CDetectorEqualizerTest::testPersist ***");

    double scales[] = {1.0, 2.1, 3.2};

    model::CDetectorEqualizer origEqualizer;

    test::CRandomNumbers rng;

    TDoubleVec logp;
    rng.generateGammaSamples(1.0, 3.1, 1000, logp);

    for (std::size_t i = 0u; i < boost::size(scales); ++i) {
        rng.generateGammaSamples(1.0, scales[i], 1000, logp);

        for (std::size_t j = 0u; j < logp.size(); ++j) {
            if (-logp[j] <= THRESHOLD) {
                double p = std::exp(-logp[j]);
                origEqualizer.add(static_cast<int>(i), p);
            }
        }
    }

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origEqualizer.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG(<< "equalizer XML representation:\n" << origXml);

    model::CDetectorEqualizer restoredEqualizer;
    {
        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        CPPUNIT_ASSERT(traverser.traverseSubLevel(boost::bind(
            &model::CDetectorEqualizer::acceptRestoreTraverser, &restoredEqualizer, _1)));
    }

    // Checksums should agree.
    CPPUNIT_ASSERT_EQUAL(origEqualizer.checksum(), restoredEqualizer.checksum());

    // The persist and restore should be idempotent.
    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredEqualizer.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    CPPUNIT_ASSERT_EQUAL(origXml, newXml);
}

CppUnit::Test* CDetectorEqualizerTest::suite() {
    CppUnit::TestSuite* suiteOfTests =
        new CppUnit::TestSuite("CDetectorEqualizerTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectorEqualizerTest>(
        "CDetectorEqualizerTest::testCorrect", &CDetectorEqualizerTest::testCorrect));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectorEqualizerTest>(
        "CDetectorEqualizerTest::testAge", &CDetectorEqualizerTest::testAge));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDetectorEqualizerTest>(
        "CDetectorEqualizerTest::testPersist", &CDetectorEqualizerTest::testPersist));

    return suiteOfTests;
}
