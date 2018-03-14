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

#include "CMathsMemoryTest.h"

#include <maths/CConstantPrior.h>
#include <maths/CBasicStatistics.h>
#include <maths/CBjkstUniqueValues.h>
#include <maths/CPrior.h>
#include <maths/CLogNormalMeanPrecConjugate.h>
#include <maths/CGammaRateConjugate.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/CPoissonMeanConjugate.h>
#include <maths/CMultinomialConjugate.h>
#include <maths/CMultimodalPrior.h>
#include <maths/CTimeSeriesDecomposition.h>
#include <maths/CTools.h>
#include <maths/CXMeansOnline1d.h>

using namespace ml;
using namespace maths;

void CMathsMemoryTest::testTimeSeriesDecompositions(void) {
    CTimeSeriesDecomposition decomp(0.95, 3600, 55);

    core_t::TTime time;
    time = 140390672;

    for (unsigned i = 0; i < 600000; i += 600) {
        decomp.addPoint(time + i, (0.55 * (0.2 + (i % 86400))));
    }

    core::CMemoryUsage mem;
    decomp.debugMemoryUsage(mem.addChild());
    CPPUNIT_ASSERT_EQUAL(decomp.memoryUsage(), mem.usage());
}

void CMathsMemoryTest::testPriors(void) {
    CConstantPrior::TOptionalDouble d;
    CConstantPrior                  constantPrior(d);
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), constantPrior.memoryUsage());

    CGammaRateConjugate::TWeightStyleVec weightStyles;
    CGammaRateConjugate::TDoubleVec      samples;
    CGammaRateConjugate::TDoubleVecVec   weights;

    weightStyles.push_back(maths_t::E_SampleCountWeight);
    samples.push_back(0.996);
    CGammaRateConjugate::TDoubleVec weight;
    weight.push_back(0.2);
    weights.push_back(weight);

    CGammaRateConjugate gammaRateConjugate(maths_t::E_ContinuousData, 0.0, 0.9, 0.8, 0.7);
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), gammaRateConjugate.memoryUsage());
    gammaRateConjugate.addSamples(weightStyles, samples, weights);
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), gammaRateConjugate.memoryUsage());


    CLogNormalMeanPrecConjugate logNormalConjugate(maths_t::E_ContinuousData, 0.0, 0.9, 0.8, 0.7, 0.2);
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), logNormalConjugate.memoryUsage());
    logNormalConjugate.addSamples(weightStyles, samples, weights);
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), logNormalConjugate.memoryUsage());


    CPoissonMeanConjugate poissonConjugate(0.0, 0.8, 0.7, 0.3);
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), poissonConjugate.memoryUsage());
    poissonConjugate.addSamples(weightStyles, samples, weights);
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), poissonConjugate.memoryUsage());


    CNormalMeanPrecConjugate normalConjugate(maths_t::E_ContinuousData, 0.0, 0.9, 0.8, 0.7, 0.2);
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), normalConjugate.memoryUsage());
    normalConjugate.addSamples(weightStyles, samples, weights);
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), normalConjugate.memoryUsage());


    CMultinomialConjugate multinomialConjugate;
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), multinomialConjugate.memoryUsage());
    multinomialConjugate.addSamples(weightStyles, samples, weights);
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), multinomialConjugate.memoryUsage());


    CXMeansOnline1d clusterer(maths_t::E_ContinuousData,
                              maths::CAvailableModeDistributions::ALL,
                              maths_t::E_ClustersEqualWeight);

    // Check that the clusterer has size at least as great as the sum of it's fixed members
    std::size_t clustererSize =  sizeof(maths_t::EDataType)
                                + 4 * sizeof(double)
                                + sizeof(maths_t::EClusterWeightCalc)
                                + sizeof(CClusterer1d::CIndexGenerator)
                                + sizeof(CXMeansOnline1d::TClusterVec);

    CPPUNIT_ASSERT(clusterer.memoryUsage() >= clustererSize);

    CClusterer1d::TPointPreciseDoublePrVec clusters;
    clusters.push_back(CClusterer1d::TPointPreciseDoublePr(0.1, 0.7));
    clusters.push_back(CClusterer1d::TPointPreciseDoublePr(0.01, 0.6));
    clusters.push_back(CClusterer1d::TPointPreciseDoublePr(0.9, 0.5));
    clusters.push_back(CClusterer1d::TPointPreciseDoublePr(0.6, 0.2));
    clusters.push_back(CClusterer1d::TPointPreciseDoublePr(0.3, 0.3));
    clusters.push_back(CClusterer1d::TPointPreciseDoublePr(0.4, 0.9));
    clusters.push_back(CClusterer1d::TPointPreciseDoublePr(0.7, 0.8));
    clusters.push_back(CClusterer1d::TPointPreciseDoublePr(0.8, 0.9));
    clusters.push_back(CClusterer1d::TPointPreciseDoublePr(0.2, 0.4));
    clusters.push_back(CClusterer1d::TPointPreciseDoublePr(0.3, 0.5));
    clusters.push_back(CClusterer1d::TPointPreciseDoublePr(0.3, 0.5));
    clusterer.add(clusters);

    // Check that the CMultimodalPrior increases in size
    CMultimodalPrior multimodalPrior(maths_t::E_ContinuousData, clusterer, constantPrior);

    std::size_t initialMultimodalPriorSize = multimodalPrior.memoryUsage();

    multimodalPrior.addSamples(weightStyles, samples, weights);
    CPPUNIT_ASSERT(initialMultimodalPriorSize < multimodalPrior.memoryUsage());

    core::CMemoryUsage mem;
    multimodalPrior.debugMemoryUsage(mem.addChild());
    CPPUNIT_ASSERT_EQUAL(multimodalPrior.memoryUsage(), mem.usage());
}

void CMathsMemoryTest::testBjkstVec(void) {
    typedef std::vector<maths::CBjkstUniqueValues> TBjkstValuesVec;
    {
        // Test empty
        TBjkstValuesVec    values;
        core::CMemoryUsage mem;
        mem.setName("root", 0);
        core::CMemoryDebug::dynamicSize("values", values, &mem);
        CPPUNIT_ASSERT_EQUAL(core::CMemory::dynamicSize(values), mem.usage());
    }
    {
        // Test adding values to the vector part
        TBjkstValuesVec           values;
        maths::CBjkstUniqueValues seed(3, 100);
        values.resize(5, seed);
        for (std::size_t i = 0; i < 5; i++) {
            for (int j = 0; j < 100; j++) {
                values[i].add(j);
            }
        }
        core::CMemoryUsage mem;
        mem.setName("root", 0);
        core::CMemoryDebug::dynamicSize("values", values, &mem);
        CPPUNIT_ASSERT_EQUAL(core::CMemory::dynamicSize(values), mem.usage());
    }
    {
        // Test adding values to the sketch part
        TBjkstValuesVec           values;
        maths::CBjkstUniqueValues seed(3, 100);
        values.resize(5, seed);
        for (std::size_t i = 0; i < 5; i++) {
            for (int j = 0; j < 1000; j++) {
                values[i].add(j);
            }
        }
        core::CMemoryUsage mem;
        mem.setName("root", 0);
        core::CMemoryDebug::dynamicSize("values", values, &mem);
        CPPUNIT_ASSERT_EQUAL(core::CMemory::dynamicSize(values), mem.usage());
    }
}


CppUnit::Test *CMathsMemoryTest::suite(void) {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CMathsMemoryTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CMathsMemoryTest>(
                               "CMathsMemoryTest::testPriors",
                               &CMathsMemoryTest::testPriors) );

    suiteOfTests->addTest( new CppUnit::TestCaller<CMathsMemoryTest>(
                               "CMathsMemoryTest::testTimeSeriesDecompositions",
                               &CMathsMemoryTest::testTimeSeriesDecompositions) );

    suiteOfTests->addTest( new CppUnit::TestCaller<CMathsMemoryTest>(
                               "CMathsMemoryTest::testBjkstVec",
                               &CMathsMemoryTest::testBjkstVec) );


    return suiteOfTests;
}
