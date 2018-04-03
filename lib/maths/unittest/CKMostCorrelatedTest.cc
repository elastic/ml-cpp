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

#include "CKMostCorrelatedTest.h"

#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CStopWatch.h>
#include <core/CUname.h>

#include <maths/CBasicStatistics.h>
#include <maths/CKMostCorrelated.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CSampling.h>

#include <test/CRandomNumbers.h>
#include <test/CRandomNumbersDetail.h>

#include <boost/range.hpp>

#include <vector>

using namespace ml;

namespace {

typedef std::vector<double>                                           TDoubleVec;
typedef maths::CVectorNx1<double, 2>                                  TVector2;
typedef std::vector<TVector2>                                         TVector2Vec;
typedef maths::CSymmetricMatrixNxN<double, 2>                         TMatrix2;
typedef maths::CBasicStatistics::SSampleMean<double>::TAccumulator    TMeanAccumulator;
typedef maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator TMeanVarAccumulator;

class CKMostCorrelatedForTest : public maths::CKMostCorrelated {
    public:
        typedef maths::CKMostCorrelated::SCorrelation                         TCorrelation;
        typedef maths::CKMostCorrelated::TCorrelationVec                      TCorrelationVec;
        typedef maths::CKMostCorrelated::TSizeVectorPackedBitVectorPrUMap     TSizeVectorPackedBitVectorPrUMap;
        typedef maths::CKMostCorrelated::TSizeVectorPackedBitVectorPrUMapCItr TSizeVectorPackedBitVectorPrUMapCItr;
        typedef maths::CKMostCorrelated::TMeanVarAccumulatorVec               TMeanVarAccumulatorVec;
        using maths::CKMostCorrelated::mostCorrelated;
        using maths::CKMostCorrelated::correlations;

    public:
        CKMostCorrelatedForTest(std::size_t size, double decayRate) :
            maths::CKMostCorrelated(size, decayRate)
        {}

        void mostCorrelated(TCorrelationVec &result) const {
            this->maths::CKMostCorrelated::mostCorrelated(result);
        }

        const TVectorVec &projections(void) const {
            return this->maths::CKMostCorrelated::projections();
        }

        const TSizeVectorPackedBitVectorPrUMap &projected(void) const {
            return this->maths::CKMostCorrelated::projected();
        }

        const TCorrelationVec &correlations(void) const {
            return this->maths::CKMostCorrelated::correlations();
        }

        const TMeanVarAccumulatorVec &moments(void) const {
            return this->maths::CKMostCorrelated::moments();
        }
};

double mutualInformation(const TDoubleVec &p1,
                         const TDoubleVec &p2) {
    std::size_t n = p1.size();

    double f1[]     = { 0.0, 0.0 };
    double f2[]     = { 0.0, 0.0 };
    double f12[][2] = { { 0.0, 0.0 }, { 0.0, 0.0 } };

    for (std::size_t i = 0u; i < n; ++i) {
        f1[p1[i] < 0 ? 0 : 1] += 1.0;
        f2[p2[i] < 0 ? 0 : 1] += 1.0;
        f12[p1[i] < 0 ? 0 : 1][p2[i] < 0 ? 0 : 1] += 1.0;
    }

    double I  = 0.0;
    double H1 = 0.0;
    double H2 = 0.0;
    for (std::size_t i = 0u; i < 2; ++i) {
        for (std::size_t j = 0u; j < 2; ++j) {
            if (f12[i][j] > 0.0) {
                I +=    f12[i][j] / static_cast<double>(n)
                     * ::log(f12[i][j] * static_cast<double>(n) / f1[i] / f2[j]);
            }
        }
        if (f1[i] > 0.0) {
            H1 -= f1[i] / static_cast<double>(n) * ::log(f1[i] / static_cast<double>(n));
        }
        if (f2[i] > 0.0) {
            H2 -= f2[i] / static_cast<double>(n) * ::log(f2[i] / static_cast<double>(n));
        }
    }

    return I / std::min(H1, H2);
}

void estimateCorrelation(const std::size_t trials,
                         const TVector2 &mean,
                         const TMatrix2 &covariance,
                         TMeanVarAccumulator &correlationEstimate) {
    typedef maths::CVectorNx1<maths::CFloatStorage, 10>                     TVector10;
    typedef std::vector<TVector10>                                          TVector10Vec;
    typedef maths::CBasicStatistics::SSampleMeanVar<TVector2>::TAccumulator TMeanVar2Accumulator;

    test::CRandomNumbers rng;

    TMeanVar2Accumulator sampleMoments;

    for (std::size_t t = 0u; t < trials; ++t) {
        TVector2Vec samples;
        maths::CSampling::multivariateNormalSample(mean, covariance, 50, samples);

        TVector10Vec projections;
        TDoubleVec uniform01;
        rng.generateUniformSamples(0.0, 1.0, 500, uniform01);
        for (std::size_t i = 0u; i < uniform01.size(); i += 10) {
            double v[] =
            {
                uniform01[i + 0] < 0.5 ? -1.0 : 1.0,
                uniform01[i + 1] < 0.5 ? -1.0 : 1.0,
                uniform01[i + 2] < 0.5 ? -1.0 : 1.0,
                uniform01[i + 3] < 0.5 ? -1.0 : 1.0,
                uniform01[i + 4] < 0.5 ? -1.0 : 1.0,
                uniform01[i + 5] < 0.5 ? -1.0 : 1.0,
                uniform01[i + 6] < 0.5 ? -1.0 : 1.0,
                uniform01[i + 7] < 0.5 ? -1.0 : 1.0,
                uniform01[i + 8] < 0.5 ? -1.0 : 1.0,
                uniform01[i + 9] < 0.5 ? -1.0 : 1.0

            };
            projections.push_back(TVector10(boost::begin(v), boost::end(v)));
        }

        TVector10 px(0.0);
        TVector10 py(0.0);
        for (std::size_t i = 0u; i < projections.size(); ++i) {
            sampleMoments.add(samples[i]);
            if (maths::CBasicStatistics::count(sampleMoments) > 1.0) {
                px += projections[i] * (samples[i](0) - maths::CBasicStatistics::mean(sampleMoments)(0))
                      / ::sqrt(maths::CBasicStatistics::variance(sampleMoments)(0));
                py += projections[i] * (samples[i](1) - maths::CBasicStatistics::mean(sampleMoments)(1))
                      / ::sqrt(maths::CBasicStatistics::variance(sampleMoments)(1));
            }
        }
        maths::CPackedBitVector ix(50, true);
        maths::CPackedBitVector iy(50, true);
        double correlation = CKMostCorrelatedForTest::TCorrelation::correlation(px, ix, py, iy);
        if (t % 10 == 0) {
            LOG_DEBUG("correlation = " << correlation);
        }

        correlationEstimate.add(correlation);
    }
}

}

void CKMostCorrelatedTest::testCorrelation(void) {
    LOG_DEBUG("+-----------------------------------------+")
    LOG_DEBUG("|  CKMostCorrelatedTest::testCorrelation  |")
    LOG_DEBUG("+-----------------------------------------+")

    // Check that the proposed estimator is unbiased.

    maths::CSampling::seed();

    {
        LOG_DEBUG("*** Weak Correlation ***");

        double m[] = { 10.0, 15.0 };
        double c[] = { 10.0, 2.0, 10.0 };
        TVector2 mean(boost::begin(m), boost::end(m));
        TMatrix2 covariance(boost::begin(c), boost::end(c));

        TMeanVarAccumulator correlationEstimate;
        estimateCorrelation(100, mean, covariance, correlationEstimate);
        LOG_DEBUG("correlationEstimate = " << correlationEstimate);

        double sd = ::sqrt(maths::CBasicStatistics::variance(correlationEstimate));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.2, maths::CBasicStatistics::mean(correlationEstimate), 3.0 * sd / 10.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sd, 0.5);
    }
    {
        LOG_DEBUG("*** Medium Correlation ***");

        double m[] = { 10.0, 15.0 };
        double c[] = { 10.0, 5.0, 10.0 };
        TVector2 mean(boost::begin(m), boost::end(m));
        TMatrix2 covariance(boost::begin(c), boost::end(c));

        TMeanVarAccumulator correlationEstimate;
        estimateCorrelation(100, mean, covariance, correlationEstimate);
        LOG_DEBUG("correlation = " << correlationEstimate);

        double sd = ::sqrt(maths::CBasicStatistics::variance(correlationEstimate));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, maths::CBasicStatistics::mean(correlationEstimate), 3.0 * sd / 10.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sd, 0.42);
    }
    {
        LOG_DEBUG("*** Strong Correlation ***");

        double m[] = { 10.0, 15.0 };
        double c[] = { 10.0, 9.0, 10.0 };
        TVector2 mean(boost::begin(m), boost::end(m));
        TMatrix2 covariance(boost::begin(c), boost::end(c));

        TMeanVarAccumulator correlationEstimate;
        estimateCorrelation(100, mean, covariance, correlationEstimate);
        LOG_DEBUG("correlation = " << correlationEstimate);

        double sd = ::sqrt(maths::CBasicStatistics::variance(correlationEstimate));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.9, maths::CBasicStatistics::mean(correlationEstimate), 3.0 * sd / 10.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sd, 0.2);
    }
}

void CKMostCorrelatedTest::testNextProjection(void) {
    LOG_DEBUG("+--------------------------------------------+")
    LOG_DEBUG("|  CKMostCorrelatedTest::testNextProjection  |")
    LOG_DEBUG("+--------------------------------------------+")

    // Test that aging happens correctly and that the projections
    // are have low mutual information.

    typedef std::vector<TDoubleVec> TDoubleVecVec;

    maths::CSampling::seed();

    double combinations[][2] =
    {
        { 1.0, 0.0 },
        { 0.9, 0.1 },
        { 0.5, 0.5 },
        { 0.1, 0.9 },
        { 0.0, 1.0 }
    };

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateUniformSamples(0.0, 10.0, 200, samples);

    std::size_t variables = samples.size() / 20;

    CKMostCorrelatedForTest mostCorrelated(10, 0.01);
    mostCorrelated.addVariables((variables * boost::size(combinations)) / 2);

    CKMostCorrelatedForTest::TVectorVec p1 = mostCorrelated.projections();
    LOG_DEBUG("projections 1 = ");
    for (std::size_t i = 0u; i < p1.size(); ++i) {
        LOG_DEBUG("  " << core::CContainerPrinter::print(p1[i]));
    }
    CPPUNIT_ASSERT(!p1.empty());
    CPPUNIT_ASSERT_EQUAL(std::size_t(10), p1[0].dimension());
    TDoubleVecVec projections1(10, TDoubleVec(p1.size()));
    for (std::size_t i = 0u; i < p1.size(); ++i) {
        for (std::size_t j = 0u; j < p1[i].dimension(); ++j) {
            projections1[j][i] = p1[i](j);
        }
    }

    TMeanAccumulator I1;
    for (std::size_t i = 0u; i < projections1.size(); ++i) {
        for (std::size_t j = 0u; j < i; ++j) {
            I1.add(mutualInformation(projections1[i], projections1[j]));
        }
    }
    LOG_DEBUG("I1 = " << maths::CBasicStatistics::mean(I1));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(I1) < 0.1);

    for (std::size_t i = 0u; i < 19; ++i) {
        for (std::size_t j = 0u, X = 0u; j < variables; j += 2) {
            for (std::size_t k = 0u; k < boost::size(combinations); ++k, ++X) {
                double x =  combinations[k][0] * samples[i * variables + j]
                           + combinations[k][1] * samples[i * variables + j + 1];
                mostCorrelated.add(X, x);
            }
        }
        mostCorrelated.capture();
    }

    // This should trigger the next projection to be generated.
    for (std::size_t i = 0u, X = 0u; i < variables; i += 2) {
        for (std::size_t j = 0u; j < boost::size(combinations); ++j, ++X) {
            double x =  combinations[j][0] * samples[19 * variables + i]
                       + combinations[j][1] * samples[19 * variables + i + 1];
            mostCorrelated.add(X, x);
        }
    }

    CKMostCorrelatedForTest::TCorrelationVec correlations1 = mostCorrelated.correlations();
    CKMostCorrelatedForTest::TMeanVarAccumulatorVec moments1 = mostCorrelated.moments();

    mostCorrelated.capture();

    CKMostCorrelatedForTest::TCorrelationVec correlations2 = mostCorrelated.correlations();
    CKMostCorrelatedForTest::TMeanVarAccumulatorVec moments2 = mostCorrelated.moments();

    CKMostCorrelatedForTest::TVectorVec p2 = mostCorrelated.projections();
    LOG_DEBUG("projections 2 = ");
    for (std::size_t i = 0u; i < p2.size(); ++i) {
        LOG_DEBUG("  " << core::CContainerPrinter::print(p2[i]));
    }
    CPPUNIT_ASSERT(!p2.empty());
    CPPUNIT_ASSERT_EQUAL(std::size_t(10), p2[0].dimension());
    TDoubleVecVec projections2(10, TDoubleVec(p2.size()));
    for (std::size_t i = 0u; i < p2.size(); ++i) {
        for (std::size_t j = 0u; j < p2[i].dimension(); ++j) {
            projections2[j][i] = p2[i](j);
        }
    }

    TMeanAccumulator I2;
    for (std::size_t i = 0u; i < projections2.size(); ++i) {
        for (std::size_t j = 0u; j < i; ++j) {
            I2.add(mutualInformation(projections2[i], projections2[j]));
        }
    }
    LOG_DEBUG("I2 = " << maths::CBasicStatistics::mean(I2));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(I2) < 0.1);
    TMeanAccumulator I12;
    for (std::size_t i = 0u; i < projections1.size(); ++i) {
        for (std::size_t j = 0u; j < projections2.size(); ++j) {
            I12.add(mutualInformation(projections1[i], projections2[j]));
        }
    }
    LOG_DEBUG("I12 = " << maths::CBasicStatistics::mean(I12));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(I12) < 0.1);

    for (std::size_t i = 0u; i < moments1.size(); ++i) {
        CPPUNIT_ASSERT(  maths::CBasicStatistics::count(moments1[i])
                         > maths::CBasicStatistics::count(moments2[i]));
    }
    for (std::size_t i = 0u; i < correlations2.size(); ++i) {
        CPPUNIT_ASSERT(maths::CBasicStatistics::count(correlations2[i].s_Correlation) > 0.0);
        CPPUNIT_ASSERT(maths::CBasicStatistics::count(correlations2[i].s_Correlation) < 1.0);
    }
}

void CKMostCorrelatedTest::testMostCorrelated(void) {
    LOG_DEBUG("+--------------------------------------------+")
    LOG_DEBUG("|  CKMostCorrelatedTest::testMostCorrelated  |")
    LOG_DEBUG("+--------------------------------------------+")

    // Check the variables with the highest estimated correlation emerge.

    typedef maths::CBasicStatistics::COrderStatisticsHeap<CKMostCorrelatedForTest::TCorrelation> TMaxCorrelationAccumulator;

    maths::CSampling::seed();

    double combinations[][2] =
    {
        { 1.0, 0.0 },
        { 0.9, 0.1 },
        { 0.5, 0.5 },
        { 0.1, 0.9 },
        { 0.0, 1.0 }
    };

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateUniformSamples(0.0, 10.0, 1900, samples);

    std::size_t variables = samples.size() / 19;

    CKMostCorrelatedForTest mostCorrelated(100, 0.0);
    mostCorrelated.addVariables((variables * boost::size(combinations)) / 2);

    for (std::size_t i = 0u; i < 19; ++i) {
        for (std::size_t j = 0u, X = 0u; j < variables; j += 2) {
            for (std::size_t k = 0u; k < boost::size(combinations); ++k, ++X) {
                double x =  combinations[k][0] * samples[i * variables + j]
                           + combinations[k][1] * samples[i * variables + j + 1];
                mostCorrelated.add(X, x);
            }
        }
        mostCorrelated.capture();
    }

    TMaxCorrelationAccumulator expected(200);
    for (CKMostCorrelatedForTest::TSizeVectorPackedBitVectorPrUMapCItr x = mostCorrelated.projected().begin();
         x != mostCorrelated.projected().end();
         ++x) {
        std::size_t X = x->first;
        CKMostCorrelatedForTest::TSizeVectorPackedBitVectorPrUMapCItr y = x;
        while (++y != mostCorrelated.projected().end()) {
            std::size_t Y = y->first;
            CKMostCorrelatedForTest::TCorrelation cxy(X, x->second.first, x->second.second,
                                                      Y, y->second.first, y->second.second);
            expected.add(cxy);
        }
    }
    expected.sort();
    LOG_DEBUG("most correlated = " << expected);

    CKMostCorrelatedForTest::TCorrelationVec actual;
    mostCorrelated.mostCorrelated(actual);

    CPPUNIT_ASSERT_EQUAL(core::CContainerPrinter::print(expected),
                         core::CContainerPrinter::print(actual));
}

void CKMostCorrelatedTest::testRemoveVariables(void) {
    LOG_DEBUG("+---------------------------------------------+")
    LOG_DEBUG("|  CKMostCorrelatedTest::testRemoveVariables  |")
    LOG_DEBUG("+---------------------------------------------+")

    // Test we correctly remove correlated pairs which include a variable
    // to prune.
    //
    // For ten variables [0, ..., 9] create correlated pairs { (0, 1), (2, 3),
    // (4, 5), (6, 7), (8, 9) } and remove variables 2 and 5.

    maths::CSampling::seed();

    double combinations[][2] =
    {
        { 0.8, 0.2 }
    };

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateUniformSamples(0.0, 10.0, 2000, samples);

    for (std::size_t i = 0u; i < samples.size(); i += 10) {
        for (std::size_t j = 0u; j < 10; j += 2) {
            samples[i + j + 1] =  combinations[0][0] * samples[i + j]
                                 + combinations[0][1] * samples[i + j + 1];
        }
    }

    CKMostCorrelatedForTest mostCorrelated(10, 0.0);
    mostCorrelated.addVariables(10);

    for (std::size_t i = 0u; i < samples.size(); i += 10) {
        for (std::size_t j = 0u; j < 10; ++j) {
            mostCorrelated.add(j, samples[i + j]);
        }
        mostCorrelated.capture();
    }

    CKMostCorrelatedForTest::TSizeSizePrVec correlatedPairs;
    mostCorrelated.mostCorrelated(correlatedPairs);
    LOG_DEBUG("correlatedPairs = " << core::CContainerPrinter::print(correlatedPairs));

    std::size_t remove_[] = { 2, 5 };
    CKMostCorrelatedForTest::TSizeVec remove(boost::begin(remove_), boost::end(remove_));
    mostCorrelated.removeVariables(remove);
    mostCorrelated.mostCorrelated(correlatedPairs);
    LOG_DEBUG("correlatedPairs = " << core::CContainerPrinter::print(correlatedPairs));

    for (std::size_t i = 0u; i < correlatedPairs.size(); ++i) {
        CPPUNIT_ASSERT(std::find(remove.begin(), remove.end(), correlatedPairs[i].first)  == remove.end());
        CPPUNIT_ASSERT(std::find(remove.begin(), remove.end(), correlatedPairs[i].second) == remove.end());
    }
}

void CKMostCorrelatedTest::testAccuracy(void) {
    LOG_DEBUG("+--------------------------------------+")
    LOG_DEBUG("|  CKMostCorrelatedTest::testAccuracy  |")
    LOG_DEBUG("+--------------------------------------+")

    // Check that we consistently find the most correlated pairs of variables.
    //
    // For ten variables [0, ..., 9] create correlated pairs { (0, 1), (2, 3),
    // (4, 5), (6, 7), (8, 9) } and check these are consistently identified.

    maths::CSampling::seed();

    double combinations[][2] =
    {
        { 0.8, 0.2 }
    };

    test::CRandomNumbers rng;

    for (std::size_t t = 0u; t < 10; ++t) {
        LOG_DEBUG("*** test = " << t + 1 << " ***");

        TDoubleVec samples;
        rng.generateUniformSamples(0.0, 10.0, 2000, samples);


        for (std::size_t i = 0u; i < samples.size(); i += 10) {
            for (std::size_t j = 0u; j < 10; j += 2) {
                samples[i + j + 1] =  combinations[0][0] * samples[i + j]
                                     + combinations[0][1] * samples[i + j + 1];
            }
        }

        CKMostCorrelatedForTest mostCorrelated(10, 0.0);
        mostCorrelated.addVariables(10);

        for (std::size_t i = 0u; i < samples.size(); i += 10) {
            for (std::size_t j = 0u; j < 10; ++j) {
                mostCorrelated.add(j, samples[i + j]);
            }
            mostCorrelated.capture();

            if ((i + 10) % 200 == 0) {
                CKMostCorrelatedForTest::TSizeSizePrVec correlatedPairs;
                mostCorrelated.mostCorrelated(correlatedPairs);
                TDoubleVec correlations;
                mostCorrelated.correlations(correlations);
                LOG_DEBUG("correlatedPairs = " << core::CContainerPrinter::print(correlatedPairs.begin(),
                                                                                 correlatedPairs.begin() + 5));
                LOG_DEBUG("correlations = " << core::CContainerPrinter::print(correlations.begin(),
                                                                              correlations.begin() + 5));
                std::sort(correlatedPairs.begin(), correlatedPairs.begin() + 5);
                CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]"),
                                     core::CContainerPrinter::print(correlatedPairs.begin(),
                                                                    correlatedPairs.begin() + 5));
            }
        }
    }
}

void CKMostCorrelatedTest::testStability(void) {
    LOG_DEBUG("+---------------------------------------+")
    LOG_DEBUG("|  CKMostCorrelatedTest::testStability  |")
    LOG_DEBUG("+---------------------------------------+")

    // For twenty variables [0, ..., 19] create correlated pairs { (0, 1),
    // (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17),
    // (18, 19) } with correlations of { (0, 1), (2, 3), (4, 5), (6, 7),
    // (8, 9) } equal and all greater than { (10, 11), (12, 13), (14, 15),
    // (16, 17), (18, 19) }. Test we correctly and stably find all correlated
    // pairs and in the right order.

    maths::CSampling::seed();

    double combinations[][2] =
    {
        { 0.8, 0.2 },
        { 6.0, 4.0 }
    };

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateUniformSamples(0.0, 10.0, 16000, samples);

    for (std::size_t i = 0u; i < samples.size(); i += 20) {
        for (std::size_t j = 0u; j < 10; j += 2) {
            samples[i + j + 1] =  combinations[0][0] * samples[i + j]
                                 + combinations[0][1] * samples[i + j + 1];
        }
        for (std::size_t j = 10u; j < 20; j += 2) {
            samples[i + j + 1] =  combinations[1][0] * samples[i + j]
                                 + combinations[1][1] * samples[i + j + 1];
        }
    }

    CKMostCorrelatedForTest mostCorrelated(10, 0.0);
    mostCorrelated.addVariables(20);

    for (std::size_t i = 0u; i < samples.size(); i += 20) {
        for (std::size_t j = 0u; j < 20; ++j) {
            mostCorrelated.add(j, samples[i + j]);
        }
        mostCorrelated.capture();

        if (i > 800 && (i + 20) % 400 == 0) {
            CKMostCorrelatedForTest::TSizeSizePrVec correlatedPairs;
            mostCorrelated.mostCorrelated(correlatedPairs);
            TDoubleVec correlations;
            mostCorrelated.correlations(correlations);
            LOG_DEBUG("correlatedPairs = " << core::CContainerPrinter::print(correlatedPairs));
            LOG_DEBUG("correlations = " << core::CContainerPrinter::print(correlations));
            std::sort(correlatedPairs.begin(),     correlatedPairs.begin() + 5);
            std::sort(correlatedPairs.begin() + 5, correlatedPairs.begin() + 10);
            CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), "
                                             "(10, 11), (12, 13), (14, 15), (16, 17), (18, 19)]"),
                                 core::CContainerPrinter::print(correlatedPairs));
        }
    }
}

void CKMostCorrelatedTest::testChangingCorrelation(void) {
    LOG_DEBUG("+-------------------------------------------------+")
    LOG_DEBUG("|  CKMostCorrelatedTest::testChangingCorrelation  |")
    LOG_DEBUG("+-------------------------------------------------+")

    // Test that we correctly identify a newly emerging correlation.
    //
    // For ten variables [0, ..., 9] create correlated pairs { (0, 1), (2, 3),
    // (4, 5), (6, 7), (8, 9) }. The pair (8, 9) starts off uncorrelated but
    // becomes correlated later on.

    maths::CSampling::seed();

    double combinations[][2] =
    {
        { 0.6, 0.4 }
    };

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateUniformSamples(0.0, 10.0, 4000, samples);

    for (std::size_t i = 0u; i < samples.size(); i += 10) {
        for (std::size_t j = 0u; j < 8; j += 2) {
            samples[i + j + 1] =  combinations[0][0] * samples[i + j]
                                 + combinations[0][1] * samples[i + j + 1];
        }
        if (i >= samples.size() / 3) {
            samples[i + 9] =  combinations[0][0] * samples[i + 8]
                             + combinations[0][1] * samples[i + 9];
        }
    }

    CKMostCorrelatedForTest mostCorrelated(10, 0.0);
    mostCorrelated.addVariables(10);

    for (std::size_t i = 0u; i < samples.size(); i += 10) {
        for (std::size_t j = 0u; j < 10; ++j) {
            mostCorrelated.add(j, samples[i + j]);
        }
        mostCorrelated.capture();
    }
    LOG_DEBUG("correlations = " << core::CContainerPrinter::print(mostCorrelated.correlations()));

    bool present = false;
    for (std::size_t i = 0u; i < mostCorrelated.correlations().size(); ++i) {
        if (   mostCorrelated.correlations()[i].s_X == 8 &&
               mostCorrelated.correlations()[i].s_Y == 9) {
            CPPUNIT_ASSERT(maths::CBasicStatistics::mean(mostCorrelated.correlations()[i].s_Correlation) > 0.7);
            present = true;
        }
    }
    CPPUNIT_ASSERT(present);
}

void CKMostCorrelatedTest::testMissingData(void) {
    LOG_DEBUG("+-----------------------------------------+")
    LOG_DEBUG("|  CKMostCorrelatedTest::testMissingData  |")
    LOG_DEBUG("+-----------------------------------------+")

    // Test the case that some of the metric values are missing.
    //
    // For ten variables [0, ..., 9] create correlated pairs { (0, 1), (2, 3),
    // (4, 5), (6, 7), (8, 9) }. We drop 20% of values at random from variables
    // 4 and 6 and test we find the corresponding correlated pairs but reduce
    // their estimated correlation.

    maths::CSampling::seed();

    double combinations[][2] =
    {
        { 0.8, 0.2 }
    };

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateUniformSamples(0.0, 10.0, 4000, samples);

    for (std::size_t i = 0u; i < samples.size(); i += 10) {
        for (std::size_t j = 0u; j < 10; j += 2) {
            samples[i + j + 1] =  combinations[0][0] * samples[i + j]
                                 + combinations[0][1] * samples[i + j + 1];
        }
    }

    CKMostCorrelatedForTest mostCorrelated(10, 0.0);
    mostCorrelated.addVariables(10);

    for (std::size_t i = 0u; i < samples.size(); i += 10) {
        for (std::size_t j = 0u; j < 10; ++j) {
            if (j == 4 || j == 6) {
                TDoubleVec test;
                rng.generateUniformSamples(0.0, 1.0, 1, test);
                if (test[0] < 0.2) {
                    continue;
                }
            }
            mostCorrelated.add(j, samples[i + j]);
        }
        mostCorrelated.capture();

        if (i > 1000 && (i + 10) % 200 == 0) {
            CKMostCorrelatedForTest::TSizeSizePrVec correlatedPairs;
            mostCorrelated.mostCorrelated(correlatedPairs);
            TDoubleVec correlations;
            mostCorrelated.correlations(correlations);
            LOG_DEBUG("correlatedPairs = " << core::CContainerPrinter::print(correlatedPairs.begin(),
                                                                             correlatedPairs.begin() + 5));
            LOG_DEBUG("correlations = " << core::CContainerPrinter::print(correlations.begin(),
                                                                          correlations.begin() + 5));
            std::sort(correlatedPairs.begin(), correlatedPairs.begin() + 3);
            std::sort(correlatedPairs.begin() + 3, correlatedPairs.begin() + 5);
            CPPUNIT_ASSERT_EQUAL(std::string("[(0, 1), (2, 3), (8, 9), (4, 5), (6, 7)]"),
                                 core::CContainerPrinter::print(correlatedPairs.begin(), correlatedPairs.begin() + 5));
        }
    }
}

void CKMostCorrelatedTest::testScale(void) {
    LOG_DEBUG("+-----------------------------------+")
    LOG_DEBUG("|  CKMostCorrelatedTest::testScale  |")
    LOG_DEBUG("+-----------------------------------+")

    // Test runtime is approximately linear in the number of variables
    // if we look for O(number of variables) correlations.

    typedef std::vector<std::size_t> TSizeVec;
    typedef std::vector<TDoubleVec>  TDoubleVecVec;

    maths::CSampling::seed();

    test::CRandomNumbers rng;

    std::size_t n[] = { 200, 400, 800, 1600, 3200 };
    uint64_t elapsed[5];

    for (std::size_t s = 0u; s < boost::size(n); ++s) {
        double proportions[] = { 0.2, 0.3, 0.5 };
        std::size_t b = 200;
        std::size_t ns[] =
        {
            static_cast<std::size_t>(static_cast<double>(n[s] * b) * proportions[0]),
            static_cast<std::size_t>(static_cast<double>(n[s] * b) * proportions[1]),
            static_cast<std::size_t>(static_cast<double>(n[s] * b) * proportions[2])
        };
        TDoubleVec scales;
        rng.generateUniformSamples(10.0, 40.0, n[s], scales);

        TSizeVec labels;
        for (std::size_t i = 0; i < n[s]; ++i) {
            labels.push_back(i);
        }
        rng.random_shuffle(labels.begin(), labels.end());

        TDoubleVec uniform;
        rng.generateUniformSamples(0.0, 100.0, ns[0], uniform);
        TDoubleVec gamma;
        rng.generateGammaSamples(10.0, 10.0, ns[1], gamma);
        TDoubleVec normal;
        rng.generateNormalSamples(50.0, 20.0, ns[2], normal);

        TDoubleVecVec samples(b, TDoubleVec(n[s]));
        const TDoubleVec *samples_[] = { &uniform, &gamma, &normal };
        for (std::size_t i = 0u; i < b; ++i) {
            for (std::size_t j = 0u, l = 0u; j < 3; ++j) {
                std::size_t m = samples_[j]->size() / b;
                for (std::size_t k = 0u; k < m; ++k, ++l) {
                    samples[i][labels[l]] = scales[k] * (*samples_[j])[i * m + k];
                }
            }
        }

        double weights[][2] =
        {
            { 0.65, 0.35 },
            { 0.35, 0.65 }
        };

        CKMostCorrelatedForTest mostCorrelated(n[s], 0.0);
        mostCorrelated.addVariables(n[s]);

        core::CStopWatch watch;

        watch.start();
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            for (std::size_t j = 0u; j < samples[i].size(); j += 2) {
                double x = weights[0][0] * samples[i][j] + weights[0][1] * samples[i][j + 1];
                double y = weights[1][0] * samples[i][j] + weights[1][1] * samples[i][j + 1];
                mostCorrelated.add(j,   x);
                mostCorrelated.add(j + 1, y);
            }
            mostCorrelated.capture();
        }
        elapsed[s] = watch.stop();

        LOG_DEBUG("elapsed time = " << elapsed[s] << "ms");

        //std::vector<std::pair<std::size_t, std::size_t> > pairs;
        //mostCorrelated.mostCorrelated(n[s] / 2, pairs);
        //LOG_DEBUG("pairs = " << core::CContainerPrinter::print(pairs));
        //TDoubleVec correlations;
        //mostCorrelated.correlations(n[s] / 2, correlations);
        //LOG_DEBUG("correlations = " << core::CContainerPrinter::print(correlations));
    }

    LOG_DEBUG("elapsed times = " << core::CContainerPrinter::print(elapsed));

    // Test that the slope is subquadratic
    TMeanVarAccumulator slope;
    for (std::size_t i = 1u; i < boost::size(elapsed); ++i) {
        slope.add(static_cast<double>(elapsed[i]) / static_cast<double>(elapsed[i - 1]));
    }
    double exponent = ::log(maths::CBasicStatistics::mean(slope)) / ::log(2.0);
    LOG_DEBUG("exponent = " << exponent);
    double sdRatio = ::sqrt(maths::CBasicStatistics::variance(slope))
                     / maths::CBasicStatistics::mean(slope);
    LOG_DEBUG("sdRatio = " << sdRatio);
    if (core::CUname::nodeName().compare(0, 3, "vm-") == 0) {
        // Allow more leeway on a VM as non-linearity is most likely due to the
        // VM stalling
        CPPUNIT_ASSERT(exponent < 2.0);
        CPPUNIT_ASSERT(sdRatio < 0.75);
    } else   {
        CPPUNIT_ASSERT(exponent < 1.75);
        CPPUNIT_ASSERT(sdRatio < 0.5);
    }
}

void CKMostCorrelatedTest::testPersistence(void) {
    LOG_DEBUG("+-----------------------------------------+")
    LOG_DEBUG("|  CKMostCorrelatedTest::testPersistence  |")
    LOG_DEBUG("+-----------------------------------------+")

    // Check that persistence is idempotent.

    maths::CSampling::seed();

    double combinations[][2] =
    {
        { 0.8, 0.2 }
    };

    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateUniformSamples(0.0, 10.0, 4000, samples);

    for (std::size_t i = 0u; i < samples.size(); i += 10) {
        for (std::size_t j = 0u; j < 10; j += 2) {
            samples[i + j + 1] =  combinations[0][0] * samples[i + j]
                                 + combinations[0][1] * samples[i + j + 1];
        }
    }

    maths::CKMostCorrelated origMostCorrelated(10, 0.001);
    origMostCorrelated.addVariables(10);

    for (std::size_t i = 0u; i < samples.size(); i += 10) {
        for (std::size_t j = 0u; j < 10; ++j) {
            origMostCorrelated.add(j, samples[i + j]);
        }
        origMostCorrelated.capture();
    }

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origMostCorrelated.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }
    LOG_DEBUG("original k-most correlated XML = " << origXml);

    // Restore the XML into a new sketch.
    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);
    maths::CKMostCorrelated restoredMostCorrelated(10, 0.001);
    CPPUNIT_ASSERT(traverser.traverseSubLevel(
                       boost::bind(&maths::CKMostCorrelated::acceptRestoreTraverser,
                                   &restoredMostCorrelated,
                                   _1)));

    LOG_DEBUG("orig checksum = " << origMostCorrelated.checksum()
              << ", new checksum = " << restoredMostCorrelated.checksum());
    CPPUNIT_ASSERT_EQUAL(origMostCorrelated.checksum(),
                         restoredMostCorrelated.checksum());

    std::string newXml;
    core::CRapidXmlStatePersistInserter inserter("root");
    restoredMostCorrelated.acceptPersistInserter(inserter);
    inserter.toXml(newXml);

    CPPUNIT_ASSERT_EQUAL(origXml, newXml);
}

CppUnit::Test *CKMostCorrelatedTest::suite(void) {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CKMostCorrelatedTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CKMostCorrelatedTest>(
                               "CKMostCorrelatedTest::testCorrelation",
                               &CKMostCorrelatedTest::testCorrelation) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CKMostCorrelatedTest>(
                               "CKMostCorrelatedTest::testNextProjection",
                               &CKMostCorrelatedTest::testNextProjection) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CKMostCorrelatedTest>(
                               "CKMostCorrelatedTest::testMostCorrelated",
                               &CKMostCorrelatedTest::testMostCorrelated) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CKMostCorrelatedTest>(
                               "CKMostCorrelatedTest::testRemoveVariables",
                               &CKMostCorrelatedTest::testRemoveVariables) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CKMostCorrelatedTest>(
                               "CKMostCorrelatedTest::testAccuracy",
                               &CKMostCorrelatedTest::testAccuracy) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CKMostCorrelatedTest>(
                               "CKMostCorrelatedTest::testStability",
                               &CKMostCorrelatedTest::testStability) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CKMostCorrelatedTest>(
                               "CKMostCorrelatedTest::testChangingCorrelation",
                               &CKMostCorrelatedTest::testChangingCorrelation) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CKMostCorrelatedTest>(
                               "CKMostCorrelatedTest::testMissingData",
                               &CKMostCorrelatedTest::testMissingData) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CKMostCorrelatedTest>(
                               "CKMostCorrelatedTest::testScale",
                               &CKMostCorrelatedTest::testScale) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CKMostCorrelatedTest>(
                               "CKMostCorrelatedTest::testPersistence",
                               &CKMostCorrelatedTest::testPersistence) );

    return suiteOfTests;
}
