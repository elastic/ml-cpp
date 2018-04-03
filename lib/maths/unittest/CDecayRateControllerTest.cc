/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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

#include "CDecayRateControllerTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CDecayRateController.h>

#include <test/CRandomNumbers.h>

#include "TestUtils.h"

#include <boost/bind.hpp>

using namespace ml;
using namespace handy_typedefs;

void CDecayRateControllerTest::testLowCov() {
    LOG_DEBUG("+----------------------------------------+");
    LOG_DEBUG("|  CDecayRateControllerTest::testLowCov  |");
    LOG_DEBUG("+----------------------------------------+");

    // Supply small but biased errors so we increase the decay
    // rate to its maximum then gradually reduce the error to
    // less than the coefficient of variation cutoff to control
    // and make sure the decay rate reverts to typical.

    maths::CDecayRateController controller(maths::CDecayRateController::E_PredictionBias, 1);

    double decayRate{0.0005};
    for (std::size_t i = 0u; i < 1000; ++i) {
        double multiplier{controller.multiplier({10000.0}, {{1.0}}, 3600, 1.0, 0.0005)};
        decayRate *= multiplier;
    }
    LOG_DEBUG("Controlled decay = " << decayRate);
    CPPUNIT_ASSERT(decayRate > 0.0005);

    for (std::size_t i = 0u; i < 1000; ++i) {
        double multiplier{controller.multiplier({10000.0}, {{0.0}}, 3600, 1.0, 0.0005)};
        decayRate *= multiplier;
    }
    LOG_DEBUG("Controlled decay = " << decayRate);
    CPPUNIT_ASSERT(decayRate < 0.0005);
}

void CDecayRateControllerTest::testOrderedErrors() {
    LOG_DEBUG("+-----------------------------------------------+");
    LOG_DEBUG("|  CDecayRateControllerTest::testOrderedErrors  |");
    LOG_DEBUG("+-----------------------------------------------+");

    // Test that if we add a number of ordered samples, such
    // that overall they don't have bias, the decay rate is
    // not increased.

    using TDouble1VecVec = std::vector<TDouble1Vec>;

    maths::CDecayRateController controller(maths::CDecayRateController::E_PredictionBias, 1);

    double decayRate{0.0005};
    TDouble1VecVec predictionErrors;
    for (std::size_t i = 0u; i < 500; ++i) {
        for (int j = 0; j < 100; ++j) {
            predictionErrors.push_back({static_cast<double>(j - 50)});
        }
        double multiplier{controller.multiplier({100.0}, predictionErrors, 3600, 1.0, 0.0005)};
        decayRate *= multiplier;
    }
    LOG_DEBUG("Controlled decay = " << decayRate);
    CPPUNIT_ASSERT(decayRate <= 0.0005);

}

void CDecayRateControllerTest::testPersist() {
    LOG_DEBUG("+-----------------------------------------+");
    LOG_DEBUG("|  CDecayRateControllerTest::testPersist  |");
    LOG_DEBUG("+-----------------------------------------+");

    using TDoubleVec = std::vector<double>;

    test::CRandomNumbers rng;

    TDoubleVec values;
    rng.generateUniformSamples(1000.0, 1010.0, 1000, values);

    TDoubleVec errors;
    rng.generateUniformSamples(-2.0, 6.0, 1000, errors);

    maths::CDecayRateController origController(maths::CDecayRateController::E_PredictionBias, 1);
    for (std::size_t i = 0u; i < values.size(); ++i) {
        origController.multiplier({values[i]}, {{errors[i]}}, 3600, 1.0, 0.0005);
    }

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origController.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }
    LOG_TRACE("Controller XML = " << origXml);
    LOG_TRACE("Controller XML size = " << origXml.size());

    // Restore the XML into a new sketch.
    {
        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        maths::CDecayRateController restoredController;
        CPPUNIT_ASSERT_EQUAL(true, traverser.traverseSubLevel(
                                 boost::bind(&maths::CDecayRateController::acceptRestoreTraverser,
                                             &restoredController, _1)));

        LOG_DEBUG("orig checksum = " << origController.checksum()
                  << ", new checksum = " << restoredController.checksum());
        CPPUNIT_ASSERT_EQUAL(origController.checksum(),
                             restoredController.checksum());
    }
}

CppUnit::Test *CDecayRateControllerTest::suite() {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CDecayRateControllerTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CDecayRateControllerTest>(
                               "CDecayRateControllerTest::testLowCov",
                               &CDecayRateControllerTest::testLowCov) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CDecayRateControllerTest>(
                               "CDecayRateControllerTest::testOrderedErrors",
                               &CDecayRateControllerTest::testOrderedErrors) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CDecayRateControllerTest>(
                               "CDecayRateControllerTest::testPersist",
                               &CDecayRateControllerTest::testPersist) );

    return suiteOfTests;
}
