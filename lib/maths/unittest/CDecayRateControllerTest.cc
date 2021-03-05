/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CDecayRateController.h>

#include <test/CRandomNumbers.h>

#include "TestUtils.h"

#include <boost/test/unit_test.hpp>

#include <vector>

BOOST_AUTO_TEST_SUITE(CDecayRateControllerTest)

using namespace ml;
using namespace handy_typedefs;
using TDoubleVec = std::vector<double>;

BOOST_AUTO_TEST_CASE(testLowCov) {
    // Supply small but biased errors so we increase the decay rate to its maximum
    // then gradually reduce the error to less than the coefficient of variation
    // cutoff to control and make sure the decay rate reverts to typical.

    maths::CDecayRateController controller(maths::CDecayRateController::E_PredictionBias, 1);

    double decayRate{0.0005};
    for (std::size_t i = 0; i < 1000; ++i) {
        double multiplier{controller.multiplier({10000.0}, {{1.0}}, 3600, 1.0, 0.0005)};
        decayRate *= multiplier;
    }
    LOG_DEBUG(<< "Controlled decay = " << decayRate);
    BOOST_TEST_REQUIRE(decayRate > 0.0005);

    for (std::size_t i = 0; i < 1000; ++i) {
        double multiplier{controller.multiplier({10000.0}, {{0.0}}, 3600, 1.0, 0.0005)};
        decayRate *= multiplier;
    }
    LOG_DEBUG(<< "Controlled decay = " << decayRate);
    BOOST_TEST_REQUIRE(decayRate < 0.0005);
}

BOOST_AUTO_TEST_CASE(testOrderedErrors) {
    // Test that if we add a number of ordered samples, such that overall they don't
    // have bias, the decay rate is not increased.

    using TDouble1VecVec = std::vector<TDouble1Vec>;

    maths::CDecayRateController controller(maths::CDecayRateController::E_PredictionBias, 1);

    double decayRate{0.0005};
    TDouble1VecVec predictionErrors;
    for (std::size_t i = 0; i < 500; ++i) {
        for (int j = 0; j < 100; ++j) {
            predictionErrors.push_back({static_cast<double>(j - 50)});
        }
        double multiplier{controller.multiplier({100.0}, predictionErrors, 3600, 1.0, 0.0005)};
        decayRate *= multiplier;
    }
    LOG_DEBUG(<< "Controlled decay = " << decayRate);
    BOOST_TEST_REQUIRE(decayRate <= 0.0005);
}

BOOST_AUTO_TEST_CASE(testPersist) {
    // Test persist and restore preserves checksums.

    test::CRandomNumbers rng;

    TDoubleVec values;
    rng.generateUniformSamples(1000.0, 1010.0, 1000, values);

    TDoubleVec errors;
    rng.generateUniformSamples(-2.0, 6.0, 1000, errors);

    maths::CDecayRateController origController(
        maths::CDecayRateController::E_PredictionBias, 1);
    for (std::size_t i = 0; i < values.size(); ++i) {
        origController.multiplier({values[i]}, {{errors[i]}}, 3600, 1.0, 0.0005);
    }

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origController.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }
    LOG_TRACE(<< "Controller XML = " << origXml);
    LOG_DEBUG(<< "Controller XML size = " << origXml.size());

    // Restore the XML into a new controller.
    {
        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        maths::CDecayRateController restoredController;
        BOOST_REQUIRE_EQUAL(true, traverser.traverseSubLevel(std::bind(
                                      &maths::CDecayRateController::acceptRestoreTraverser,
                                      &restoredController, std::placeholders::_1)));

        LOG_DEBUG(<< "orig checksum = " << origController.checksum()
                  << ", new checksum = " << restoredController.checksum());
        BOOST_REQUIRE_EQUAL(origController.checksum(), restoredController.checksum());
    }
}

BOOST_AUTO_TEST_CASE(testBehaviourAfterPersistAndRestore) {
    // Test that we get the same decisions after persisting and restoring.

    test::CRandomNumbers rng;

    TDoubleVec values;
    rng.generateUniformSamples(1000.0, 1010.0, 1000, values);

    TDoubleVec errors;
    rng.generateUniformSamples(-2.0, 6.0, 1000, errors);

    maths::CDecayRateController origController{
        maths::CDecayRateController::E_PredictionBias, 1};
    maths::CDecayRateController restoredController;
    for (std::size_t i = 0; i < 500; ++i) {
        origController.multiplier({values[i]}, {{errors[i]}}, 3600, 1.0, 0.0005);
    }

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origController.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    // Restore the XML into a new controller.
    {
        core::CRapidXmlParser parser;
        BOOST_TEST_REQUIRE(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        BOOST_REQUIRE_EQUAL(true, traverser.traverseSubLevel(std::bind(
                                      &maths::CDecayRateController::acceptRestoreTraverser,
                                      &restoredController, std::placeholders::_1)));
    }

    for (std::size_t i = 500; i < values.size(); ++i) {
        BOOST_REQUIRE_CLOSE(
            origController.multiplier({values[i]}, {{errors[i]}}, 3600, 1.0, 0.0005),
            restoredController.multiplier({values[i]}, {{errors[i]}}, 3600, 1.0, 0.0005),
            0.001);
    }
}

BOOST_AUTO_TEST_SUITE_END()
