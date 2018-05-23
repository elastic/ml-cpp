/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CExpandingWindowTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/Constants.h>

#include <maths/CBasicStatistics.h>
#include <maths/CExpandingWindow.h>

#include <test/CRandomNumbers.h>

#include <boost/math/constants/constants.hpp>

#include <vector>

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;
using TTimeVec = std::vector<core_t::TTime>;
using TTimeCRng = core::CVectorRange<const TTimeVec>;
using TFloatMeanAccumulator =
    maths::CBasicStatistics::SSampleMean<maths::CFloatStorage>::TAccumulator;
using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;

TTimeVec BUCKET_LENGTHS{300, 600, 1800, 3600};
}

void CExpandingWindowTest::testPersistence() {
    // Test persist and restore is idempotent.

    core_t::TTime bucketLength{300};
    std::size_t size{336};
    double decayRate{0.01};

    test::CRandomNumbers rng;

    maths::CExpandingWindow origWindow{bucketLength, TTimeCRng{BUCKET_LENGTHS, 0, 4},
                                       size, decayRate};

    TDoubleVec values;
    rng.generateUniformSamples(0.0, 10.0, size, values);
    for (core_t::TTime time = 0; time < static_cast<core_t::TTime>(size) * bucketLength;
         time += bucketLength) {
        double value{values[time / bucketLength]};
        origWindow.add(time, value);
    }

    std::string origXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        origWindow.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }
    LOG_TRACE(<< "Window XML = " << origXml);
    LOG_DEBUG(<< "Window XML size = " << origXml.size());

    // Restore the XML into a new window.
    {
        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        maths::CExpandingWindow restoredWindow{
            bucketLength, TTimeCRng{BUCKET_LENGTHS, 0, 4}, size, decayRate};
        CPPUNIT_ASSERT_EQUAL(
            true, traverser.traverseSubLevel(boost::bind(&maths::CExpandingWindow::acceptRestoreTraverser,
                                                         &restoredWindow, _1)));

        LOG_DEBUG(<< "orig checksum = " << origWindow.checksum()
                  << ", new checksum = " << restoredWindow.checksum());
        CPPUNIT_ASSERT_EQUAL(origWindow.checksum(), restoredWindow.checksum());
    }
}

CppUnit::Test* CExpandingWindowTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CExpandingWindowTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CExpandingWindowTest>(
        "CExpandingWindowTest::testPersistence", &CExpandingWindowTest::testPersistence));

    return suiteOfTests;
}
