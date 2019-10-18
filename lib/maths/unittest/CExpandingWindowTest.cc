/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/Constants.h>

#include <maths/CBasicStatistics.h>
#include <maths/CExpandingWindow.h>

#include <test/CRandomNumbers.h>
#include <test/BoostTestCloseAbsolute.h>

#include <boost/test/unit_test.hpp>
#include <boost/math/constants/constants.hpp>

#include <vector>

BOOST_AUTO_TEST_SUITE(CExpandingWindowTest)

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

BOOST_AUTO_TEST_CASE(testBasicUsage) {
    // 1) Check compressed and uncompressed storage formats produce the
    //    same results.
    // 2) Check multiple rounds of bucket compression work as expected.
    // 3) Check extension beyond the end of the maximum window length
    //    works as expected.

    core_t::TTime bucketLength{300};
    std::size_t size{336};
    double decayRate{0.01};

    test::CRandomNumbers rng;
    TDoubleVec values;
    rng.generateUniformSamples(0.0, 10.0, size, values);

    for (auto startTime : {0, 100000}) {
        LOG_DEBUG(<< "Testing start time " << startTime);

        maths::CExpandingWindow compressed{
            bucketLength, TTimeCRng{BUCKET_LENGTHS, 0, 4}, size, decayRate, true};
        maths::CExpandingWindow uncompressed{
            bucketLength, TTimeCRng{BUCKET_LENGTHS, 0, 4}, size, decayRate, false};
        TFloatMeanAccumulatorVec expected300(size);
        TFloatMeanAccumulatorVec expected1800(size);

        compressed.initialize(startTime);
        uncompressed.initialize(startTime);

        for (core_t::TTime time = startTime;
             time < static_cast<core_t::TTime>(size) * bucketLength; time += bucketLength) {
            double value{values[(time - startTime) / bucketLength]};
            compressed.add(time, value);
            uncompressed.add(time, value);
            expected300[(time - startTime) / 300].add(value);
            expected1800[(time - startTime) / 1800].add(value);
            if (((time - startTime) / bucketLength) % 3 == 0) {
                BOOST_CHECK_EQUAL(uncompressed.checksum(), compressed.checksum());
            }
        }
        BOOST_CHECK_EQUAL(core::CContainerPrinter::print(expected300),
                             core::CContainerPrinter::print(uncompressed.values()));
        BOOST_CHECK_EQUAL(core::CContainerPrinter::print(expected300),
                             core::CContainerPrinter::print(compressed.values()));

        core_t::TTime time{startTime + static_cast<core_t::TTime>(600 * size + 1)};
        compressed.add(time, 5.0, 0.9);
        uncompressed.add(time, 5.0, 0.9);
        expected1800[(time - startTime) / 1800].add(5.0, 0.9);

        BOOST_CHECK_EQUAL(core::CContainerPrinter::print(expected1800),
                             core::CContainerPrinter::print(uncompressed.values()));
        BOOST_CHECK_EQUAL(core::CContainerPrinter::print(expected1800),
                             core::CContainerPrinter::print(compressed.values()));
    }

    LOG_DEBUG(<< "Testing multiple rounds of compression");

    TDoubleVec times;
    rng.generateUniformSamples(0.0, static_cast<double>(3600 * size), 1000, times);
    rng.generateUniformSamples(0.0, 10.0, 1000, values);
    std::sort(times.begin(), times.end());

    maths::CExpandingWindow window{bucketLength, TTimeCRng{BUCKET_LENGTHS, 0, 4},
                                   size, decayRate, true};
    TFloatMeanAccumulatorVec expected300(size);
    TFloatMeanAccumulatorVec expected600(size);
    TFloatMeanAccumulatorVec expected1800(size);
    TFloatMeanAccumulatorVec expected3600(size);

    window.initialize(0);

    for (std::size_t i = 0; i < times.size(); ++i) {
        core_t::TTime time{static_cast<core_t::TTime>(times[i])};
        LOG_TRACE(<< "Adding " << values[i] << " at " << time);

        window.add(time, values[i]);
        window.propagateForwardsByTime(1.0);

        expected300[(time / 300) % size].add(values[i]);
        expected600[(time / 600) % size].add(values[i]);
        expected1800[(time / 1800) % size].add(values[i]);
        expected3600[(time / 3600) % size].add(values[i]);
        double factor{std::exp(-decayRate)};
        for (std::size_t j = 0; j < size; ++j) {
            expected300[j].age(factor);
            expected600[j].age(factor);
            expected1800[j].age(factor);
            expected3600[j].age(factor);
        }

        const TFloatMeanAccumulatorVec& expected{[&](core_t::TTime) -> const TFloatMeanAccumulatorVec& {
            if (time < static_cast<core_t::TTime>(size * 300)) {
                return expected300;
            } else if (time < static_cast<core_t::TTime>(size * 600)) {
                return expected600;
            } else if (time < static_cast<core_t::TTime>(size * 1800)) {
                return expected1800;
            }
            return expected3600;
        }(time)};
        TFloatMeanAccumulatorVec actual{window.values()};

        for (std::size_t j = 0; j < size; ++j) {
            BOOST_CHECK_CLOSE_ABSOLUTE(maths::CBasicStatistics::count(expected[j]),
                                         maths::CBasicStatistics::count(actual[j]), 1e-5);
            BOOST_CHECK_CLOSE_ABSOLUTE(maths::CBasicStatistics::mean(expected[j]),
                                         maths::CBasicStatistics::mean(actual[j]), 1e-5);
        }
    }

    LOG_DEBUG(<< "Testing overflow");

    window.add(static_cast<core_t::TTime>(size * 3600 + 1), 0.1);
    BOOST_CHECK_EQUAL(static_cast<core_t::TTime>(size * 3600), window.startTime());
    BOOST_CHECK_EQUAL(static_cast<core_t::TTime>(size * 3900), window.endTime());

    TFloatMeanAccumulatorVec expected(size);
    expected[0].add(0.1);
    TFloatMeanAccumulatorVec actual{window.values()};

    BOOST_CHECK_EQUAL(core::CContainerPrinter::print(expected),
                         core::CContainerPrinter::print(actual));
}

BOOST_AUTO_TEST_CASE(testValuesMinusPrediction) {
    // Test we get back the values we expect.

    maths::CExpandingWindow::TPredictor trend = [](core_t::TTime time) {
        return 10.0 * std::sin(boost::math::double_constants::two_pi *
                               static_cast<double>(time) /
                               static_cast<double>(core::constants::DAY));
    };

    core_t::TTime bucketLength{300};
    std::size_t size{336};
    double decayRate{0.01};

    test::CRandomNumbers rng;
    TDoubleVec noise;
    rng.generateUniformSamples(0.0, 2.0, size, noise);

    maths::CExpandingWindow window{bucketLength, TTimeCRng{BUCKET_LENGTHS, 0, 4},
                                   size, decayRate, true};

    TFloatMeanAccumulatorVec expected(size);
    for (core_t::TTime time = 0; time < static_cast<core_t::TTime>(size) * bucketLength;
         time += bucketLength) {
        window.add(time, trend(time) + noise[time / bucketLength]);
        expected[time / bucketLength].add(noise[time / bucketLength]);
    }
    TFloatMeanAccumulatorVec actual{window.valuesMinusPrediction(trend)};

    for (std::size_t i = 0; i < size; ++i) {
        BOOST_CHECK_EQUAL(maths::CBasicStatistics::count(expected[i]),
                             maths::CBasicStatistics::count(actual[i]));
        BOOST_CHECK_CLOSE_ABSOLUTE(maths::CBasicStatistics::mean(expected[i]),
                                     maths::CBasicStatistics::mean(actual[i]), 1e-5);
    }
}

BOOST_AUTO_TEST_CASE(testPersistence) {
    // Test persist and restore is idempotent.

    core_t::TTime bucketLength{300};
    std::size_t size{336};
    double decayRate{0.01};

    test::CRandomNumbers rng;

    for (auto compressed : {true, false}) {
        maths::CExpandingWindow origWindow{
            bucketLength, TTimeCRng{BUCKET_LENGTHS, 0, 4}, size, decayRate, compressed};

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
            BOOST_TEST(parser.parseStringIgnoreCdata(origXml));
            core::CRapidXmlStateRestoreTraverser traverser(parser);
            maths::CExpandingWindow restoredWindow{
                bucketLength, TTimeCRng{BUCKET_LENGTHS, 0, 4}, size, decayRate, compressed};
            BOOST_CHECK_EQUAL(true, traverser.traverseSubLevel(std::bind(
                                           &maths::CExpandingWindow::acceptRestoreTraverser,
                                           &restoredWindow, std::placeholders::_1)));

            LOG_DEBUG(<< "orig checksum = " << origWindow.checksum()
                      << ", new checksum = " << restoredWindow.checksum());
            BOOST_CHECK_EQUAL(origWindow.checksum(), restoredWindow.checksum());
        }
    }
}


BOOST_AUTO_TEST_SUITE_END()
