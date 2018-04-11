/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CDataSemanticsTest.h"

#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <config/CDataSemantics.h>

#include <test/CRandomNumbers.h>
#include <test/CRandomNumbersDetail.h>

#include <boost/range.hpp>

using namespace ml;

using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TStrVec = std::vector<std::string>;

void CDataSemanticsTest::testBinary() {
    LOG_DEBUG(<< "");
    LOG_DEBUG(<< "+----------------------------------+");
    LOG_DEBUG(<< "|  CDataSemanticsTest::testBinary  |");
    LOG_DEBUG(<< "+----------------------------------+");

    // Try a numeric and non-numeric example of a binary variable.

    std::string categories[][2] = {{"false", "true"}, {"0", "1"}};

    test::CRandomNumbers rng;

    TSizeVec v;
    rng.generateUniformSamples(0, 2, 100, v);

    for (std::size_t i = 0u; i < boost::size(categories); ++i) {
        config::CDataSemantics semantics;
        CPPUNIT_ASSERT_EQUAL(config_t::E_UndeterminedType, semantics.type());
        for (std::size_t j = 0u; j < v.size(); ++j) {
            semantics.add(categories[i][v[j]]);
        }
        semantics.computeType();
        CPPUNIT_ASSERT_EQUAL(config_t::E_Binary, semantics.type());
    }
}

void CDataSemanticsTest::testNonNumericCategorical() {
    LOG_DEBUG(<< "");
    LOG_DEBUG(<< "+-------------------------------------------------+");
    LOG_DEBUG(<< "|  CDataSemanticsTest::testNonNumericCategorical  |");
    LOG_DEBUG(<< "+-------------------------------------------------+");

    // Test we identify non-numerical non-binary data as categorical.

    test::CRandomNumbers rng;

    TStrVec categories;
    rng.generateWords(10, 100, categories);
    LOG_DEBUG(<< "categories = " << core::CContainerPrinter::print(categories));

    TSizeVec samples;
    rng.generateUniformSamples(0, categories.size(), 1000, samples);

    config::CDataSemantics semantics;

    for (std::size_t i = 0u; i < samples.size(); ++i) {
        semantics.add(categories[samples[i]]);
        if (i > 10) {
            semantics.computeType();
            CPPUNIT_ASSERT_EQUAL(config_t::E_Categorical, semantics.type());
        }
    }
}

void CDataSemanticsTest::testNumericCategorical() {
    LOG_DEBUG(<< "");
    LOG_DEBUG(<< "+----------------------------------------------+");
    LOG_DEBUG(<< "|  CDataSemanticsTest::testNumericCategorical  |");
    LOG_DEBUG(<< "+----------------------------------------------+");

    // Test plausible http status code distribution is correctly
    // identified as categorical.

    double codes[] = {200, 201, 202, 303, 400, 403, 404, 500, 501, 503, 506, 598, 599};
    double frequencies[] = {0.7715, 0.03, 0.05, 0.001, 0.005, 0.041, 0.061, 0.002, 0.0005, 0.021, 0.001, 0.002, 0.014};

    test::CRandomNumbers rng;

    TDoubleVec status;
    rng.generateMultinomialSamples(
        TDoubleVec(boost::begin(codes), boost::end(codes)), TDoubleVec(boost::begin(frequencies), boost::end(frequencies)), 5000, status);

    config::CDataSemantics semantics;

    for (std::size_t i = 0u; i < status.size(); ++i) {
        semantics.add(core::CStringUtils::typeToString(static_cast<unsigned int>(status[i])));
    }
    semantics.computeType();
    LOG_DEBUG(<< "type = " << semantics.type());
    CPPUNIT_ASSERT_EQUAL(config_t::E_Categorical, semantics.type());
}

void CDataSemanticsTest::testInteger() {
    LOG_DEBUG(<< "");
    LOG_DEBUG(<< "+-----------------------------------+");
    LOG_DEBUG(<< "|  CDataSemanticsTest::testInteger  |");
    LOG_DEBUG(<< "+-----------------------------------+");

    // Test a variety of uni- and multi-modal distributions.

    test::CRandomNumbers rng;

    TDoubleVec samples;
    {
        LOG_DEBUG(<< "*** uniform ***");

        rng.generateUniformSamples(0.0, 25.0, 500, samples);

        config::CDataSemantics semantics;
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            if (i % 10 == 0) {
                LOG_DEBUG(<< "    adding " << static_cast<int>(samples[i]));
            }
            semantics.add(core::CStringUtils::typeToString(static_cast<int>(samples[i])));
            //if ((i + 1) % 500 == 0)
            //{
            //    semantics.computeType();
            //    LOG_DEBUG(<< "  type = " << semantics.type());
            //    CPPUNIT_ASSERT_EQUAL(config_t::E_PositiveInteger, semantics.type());
            //}
        }
        semantics.computeType();
        LOG_DEBUG(<< "  type = " << semantics.type());
        CPPUNIT_ASSERT_EQUAL(config_t::E_PositiveInteger, semantics.type());
    }
    {
        LOG_DEBUG(<< "*** normal ***");

        rng.generateNormalSamples(-10.0, 100.0, 500, samples);

        config::CDataSemantics semantics;
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            if (i % 10 == 0) {
                LOG_DEBUG(<< "    adding " << static_cast<int>(samples[i]));
            }
            semantics.add(core::CStringUtils::typeToString(static_cast<int>(samples[i])));
            if ((i + 1) % 100 == 0) {
                semantics.computeType();
                LOG_DEBUG(<< "  type = " << semantics.type());
                CPPUNIT_ASSERT_EQUAL(config_t::E_Integer, semantics.type());
            }
        }
    }
    {
        LOG_DEBUG(<< "*** log-normal ***");

        rng.generateLogNormalSamples(0.1, 2.0, 500, samples);

        config::CDataSemantics semantics;
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            if (i % 10 == 0) {
                LOG_DEBUG(<< "    adding " << static_cast<int>(samples[i]));
            }
            semantics.add(core::CStringUtils::typeToString(static_cast<int>(samples[i])));
            if ((i + 1) % 100 == 0) {
                semantics.computeType();
                LOG_DEBUG(<< "  type = " << semantics.type());
                CPPUNIT_ASSERT_EQUAL(config_t::E_PositiveInteger, semantics.type());
            }
        }
    }
    {
        LOG_DEBUG(<< "*** mixture ***");

        TDoubleVec modeSamples;
        rng.generateNormalSamples(-100.0, 1000.0, 40, modeSamples);
        samples.assign(modeSamples.begin(), modeSamples.end());
        rng.generateUniformSamples(-10.0, 20.0, 50, modeSamples);
        samples.insert(samples.end(), modeSamples.begin(), modeSamples.end());
        rng.generateLogNormalSamples(2.0, 1.0, 20, modeSamples);
        samples.insert(samples.end(), modeSamples.begin(), modeSamples.end());
        rng.random_shuffle(samples.begin(), samples.end());

        config::CDataSemantics semantics;
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            if (i % 2 == 0) {
                LOG_DEBUG(<< "    adding " << static_cast<int>(samples[i]));
            }
            semantics.add(core::CStringUtils::typeToString(static_cast<int>(samples[i])));
            if ((i + 1) % 10 == 0) {
                semantics.computeType();
                LOG_DEBUG(<< "  type = " << semantics.type());
                if (i > 30) {
                    CPPUNIT_ASSERT_EQUAL(config_t::E_Integer, semantics.type());
                }
            }
        }
    }
}

void CDataSemanticsTest::testReal() {
    LOG_DEBUG(<< "");
    LOG_DEBUG(<< "+--------------------------------+");
    LOG_DEBUG(<< "|  CDataSemanticsTest::testReal  |");
    LOG_DEBUG(<< "+--------------------------------+");

    // Test a variety of uni- and multi-modal distributions.

    test::CRandomNumbers rng;

    TDoubleVec samples;
    {
        LOG_DEBUG(<< "*** uniform ***");

        rng.generateUniformSamples(0.0, 10.0, 500, samples);

        config::CDataSemantics semantics;
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            if (i % 10 == 0) {
                LOG_DEBUG(<< "    adding " << samples[i]);
            }
            semantics.add(core::CStringUtils::typeToString(samples[i]));
            if ((i + 1) % 50 == 0) {
                semantics.computeType();
                LOG_DEBUG(<< "  type = " << semantics.type());
                CPPUNIT_ASSERT_EQUAL(config_t::E_PositiveReal, semantics.type());
            }
        }
    }
    {
        LOG_DEBUG(<< "*** normal ***");

        rng.generateNormalSamples(-10.0, 100.0, 500, samples);

        config::CDataSemantics semantics;
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            if (i % 10 == 0) {
                LOG_DEBUG(<< "    adding " << samples[i]);
            }
            semantics.add(core::CStringUtils::typeToString(samples[i]));
            if ((i + 1) % 50 == 0) {
                semantics.computeType();
                LOG_DEBUG(<< "  type = " << semantics.type());
                CPPUNIT_ASSERT_EQUAL(config_t::E_Real, semantics.type());
            }
        }
    }
    {
        LOG_DEBUG(<< "*** log-normal ***");

        rng.generateLogNormalSamples(0.1, 1.5, 500, samples);

        config::CDataSemantics semantics;
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            if (i % 10 == 0) {
                LOG_DEBUG(<< "    adding " << samples[i]);
            }
            semantics.add(core::CStringUtils::typeToString(samples[i]));
            if ((i + 1) % 50 == 0) {
                semantics.computeType();
                LOG_DEBUG(<< "  type = " << semantics.type());
                CPPUNIT_ASSERT_EQUAL(config_t::E_PositiveReal, semantics.type());
            }
        }
    }
    {
        LOG_DEBUG(<< "*** mixture ***");

        TDoubleVec modeSamples;
        rng.generateNormalSamples(-100.0, 1000.0, 20, modeSamples);
        samples.assign(modeSamples.begin(), modeSamples.end());
        rng.generateUniformSamples(-10.0, 10.0, 30, modeSamples);
        samples.insert(samples.end(), modeSamples.begin(), modeSamples.end());
        rng.generateLogNormalSamples(2.0, 1.0, 10, modeSamples);
        samples.insert(samples.end(), modeSamples.begin(), modeSamples.end());
        rng.random_shuffle(samples.begin(), samples.end());

        config::CDataSemantics semantics;
        for (std::size_t i = 0u; i < samples.size(); ++i) {
            LOG_DEBUG(<< "    adding " << samples[i]);
            semantics.add(core::CStringUtils::typeToString(samples[i]));
            if ((i + 1) % 5 == 0) {
                semantics.computeType();
                LOG_DEBUG(<< "  type = " << semantics.type());
                if (i > 25) {
                    CPPUNIT_ASSERT_EQUAL(config_t::E_Real, semantics.type());
                }
            }
        }
    }
}

CppUnit::Test* CDataSemanticsTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CDataSemanticsTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CDataSemanticsTest>("CDataSemanticsTest::testBinary", &CDataSemanticsTest::testBinary));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataSemanticsTest>("CDataSemanticsTest::testNonNumericCategorical",
                                                                      &CDataSemanticsTest::testNonNumericCategorical));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataSemanticsTest>("CDataSemanticsTest::testNumericCategorical",
                                                                      &CDataSemanticsTest::testNumericCategorical));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataSemanticsTest>("CDataSemanticsTest::testInteger", &CDataSemanticsTest::testInteger));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataSemanticsTest>("CDataSemanticsTest::testReal", &CDataSemanticsTest::testReal));

    return suiteOfTests;
}
