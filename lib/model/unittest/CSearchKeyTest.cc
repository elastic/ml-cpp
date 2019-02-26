/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CSearchKeyTest.h"

#include <model/CSearchKey.h>

#include <test/CRandomNumbers.h>

#include <string>
#include <vector>

using namespace ml;
using namespace model;

void CSearchKeyTest::testSimpleCountComesFirst() {
    using TStrVec = std::vector<std::string>;
    using TExcludeFrequentVec = std::vector<model_t::EExcludeFrequent>;
    using TFunctionVec = std::vector<function_t::EFunction>;
    test::CRandomNumbers rng;
    TExcludeFrequentVec excludeFrequents{model_t::E_XF_None, model_t::E_XF_By,
                                         model_t::E_XF_Over, model_t::E_XF_Both};
    TFunctionVec functions{function_t::E_IndividualCount, function_t::E_IndividualMetric,
                           function_t::E_PopulationMetricMean, function_t::E_PopulationRare};

    for (std::size_t i = 0; i < 100; ++i) {
        function_t::EFunction function{functions[i % functions.size()]};
        bool useNull{i % 2 == 0};
        model_t::EExcludeFrequent excludeFrequent{
            excludeFrequents[i % excludeFrequents.size()]};
        TStrVec fields;
        rng.generateWords(10, 3, fields);
        CSearchKey key{static_cast<int>(i + 1),
                       function,
                       useNull,
                       excludeFrequent,
                       fields[0],
                       fields[1],
                       fields[2]};
        CPPUNIT_ASSERT(CSearchKey::simpleCountKey() < key);
    }
}

CppUnit::Test* CSearchKeyTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CSearchKeyTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CSearchKeyTest>(
        "CSearchKeyTest::testSimpleCountComesFirst", &CSearchKeyTest::testSimpleCountComesFirst));

    return suiteOfTests;
}
