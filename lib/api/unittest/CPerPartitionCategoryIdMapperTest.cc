/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>

#include <api/CPerPartitionCategoryIdMapper.h>

#include <boost/test/unit_test.hpp>

#include <sstream>

BOOST_AUTO_TEST_SUITE(CPerPartitionCategoryIdMapperTest)

namespace {
void persistAndRestore(const ml::api::CPerPartitionCategoryIdMapper& persistFrom,
                       ml::api::CPerPartitionCategoryIdMapper& restoreTo) {
    std::stringstream jsonStrm;
    {
        ml::core::CJsonStatePersistInserter inserter(jsonStrm);
        persistFrom.acceptPersistInserter(inserter);
    }

    LOG_DEBUG(<< "JSON representation is: " << jsonStrm.str());

    jsonStrm.seekg(0);
    ml::core::CJsonStateRestoreTraverser traverser(jsonStrm);
    BOOST_TEST_REQUIRE(restoreTo.acceptRestoreTraverser(traverser));
}
}

BOOST_AUTO_TEST_CASE(testLocalToGlobal) {
    std::string p1{"p1"};
    std::string p2{"p2"};

    auto assertions = [&p1, &p2](ml::api::CPerPartitionCategoryIdMapper& categoryIdMapper) {
        BOOST_REQUIRE_EQUAL(
            ml::api::CGlobalCategoryId(-2, p1, ml::model::CLocalCategoryId{-2}),
            categoryIdMapper.map("p1", ml::model::CLocalCategoryId{-2}));
        BOOST_REQUIRE_EQUAL(
            ml::api::CGlobalCategoryId(-2, p2, ml::model::CLocalCategoryId{-2}),
            categoryIdMapper.map("p2", ml::model::CLocalCategoryId{-2}));
        BOOST_REQUIRE_EQUAL(
            ml::api::CGlobalCategoryId(-1, p1, ml::model::CLocalCategoryId{-1}),
            categoryIdMapper.map("p1", ml::model::CLocalCategoryId{-1}));
        BOOST_REQUIRE_EQUAL(
            ml::api::CGlobalCategoryId(-1, p2, ml::model::CLocalCategoryId{-1}),
            categoryIdMapper.map("p2", ml::model::CLocalCategoryId{-1}));
        BOOST_REQUIRE_EQUAL(
            ml::api::CGlobalCategoryId(1, p1, ml::model::CLocalCategoryId{1}),
            categoryIdMapper.map("p1", ml::model::CLocalCategoryId{1}));
        BOOST_REQUIRE_EQUAL(
            ml::api::CGlobalCategoryId(2, p1, ml::model::CLocalCategoryId{2}),
            categoryIdMapper.map("p1", ml::model::CLocalCategoryId{2}));
        BOOST_REQUIRE_EQUAL(
            ml::api::CGlobalCategoryId(3, p2, ml::model::CLocalCategoryId{1}),
            categoryIdMapper.map("p2", ml::model::CLocalCategoryId{1}));
        BOOST_REQUIRE_EQUAL(
            ml::api::CGlobalCategoryId(4, p2, ml::model::CLocalCategoryId{2}),
            categoryIdMapper.map("p2", ml::model::CLocalCategoryId{2}));
        BOOST_REQUIRE_EQUAL(
            ml::api::CGlobalCategoryId(5, p1, ml::model::CLocalCategoryId{3}),
            categoryIdMapper.map("p1", ml::model::CLocalCategoryId{3}));
        BOOST_REQUIRE_EQUAL(
            ml::api::CGlobalCategoryId(2, p1, ml::model::CLocalCategoryId{2}),
            categoryIdMapper.map("p1", ml::model::CLocalCategoryId{2}));
        BOOST_REQUIRE_EQUAL(
            ml::api::CGlobalCategoryId(3, p2, ml::model::CLocalCategoryId{1}),
            categoryIdMapper.map("p2", ml::model::CLocalCategoryId{1}));
    };

    ml::api::CPerPartitionCategoryIdMapper origCategoryIdMapper;

    assertions(origCategoryIdMapper);

    ml::api::CPerPartitionCategoryIdMapper restoredCategoryIdMapper;
    persistAndRestore(origCategoryIdMapper, restoredCategoryIdMapper);

    assertions(restoredCategoryIdMapper);

    // Restore should have remembered the highest ever global ID so the next one is 6
    BOOST_REQUIRE_EQUAL(
        ml::api::CGlobalCategoryId(6, p2, ml::model::CLocalCategoryId{3}),
        restoredCategoryIdMapper.map("p2", ml::model::CLocalCategoryId{3}));
}

BOOST_AUTO_TEST_SUITE_END()
