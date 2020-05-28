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
    auto assertions = [](ml::api::CPerPartitionCategoryIdMapper& categoryIdMapper) {
        BOOST_REQUIRE_EQUAL(
            -2, categoryIdMapper.globalCategoryIdForLocalCategoryId("p1", -2));
        BOOST_REQUIRE_EQUAL("-2", categoryIdMapper.printMapping("p1", -2));
        BOOST_REQUIRE_EQUAL(
            -2, categoryIdMapper.globalCategoryIdForLocalCategoryId("p2", -2));
        BOOST_REQUIRE_EQUAL("-2", categoryIdMapper.printMapping("p2", -2));
        BOOST_REQUIRE_EQUAL(
            -1, categoryIdMapper.globalCategoryIdForLocalCategoryId("p1", -1));
        BOOST_REQUIRE_EQUAL("-1", categoryIdMapper.printMapping("p1", -1));
        BOOST_REQUIRE_EQUAL(
            -1, categoryIdMapper.globalCategoryIdForLocalCategoryId("p2", -1));
        BOOST_REQUIRE_EQUAL("-1", categoryIdMapper.printMapping("p2", -1));
        BOOST_REQUIRE_EQUAL(1, categoryIdMapper.globalCategoryIdForLocalCategoryId("p1", 1));
        BOOST_REQUIRE_EQUAL("p1/1;1", categoryIdMapper.printMapping("p1", 1));
        BOOST_REQUIRE_EQUAL(2, categoryIdMapper.globalCategoryIdForLocalCategoryId("p1", 2));
        BOOST_REQUIRE_EQUAL("p1/2;2", categoryIdMapper.printMapping("p1", 2));
        BOOST_REQUIRE_EQUAL(3, categoryIdMapper.globalCategoryIdForLocalCategoryId("p2", 1));
        BOOST_REQUIRE_EQUAL("p2/1;3", categoryIdMapper.printMapping("p2", 1));
        BOOST_REQUIRE_EQUAL(4, categoryIdMapper.globalCategoryIdForLocalCategoryId("p2", 2));
        BOOST_REQUIRE_EQUAL("p2/2;4", categoryIdMapper.printMapping("p2", 2));
        BOOST_REQUIRE_EQUAL(5, categoryIdMapper.globalCategoryIdForLocalCategoryId("p1", 3));
        BOOST_REQUIRE_EQUAL("p1/3;5", categoryIdMapper.printMapping("p1", 3));
        BOOST_REQUIRE_EQUAL(2, categoryIdMapper.globalCategoryIdForLocalCategoryId("p1", 2));
        BOOST_REQUIRE_EQUAL("p1/2;2", categoryIdMapper.printMapping("p1", 2));
        BOOST_REQUIRE_EQUAL(3, categoryIdMapper.globalCategoryIdForLocalCategoryId("p2", 1));
        BOOST_REQUIRE_EQUAL("p2/1;3", categoryIdMapper.printMapping("p2", 1));
    };

    ml::api::CPerPartitionCategoryIdMapper origCategoryIdMapper;

    assertions(origCategoryIdMapper);

    ml::api::CPerPartitionCategoryIdMapper restoredCategoryIdMapper;
    persistAndRestore(origCategoryIdMapper, restoredCategoryIdMapper);

    assertions(restoredCategoryIdMapper);

    // Restore should have remembered the highest ever global ID so the next one is 6
    BOOST_REQUIRE_EQUAL(
        6, restoredCategoryIdMapper.globalCategoryIdForLocalCategoryId("p2", 3));
    BOOST_REQUIRE_EQUAL("p2/3;6", restoredCategoryIdMapper.printMapping("p2", 3));
}

BOOST_AUTO_TEST_CASE(testGlobalToLocal) {
    auto assertions = [](const ml::api::CPerPartitionCategoryIdMapper& categoryIdMapper) {
        BOOST_REQUIRE_EQUAL("", categoryIdMapper.categorizerKeyForGlobalCategoryId(-2));
        BOOST_REQUIRE_EQUAL(-2, categoryIdMapper.localCategoryIdForGlobalCategoryId(-2));
        BOOST_REQUIRE_EQUAL("-2", categoryIdMapper.printMapping(-2));
        BOOST_REQUIRE_EQUAL("", categoryIdMapper.categorizerKeyForGlobalCategoryId(-1));
        BOOST_REQUIRE_EQUAL(-1, categoryIdMapper.localCategoryIdForGlobalCategoryId(-1));
        BOOST_REQUIRE_EQUAL("-1", categoryIdMapper.printMapping(-1));
        BOOST_REQUIRE_EQUAL("p1", categoryIdMapper.categorizerKeyForGlobalCategoryId(1));
        BOOST_REQUIRE_EQUAL(1, categoryIdMapper.localCategoryIdForGlobalCategoryId(1));
        BOOST_REQUIRE_EQUAL("p1/1;1", categoryIdMapper.printMapping(1));
        BOOST_REQUIRE_EQUAL("p1", categoryIdMapper.categorizerKeyForGlobalCategoryId(2));
        BOOST_REQUIRE_EQUAL(2, categoryIdMapper.localCategoryIdForGlobalCategoryId(2));
        BOOST_REQUIRE_EQUAL("p1/2;2", categoryIdMapper.printMapping(2));
        BOOST_REQUIRE_EQUAL("p2", categoryIdMapper.categorizerKeyForGlobalCategoryId(3));
        BOOST_REQUIRE_EQUAL(1, categoryIdMapper.localCategoryIdForGlobalCategoryId(3));
        BOOST_REQUIRE_EQUAL("p2/1;3", categoryIdMapper.printMapping(3));
        BOOST_REQUIRE_EQUAL("p3", categoryIdMapper.categorizerKeyForGlobalCategoryId(4));
        BOOST_REQUIRE_EQUAL(1, categoryIdMapper.localCategoryIdForGlobalCategoryId(4));
        BOOST_REQUIRE_EQUAL("p3/1;4", categoryIdMapper.printMapping(4));
        BOOST_REQUIRE_EQUAL("p1", categoryIdMapper.categorizerKeyForGlobalCategoryId(5));
        BOOST_REQUIRE_EQUAL(3, categoryIdMapper.localCategoryIdForGlobalCategoryId(5));
        BOOST_REQUIRE_EQUAL("p1/3;5", categoryIdMapper.printMapping(5));
        BOOST_REQUIRE_EQUAL("p3", categoryIdMapper.categorizerKeyForGlobalCategoryId(6));
        BOOST_REQUIRE_EQUAL(2, categoryIdMapper.localCategoryIdForGlobalCategoryId(6));
        BOOST_REQUIRE_EQUAL("p3/2;6", categoryIdMapper.printMapping(6));
    };

    ml::api::CPerPartitionCategoryIdMapper origCategoryIdMapper;

    // We need to map local to global first to populate
    BOOST_REQUIRE_EQUAL(1, origCategoryIdMapper.globalCategoryIdForLocalCategoryId("p1", 1));
    BOOST_REQUIRE_EQUAL(2, origCategoryIdMapper.globalCategoryIdForLocalCategoryId("p1", 2));
    BOOST_REQUIRE_EQUAL(3, origCategoryIdMapper.globalCategoryIdForLocalCategoryId("p2", 1));
    BOOST_REQUIRE_EQUAL(4, origCategoryIdMapper.globalCategoryIdForLocalCategoryId("p3", 1));
    BOOST_REQUIRE_EQUAL(5, origCategoryIdMapper.globalCategoryIdForLocalCategoryId("p1", 3));
    BOOST_REQUIRE_EQUAL(6, origCategoryIdMapper.globalCategoryIdForLocalCategoryId("p3", 2));

    // Now test
    assertions(origCategoryIdMapper);

    ml::api::CPerPartitionCategoryIdMapper restoredCategoryIdMapper;
    persistAndRestore(origCategoryIdMapper, restoredCategoryIdMapper);

    assertions(restoredCategoryIdMapper);

    // Restore should have remembered the highest ever global ID so the next one is 7
    BOOST_REQUIRE_EQUAL(
        7, restoredCategoryIdMapper.globalCategoryIdForLocalCategoryId("p2", 2));
    BOOST_REQUIRE_EQUAL("p2/2;7", restoredCategoryIdMapper.printMapping("p2", 2));
}

BOOST_AUTO_TEST_SUITE_END()
