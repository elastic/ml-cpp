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

    const std::string partitionFieldValue1{"p1"};
    const std::string partitionFieldValue2{"p1"};

    int highestGlobalId{0};
    ml::api::CPerPartitionCategoryIdMapper::TNextGlobalIdSupplier nextGlobalIdSupplier{
        [&highestGlobalId]() { return ++highestGlobalId; }};

    auto assertions = [&partitionFieldValue1, &partitionFieldValue2](
                          ml::api::CPerPartitionCategoryIdMapper& categoryIdMapper1,
                          ml::api::CPerPartitionCategoryIdMapper& categoryIdMapper2) {
        BOOST_REQUIRE_EQUAL(
            ml::api::CGlobalCategoryId::hardFailure(),
            categoryIdMapper1.map(ml::model::CLocalCategoryId::hardFailure()));
        BOOST_REQUIRE_EQUAL(
            ml::api::CGlobalCategoryId::hardFailure(),
            categoryIdMapper2.map(ml::model::CLocalCategoryId::hardFailure()));
        BOOST_REQUIRE_EQUAL(ml::api::CGlobalCategoryId(1, partitionFieldValue1,
                                                       ml::model::CLocalCategoryId{1}),
                            categoryIdMapper1.map(ml::model::CLocalCategoryId{1}));
        BOOST_REQUIRE_EQUAL(ml::api::CGlobalCategoryId(2, partitionFieldValue1,
                                                       ml::model::CLocalCategoryId{2}),
                            categoryIdMapper1.map(ml::model::CLocalCategoryId{2}));
        BOOST_REQUIRE_EQUAL(ml::api::CGlobalCategoryId(3, partitionFieldValue2,
                                                       ml::model::CLocalCategoryId{1}),
                            categoryIdMapper2.map(ml::model::CLocalCategoryId{1}));
        BOOST_REQUIRE_EQUAL(ml::api::CGlobalCategoryId(2, partitionFieldValue1,
                                                       ml::model::CLocalCategoryId{2}),
                            categoryIdMapper1.map(ml::model::CLocalCategoryId{2}));
        BOOST_REQUIRE_EQUAL(ml::api::CGlobalCategoryId(3, partitionFieldValue2,
                                                       ml::model::CLocalCategoryId{1}),
                            categoryIdMapper2.map(ml::model::CLocalCategoryId{1}));
        BOOST_REQUIRE_EQUAL(ml::api::CGlobalCategoryId(4, partitionFieldValue1,
                                                       ml::model::CLocalCategoryId{3}),
                            categoryIdMapper1.map(ml::model::CLocalCategoryId{3}));
        BOOST_REQUIRE_EQUAL(ml::api::CGlobalCategoryId(5, partitionFieldValue2,
                                                       ml::model::CLocalCategoryId{2}),
                            categoryIdMapper2.map(ml::model::CLocalCategoryId{2}));
        BOOST_REQUIRE_EQUAL(
            ml::api::CGlobalCategoryId::softFailure(),
            categoryIdMapper2.map(ml::model::CLocalCategoryId::softFailure()));
        BOOST_REQUIRE_EQUAL(
            ml::api::CGlobalCategoryId::softFailure(),
            categoryIdMapper1.map(ml::model::CLocalCategoryId::softFailure()));
    };

    ml::api::CPerPartitionCategoryIdMapper origCategoryIdMapper1{
        partitionFieldValue1, nextGlobalIdSupplier};
    ml::api::CPerPartitionCategoryIdMapper origCategoryIdMapper2{
        partitionFieldValue2, nextGlobalIdSupplier};

    assertions(origCategoryIdMapper1, origCategoryIdMapper2);

    ml::api::CPerPartitionCategoryIdMapper restoredCategoryIdMapper1{
        partitionFieldValue1, nextGlobalIdSupplier};
    persistAndRestore(origCategoryIdMapper1, restoredCategoryIdMapper1);
    ml::api::CPerPartitionCategoryIdMapper restoredCategoryIdMapper2{
        partitionFieldValue2, nextGlobalIdSupplier};
    persistAndRestore(origCategoryIdMapper2, restoredCategoryIdMapper2);

    assertions(restoredCategoryIdMapper1, restoredCategoryIdMapper2);

    BOOST_REQUIRE_EQUAL(ml::api::CGlobalCategoryId(6, partitionFieldValue2,
                                                   ml::model::CLocalCategoryId{3}),
                        restoredCategoryIdMapper2.map(ml::model::CLocalCategoryId{3}));
    BOOST_REQUIRE_EQUAL(ml::api::CGlobalCategoryId(7, partitionFieldValue1,
                                                   ml::model::CLocalCategoryId{4}),
                        restoredCategoryIdMapper1.map(ml::model::CLocalCategoryId{4}));
}

BOOST_AUTO_TEST_SUITE_END()
