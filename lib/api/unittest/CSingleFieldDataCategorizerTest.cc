/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>

#include <model/CLimits.h>
#include <model/CTokenListDataCategorizer.h>
#include <model/CTokenListReverseSearchCreator.h>

#include <api/CCategoryIdMapper.h>
#include <api/CFieldDataCategorizer.h>
#include <api/CJsonOutputWriter.h>
#include <api/CNoopCategoryIdMapper.h>
#include <api/CPerPartitionCategoryIdMapper.h>
#include <api/CSingleFieldDataCategorizer.h>

#include <boost/test/unit_test.hpp>

#include <memory>
#include <sstream>

BOOST_AUTO_TEST_SUITE(CSingleFieldDataCategorizerTest)

namespace {

void checkPersistAndRestore(bool inBackgroundFirst,
                            const ml::api::CSingleFieldDataCategorizer& persistFrom,
                            ml::api::CSingleFieldDataCategorizer& restoreTo) {
    std::stringstream origJsonStrm;
    {
        ml::core::CJsonStatePersistInserter inserter{origJsonStrm};
        auto persistFunc = inBackgroundFirst
                               ? persistFrom.makeBackgroundPersistFunc()
                               : persistFrom.makeForegroundPersistFunc();
        // This is a quirk of the fact that the CSingleFieldDataCategorizer
        // is persisted at the same level as other tags - it cannot be
        // first in the object it's part of.
        inserter.insertValue("a", "1");
        persistFunc(inserter);
    }

    std::string origJson{origJsonStrm.str()};

    origJsonStrm.seekg(0);
    ml::core::CJsonStateRestoreTraverser traverser(origJsonStrm);
    BOOST_TEST_REQUIRE(restoreTo.acceptRestoreTraverser(traverser));

    std::stringstream rePersistJsonStrm;
    {
        ml::core::CJsonStatePersistInserter inserter{rePersistJsonStrm};
        auto persistFunc = inBackgroundFirst ? restoreTo.makeForegroundPersistFunc()
                                             : restoreTo.makeBackgroundPersistFunc();
        inserter.insertValue("a", "1");
        persistFunc(inserter);
    }

    std::string rePersistedJson{rePersistJsonStrm.str()};

    BOOST_REQUIRE_EQUAL(origJson, rePersistedJson);
}
}

BOOST_AUTO_TEST_CASE(testPersistNotPerPartition) {

    ml::model::CLimits limits;
    std::ostringstream outputStrm;
    ml::core::CJsonOutputStreamWrapper wrappedOutputStream{outputStrm};
    ml::api::CJsonOutputWriter jsonOutputWriter{"job", wrappedOutputStream};

    ml::api::CCategoryIdMapper::TCategoryIdMapperPtr idMapper{
        std::make_shared<ml::api::CNoopCategoryIdMapper>()};
    auto localCategorizer = std::make_unique<ml::api::CFieldDataCategorizer::TTokenListDataCategorizerKeepsFields>(
        limits, std::make_shared<ml::model::CTokenListReverseSearchCreator>("message"),
        0.7, "message");

    ml::api::CSingleFieldDataCategorizer origGlobalCategorizer{
        "", std::move(localCategorizer), std::move(idMapper)};

    ml::model::CDataCategorizer::TStrStrUMap fields;
    fields["message"] = "2015-10-18 18:01:51,963 INFO [main] org.mortbay.log: jetty-6.1.26\r";
    BOOST_REQUIRE_EQUAL(ml::api::CGlobalCategoryId{1},
                        origGlobalCategorizer.computeAndUpdateCategory(
                            false, fields, ml::api::CSingleFieldDataCategorizer::TOptionalTime{},
                            fields["message"], fields["message"],
                            limits.resourceMonitor(), jsonOutputWriter));

    fields["message"] = "2015-10-18 18:01:52,728 INFO [main] org.mortbay.log: Started HttpServer2$SelectChannelConnectorWithSafeStartup@0.0.0.0:62267\r";
    BOOST_REQUIRE_EQUAL(ml::api::CGlobalCategoryId{2},
                        origGlobalCategorizer.computeAndUpdateCategory(
                            false, fields, ml::api::CSingleFieldDataCategorizer::TOptionalTime{},
                            fields["message"], fields["message"],
                            limits.resourceMonitor(), jsonOutputWriter));

    idMapper = std::make_shared<ml::api::CNoopCategoryIdMapper>();
    localCategorizer = std::make_unique<ml::api::CFieldDataCategorizer::TTokenListDataCategorizerKeepsFields>(
        limits, std::make_shared<ml::model::CTokenListReverseSearchCreator>("message"),
        0.7, "message");
    ml::api::CSingleFieldDataCategorizer restoredFromBackgroundStateGlobalCategorizer{
        "", std::move(localCategorizer), std::move(idMapper)};

    checkPersistAndRestore(true, origGlobalCategorizer,
                           restoredFromBackgroundStateGlobalCategorizer);

    idMapper = std::make_shared<ml::api::CNoopCategoryIdMapper>();
    localCategorizer = std::make_unique<ml::api::CFieldDataCategorizer::TTokenListDataCategorizerKeepsFields>(
        limits, std::make_shared<ml::model::CTokenListReverseSearchCreator>("message"),
        0.7, "message");
    ml::api::CSingleFieldDataCategorizer restoredFromForegroundStateGlobalCategorizer{
        "", std::move(localCategorizer), std::move(idMapper)};

    checkPersistAndRestore(false, origGlobalCategorizer,
                           restoredFromForegroundStateGlobalCategorizer);
}

BOOST_AUTO_TEST_CASE(testPersistPerPartition) {

    ml::model::CLimits limits;
    std::ostringstream outputStrm;
    ml::core::CJsonOutputStreamWrapper wrappedOutputStream{outputStrm};
    ml::api::CJsonOutputWriter jsonOutputWriter{"job", wrappedOutputStream};

    int highestGlobalId{0};

    ml::api::CCategoryIdMapper::TCategoryIdMapperPtr idMapper{
        std::make_shared<ml::api::CPerPartitionCategoryIdMapper>(
            "vmware", [&highestGlobalId]() { return ++highestGlobalId; })};
    auto localCategorizer = std::make_unique<ml::api::CFieldDataCategorizer::TTokenListDataCategorizerKeepsFields>(
        limits, std::make_shared<ml::model::CTokenListReverseSearchCreator>("message"),
        0.7, "message");

    ml::api::CSingleFieldDataCategorizer origGlobalCategorizer{
        "event.dataset", std::move(localCategorizer), std::move(idMapper)};

    ml::model::CDataCategorizer::TStrStrUMap fields;
    fields["event.dataset"] = "vmware";
    fields["message"] = "Vpxa: [49EC0B90 verbose 'VpxaHalCnxHostagent' opID=WFU-ddeadb59] [WaitForUpdatesDone] Received callback";
    BOOST_REQUIRE_EQUAL(
        ml::api::CGlobalCategoryId(1, fields["event.dataset"], ml::model::CLocalCategoryId{1}),
        origGlobalCategorizer.computeAndUpdateCategory(
            false, fields, ml::api::CSingleFieldDataCategorizer::TOptionalTime{},
            fields["message"], fields["message"], limits.resourceMonitor(), jsonOutputWriter));

    fields["message"] = "Vpxa: [49EC0B90 verbose 'Default' opID=WFU-ddeadb59] [VpxaHalVmHostagent] 11: GuestInfo changed 'guest.disk";
    BOOST_REQUIRE_EQUAL(
        ml::api::CGlobalCategoryId(2, fields["event.dataset"], ml::model::CLocalCategoryId{2}),
        origGlobalCategorizer.computeAndUpdateCategory(
            false, fields, ml::api::CSingleFieldDataCategorizer::TOptionalTime{},
            fields["message"], fields["message"], limits.resourceMonitor(), jsonOutputWriter));

    idMapper = std::make_shared<ml::api::CPerPartitionCategoryIdMapper>(
        "event.dataset", [&highestGlobalId]() { return ++highestGlobalId; });
    localCategorizer = std::make_unique<ml::api::CFieldDataCategorizer::TTokenListDataCategorizerKeepsFields>(
        limits, std::make_shared<ml::model::CTokenListReverseSearchCreator>("message"),
        0.7, "message");
    ml::api::CSingleFieldDataCategorizer restoredFromBackgroundStateGlobalCategorizer{
        "event.dataset", std::move(localCategorizer), std::move(idMapper)};

    checkPersistAndRestore(true, origGlobalCategorizer,
                           restoredFromBackgroundStateGlobalCategorizer);

    idMapper = std::make_shared<ml::api::CPerPartitionCategoryIdMapper>(
        "event.dataset", [&highestGlobalId]() { return ++highestGlobalId; });
    localCategorizer = std::make_unique<ml::api::CFieldDataCategorizer::TTokenListDataCategorizerKeepsFields>(
        limits, std::make_shared<ml::model::CTokenListReverseSearchCreator>("message"),
        0.7, "message");
    ml::api::CSingleFieldDataCategorizer restoredFromForegroundStateGlobalCategorizer{
        "event.dataset", std::move(localCategorizer), std::move(idMapper)};

    checkPersistAndRestore(false, origGlobalCategorizer,
                           restoredFromForegroundStateGlobalCategorizer);
}

BOOST_AUTO_TEST_SUITE_END()
