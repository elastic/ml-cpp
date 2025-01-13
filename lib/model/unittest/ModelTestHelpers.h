/*
* Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
* or more contributor license agreements. Licensed under the Elastic License
* 2.0 and the following additional limitation. Functionality enabled by the
* files subject to the Elastic License 2.0 may only be used in production when
* invoked by an Elasticsearch process with a license key installed that permits
* use of machine learning features. You may not use this file except in
* compliance with the Elastic License 2.0 and the foregoing additional
* limitation.
*/

#ifndef INCLUDED_ml_ModelTestHelpers_h
#define INCLUDED_ml_ModelTestHelpers_h

#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>

#include <model/CDataGatherer.h>
#include <model/CSearchKey.h>
#include <model/ModelTypes.h>
#include <model/SModelParams.h>

#include <boost/test/unit_test.hpp>

namespace ml {
namespace model {

const CSearchKey KEY;
const std::string EMPTY_STRING;

static void testPersistence(const SModelParams& params,
                            const CDataGatherer& origGatherer,
                            model_t::EAnalysisCategory category) {
    // Test persistence. (We check for idempotency.)
    std::ostringstream origJson;
    core::CJsonStatePersistInserter::persist(
        origJson, [&origGatherer](core::CJsonStatePersistInserter& inserter) {
            origGatherer.acceptPersistInserter(inserter);
        });

    LOG_DEBUG(<< "gatherer JSON size " << origJson.str().size());
    LOG_TRACE(<< "gatherer JSON representation:\n" << origJson.str());

    // Restore the JSON into a new filter
    // The traverser expects the state json in a embedded document
    std::istringstream origJsonStrm{"{\"topLevel\" : " + origJson.str() + "}"};
    core::CJsonStateRestoreTraverser traverser(origJsonStrm);

    CDataGatherer restoredGatherer(category, model_t::E_None, params, EMPTY_STRING,
                                   EMPTY_STRING, EMPTY_STRING, EMPTY_STRING,
                                   EMPTY_STRING, {}, KEY, traverser);

    BOOST_REQUIRE_EQUAL(origGatherer.checksum(), restoredGatherer.checksum());

    // The JSON representation of the new filter should be the
    // same as the original
    std::ostringstream newJson;
    core::CJsonStatePersistInserter::persist(
        newJson, [&restoredGatherer](core::CJsonStatePersistInserter& inserter) {
            restoredGatherer.acceptPersistInserter(inserter);
        });
    BOOST_REQUIRE_EQUAL(origJson.str(), newJson.str());
}

static void testGathererAttributes(const CDataGatherer& gatherer,
                                   core_t::TTime startTime,
                                   core_t::TTime bucketLength) {

    BOOST_REQUIRE_EQUAL(1, gatherer.numberActivePeople());
    BOOST_REQUIRE_EQUAL(1, gatherer.numberByFieldValues());
    BOOST_REQUIRE_EQUAL(std::string("p"), gatherer.personName(0));
    BOOST_REQUIRE_EQUAL(std::string("-"), gatherer.personName(1));
    std::size_t pid;
    BOOST_TEST_REQUIRE(gatherer.personId("p", pid));
    BOOST_REQUIRE_EQUAL(0, pid);
    BOOST_TEST_REQUIRE(!gatherer.personId("a.n.other p", pid));

    BOOST_REQUIRE_EQUAL(0, gatherer.numberActiveAttributes());
    BOOST_REQUIRE_EQUAL(0, gatherer.numberOverFieldValues());

    BOOST_REQUIRE_EQUAL(startTime, gatherer.currentBucketStartTime());
    BOOST_REQUIRE_EQUAL(bucketLength, gatherer.bucketLength());
}
}
}

#endif //INCLUDED_ml_ModelTestHelpers_h
