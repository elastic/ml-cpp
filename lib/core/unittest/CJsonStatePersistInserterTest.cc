/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CJsonStatePersistInserter.h>
#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <boost/test/unit_test.hpp>

#include <sstream>

BOOST_AUTO_TEST_SUITE(CJsonStatePersistInserterTest)


namespace {
//! Persist state as JSON with meaningful tag names.
class CReadableJsonStatePersistInserter : public ml::core::CJsonStatePersistInserter {
public:
    explicit CReadableJsonStatePersistInserter(std::ostream& outputStream)
        : ml::core::CJsonStatePersistInserter(outputStream) {}
    virtual bool readableTags() const { return true; }
};

const ml::core::TPersistenceTag LEVEL1A_TAG{"a", "level1A"};
const ml::core::TPersistenceTag LEVEL1B_TAG{"b", "level1B"};
const ml::core::TPersistenceTag LEVEL1C_TAG{"c", "level1C"};

const ml::core::TPersistenceTag LEVEL2A_TAG{"a", "level2A"};
const ml::core::TPersistenceTag LEVEL2B_TAG{"b", "level2B"};

void insert2ndLevel(ml::core::CStatePersistInserter& inserter) {
    inserter.insertValue(LEVEL2A_TAG, 3.14, ml::core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(LEVEL2B_TAG, 'z');
}
}

BOOST_AUTO_TEST_CASE(testPersist) {
    {
        std::ostringstream strm;

        {
            ml::core::CJsonStatePersistInserter inserter(strm);

            inserter.insertValue(LEVEL1A_TAG, "a");
            inserter.insertValue(LEVEL1B_TAG, 25);
            inserter.insertLevel(LEVEL1C_TAG, &insert2ndLevel);
        }

        std::string json(strm.str());
        ml::core::CStringUtils::trimWhitespace(json);

        LOG_DEBUG(<< "JSON is: " << json);

        BOOST_CHECK_EQUAL(std::string("{\"a\":\"a\",\"b\":\"25\",\"c\":{\"a\":\"3.14\",\"b\":\"z\"}}"),
                             json);
    }

    {
        // Test persistence with meaningful tag names
        std::ostringstream strm;

        {
            CReadableJsonStatePersistInserter inserter(strm);

            inserter.insertValue(LEVEL1A_TAG, "a");
            inserter.insertValue(LEVEL1B_TAG, 25);
            inserter.insertLevel(LEVEL1C_TAG, &insert2ndLevel);
        }

        std::string json(strm.str());
        ml::core::CStringUtils::trimWhitespace(json);

        LOG_DEBUG(<< "JSON is: " << json);

        BOOST_CHECK_EQUAL(std::string("{\"level1A\":\"a\",\"level1B\":\"25\",\"level1C\":{\"level2A\":\"3.14\",\"level2B\":\"z\"}}"),
                             json);
    }
}

BOOST_AUTO_TEST_SUITE_END()
