/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CRapidXmlStatePersistInserter.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CRapidXmlStatePersistInserterTest)


namespace {
//! Persist state as XML with meaningful tag names.
class CReadableXmlStatePersistInserter : public ml::core::CRapidXmlStatePersistInserter {
public:
    explicit CReadableXmlStatePersistInserter(const std::string& rootName,
                                              ml::core::CRapidXmlStatePersistInserter::TStrStrMap& rootAttributes)
        : ml::core::CRapidXmlStatePersistInserter(rootName, rootAttributes) {}
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
    ml::core::CRapidXmlStatePersistInserter::TStrStrMap rootAttributes;
    rootAttributes["attr1"] = "attrVal1";
    rootAttributes["attr2"] = "attrVal2";

    {
        ml::core::CRapidXmlStatePersistInserter inserter("root", rootAttributes);

        inserter.insertValue(LEVEL1A_TAG, "a");
        inserter.insertValue(LEVEL1B_TAG, 25);
        inserter.insertLevel(LEVEL1C_TAG, &insert2ndLevel);

        std::string xml;
        inserter.toXml(xml);

        LOG_DEBUG(<< "XML is: " << xml);

        inserter.toXml(false, xml);
        BOOST_CHECK_EQUAL(std::string("<root attr1=\"attrVal1\" "
                                         "attr2=\"attrVal2\"><a>a</a><b>25</b><c><a>3.14</"
                                         "a><b>z</b></c></root>"),
                             xml);
    }

    {
        // Test persistence with meaningful tag names

        CReadableXmlStatePersistInserter inserter("root", rootAttributes);

        inserter.insertValue(LEVEL1A_TAG, "a");
        inserter.insertValue(LEVEL1B_TAG, 25);
        inserter.insertLevel(LEVEL1C_TAG, &insert2ndLevel);

        std::string xml;
        inserter.toXml(xml);

        LOG_DEBUG(<< "XML is: " << xml);

        inserter.toXml(false, xml);

        BOOST_CHECK_EQUAL(std::string("<root attr1=\"attrVal1\" "
                                         "attr2=\"attrVal2\"><level1A>a</level1A><level1B>25</level1B><level1C><level2A>3.14</"
                                         "level2A><level2B>z</level2B></level1C></root>"),
                             xml);
    }
}

BOOST_AUTO_TEST_SUITE_END()
