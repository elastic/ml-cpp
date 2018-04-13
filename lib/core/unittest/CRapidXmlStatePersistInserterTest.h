/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CRapidXmlStatePersistInserterTest_h
#define INCLUDED_CRapidXmlStatePersistInserterTest_h

#include <cppunit/extensions/HelperMacros.h>


class CRapidXmlStatePersistInserterTest : public CppUnit::TestFixture
{
    public:
        void testPersist();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CRapidXmlStatePersistInserterTest_h

