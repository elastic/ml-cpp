/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CXmlNodeWithChildrenTest_h
#define INCLUDED_CXmlNodeWithChildrenTest_h

#include <cppunit/extensions/HelperMacros.h>

class CXmlNodeWithChildrenTest : public CppUnit::TestFixture
{
    public:
        void testNodeHierarchyToXml(void);
        void testParserToNodeHierarchy(void);
        void testPerformanceNoPool(void);
        void testPerformanceWithPool(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CXmlNodeWithChildrenTest_h

