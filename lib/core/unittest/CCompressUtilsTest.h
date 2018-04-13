/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CCompressUtilsTest_h
#define INCLUDED_CCompressUtilsTest_h

#include <cppunit/extensions/HelperMacros.h>


class CCompressUtilsTest : public CppUnit::TestFixture
{
    public:
        void testEmptyAdd();
        void testOneAdd();
        void testManyAdds();
        void testLengthOnly();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CCompressUtilsTest_h

