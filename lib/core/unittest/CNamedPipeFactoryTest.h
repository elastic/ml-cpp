/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CNamedPipeFactoryTest_h
#define INCLUDED_CNamedPipeFactoryTest_h

#include <cppunit/extensions/HelperMacros.h>

class CNamedPipeFactoryTest : public CppUnit::TestFixture
{
    public:
        void testServerIsCppReader(void);
        void testServerIsCReader(void);
        void testServerIsCppWriter(void);
        void testServerIsCWriter(void);
        void testCancelBlock(void);
        void testErrorIfRegularFile(void);
        void testErrorIfSymlink(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CNamedPipeFactoryTest_h

