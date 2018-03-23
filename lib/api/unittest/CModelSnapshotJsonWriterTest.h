/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CModelSnapshotJsonWriterTest_h
#define INCLUDED_CModelSnapshotJsonWriterTest_h

#include <cppunit/extensions/HelperMacros.h>


class CModelSnapshotJsonWriterTest : public CppUnit::TestFixture
{
    public:
        void testWrite(void);

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CModelSnapshotJsonWriterTest_h

