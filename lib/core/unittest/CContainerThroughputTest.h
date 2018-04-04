/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CContainerThroughputTest_h
#define INCLUDED_CContainerThroughputTest_h

#include <cppunit/extensions/HelperMacros.h>


class CContainerThroughputTest : public CppUnit::TestFixture
{
    public:
        struct SContent
        {
            SContent(size_t count);

            size_t s_Size;
            void   *s_Ptr;
            double s_Double;
        };

    public:
        void testVector();
        void testList();
        void testDeque();
        void testMap();
        void testCircBuf();
        void testMultiIndex();

        void setUp();

        static CppUnit::Test *suite();

    private:
        static const size_t FILL_SIZE;
        static const size_t TEST_SIZE;
};

#endif // INCLUDED_CContainerThroughputTest_h

