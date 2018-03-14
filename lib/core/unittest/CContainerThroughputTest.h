/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#ifndef INCLUDED_CContainerThroughputTest_h
#define INCLUDED_CContainerThroughputTest_h

#include <cppunit/extensions/HelperMacros.h>


class CContainerThroughputTest : public CppUnit::TestFixture {
    public:
        struct SContent {
            SContent(size_t count);

            size_t s_Size;
            void   *s_Ptr;
            double s_Double;
        };

    public:
        void testVector(void);
        void testList(void);
        void testDeque(void);
        void testMap(void);
        void testCircBuf(void);
        void testMultiIndex(void);

        void setUp(void);

        static CppUnit::Test *suite();

    private:
        static const size_t FILL_SIZE;
        static const size_t TEST_SIZE;
};

#endif // INCLUDED_CContainerThroughputTest_h

