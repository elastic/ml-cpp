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

#ifndef INCLUDED_CKMeansFastTest_h
#define INCLUDED_CKMeansFastTest_h

#include <cppunit/extensions/HelperMacros.h>

class CKMeansFastTest : public CppUnit::TestFixture
{
    public:
        void testDataPropagation();
        void testFilter();
        void testCentroids();
        void testClosestPoints();
        void testRun();
        void testRunWithSphericalClusters();
        void testPlusPlus();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CKMeansFastTest_h
