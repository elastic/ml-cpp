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

#ifndef INCLUDED_CInformationCriteriaTest_h
#define INCLUDED_CInformationCriteriaTest_h

#include <cppunit/extensions/HelperMacros.h>

class CInformationCriteriaTest : public CppUnit::TestFixture
{
    public:
        void testSphericalGaussian(void);
        void testSphericalGaussianWithSphericalCluster(void);
        void testGaussian(void);
        void testGaussianWithSphericalCluster(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CInformationCriteriaTest_h
