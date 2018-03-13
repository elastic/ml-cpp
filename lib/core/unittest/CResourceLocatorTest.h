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
#ifndef INCLUDED_CResourceLocatorTest_h
#define INCLUDED_CResourceLocatorTest_h

#include <cppunit/extensions/HelperMacros.h>

class CResourceLocatorTest : public CppUnit::TestFixture {
public:
    void testResourceDir(void);
    void testLogDir(void);
    void testSrcRootDir(void);

    static CppUnit::Test *suite();
};

#endif// INCLUDED_CResourceLocatorTest_h
