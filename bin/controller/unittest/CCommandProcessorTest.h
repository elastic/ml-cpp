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
#ifndef INCLUDED_CCommandProcessorTest_h
#define INCLUDED_CCommandProcessorTest_h

#include <cppunit/extensions/HelperMacros.h>

class CCommandProcessorTest : public CppUnit::TestFixture {
public:
    void testStartPermitted(void);
    void testStartNonPermitted(void);
    void testStartNonExistent(void);
    void testKillDisallowed(void);
    void testInvalidVerb(void);

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CCommandProcessorTest_h
