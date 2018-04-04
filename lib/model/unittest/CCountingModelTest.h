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

#ifndef INCLUDED_CCountingModelTest_h
#define INCLUDED_CCountingModelTest_h

#include <model/CResourceMonitor.h>

#include <cppunit/extensions/HelperMacros.h>

class CCountingModelTest : public CppUnit::TestFixture {
public:
    void testSkipSampling();
    void testCheckScheduledEvents();
    static CppUnit::Test* suite();

private:
    ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CCountingModelTest_h
