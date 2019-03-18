/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CAnomalyJobTest_h
#define INCLUDED_CAnomalyJobTest_h

#include <core/CoreTypes.h>

#include <cppunit/extensions/HelperMacros.h>

class CAnomalyJobTest : public CppUnit::TestFixture {
public:
    void testLicense();
    void testBadTimes();
    void testOutOfSequence();
    void testControlMessages();
    void testSkipTimeControlMessage();
    void testIsPersistenceNeeded();
    void testModelPlot();
    void testInterimResultEdgeCases();
    void testRestoreFailsWithEmptyStream();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CAnomalyJobTest_h
