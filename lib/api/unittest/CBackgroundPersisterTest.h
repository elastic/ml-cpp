/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CBackgroundPersisterTest_h
#define INCLUDED_CBackgroundPersisterTest_h

#include <cppunit/extensions/HelperMacros.h>

#include <string>

class CBackgroundPersisterTest : public CppUnit::TestFixture {
public:
    void testDetectorPersistBy();
    void testDetectorPersistOver();
    void testDetectorPersistPartition();
    void testCategorizationOnlyPersist();
    void testDetectorBackgroundPersistStaticsConsistency();

    static CppUnit::Test* suite();

private:
    void foregroundBackgroundCompCategorizationAndAnomalyDetection(const std::string& configFileName);
    void foregroundBackgroundCompAnomalyDetectionAfterStaticsUpdate(const std::string& configFileName);
};

#endif // INCLUDED_CBackgroundPersisterTest_h
