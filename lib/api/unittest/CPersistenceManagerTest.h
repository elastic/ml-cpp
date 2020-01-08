/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CPersistenceManagerTest_h
#define INCLUDED_CPersistenceManagerTest_h

#include <cppunit/extensions/HelperMacros.h>

#include <string>

class CPersistenceManagerTest : public CppUnit::TestFixture {
public:
    void testDetectorPersistBy();
    void testDetectorPersistOver();
    void testDetectorPersistPartition();
    void testBackgroundPersistCategorizationConsistency();
    void testCategorizationOnlyPersist();
    void testDetectorBackgroundPersistStaticsConsistency();

    static CppUnit::Test* suite();

private:
    void foregroundBackgroundCompCategorizationAndAnomalyDetection(const std::string& configFileName);
    void foregroundBackgroundCompAnomalyDetectionAfterStaticsUpdate(const std::string& configFileName);
};

#endif // INCLUDED_CPersistenceManagerTest_h
