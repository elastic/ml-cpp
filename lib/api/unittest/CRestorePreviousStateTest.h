/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CRestorePreviousStateTest_h
#define INCLUDED_CRestorePreviousStateTest_h

#include <cppunit/extensions/HelperMacros.h>

class CRestorePreviousStateTest : public CppUnit::TestFixture {
public:
    void testRestoreDetectorBy();
    void testRestoreDetectorOver();
    void testRestoreDetectorPartition();
    void testRestoreDetectorDc();
    void testRestoreDetectorCount();
    void testRestoreNormalizer();
    void testRestoreCategorizer();

    static CppUnit::Test* suite();

private:
    void anomalyDetectorRestoreHelper(const std::string& stateFile,
                                      const std::string& configFileName,
                                      bool isSymmetric,
                                      int latencyBuckets);

    void categorizerRestoreHelper(const std::string& stateFile, bool isSymmetric);

    std::string stripDocIds(const std::string& peristedState);
};

#endif // INCLUDED_CRestorePreviousStateTest_h
