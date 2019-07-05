/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CDataFrameAnalysisSpecificationTest_h
#define INCLUDED_CDataFrameAnalysisSpecificationTest_h

#include <cppunit/extensions/HelperMacros.h>

class CDataFrameAnalysisSpecificationTest : public CppUnit::TestFixture {
public:
    void testCreate();
    void testRunAnalysis();
    void testTempDirDiskUsage();

    static CppUnit::Test* suite();

private:
    std::string createSpecJsonForTempDirDiskUsageTest(bool, bool);
    std::string jsonString(size_t rows,
                           size_t cols,
                           size_t memoryLimit,
                           size_t numberThreads,
                           bool diskUsageAllowed,
                           const std::string& tempDir,
                           const std::string& resultField,
                           const std::string& analysis_name,
                           const std::string& analysis_parameters) const;
};

#endif // INCLUDED_CDataFrameAnalysisSpecificationTest_h
