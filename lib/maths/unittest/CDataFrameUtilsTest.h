/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CDataFrameUtilsTest_h
#define INCLUDED_CDataFrameUtilsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CDataFrameUtilsTest : public CppUnit::TestFixture {
public:
    void testColumnDataTypes();
    void testStandardizeColumns();
    void testColumnQuantiles();
    void testColumnQuantilesWithEncoding();
    void testStratifiedCrossValidationRowMasks();
    void testCrossValidationRowMasks();
    void testMicWithColumn();
    void testCategoryFrequencies();
    void testMeanValueOfTargetForCategories();
    void testMeanValueOfTargetForCategoriesWithMissing();
    void testCategoryMicWithColumn();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CDataFrameUtilsTest_h
