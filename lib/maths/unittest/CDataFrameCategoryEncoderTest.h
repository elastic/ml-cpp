/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CDataFrameCategoryEncoderTest_h
#define INCLUDED_CDataFrameCategoryEncoderTest_h

#include <cppunit/extensions/HelperMacros.h>

class CDataFrameCategoryEncoderTest : public CppUnit::TestFixture {
public:
    void testOneHotEncoding();
    void testMeanValueEncoding();
    void testRareCategories();
    void testCorrelatedFeatures();
    void testWithRowMask();
    void testEncodingOfCategoricalTarget();
    void testEncodedDataFrameRowRef();
    void testUnseenCategoryEncoding();
    void testDiscardNuisanceFeatures();
    void testPersistRestore();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CDataFrameCategoryEncoderTest_h
