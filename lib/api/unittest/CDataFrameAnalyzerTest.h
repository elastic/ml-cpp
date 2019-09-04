/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CDataFrameAnalyzerTest_h
#define INCLUDED_CDataFrameAnalyzerTest_h

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>

#include <cppunit/extensions/HelperMacros.h>

class CDataFrameAnalyzerTest : public CppUnit::TestFixture {
public:
    void testWithoutControlMessages();
    void testRunOutlierDetection();
    void testRunOutlierDetectionPartitioned();
    void testRunOutlierFeatureInfluences();
    void testRunOutlierDetectionWithParams();
    void testRunBoostedTreeTraining();
    void testRunBoostedTreeTrainingWithStateRecovery();
    void testRunBoostedTreeTrainingWithParams();
    void testRunBoostedTreeTrainingWithRowsMissingTargetValue();
    void testFlushMessage();
    void testErrors();
    void testRoundTripDocHashes();
    void testCategoricalFields();

    static CppUnit::Test* suite();

private:
    rapidjson::GenericDocument<rapidjson::UTF8<>>
    stringToJsonDocument(const std::string& inputString) const;

    std::string jsonObjectToString(const rapidjson::GenericValue<rapidjson::UTF8<>>& jsonObject);
};

#endif // INCLUDED_CDataFrameAnalyzerTest_h
