/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CFieldConfigTest_h
#define INCLUDED_CFieldConfigTest_h

#include <cppunit/extensions/HelperMacros.h>

#include <api/CFieldConfig.h>

#include <functional>

class CFieldConfigTest : public CppUnit::TestFixture {
public:
    using TInitFromFileFunc =
        std::function<bool(ml::api::CFieldConfig*, const std::string&)>;

public:
    void testTrivial();
    void testValid();
    void testInvalid();
    void testValidSummaryCountFieldName();
    void testValidClauses();
    void testInvalidClauses();
    void testFieldOptions();
    void testValidPopulationClauses();
    void testValidPopulation();
    void testDefaultCategorizationField();
    void testCategorizationFieldWithFilters();
    void testExcludeFrequentClauses();
    void testExcludeFrequent();
    void testSlashes();
    void testBracketPercent();
    void testClauseTokenise();
    void testUtf8Bom();
    void testAddByOverPartitionInfluencers();
    void testAddOptions();
    void testScheduledEvents();

    static CppUnit::Test* suite();

private:
    void testValidFile(TInitFromFileFunc initFunc, const std::string& fileName);
    void testInvalidFile(TInitFromFileFunc initFunc, const std::string& fileName);
    void testValidSummaryCountFieldNameFile(TInitFromFileFunc initFunc,
                                            const std::string& fileName);
    void testValidPopulationFile(TInitFromFileFunc initFunc, const std::string& fileName);
    void testDefaultCategorizationFieldFile(TInitFromFileFunc initFunc,
                                            const std::string& fileName);
    void testExcludeFrequentFile(TInitFromFileFunc initFunc, const std::string& fileName);
    void testSlashesFile(TInitFromFileFunc initFunc, const std::string& fileName);
    void testBracketPercentFile(TInitFromFileFunc initFunc, const std::string& fileName);
};

#endif // INCLUDED_CFieldConfigTest_h
