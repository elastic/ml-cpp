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
#ifndef INCLUDED_CFieldConfigTest_h
#define INCLUDED_CFieldConfigTest_h

#include <cppunit/extensions/HelperMacros.h>

#include <api/CFieldConfig.h>

#include <functional>


class CFieldConfigTest : public CppUnit::TestFixture {
    public:
        using TInitFromFileFunc = std::function<bool(
                                                    ml::api::CFieldConfig *,
                                                    const std::string &)>;

    public:
        void testTrivial(void);
        void testValid(void);
        void testInvalid(void);
        void testValidSummaryCountFieldName(void);
        void testValidClauses(void);
        void testInvalidClauses(void);
        void testFieldOptions(void);
        void testValidPopulationClauses(void);
        void testValidPopulation(void);
        void testDefaultCategorizationField(void);
        void testCategorizationFieldWithFilters(void);
        void testExcludeFrequentClauses(void);
        void testExcludeFrequent(void);
        void testSlashes(void);
        void testBracketPercent(void);
        void testClauseTokenise(void);
        void testUtf8Bom(void);
        void testAddByOverPartitionInfluencers(void);
        void testAddOptions(void);
        void testScheduledEvents(void);

        static CppUnit::Test *suite();

    private:
        void testValidFile(TInitFromFileFunc initFunc,
                           const std::string &fileName);
        void testInvalidFile(TInitFromFileFunc initFunc,
                             const std::string &fileName);
        void testValidSummaryCountFieldNameFile(TInitFromFileFunc initFunc,
                                                const std::string &fileName);
        void testValidPopulationFile(TInitFromFileFunc initFunc,
                                     const std::string &fileName);
        void testDefaultCategorizationFieldFile(TInitFromFileFunc initFunc,
                                                const std::string &fileName);
        void testExcludeFrequentFile(TInitFromFileFunc initFunc,
                                     const std::string &fileName);
        void testSlashesFile(TInitFromFileFunc initFunc,
                             const std::string &fileName);
        void testBracketPercentFile(TInitFromFileFunc initFunc,
                                    const std::string &fileName);
};

#endif // INCLUDED_CFieldConfigTest_h

