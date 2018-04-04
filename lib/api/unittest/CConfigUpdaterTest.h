/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CConfigUpdaterTest_h
#define INCLUDED_CConfigUpdaterTest_h

#include <cppunit/extensions/HelperMacros.h>


class CConfigUpdaterTest : public CppUnit::TestFixture
{
    public:
        void testUpdateGivenUpdateCannotBeParsed();
        void testUpdateGivenUnknownStanzas();
        void testUpdateGivenModelPlotConfig();
        void testUpdateGivenDetectorRules();
        void testUpdateGivenRulesWithInvalidDetectorIndex();
        void testUpdateGivenFilters();
        void testUpdateGivenScheduledEvents();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CConfigUpdaterTest_h

