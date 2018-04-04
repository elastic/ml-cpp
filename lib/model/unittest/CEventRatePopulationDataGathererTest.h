/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CEventRatePopulationDataGathererTest_h
#define INCLUDED_CEventRatePopulationDataGathererTest_h

#include <model/CResourceMonitor.h>

#include <cppunit/extensions/HelperMacros.h>


class CEventRatePopulationDataGathererTest : public CppUnit::TestFixture
{
    public:
        void testAttributeCounts();
        void testAttributeIndicator();
        void testUniqueValueCounts();
        void testCompressedLength();
        void testRemovePeople();
        void testRemoveAttributes();
        void testPersistence();

        static CppUnit::Test *suite();

    private:
        ml::model::CResourceMonitor m_ResourceMonitor;
};

#endif // INCLUDED_CEventRatePopulationDataGathererTest_h
