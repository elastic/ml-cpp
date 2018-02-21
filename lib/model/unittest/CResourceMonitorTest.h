/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CResourceMonitorTest_h
#define INCLUDED_CResourceMonitorTest_h

#include <model/CResourceMonitor.h>

#include <cppunit/extensions/HelperMacros.h>

namespace ml
{
namespace model
{
class CAnomalyDetector;
}
}

class CResourceMonitorTest : public CppUnit::TestFixture
{
    public:
        void setUp(void);

        void testMonitor(void);
        void testPruning(void);
        void testExtraMemory(void);

        static CppUnit::Test *suite();

    private:
        void reportCallback(const ml::model::CResourceMonitor::SResults &results);

        void addTestData(ml::core_t::TTime &firstTime, const ml::core_t::TTime bucketLength,
                     const std::size_t buckets, const std::size_t newPeoplePerBucket,
                     std::size_t &startOffset, ml::model::CAnomalyDetector &detector,
                     ml::model::CResourceMonitor &monitor);

    private:
        ml::model::CResourceMonitor::SResults m_CallbackResults;
};

#endif // INCLUDED_CResourceMonitorTest_h
