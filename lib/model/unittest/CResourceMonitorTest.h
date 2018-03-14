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
#ifndef INCLUDED_CResourceMonitorTest_h
#define INCLUDED_CResourceMonitorTest_h

#include <model/CResourceMonitor.h>

#include <cppunit/extensions/HelperMacros.h>

namespace ml {
namespace model {
class CAnomalyDetector;
}
}

class CResourceMonitorTest : public CppUnit::TestFixture {
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
