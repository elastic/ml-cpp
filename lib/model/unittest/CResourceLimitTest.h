/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CResourceLimitTest_h
#define INCLUDED_CResourceLimitTest_h

#include <core/CoreTypes.h>

#include <cppunit/extensions/HelperMacros.h>

namespace ml
{
namespace model
{
class CAnomalyDetector;
class CResourceMonitor;
}
}

class CResultWriter;

class CResourceLimitTest : public CppUnit::TestFixture
{
    public:
        void testLimitBy(void);
        void testLimitByOver(void);
        void testLargeAllocations(void);

        static CppUnit::Test *suite(void);

    private:
        void importCsvDataWithLimiter(ml::core_t::TTime firstTime,
                                      ml::core_t::TTime bucketLength,
                                      CResultWriter &outputResults,
                                      const std::string &fileName,
                                      ml::model::CAnomalyDetector &detector,
                                      std::size_t limitCutoff,
                                      ml::model::CResourceMonitor &resourceMonitor);
};

#endif // INCLUDED_CResourceLimitTest_h

