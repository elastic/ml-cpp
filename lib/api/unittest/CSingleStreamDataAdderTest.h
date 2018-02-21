/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CSingleStreamDataAdderTest_h
#define INCLUDED_CSingleStreamDataAdderTest_h

#include <cppunit/extensions/HelperMacros.h>


class CSingleStreamDataAdderTest : public CppUnit::TestFixture
{
    public:
        void testDetectorPersistBy(void);
        void testDetectorPersistOver(void);
        void testDetectorPersistPartition(void);
        void testDetectorPersistDc(void);
        void testDetectorPersistCount(void);
        void testDetectorPersistCategorization(void);

        static CppUnit::Test *suite();

    private:
        void detectorPersistHelper(const std::string &configFileName,
                                   const std::string &inputFilename,
                                   int latencyBuckets,
                                   const std::string &timeFormat = std::string());
};

#endif // INCLUDED_CSingleStreamDataAdderTest_h

