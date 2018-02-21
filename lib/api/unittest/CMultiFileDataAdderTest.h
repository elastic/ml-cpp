/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CMultiFileDataAdderTest_h
#define INCLUDED_CMultiFileDataAdderTest_h

#include <cppunit/extensions/HelperMacros.h>


class CMultiFileDataAdderTest : public CppUnit::TestFixture
{
    public:
        void testSimpleWrite(void);
        void testDetectorPersistBy(void);
        void testDetectorPersistOver(void);
        void testDetectorPersistPartition(void);
        void testDetectorPersistDc(void);
        void testDetectorPersistCount(void);

        static CppUnit::Test *suite();

    private:
        void detectorPersistHelper(const std::string &configFileName,
                                   const std::string &inputFilename,
                                   int latencyBuckets,
                                   const std::string &timeFormat = std::string());
};

#endif // INCLUDED_CMultiFileDataAdderTest_h

