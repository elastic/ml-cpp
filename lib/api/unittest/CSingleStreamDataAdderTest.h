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

