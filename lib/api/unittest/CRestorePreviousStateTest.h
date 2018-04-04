/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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
#ifndef INCLUDED_CRestorePreviousStateTest_h
#define INCLUDED_CRestorePreviousStateTest_h

#include <cppunit/extensions/HelperMacros.h>


class CRestorePreviousStateTest : public CppUnit::TestFixture
{
    public:
        void testRestoreDetectorBy();
        void testRestoreDetectorOver();
        void testRestoreDetectorPartition();
        void testRestoreDetectorDc();
        void testRestoreDetectorCount();
        void testRestoreNormalizer();
        void testRestoreCategorizer();

        static CppUnit::Test *suite();

    private:
        void anomalyDetectorRestoreHelper(const std::string &stateFile,
                                          const std::string &configFileName,
                                          bool isSymmetric,
                                          int latencyBuckets);

        void categorizerRestoreHelper(const std::string &stateFile,
                                    bool isSymmetric);

        std::string stripDocIds(const std::string &peristedState);
};

#endif // INCLUDED_CRestorePreviousStateTest_h
