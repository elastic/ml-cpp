/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2018 Elasticsearch BV. All Rights Reserved.
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

#ifndef INCLUDED_CTimeSeriesChangeDetectorTest_h
#define INCLUDED_CTimeSeriesChangeDetectorTest_h

#include <cppunit/extensions/HelperMacros.h>

#include <core/CoreTypes.h>

#include <maths/CTimeSeriesChangeDetector.h>

class CTimeSeriesChangeDetectorTest : public CppUnit::TestFixture
{
    public:
        void testNoChange();
        void testLevelShift();
        void testLinearScale();
        void testTimeShift();
        void testPersist();

        static CppUnit::Test *suite();

    private:
        using TGenerator = std::function<double (ml::core_t::TTime)>;
        using TGeneratorVec = std::vector<TGenerator>;
        using TChange = std::function<double (TGenerator generator, ml::core_t::TTime)>;

    private:
        void testChange(const TGeneratorVec &trends,
                        ml::maths::SChangeDescription::EDescription description,
                        TChange applyChange,
                        double expectedChange,
                        double expectedMeanBucketsToDetectChange);
};

#endif // INCLUDED_CTimeSeriesChangeDetectorTest_h
