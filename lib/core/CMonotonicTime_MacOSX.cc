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
#include <core/CMonotonicTime.h>

#include <core/CLogger.h>

#include <mach/mach_time.h>


namespace ml
{
namespace core
{


CMonotonicTime::CMonotonicTime(void)
    : m_ScalingFactor1(1),
      m_ScalingFactor2(1000000),
      m_ScalingFactor3(1)
{
    mach_timebase_info_data_t info;
    if (::mach_timebase_info(&info) != 0)
    {
        // Assume numerator and denominator for nanoseconds are both 1 (which is
        // true on a 2010 MacBook Pro)
        LOG_ERROR("Failed to get time base info");
    }
    else
    {
        m_ScalingFactor1 = info.numer;
        m_ScalingFactor2 *= info.denom;
        m_ScalingFactor3 *= info.denom;
    }
}

uint64_t CMonotonicTime::milliseconds(void) const
{
    return ::mach_absolute_time() * m_ScalingFactor1 / m_ScalingFactor2;
}

uint64_t CMonotonicTime::nanoseconds(void) const
{
    return ::mach_absolute_time() * m_ScalingFactor1 / m_ScalingFactor3;
}


}
}

