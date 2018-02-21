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
#ifndef INCLUDED_ml_core_CMonotonicTime_h
#define INCLUDED_ml_core_CMonotonicTime_h

#include <core/ImportExport.h>

#include <stdint.h>


namespace ml
{
namespace core
{

//! \brief
//! Get a time that should never decrease
//!
//! DESCRIPTION:\n
//! Encapsulates the OS specific methods that obtain a monotonically
//! increasing time.  This is useful for timing the duration of other
//! processing in the face of system clock adjustments.
//!
//! The time is measured since some arbitrary fixed point in the past,
//! so the only meaningful thing to do with the values obtained is to
//! subtract two to measure the interval between them.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The functions to obtain the necessary time vary greatly between
//! different operating systems.
//!
//! Additionally, many operating systems have different functions
//! that could do the job, but sometimes there is a high cost of
//! making the timing call itself.
//!
//! For example, on Windows QueryPerformanceCounter() is very accurate
//! but very slow - 10 million calls to QueryPerformanceCounter() take
//! 130 times as long as 10 million calls to the less accurate function
//! GetTickCount64().
//!
//! On Linux, the CLOCK_MONOTONIC apparently IS affected by NTP
//! adjustments.  However, the CLOCK_MONOTONIC_RAW (which isn't) suffers
//! from three problems:
//! 1) It's only available in kernel 2.6.28 and higher
//! 2) It's slow to obtain
//! 3) It's not very accurate
//! Therefore, on Linux, we use the CLOCK_MONOTONIC even though it may
//! occasionally creep backwards slightly.
//!
//! For platforms using clock_gettime(), there is a further fallback to
//! CLOCK_REALTIME in the event of CLOCK_MONOTONIC not being available.
//!
class CORE_EXPORT CMonotonicTime
{
    public:
        //! Initialise any required scaling factors
        CMonotonicTime(void);

        //! Get the number of milliseconds since some fixed point in the past
        uint64_t milliseconds(void) const;

        //! Get the number of nanoseconds since some fixed point in the past
        uint64_t nanoseconds(void) const;

    private:
        //! Operating system specific scaling factors
        uint64_t m_ScalingFactor1;
        uint64_t m_ScalingFactor2;
        uint64_t m_ScalingFactor3;
};


}
}

#endif // INCLUDED_ml_core_CMonotonicTime_h

