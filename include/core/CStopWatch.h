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
#ifndef INCLUDED_ml_core_CStopWatch_h
#define INCLUDED_ml_core_CStopWatch_h

#include <core/CMonotonicTime.h>
#include <core/ImportExport.h>

#include <stdint.h>


namespace ml {
namespace core {


//! \brief
//! Can be used for timing within a program
//!
//! DESCRIPTION:\n
//! Can be used for timing within a program when intervals must be
//! measured more accurately than to the nearest second.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The interface mirrors the stop watch functionality you would expect
//! on a cheap digital watch.
//!
//! A 64 bit counter is used to store the elapsed milliseconds, because
//! 2^32 milliseconds is less than 50 days, and, in production, our
//! processes should be able to run for longer than this.
//!
class CORE_EXPORT CStopWatch {
    public:
        //! Construct a stop watch, optionally starting it immediately
        CStopWatch(bool startRunning = false);

        //! Start the stop watch
        void     start(void);

        //! Stop the stop watch and retrieve the accumulated reading
        uint64_t stop(void);

        //! Retrieve the accumulated reading from the stop watch without
        //! stopping it.  (Not const because it may trigger a reset if the
        //! system clock has been adjusted.)
        uint64_t lap(void);

        //! Is the stop watch running?
        bool     isRunning(void) const;

        //! Reset the stop watch, optionally starting it immediately
        void     reset(bool startRunning = false);

    private:
        //! Calculate the difference between two monotonic times, with sanity
        //! checking just in case the timer does go backwards somehow, and
        //! return the answer in milliseconds
        uint64_t calcDuration(void);

    private:
        //! Is the stop watch currently running?
        bool           m_IsRunning;

        //! Monotonic timer - should not go backwards if the user sets the clock
        CMonotonicTime m_MonotonicTime;

        //! Monotonic time (in milliseconds since some arbitrary time in the
        //! past) when the stop watch was last started
        uint64_t       m_Start;

        //! Time (in milliseconds) accumulated over previous runs of the stop
        //! watch since the last reset
        uint64_t       m_AccumulatedTime;
};


}
}

#endif // INCLUDED_ml_core_CStopWatch_h

