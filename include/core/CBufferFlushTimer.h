/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CBufferFlushTimer_h
#define INCLUDED_ml_core_CBufferFlushTimer_h

#include <core/CoreTypes.h>
#include <core/ImportExport.h>


namespace ml
{
namespace core
{

//! \brief
//! Class to hide the complexity of establishing when
//! and how to flush a buffer
//!
//! DESCRIPTION:\n
//! Class to hide the complexity of establishing when
//! and how to flush a buffer
//!
//! IMPLEMENTATION DECISIONS:\n
//! In typical live use, the buffer is flushed
//! at the maxTime - bufferDelay.
//!
//! If no events have arrived this breaks as events
//! between maxTime and maxTime - bufferDelay never
//! get flushed.
//!
//! To avoid the method will return maxtime if
//! maxTime hasn't changed for bufferDelay clock seconds.
//!
class CORE_EXPORT CBufferFlushTimer
{
    public:
        CBufferFlushTimer(void);

        core_t::TTime flushTime(core_t::TTime bufferDelay,
                                core_t::TTime bufferMaxTime);

    private:
        //! The last reported 'max time'
        core_t::TTime m_LastMaxTime;

        //! The last actual clock time of the flush
        core_t::TTime m_LastFlushTime;
};

}
}

#endif // INCLUDED_ml_core_CBufferFlushTimer_h

