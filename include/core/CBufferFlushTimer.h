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
#ifndef INCLUDED_ml_core_CBufferFlushTimer_h
#define INCLUDED_ml_core_CBufferFlushTimer_h

#include <core/CoreTypes.h>
#include <core/ImportExport.h>

namespace ml {
namespace core {

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
class CORE_EXPORT CBufferFlushTimer {
public:
    CBufferFlushTimer();

    core_t::TTime flushTime(core_t::TTime bufferDelay, core_t::TTime bufferMaxTime);

private:
    //! The last reported 'max time'
    core_t::TTime m_LastMaxTime;

    //! The last actual clock time of the flush
    core_t::TTime m_LastFlushTime;
};
}
}

#endif // INCLUDED_ml_core_CBufferFlushTimer_h
