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
#ifndef INCLUDED_ml_core_CThreadFarmReceiver_h
#define INCLUDED_ml_core_CThreadFarmReceiver_h

#include <core/CLogger.h>

namespace ml {
namespace core {

//! \brief
//! A receiver used by the CThreadFarm class
//!
//! DESCRIPTION:\n
//! Acts as a conduit between the message queues in the CThreadFarm
//! and arbitrary processor classes
//!
//! IMPLEMENTATION DECISIONS:\n
//! Only stores a reference to the processor in case it's expensive to copy
//!
template<typename THREADFARM, typename PROCESSOR, typename MESSAGE, typename RESULT>
class CThreadFarmReceiver {
public:
    CThreadFarmReceiver(PROCESSOR& processor, THREADFARM& threadFarm) : m_Processor(processor), m_ThreadFarm(threadFarm) {}

    virtual ~CThreadFarmReceiver() {}

    void processMsg(const MESSAGE& msg, size_t /* backlog */) {
        RESULT result;

        m_Processor.msgToResult(msg, result);

        m_ThreadFarm.addResult(result);
    }

private:
    PROCESSOR& m_Processor;
    THREADFARM& m_ThreadFarm;
};
}
}

#endif // INCLUDED_ml_core_CThreadFarmReceiver_h
