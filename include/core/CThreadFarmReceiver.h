/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CThreadFarmReceiver_h
#define INCLUDED_ml_core_CThreadFarmReceiver_h

#include <core/CLogger.h>


namespace ml
{
namespace core
{


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
class CThreadFarmReceiver
{
    public:
        CThreadFarmReceiver(PROCESSOR &processor,
                            THREADFARM &threadFarm)
            : m_Processor(processor),
              m_ThreadFarm(threadFarm)
        {
        }

        virtual ~CThreadFarmReceiver()
        {
        }

        void processMsg(const MESSAGE &msg, size_t /* backlog */)
        {
            RESULT result;

            m_Processor.msgToResult(msg, result);

            m_ThreadFarm.addResult(result);
        }

    private:
        PROCESSOR   &m_Processor;
        THREADFARM  &m_ThreadFarm;
};


}
}

#endif // INCLUDED_ml_core_CThreadFarmReceiver_h

