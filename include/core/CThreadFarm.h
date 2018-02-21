/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CThreadFarm_h
#define INCLUDED_ml_core_CThreadFarm_h

#include <core/CLogger.h>
#include <core/CMutex.h>
#include <core/CMessageQueue.h>
#include <core/CNonCopyable.h>
#include <core/CScopedLock.h>
#include <core/CThread.h>
#include <core/CThreadFarmReceiver.h>

#include <boost/shared_ptr.hpp>

#include <vector>
#include <string>

#include <stdint.h>


namespace ml
{
namespace core
{


//! \brief
//! A means to have multiple threads work on some input
//!
//! DESCRIPTION:\n
//! Before the thread farm is started, one or more processors must be added to
//! it.  Then, when each message is added to the running thread farm, a copy of
//! it is passed to each of the processors via their own separate message
//! queues.  The processors then do some work (presumably different for each
//! processor otherwise there's no point), and pass the results back to the
//! thread farm which in turn passes all the results to a handler as soon as
//! it gets them.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The type used for messages should be cheap to copy.
//! The handler must have a processResult() method.
//! Each processor must have a msgToResult() method.
//! The message type must have a copy constructor.
//! The result type must have both a default constructor and a copy constructor.
//!
template<typename HANDLER, typename PROCESSOR, typename MESSAGE, typename RESULT>
class CThreadFarm : private CNonCopyable
{
    public:
        CThreadFarm(HANDLER &handler, const std::string &name)
            : m_Handler(handler),
              m_Pending(0),
              m_LastPrint(0),
              m_MessagesAdded(0),
              m_Started(false),
              m_Name(name)
        {
        }

        virtual ~CThreadFarm(void)
        {
            // Shared_ptr cleans up
        }

        //! Add a processor
        bool addProcessor(PROCESSOR &processor)
        {
            if (m_Started == true)
            {
                LOG_ERROR("Can't add receiver to running " << m_Name <<
                          " thread farm");
                return false;
            }

            TReceiverP receiver(new TReceiver(processor, *this));

            TMessageQueueP  mq(new CMessageQueue<MESSAGE, TReceiver>(*receiver));

            m_MessageQueues.push_back(mq);
            m_Receivers.push_back(receiver);

            return true;
        }

        //! Add some work, and find out how many results are pending
        //! following the addition
        bool addMessage(const MESSAGE &msg, size_t &pending)
        {
            CScopedLock lock(m_Mutex);

            if (m_Started == false)
            {
                LOG_ERROR("Can't add message to the " << m_Name <<
                          " thread farm because it's not running.  Call 'start'");
                return false;
            }

            for (TMessageQueuePVecItr itr = m_MessageQueues.begin();
                 itr != m_MessageQueues.end();
                 ++itr)
            {
                (*itr)->dispatchMsg(msg);
                ++m_Pending;
            }

            ++m_MessagesAdded;
            if (m_MessagesAdded % 1000 == 0)
            {
                LOG_INFO("Added message " << m_MessagesAdded << " to the " <<
                         m_Name << " thread farm; pending count now " <<
                         m_Pending);
            }

            pending = m_Pending;

            return true;
        }

        //! Add some work
        bool addMessage(const MESSAGE &msg)
        {
            size_t dummy = 0;
            return this->addMessage(msg, dummy);
        }

        //! Initialise - create the receiving threads
        bool start(void)
        {
            if (m_Started == true)
            {
                LOG_ERROR("Can't start the " << m_Name <<
                          " thread farm because it's already running.");
                return false;
            }

            size_t count(1);
            for (TMessageQueuePVecItr itr = m_MessageQueues.begin();
                 itr != m_MessageQueues.end();
                 ++itr)
            {
                if ((*itr)->start() == false)
                {
                    LOG_ERROR("Unable to start message queue " << count <<
                              " for the " << m_Name << " thread farm");
                    return false;
                }

                ++count;
            }

            m_Started = true;

            return true;
        }

        //! Shutdown - kill threads
        bool stop(void)
        {
            if (m_Started == false)
            {
                LOG_ERROR("Can't stop the " << m_Name <<
                          " thread farm because it's not running.");
                return false;
            }

            size_t count(1);
            for (TMessageQueuePVecItr itr = m_MessageQueues.begin();
                 itr != m_MessageQueues.end();
                 ++itr)
            {
                if ((*itr)->stop() == false)
                {
                    LOG_ERROR("Unable to stop message queue " << count <<
                              " for the " << m_Name << " thread farm");
                    return false;
                }

                LOG_DEBUG("Stopped message queue " << count <<
                          " for the " << m_Name << " thread farm");
                ++count;
            }

            m_Started = false;

            // Reset counters in case of restart
            m_MessagesAdded = 0;
            m_LastPrint = 0;

            if (m_Pending != 0)
            {
                LOG_ERROR("Inconsistency - " << m_Pending <<
                          " pending messages after stopping the " << m_Name <<
                          " thread farm");
                m_Pending = 0;
            }

            return true;
        }

    private:
        //! This should only be called by our friend the CThreadFarmReceiver
        //! otherwise the pending count will get messed up
        void addResult(const RESULT &result)
        {
            CScopedLock lock(m_Mutex);

            if (m_Pending <= 0)
            {
                LOG_ERROR("Inconsistency - result added with " << m_Pending <<
                          " pending messages in the " << m_Name <<
                          " thread farm");
                return;
            }

            m_Handler.processResult(result);

            --m_Pending;

            // Log how much work is outstanding every so often
            if ((m_Pending % 10000) == 0 && m_Pending != m_LastPrint)
            {
                LOG_INFO("Pending count now " << m_Pending << " for the " <<
                         m_Name << " thread farm");
                m_LastPrint = m_Pending;
            }

            if (m_Pending == 0)
            {
                //m_Handler.allComplete();
            }
        }

    private:
        //! Reference to the object that will handle the results
        HANDLER           &m_Handler;

        typedef CThreadFarm<HANDLER, PROCESSOR, MESSAGE, RESULT>             TThreadFarm;

        typedef CThreadFarmReceiver<TThreadFarm, PROCESSOR, MESSAGE, RESULT> TReceiver;
        typedef boost::shared_ptr<TReceiver>                                 TReceiverP;
        typedef std::vector<TReceiverP>                                      TReceiverPVec;
        typedef typename TReceiverPVec::iterator                             TReceiverPVecItr;

        typedef boost::shared_ptr< CMessageQueue<MESSAGE, TReceiver> >       TMessageQueueP;
        typedef std::vector<TMessageQueueP>                                  TMessageQueuePVec;
        typedef typename TMessageQueuePVec::iterator                         TMessageQueuePVecItr;

        TReceiverPVec     m_Receivers;

        //! We want the message queues destroyed before the receivers
        TMessageQueuePVec m_MessageQueues;

        //! How many results are pending?
        size_t            m_Pending;

        //! What was the pending value when we last printed it?
        uint64_t          m_LastPrint;

        //! How many messages have been added to the farm?
        uint64_t          m_MessagesAdded;

        //! Is the farm started?
        bool              m_Started;

        //! Protect members from multi-threaded access
        CMutex            m_Mutex;

        //! Purely for better logging messages
        std::string       m_Name;

    friend class CThreadFarmReceiver<TThreadFarm, PROCESSOR, MESSAGE, RESULT>;
};


}
}

#endif // INCLUDED_ml_core_CThreadFarm_h

