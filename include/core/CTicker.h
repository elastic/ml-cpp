/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CTicker_h
#define INCLUDED_ml_core_CTicker_h

#include <core/CCondition.h>
#include <core/CMutex.h>
#include <core/CScopedLock.h>
#include <core/CThread.h>
#include <core/ImportExport.h>


namespace ml
{
namespace core
{


//! \brief
//! A generic class that calls a method 'tick' on the
//! target template every xx milliseconds.
//!
//! DESCRIPTION:\n
//! A generic class that calls a method 'tick' on the
//! target template every xx milliseconds.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The method is templated - we could use boost function
//! objects for this but there is complexity with scope.
//!
template<typename RECEIVER>
class CTicker : public CThread
{
    public:
        //! Timeout is in milliseconds
        CTicker(uint32_t timeOut, RECEIVER &receiver)
            : m_Condition(m_Mutex),
              m_Quit(false),
              m_TimeOut(timeOut),
              m_Receiver(receiver)
        {
        }

        //! Destructor will stop the ticker thread if it's already running
        ~CTicker()
        {
            if (this->isStarted())
            {
                this->stop();
            }
        }

    protected:
        void run()
        {
            CScopedLock lock(m_Mutex);

            while (!m_Quit)
            {
                m_Condition.wait(m_TimeOut);

                // Call receiver
                m_Receiver.tick();
            }

            // Reset quit flag to false in case we're restarted
            m_Quit = false;
        }

        void shutdown()
        {
            CScopedLock lock(m_Mutex);

            m_Quit = true;
            m_Condition.signal();
        }

    private:
        CMutex     m_Mutex;
        CCondition m_Condition;

        //! Should the ticker quit?
        bool       m_Quit;

        //! How often (in milliseconds) should the ticker tick?
        uint32_t   m_TimeOut;

        //! Reference to the object whose tick() method will be called
        RECEIVER   &m_Receiver;
};


}
}

#endif // INCLUDED_ml_core_CTicker_h

