/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CMessageQueue_h
#define INCLUDED_ml_core_CMessageQueue_h

#include <core/CCondition.h>
#include <core/CLogger.h>
#include <core/CMutex.h>
#include <core/CScopedLock.h>
#include <core/CStopWatch.h>
#include <core/CThread.h>

#include <boost/circular_buffer.hpp>

#include <functional>
#include <memory>
#include <queue>

namespace ml {
namespace core {

//! \brief
//! A thread safe message queue.
//!
//! DESCRIPTION:\n
//! A thread safe message queue.
//!
//! The queue of items to be processed can grow to any
//! size.  This means this queue is the correct choice
//! for input being received from another process, where
//! it is important we don't block network buffers.
//!
//! Optionally, timing may be enabled so that the producer
//! can get an idea of how long the consumer is taking to
//! consume each item.  This can then be used to ask the
//! external process that is the original source of the
//! items to slow down.
//!
//! For queues of items being passed between threads in
//! the same application CBlockingMessageQueue is likely
//! to be a better choice.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The 'receive' function is executed in a different
//! thread to 'send'.
//!
//! The RECEIVER type must implement a method called
//! processMsg() taking 2 arguments, namely a MESSAGE
//! or const MESSAGE & and a size_t.
//!
//! Timing is enabled via a template parameter, such
//! that timing code can be completely compiled out when
//! not required.
//!
template<typename MESSAGE, typename RECEIVER, size_t NUM_TO_TIME = 0>
class CMessageQueue {
public:
    //! Prototype for function to be called on queue shutdown
    using TShutdownFunc = std::function<void()>;

public:
    CMessageQueue(RECEIVER& receiver,
                  const TShutdownFunc& shutdownFunc = &CMessageQueue::defaultShutdownFunc)
        : m_Thread(*this), m_Condition(m_Mutex), m_Receiver(receiver),
          m_ShutdownFunc(shutdownFunc),
          // If timing is enabled, we need a buffer one bigger than the
          // number of times to average over.  If timing is disabled, the
          // buffer can have capacity zero.
          m_Readings((NUM_TO_TIME > 0) ? (NUM_TO_TIME + 1) : 0) {}

    virtual ~CMessageQueue() {}

    //! Initialise - create the receiving thread
    bool start() {
        CScopedLock lock(m_Mutex);

        if (m_Thread.start() == false) {
            LOG_ERROR(<< "Unable to initialise thread");
            return false;
        }

        m_Condition.wait();

        return true;
    }

    //! Shutdown - kill thread
    bool stop() {
        m_Thread.stop();

        return true;
    }

    //! Send a message to the message queue thread (from any thread)
    void dispatchMsg(const MESSAGE& msg) {
        size_t dummy(0);
        this->dispatchMsg(msg, dummy);
    }

    //! Send a message to the message queue thread (from any thread),
    //! and get the pending count at the same time
    void dispatchMsg(const MESSAGE& msg, size_t& pending) {
        CScopedLock lock(m_Mutex);

        if (!m_Thread.isRunning()) {
            pending = 0;

            // Should be fatal error
            LOG_FATAL(<< "Cannot dispatch to message queue.  Queue not initialised");
            return;
        }

        m_Queue.push(msg);
        pending = m_Queue.size();

        // If there was already work queued up, we can save the cost of
        // signalling (which is expensive as it involves kernel interaction)
        if (pending <= 1) {
            m_Condition.signal();
        }
    }

    //! Get the number of pending messages in the queue.  Note that it's
    //! much more efficient to get this when dispatching a message, as
    //! everything can then be done under a single mutex lock.  This method
    //! must be used sparingly to avoid excessive lock contention.
    size_t pending() const {
        CScopedLock lock(m_Mutex);

        return m_Queue.size();
    }

    //! Get the average time taken to process the last N items (in
    //! seconds), where N was specified when timing was enabled.  A
    //! negative return value indicates an error.
    double rollingAverageProcessingTime() const {
        if (NUM_TO_TIME == 0) {
            LOG_ERROR(<< "Message queue timing is not switched on");

            return -1.0;
        }

        CScopedLock lock(m_Mutex);

        if (m_Readings.size() < 2) {
            return -1.0;
        }

        if (m_Readings.front() > m_Readings.back()) {
            LOG_ERROR(<< "Time to process last " << NUM_TO_TIME << " messages is negative (-"
                      << (m_Readings.front() - m_Readings.back())
                      << "ms).  "
                         "Maybe the system clock has been put back?");
            return -1.0;
        }

        return double(m_Readings.back() - m_Readings.front()) * 0.001 / double(NUM_TO_TIME);
    }

private:
    //! No-op shutdown function if no other is provided
    static void defaultShutdownFunc() {}

private:
    class CMessageQueueThread : public CThread {
    public:
        CMessageQueueThread(CMessageQueue<MESSAGE, RECEIVER, NUM_TO_TIME>& messageQueue)
            : m_MessageQueue(messageQueue), m_ShuttingDown(false),
              m_IsRunning(false) {}

        //! The queue must have the mutex for this to be called
        bool isRunning() const {
            // Assumes lock
            return m_IsRunning;
        }

    protected:
        void run() {
            m_MessageQueue.m_Mutex.lock();
            m_MessageQueue.m_Condition.signal();

            m_IsRunning = true;

            while (!m_ShuttingDown) {
                m_MessageQueue.m_Condition.wait();

                while (!m_MessageQueue.m_Queue.empty()) {
                    // Start the stop watch if it's not running and it
                    // should be
                    if (NUM_TO_TIME > 0 && !m_MessageQueue.m_StopWatch.isRunning()) {
                        m_MessageQueue.m_StopWatch.start();
                    }

                    MESSAGE& msg = m_MessageQueue.m_Queue.front();

                    // Don't include the current work item in the backlog
                    size_t backlog(m_MessageQueue.m_Queue.size() - 1);

                    m_MessageQueue.m_Mutex.unlock();

                    m_MessageQueue.m_Receiver.processMsg(msg, backlog);

                    // Try to do as much deletion as possible outside
                    // the lock, so the pop() below is cheap
                    this->destroyMsgDataUnlocked(msg);

                    m_MessageQueue.m_Mutex.lock();

                    m_MessageQueue.m_Queue.pop();

                    // If the stop watch is running, update the history
                    // of readings
                    if (NUM_TO_TIME > 0 && m_MessageQueue.m_StopWatch.isRunning()) {
                        m_MessageQueue.m_Readings.push_back(
                            m_MessageQueue.m_StopWatch.lap());
                    }
                }

                // Stop the stop watch if it's running, as we're
                // probably about to go to sleep
                if (NUM_TO_TIME > 0 && m_MessageQueue.m_StopWatch.isRunning()) {
                    m_MessageQueue.m_StopWatch.stop();
                }
            }

            m_IsRunning = false;

            m_MessageQueue.m_ShutdownFunc();

            m_MessageQueue.m_Mutex.unlock();
        }

        void shutdown() {
            CScopedLock lock(m_MessageQueue.m_Mutex);

            m_ShuttingDown = true;
            m_MessageQueue.m_Condition.signal();
        }

    private:
        //! It's best to minimise work done while the mutex is locked,
        //! so ideally we'll clean up the MESSAGE object as much as
        //! possible outside the lock.  This is the most generic case,
        //! where we can't do anything.
        template<typename ANYTHING>
        void destroyMsgDataUnlocked(ANYTHING&) {
            // For an arbitrary type we have no idea how to destroy some
            // of its data without calling its destructor
        }

        //! Specialisation of the above that might delete the referenced
        //! data if the MESSAGE type is a shared pointer (if no other
        //! shared pointer points to it).
        template<typename POINTEE>
        void destroyMsgDataUnlocked(std::shared_ptr<POINTEE>& ptr) {
            ptr.reset();
        }

        // Other specialisations could potentially be added here

    private:
        CMessageQueue<MESSAGE, RECEIVER, NUM_TO_TIME>& m_MessageQueue;
        bool m_ShuttingDown;
        bool m_IsRunning;
    };

    CMessageQueueThread m_Thread;
    mutable CMutex m_Mutex;
    CCondition m_Condition;
    RECEIVER& m_Receiver;

    using TMessageQueue = std::queue<MESSAGE>;

    TMessageQueue m_Queue;

    //! Function to be called on queue shutdown
    TShutdownFunc m_ShutdownFunc;

    //! A stop watch for timing how long it takes to process messages
    CStopWatch m_StopWatch;

    using TUIntCircBuf = boost::circular_buffer<uint64_t>;

    //! Stop watch readings
    TUIntCircBuf m_Readings;

    friend class CMessageQueueThread;
};
}
}

#endif // INCLUDED_ml_core_CMessageQueue_h
