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
#ifndef INCLUDED_ml_core_CBlockingMessageQueue_h
#define INCLUDED_ml_core_CBlockingMessageQueue_h

#include <core/CCondition.h>
#include <core/CLogger.h>
#include <core/CMutex.h>
#include <core/CScopedLock.h>
#include <core/CThread.h>

#include <boost/circular_buffer.hpp>
#include <boost/shared_ptr.hpp>

#include <functional>

namespace ml {
namespace core {

//! \brief
//! A thread safe fixed size message queue.
//!
//! DESCRIPTION:\n
//! A thread safe message queue.
//!
//! The queue of items to be processed is limited to a
//! specified size.  Once the queue reaches this size,
//! attempts to add more items will block.  This is often
//! sensible behaviour for passing items between threads
//! in the same process when blocking would not cause an
//! external resource (e.g. a network buffer) to fill up.
//!
//! For queues of items being received from a different
//! application, i.e. where blocking would cause a network
//! buffer to fill up, CMessageQueue is likely to be a
//! better choice.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The 'receive' function is executed in a different
//! thread to 'send'.
//!
//! The RECEIVER type must implement a method called
//! processMsg() taking 2 arguments, namely a MESSAGE
//! or const MESSAGE & and a size_t.
//!
//! The maximum queue size is specified as a template
//! parameter, and hence cannot be changed at runtime.
//!
template<typename MESSAGE, typename RECEIVER, size_t QUEUE_CAPACITY>
class CBlockingMessageQueue {
public:
    //! Prototype for function to be called on queue shutdown
    using TShutdownFunc = std::function<void()>;

public:
    CBlockingMessageQueue(
        RECEIVER& receiver,
        const TShutdownFunc& shutdownFunc = &CBlockingMessageQueue::defaultShutdownFunc)
        : m_Thread(*this),
          m_ProducerCondition(m_Mutex),
          m_ConsumerCondition(m_Mutex),
          m_Receiver(receiver),
          m_Queue(QUEUE_CAPACITY),
          m_ShutdownFunc(shutdownFunc) {}

    virtual ~CBlockingMessageQueue(void) {}

    //! Initialise - create the receiving thread
    bool start(void) {
        CScopedLock lock(m_Mutex);

        if (m_Thread.start() == false) {
            LOG_ERROR("Unable to initialise thread");
            return false;
        }

        m_ProducerCondition.wait();

        return true;
    }

    //! Shutdown - kill thread
    bool stop(void) {
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
            LOG_FATAL("Cannot dispatch to message queue.  Queue not initialised");
            return;
        }

        for (;;) {
            // The pending count includes the item to be added
            pending = 1 + m_Queue.size();

            if (pending <= QUEUE_CAPACITY) {
                break;
            }

            m_ProducerCondition.wait();
        }

        m_Queue.push_back(msg);

        // If there was already work queued up, we can save the cost of
        // signalling (which is expensive as it involves kernel interaction)
        if (pending <= 1) {
            m_ConsumerCondition.signal();
        }
    }

    //! Get the number of pending messages in the queue.  Note that it's
    //! much more efficient to get this when dispatching a message, as
    //! everything can then be done under a single mutex lock.  This method
    //! must be used sparingly to avoid excessive lock contention.
    size_t pending(void) const {
        CScopedLock lock(m_Mutex);

        return m_Queue.size();
    }

private:
    //! No-op shutdown function if no other is provided
    static void defaultShutdownFunc(void) {}

private:
    class CMessageQueueThread : public CThread {
    public:
        CMessageQueueThread(CBlockingMessageQueue<MESSAGE, RECEIVER, QUEUE_CAPACITY>& messageQueue)
            : m_MessageQueue(messageQueue), m_ShuttingDown(false), m_IsRunning(false) {}

        //! The queue must have the mutex for this to be called
        bool isRunning(void) const {
            // Assumes lock
            return m_IsRunning;
        }

    protected:
        void run(void) {
            m_MessageQueue.m_Mutex.lock();
            m_MessageQueue.m_ProducerCondition.signal();

            m_IsRunning = true;

            while (!m_ShuttingDown) {
                m_MessageQueue.m_ConsumerCondition.wait();

                while (!m_MessageQueue.m_Queue.empty()) {
                    MESSAGE& msg = m_MessageQueue.m_Queue.front();

                    // Don't include the current work item in the backlog
                    size_t backlog(m_MessageQueue.m_Queue.size() - 1);

                    m_MessageQueue.m_Mutex.unlock();

                    m_MessageQueue.m_Receiver.processMsg(msg, backlog);

                    // Try to do as much deletion as possible outside
                    // the lock, so the pop_front() below is cheap
                    this->destroyMsgDataUnlocked(msg);

                    m_MessageQueue.m_Mutex.lock();

                    // If the queue was full, signal a thread waiting to
                    // add data
                    if (m_MessageQueue.m_Queue.size() == QUEUE_CAPACITY) {
                        // Only do this if the queue is full, as it
                        // involves kernel interaction, so is expensive
                        m_MessageQueue.m_ProducerCondition.signal();
                    }

                    m_MessageQueue.m_Queue.pop_front();
                }
            }

            m_IsRunning = false;

            m_MessageQueue.m_ShutdownFunc();

            m_MessageQueue.m_Mutex.unlock();
        }

        void shutdown(void) {
            CScopedLock lock(m_MessageQueue.m_Mutex);

            m_ShuttingDown = true;
            m_MessageQueue.m_ConsumerCondition.signal();
            m_MessageQueue.m_ProducerCondition.broadcast();
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
        void destroyMsgDataUnlocked(boost::shared_ptr<POINTEE>& ptr) {
            ptr.reset();
        }

        // Other specialisations could potentially be added here

    private:
        CBlockingMessageQueue<MESSAGE, RECEIVER, QUEUE_CAPACITY>& m_MessageQueue;
        bool m_ShuttingDown;
        bool m_IsRunning;
    };

    CMessageQueueThread m_Thread;
    mutable CMutex m_Mutex;
    CCondition m_ProducerCondition;
    CCondition m_ConsumerCondition;
    RECEIVER& m_Receiver;

    //! Using a circular buffer for the queue means that it will not do any
    //! memory allocations after construction (providing the message type
    //! does not allocate any heap memory in its constructor).
    typedef boost::circular_buffer<MESSAGE> TMessageCircBuf;

    TMessageCircBuf m_Queue;

    //! Function to be called on queue shutdown
    TShutdownFunc m_ShutdownFunc;

    friend class CMessageQueueThread;
};
}
}

#endif // INCLUDED_ml_core_CBlockingMessageQueue_h
