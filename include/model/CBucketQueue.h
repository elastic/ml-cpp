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

#ifndef INCLUDED_ml_model_CBucketQueue_h
#define INCLUDED_ml_model_CBucketQueue_h

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CPersistUtils.h>
#include <core/CoreTypes.h>

#include <boost/bind.hpp>
#include <boost/circular_buffer.hpp>

#include <string>

namespace ml {
namespace model {

//! \brief A fixed size queue with the purpose of storing bucket-associated data.
//!
//! DESCRIPTION:\n
//! A queue whose size is fixed to the number of latency buckets plus
//! one, for the latest bucket. The queue allows the storage of
//! bucket-associated data and their retrieval by providing a time.
//! The queue manages the matching of the time to the corresponding
//! bucket.
template<typename T>
class CBucketQueue {
public:
    typedef boost::circular_buffer<T> TQueue;
    typedef typename TQueue::value_type value_type;
    typedef typename TQueue::iterator iterator;
    typedef typename TQueue::const_iterator const_iterator;
    typedef typename TQueue::const_reverse_iterator const_reverse_iterator;

public:
    static const std::string BUCKET_TAG;
    static const std::string INDEX_TAG;

    //! \brief Wraps persist and restore so they can be used with boost::bind.
    template<typename F>
    class CSerializer {
    public:
        CSerializer(const T& initial = T(), const F& serializer = F()) : m_InitialValue(initial), m_Serializer(serializer) {}

        void operator()(const CBucketQueue& queue, core::CStatePersistInserter& inserter) const { queue.persist(m_Serializer, inserter); }

        bool operator()(CBucketQueue& queue, core::CStateRestoreTraverser& traverser) const {
            return queue.restore(m_Serializer, m_InitialValue, traverser);
        }

    private:
        T m_InitialValue;
        F m_Serializer;
    };

public:
    //! Constructs a new queue of size \p latencyBuckets, with bucket-length
    //! set to \p bucketLength and initialises it with \p latencyBuckets + 1
    //! up to the bucket corresponding to the supplied \p latestBucketStart.
    //!
    //! \param[in] latencyBuckets The number of buckets that are within
    //! the latency window.
    //! \param[in] bucketLength The bucket length.
    //! \param[in] latestBucketStart The start time of the latest bucket.
    CBucketQueue(std::size_t latencyBuckets, core_t::TTime bucketLength, core_t::TTime latestBucketStart, T initial = T())
        : m_Queue(latencyBuckets + 1), m_LatestBucketEnd(latestBucketStart + bucketLength - 1), m_BucketLength(bucketLength) {
        this->fill(initial);
        LOG_TRACE("Queue created :");
        LOG_TRACE("Bucket length = " << m_BucketLength);
        LOG_TRACE("LatestBucketEnd = " << m_LatestBucketEnd);
    }

    //! Pushed an item to the queue and moves the time forward
    //! by bucket length. If the \p time is earlier than the
    //! latest bucket end, the push operation is ignored.
    //!
    //! \param[in] item The item to be pushed in the queue.
    //! \param[in] time The time to which the item corresponds.
    void push(const T& item, core_t::TTime time) {
        if (time <= m_LatestBucketEnd) {
            LOG_ERROR("Push was called with early time = " << time << ", latest bucket end time = " << m_LatestBucketEnd);
            return;
        }
        m_LatestBucketEnd += m_BucketLength;
        this->push(item);
    }

    //! Pushes an item to the queue. This is only intended to be used
    //! internally and from clients that perform restoration of the queue.
    void push(const T& item) {
        m_Queue.push_front(item);
        LOG_TRACE("Queue after push -> " << core::CContainerPrinter::print(*this));
    }

    //! Returns the item in the queue that corresponds to the bucket
    //! indicated by \p time.
    T& get(core_t::TTime time) { return m_Queue[this->index(time)]; }

    //! Returns the item in the queue that corresponds to the bucket
    //! indicated by \p time.
    const T& get(core_t::TTime time) const { return m_Queue[this->index(time)]; }

    //! Returns the size of the queue.
    std::size_t size(void) const { return m_Queue.size(); }

    //! Is the queue empty?
    bool empty(void) const { return m_Queue.empty(); }

    //! Removes all items from the queue and fills with initial values
    //! Note, the queue should never be empty.
    void clear(const T& initial = T()) { this->fill(initial); }

    //! Resets the queue to \p startTime.
    //! This will clear the queue and will fill it with default items.
    void reset(core_t::TTime startTime, const T& initial = T()) {
        m_LatestBucketEnd = startTime + m_BucketLength - 1;
        this->fill(initial);
    }

    //! Returns an iterator pointing to the latest bucket and directed
    //! towards the earlier buckets.
    iterator begin(void) { return m_Queue.begin(); }

    //! Returns an iterator pointing to the end of the queue.
    iterator end(void) { return m_Queue.end(); }

    //! Returns an iterator pointing to the latest bucket and directed
    //! towards the earlier buckets.
    const_iterator begin(void) const { return m_Queue.begin(); }

    //! Returns an iterator pointing to the end of the queue.
    const_iterator end(void) const { return m_Queue.end(); }

    //! Returns a reverse_iterator pointing to the earliest bucket and directed
    //! towards the later buckets.
    const_reverse_iterator rbegin(void) const { return m_Queue.rbegin(); }

    //! Returns an iterator pointing to the end of the "reversed" queue.
    const_reverse_iterator rend(void) const { return m_Queue.rend(); }

    //! Returns the item in the queue corresponding to the earliest bucket.
    T& earliest(void) { return m_Queue.back(); }

    //! Returns the item corresponding to the latest bucket.
    T& latest(void) { return m_Queue.front(); }

    //! Returns the latest bucket end time, as tracked by the queue.
    core_t::TTime latestBucketEnd(void) const { return m_LatestBucketEnd; }

    //! Debug the memory used by this component.
    void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
        mem->setName("CBucketQueue");
        core::CMemoryDebug::dynamicSize("m_Queue", m_Queue, mem);
    }

    //! Get the memory used by this component.
    std::size_t memoryUsage(void) const { return core::CMemory::dynamicSize(m_Queue); }

    //! Prints the contents of the queue.
    std::string print(void) const { return core::CContainerPrinter::print(m_Queue); }

    //! Return the configured bucketlength of this queue
    core_t::TTime bucketLength(void) const { return m_BucketLength; }

    //! Generic persist interface that assumes the bucket items can
    //! be persisted by core::CPersistUtils
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const {
        for (std::size_t i = 0; i < m_Queue.size(); i++) {
            inserter.insertValue(INDEX_TAG, i);
            core::CPersistUtils::persist(BUCKET_TAG, m_Queue[i], inserter);
        }
    }

    //! Generic restore interface that assumes the bucket items can
    //! be restored by core::CPersistUtils
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
        std::size_t i = 0;
        do {
            if (traverser.name() == INDEX_TAG) {
                if (core::CStringUtils::stringToType(traverser.value(), i) == false) {
                    LOG_ERROR("Bad index in " << traverser.value());
                    return false;
                }
            } else if (traverser.name() == BUCKET_TAG) {
                if (i >= m_Queue.size()) {
                    LOG_WARN("Bucket queue is smaller on restore than on persist: " << i << " >= " << m_Queue.size()
                                                                                    << ".  Extra buckets will be ignored.");
                    // Restore into a temporary
                    T dummy;
                    if (!(core::CPersistUtils::restore(BUCKET_TAG, dummy, traverser))) {
                        LOG_ERROR("Invalid bucket");
                    }
                } else {
                    if (!(core::CPersistUtils::restore(BUCKET_TAG, m_Queue[i], traverser))) {
                        LOG_ERROR("Invalid bucket");
                        return false;
                    }
                }
            }
        } while (traverser.next());
        return true;
    }

private:
    //! Persist the buckets in the queue using \p bucketPersist.
    template<typename F>
    void persist(F bucketPersist, core::CStatePersistInserter& inserter) const {
        for (std::size_t i = 0; i < m_Queue.size(); i++) {
            inserter.insertValue(INDEX_TAG, i);
            inserter.insertLevel(BUCKET_TAG, boost::bind<void>(bucketPersist, boost::cref(m_Queue[i]), _1));
        }
    }

    //! Restore the buckets in the queue using \p bucketRestore.
    template<typename F>
    bool restore(F bucketRestore, const T& initial, core::CStateRestoreTraverser& traverser) {
        std::size_t i = 0;
        do {
            if (traverser.name() == INDEX_TAG) {
                if (core::CStringUtils::stringToType(traverser.value(), i) == false) {
                    LOG_DEBUG("Bad index in " << traverser.value());
                    return false;
                }
            } else if (traverser.name() == BUCKET_TAG) {
                if (i >= m_Queue.size()) {
                    LOG_WARN("Bucket queue is smaller on restore than on persist: " << i << " >= " << m_Queue.size()
                                                                                    << ".  Extra buckets will be ignored.");
                    if (traverser.hasSubLevel()) {
                        // Restore into a temporary
                        T dummy = initial;
                        if (traverser.traverseSubLevel(boost::bind<bool>(bucketRestore, dummy, _1)) == false) {
                            LOG_ERROR("Invalid bucket");
                        }
                    }
                } else {
                    m_Queue[i] = initial;
                    if (traverser.hasSubLevel()) {
                        if (traverser.traverseSubLevel(boost::bind<bool>(bucketRestore, boost::ref(m_Queue[i]), _1)) == false) {
                            LOG_ERROR("Invalid bucket");
                            return false;
                        }
                    }
                }
            }
        } while (traverser.next());
        return true;
    }

    //! Fill the queue with default constructed bucket values.
    void fill(const T& initial) {
        for (std::size_t i = 0; i < m_Queue.capacity(); ++i) {
            this->push(initial);
        }
    }

    //! Get the index of the bucket containing \p time.
    std::size_t index(core_t::TTime time) const {
        if (m_BucketLength == 0) {
            LOG_ERROR("Invalid bucketLength for queue!");
            return 0;
        }
        std::size_t index = static_cast<std::size_t>((m_LatestBucketEnd - time) / m_BucketLength);
        std::size_t size = this->size();
        if (index >= size) {
            LOG_ERROR("Time " << time << " is out of range. Returning earliest bucket index.");
            return size - 1;
        }
        return index;
    }

private:
    TQueue m_Queue;
    core_t::TTime m_LatestBucketEnd;
    core_t::TTime m_BucketLength;
};

template<typename T>
const std::string CBucketQueue<T>::BUCKET_TAG("a");

template<typename T>
const std::string CBucketQueue<T>::INDEX_TAG("b");
}
}

#endif // INCLUDED_ml_model_CBucketQueue_h
