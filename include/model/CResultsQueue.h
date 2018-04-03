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
#ifndef INCLUDED_ml_model_CResultsQueue_h
#define INCLUDED_ml_model_CResultsQueue_h

#include <model/CBucketQueue.h>
#include <model/CHierarchicalResults.h>

namespace ml {
namespace model {
class CHierarchicalResults;

//! \brief A queue for CHierarchicalResults objects.
//!
//! DESCRIPTION:\n
//! A queue for CHierarchicalResults objects that handles
//! overlapping bucket result selection
class MODEL_EXPORT CResultsQueue {
public:
    typedef CBucketQueue<CHierarchicalResults> THierarchicalResultsQueue;

public:
    //! Constructor
    CResultsQueue(std::size_t delayBuckets, core_t::TTime bucketLength);

    //! Reset the underlying queue
    void reset(core_t::TTime time);

    //! Have we got unsent items in the queue?
    bool hasInterimResults(void) const;

    //! Push to the underlying queue
    void push(const CHierarchicalResults& item, core_t::TTime time);

    //! Push to the underlying queue
    void push(const CHierarchicalResults& item);

    //! Get a result from the queue
    const CHierarchicalResults& get(core_t::TTime time) const;

    //! Get a result from the queue
    CHierarchicalResults& get(core_t::TTime time);

    //! Returns the size of the queue.
    std::size_t size(void) const;

    //! Get the latest result from the queue
    CHierarchicalResults& latest(void);

    //! Returns the latest bucket end time, as tracked by the queue
    core_t::TTime latestBucketEnd(void) const;

    //! Select which queued result object to output, based on anomaly score
    //! and which have been output most recently
    core_t::TTime chooseResultTime(core_t::TTime bucketStartTime, core_t::TTime bucketLength, model::CHierarchicalResults& results);

    //! Standard persistence
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Standard restoration
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

private:
    //! The collection of results objects
    THierarchicalResultsQueue m_Results;

    //! Which of the previous results did we output?
    size_t m_LastResultsIndex;
};

} // model
} // ml

#endif // INCLUDED_ml_model_CResultsQueue_h
