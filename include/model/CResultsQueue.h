/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
    using THierarchicalResultsQueue = CBucketQueue<CHierarchicalResults>;

public:
    //! Constructor
    CResultsQueue(std::size_t delayBuckets, core_t::TTime bucketLength);

    //! Reset the underlying queue
    void reset(core_t::TTime time);

    //! Have we got unsent items in the queue?
    bool hasInterimResults() const;

    //! Push to the underlying queue
    void push(const CHierarchicalResults& item, core_t::TTime time);

    //! Push to the underlying queue
    void push(const CHierarchicalResults& item);

    //! Get a result from the queue
    const CHierarchicalResults& get(core_t::TTime time) const;

    //! Get a result from the queue
    CHierarchicalResults& get(core_t::TTime time);

    //! Returns the size of the queue.
    std::size_t size() const;

    //! Get the latest result from the queue
    CHierarchicalResults& latest();

    //! Returns the latest bucket end time, as tracked by the queue
    core_t::TTime latestBucketEnd() const;

    //! Select which queued result object to output, based on anomaly score
    //! and which have been output most recently
    core_t::TTime chooseResultTime(core_t::TTime bucketStartTime,
                                   core_t::TTime bucketLength,
                                   model::CHierarchicalResults& results);

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
