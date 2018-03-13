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
#include <model/CResultsQueue.h>

#include <core/CLogger.h>
#include <core/CPersistUtils.h>

#include <model/CHierarchicalResults.h>

namespace ml {
namespace model {

namespace {
const std::string RESULTS_TAG("a");
const std::string LAST_RESULTS_INDEX_TAG("b");
const std::string INITIALISATION_TIME_TAG("c");
}

CResultsQueue::CResultsQueue(std::size_t delayBuckets, core_t::TTime bucketLength)
    : m_Results(delayBuckets, bucketLength, 0), m_LastResultsIndex(2) {}

void CResultsQueue::push(const CHierarchicalResults &result, core_t::TTime time) {
    if (m_Results.latestBucketEnd() + 1 - m_Results.bucketLength() == 0) {
        m_Results.reset(time - m_Results.bucketLength());
        LOG_TRACE("Resetting results queue. Queue's latestBucketEnd is "
                  << m_Results.latestBucketEnd());
    }
    m_Results.push(result, time);
}

void CResultsQueue::push(const CHierarchicalResults &result) { m_Results.push(result); }

const CHierarchicalResults &CResultsQueue::get(core_t::TTime time) const {
    return m_Results.get(time);
}

CHierarchicalResults &CResultsQueue::get(core_t::TTime time) { return m_Results.get(time); }

CHierarchicalResults &CResultsQueue::latest(void) { return m_Results.latest(); }

core_t::TTime CResultsQueue::latestBucketEnd(void) const { return m_Results.latestBucketEnd(); }

std::size_t CResultsQueue::size(void) const { return m_Results.size(); }

void CResultsQueue::reset(core_t::TTime time) {
    m_Results.reset(time);
    m_LastResultsIndex = m_Results.size() - 1;
}

bool CResultsQueue::hasInterimResults(void) const {
    return m_Results.size() > 2 && m_LastResultsIndex == 0;
}

core_t::TTime CResultsQueue::chooseResultTime(core_t::TTime bucketStartTime,
                                              core_t::TTime bucketLength,
                                              model::CHierarchicalResults &results) {
    if (m_Results.size() == 1) {
        return bucketStartTime;
    }

    // Select the correct bucket to use
    LOG_TRACE("Asking for queue items at " << (bucketStartTime - bucketLength) << " and "
                                           << (bucketStartTime - (bucketLength / 2)));

    core_t::TTime resultsTime = 0;
    const model::CHierarchicalResults::TNode *node =
        m_Results.get(bucketStartTime - bucketLength).root();
    double r1 = 0.0;
    if (node) {
        r1 = node->s_NormalizedAnomalyScore;
    }
    node = m_Results.get(bucketStartTime - (bucketLength / 2)).root();
    double r2 = 0.0;
    if (node) {
        r2 = node->s_NormalizedAnomalyScore;
    }
    double r3 = 0.0;
    if (results.root()) {
        r3 = results.root()->s_NormalizedAnomalyScore;
    }

    LOG_TRACE("Testing results " << r1 << ", " << r2 << ", " << r3);

    if (m_LastResultsIndex == 0) {
        // With 3 clear buckets to look at, start choosing
        if ((r3 > r2) && (r3 > r1)) {
            // We want this guy, so choose r1 so that he can be selected next time
            resultsTime = bucketStartTime - bucketLength;
            m_LastResultsIndex = 2;
        } else {
            // Pick the bigger of 1 / 2
            if (r2 > r1) {
                resultsTime = bucketStartTime - (bucketLength / 2);
                m_LastResultsIndex = 3;
            } else {
                resultsTime = bucketStartTime - bucketLength;
                m_LastResultsIndex = 2;
            }
        }
    }
    --m_LastResultsIndex;
    return resultsTime;
}

void CResultsQueue::acceptPersistInserter(core::CStatePersistInserter &inserter) const {
    core_t::TTime initialisationTime = m_Results.latestBucketEnd() + 1 - m_Results.bucketLength();
    core::CPersistUtils::persist(INITIALISATION_TIME_TAG, initialisationTime, inserter);
    core::CPersistUtils::persist(RESULTS_TAG, m_Results, inserter);
    core::CPersistUtils::persist(LAST_RESULTS_INDEX_TAG, m_LastResultsIndex, inserter);
}

bool CResultsQueue::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser) {
    do {
        const std::string &name = traverser.name();
        if (name == RESULTS_TAG) {
            if (!core::CPersistUtils::restore(RESULTS_TAG, m_Results, traverser)) {
                return false;
            }
        } else if (name == LAST_RESULTS_INDEX_TAG) {
            if (!core::CPersistUtils::restore(
                    LAST_RESULTS_INDEX_TAG, m_LastResultsIndex, traverser)) {
                return false;
            }
        } else if (name == INITIALISATION_TIME_TAG) {
            core_t::TTime initialisationTime = 0;
            if (!core::CPersistUtils::restore(
                    INITIALISATION_TIME_TAG, initialisationTime, traverser)) {
                return false;
            }
            m_Results.reset(initialisationTime);
        }
    } while (traverser.next());
    return true;
}

}// model
}// ml
