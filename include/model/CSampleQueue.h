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

#ifndef INCLUDED_ml_model_CSampleQueue_h
#define INCLUDED_ml_model_CSampleQueue_h

#include <core/CContainerPrinter.h>
#include <core/CHashing.h>
#include <core/CIEEE754.h>
#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStringUtils.h>
#include <core/CoreTypes.h>

#include <maths/CIntegerTools.h>
#include <maths/COrderings.h>

#include <model/CFeatureData.h>
#include <model/CMetricPartialStatistic.h>
#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <boost/bind.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/optional.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace ml {
namespace model {

//! \brief A queue that manages the sampling of statistics
//!
//! DESCRIPTION:\n
//! A queue which collects measurements and creates samples when
//! there are enough measurements and the time is appropriate.
//! The queue handles the creation of samples during a latency
//! window by creating smaller sub-samples which get updated
//! as new measurements are received. The new measurements are
//! combined into existing sub-samples if their time is close enough
//! or they are placed into newly created sub-samples.
//!
//! The template STATISTIC has to comply with the requirements of
//! the CMetricPartialStatistic template.
template<typename STATISTIC>
class CSampleQueue {
private:
    typedef core::CSmallVector<double, 1> TDouble1Vec;
    typedef CMetricPartialStatistic<STATISTIC> TMetricPartialStatistic;

private:
    //! A struct grouping together the data that form a sub-sample.
    //! A sub-sample is comprised of a partial statistic and a start
    //! and an end time marking the interval range for the sub-sample.
    struct SSubSample {
        static const std::string SAMPLE_START_TAG;
        static const std::string SAMPLE_END_TAG;
        static const std::string SAMPLE_TAG;

        SSubSample(std::size_t dimension, core_t::TTime time) : s_Statistic(dimension), s_Start(time), s_End(time) {}

        void add(const TDouble1Vec& measurement, core_t::TTime time, unsigned int count) {
            s_Statistic.add(measurement, time, count);
            // Using explicit tests instead of std::min and std::max to work
            // around g++ 4.1 optimiser bug
            if (time < s_Start) {
                s_Start = time;
            }
            if (time > s_End) {
                s_End = time;
            }
        }

        //! Check if \p time overlaps the interval or doesn't extend
        //! it to be more than \p targetSpan long.
        bool isClose(core_t::TTime time, core_t::TTime targetSpan) const {
            if (time > s_End) {
                return time < s_Start + targetSpan;
            }
            if (time >= s_Start) {
                return true;
            }
            return time > s_End - targetSpan;
        }

        // This assumes that buckets are aligned to n * bucketLength
        bool isInSameBucket(core_t::TTime time, core_t::TTime bucketLength) const {
            core_t::TTime timeBucket = maths::CIntegerTools::floor(time, bucketLength);
            core_t::TTime subSampleBucket = maths::CIntegerTools::floor(s_Start, bucketLength);
            return timeBucket == subSampleBucket;
        }

        //! Combine the statistic and construct the union interval.
        const SSubSample& operator+=(const SSubSample& rhs) {
            s_Statistic += rhs.s_Statistic;
            s_Start = std::min(s_Start, rhs.s_Start);
            s_End = std::max(s_End, rhs.s_End);
            return *this;
        }

        //! Persist to a state document.
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const {
            inserter.insertLevel(SAMPLE_TAG, boost::bind(&TMetricPartialStatistic::persist, &s_Statistic, _1));
            inserter.insertValue(SAMPLE_START_TAG, s_Start);
            inserter.insertValue(SAMPLE_END_TAG, s_End);
        }

        //! Restore from a state document.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
            do {
                const std::string& name = traverser.name();
                if (name == SAMPLE_TAG) {
                    if (traverser.traverseSubLevel(boost::bind(&TMetricPartialStatistic::restore, &s_Statistic, _1)) == false) {
                        LOG_ERROR("Invalid sample value");
                        return false;
                    }
                } else if (name == SAMPLE_START_TAG) {
                    if (core::CStringUtils::stringToType(traverser.value(), s_Start) == false) {
                        LOG_ERROR("Invalid attribute identifier in " << traverser.value());
                        return false;
                    }
                } else if (name == SAMPLE_END_TAG) {
                    if (core::CStringUtils::stringToType(traverser.value(), s_End) == false) {
                        LOG_ERROR("Invalid attribute identifier in " << traverser.value());
                        return false;
                    }
                }
            } while (traverser.next());
            return true;
        }

        //! Get a checksum of this sub-sample.
        uint64_t checksum(void) const {
            uint64_t seed = maths::CChecksum::calculate(0, s_Statistic);
            seed = maths::CChecksum::calculate(seed, s_Start);
            return maths::CChecksum::calculate(seed, s_End);
        }

        //! Debug the memory used by the sub-sample.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
            mem->setName("SSubSample", sizeof(*this));
            core::CMemoryDebug::dynamicSize("s_Statistic", s_Statistic, mem);
        }

        //! Get the memory used by the sub-sample.
        std::size_t memoryUsage(void) const { return sizeof(*this) + core::CMemory::dynamicSize(s_Statistic); }

        //! Print the sub-sample for debug.
        std::string print(void) const {
            return "{[" + core::CStringUtils::typeToString(s_Start) + ", " + core::CStringUtils::typeToString(s_End) + "] -> " +
                   s_Statistic.print() + "}";
        }

        TMetricPartialStatistic s_Statistic;
        core_t::TTime s_Start;
        core_t::TTime s_End;
    };

public:
    typedef boost::circular_buffer<SSubSample> TQueue;
    typedef typename TQueue::iterator iterator;
    typedef typename TQueue::reverse_iterator reverse_iterator;
    typedef typename TQueue::const_reverse_iterator const_reverse_iterator;
    typedef std::vector<CSample> TSampleVec;
    typedef boost::optional<SSubSample> TOptionalSubSample;

public:
    static const std::string SUB_SAMPLE_TAG;

public:
    //! Constructs a new queue.
    //!
    //! \param[in] dimension The dimension of the metric statistic.
    //! \param[in] sampleCountFactor The queue attempts to keep the sub-samples
    //! size to the current sample count divided by the sampleCountFactor.
    //! \param[in] latencyBuckets The number of buckets that are in the latency window.
    //! \param[in] growthFactor The factor with which the queue's size grows whenever
    //! a new item is inserted and the queue is full.
    //! \param[in] bucketLength The bucket length.
    CSampleQueue(std::size_t dimension,
                 std::size_t sampleCountFactor,
                 std::size_t latencyBuckets,
                 double growthFactor,
                 core_t::TTime bucketLength)
        : m_Dimension(dimension),
          m_Queue(std::max(sampleCountFactor * latencyBuckets, std::size_t(1))),
          m_SampleCountFactor(sampleCountFactor),
          m_GrowthFactor(growthFactor),
          m_BucketLength(bucketLength),
          m_Latency(static_cast<core_t::TTime>(latencyBuckets) * bucketLength) {}

    //! Adds a measurement to the queue.
    //!
    //! \param[in] time The time of the measurement.
    //! \param[in] measurement The value of the measurement.
    //! \param[in] count The count of the measurement.
    //! \param[in] sampleCount The target sample count.
    void add(core_t::TTime time, const TDouble1Vec& measurement, unsigned int count, unsigned int sampleCount) {
        if (m_Queue.empty()) {
            this->pushFrontNewSubSample(measurement, time, count);
        } else if (time >= m_Queue[0].s_Start) {
            this->addAfterLatestStartTime(measurement, time, count, sampleCount);
        } else {
            this->addHistorical(measurement, time, count, sampleCount);
        }
    }

    //! Can the queue possible create samples?
    bool canSample(core_t::TTime bucketStart) const {
        core_t::TTime bucketEnd = bucketStart + m_BucketLength - 1;
        return m_Queue.empty() ? false : m_Queue.back().s_End <= bucketEnd;
    }

    //! Combines as many sub-samples as possible in order to create samples.
    //!
    //! \param[in] bucketStart The start time of the bucket to sample.
    //! \param[in] sampleCount The target sample count.
    //! \param[in] feature The feature to which the measurements correspond.
    //! \param[out] samples The newly created samples.
    void sample(core_t::TTime bucketStart, unsigned int sampleCount, model_t::EFeature feature, TSampleVec& samples) {
        core_t::TTime latencyCutoff = bucketStart + m_BucketLength - 1;
        TOptionalSubSample combinedSubSample;

        while (m_Queue.empty() == false && m_Queue.back().s_End <= latencyCutoff) {
            if (combinedSubSample) {
                *combinedSubSample += m_Queue.back();
            } else {
                combinedSubSample = TOptionalSubSample(m_Queue.back());
            }

            m_Queue.pop_back();

            double count = combinedSubSample->s_Statistic.count();
            double countIncludingNext = (m_Queue.empty()) ? count : count + m_Queue.back().s_Statistic.count();
            double countRatio = sampleCount / count;
            double countRatioIncludingNext = sampleCount / countIncludingNext;

            if (countIncludingNext >= sampleCount && (std::abs(1.0 - countRatio) <= std::abs(1.0 - countRatioIncludingNext))) {
                TDouble1Vec sample = combinedSubSample->s_Statistic.value();
                core_t::TTime sampleTime = combinedSubSample->s_Statistic.time();
                double vs = model_t::varianceScale(feature, sampleCount, count);
                samples.push_back(CSample(sampleTime, sample, vs, count));
                combinedSubSample = TOptionalSubSample();
            }
        }

        if (combinedSubSample) {
            m_Queue.push_back(*combinedSubSample);
        }
    }

    void resetBucket(core_t::TTime bucketStart) {
        // The queue is ordered in descending sub-sample start time.

        iterator firstEarlierThanBucket = std::upper_bound(m_Queue.begin(), m_Queue.end(), bucketStart, timeLater);

        // This is equivalent to lower_bound(., ., bucketStart + m_BucketLength - 1, .);
        iterator latestWithinBucket = std::upper_bound(m_Queue.begin(), m_Queue.end(), bucketStart + m_BucketLength, timeLater);

        m_Queue.erase(latestWithinBucket, firstEarlierThanBucket);
    }

    //! Returns the item in the queue at position \p index.
    const SSubSample& operator[](std::size_t index) const { return m_Queue[index]; }

    //! Returns the size of the queue.
    std::size_t size(void) const { return m_Queue.size(); }

    //! Returns the capacity of the queue.
    std::size_t capacity(void) const { return m_Queue.capacity(); }

    //! Is the queue empty?
    bool empty(void) const { return m_Queue.empty(); }

    core_t::TTime latestEnd(void) const { return m_Queue.empty() ? 0 : m_Queue.front().s_End; }

    //! \name Persistence
    //@{
    //! Persist state by passing information to the supplied inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const {
        for (const_reverse_iterator itr = m_Queue.rbegin(); itr != m_Queue.rend(); ++itr) {
            inserter.insertLevel(SUB_SAMPLE_TAG, boost::bind(&SSubSample::acceptPersistInserter, *itr, _1));
        }
    }

    //! Restore by getting information from the state document traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
        do {
            const std::string& name = traverser.name();
            if (name == SUB_SAMPLE_TAG) {
                SSubSample subSample(m_Dimension, 0);
                if (traverser.traverseSubLevel(boost::bind(&SSubSample::acceptRestoreTraverser, &subSample, _1)) == false) {
                    LOG_ERROR("Invalid sub-sample in " << traverser.value());
                    return false;
                }
                this->resizeIfFull();
                m_Queue.push_front(subSample);
            }
        } while (traverser.next());

        return true;
    }
    //@}

    //! Returns the checksum of the queue.
    uint64_t checksum(void) const { return maths::CChecksum::calculate(0, m_Queue); }

    //! Debug the memory used by the queue.
    void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
        mem->setName("CSampleQueue", sizeof(*this));
        core::CMemoryDebug::dynamicSize("m_Queue", m_Queue, mem);
    }

    //! Get the memory used by the queue.
    std::size_t memoryUsage(void) const { return sizeof(*this) + core::CMemory::dynamicSize(m_Queue); }

    //! Prints the contents of the queue.
    std::string print(void) const { return core::CContainerPrinter::print(m_Queue); }

private:
    void pushFrontNewSubSample(const TDouble1Vec& measurement, core_t::TTime time, unsigned int count) {
        this->resizeIfFull();
        SSubSample newSubSample(m_Dimension, time);
        newSubSample.s_Statistic.add(measurement, time, count);
        m_Queue.push_front(newSubSample);
    }

    void pushBackNewSubSample(const TDouble1Vec& measurement, core_t::TTime time, unsigned int count) {
        this->resizeIfFull();
        SSubSample newSubSample(m_Dimension, time);
        newSubSample.s_Statistic.add(measurement, time, count);
        m_Queue.push_back(newSubSample);
    }

    void insertNewSubSample(iterator pos, const TDouble1Vec& measurement, core_t::TTime time, unsigned int count) {
        this->resizeIfFull();
        SSubSample newSubSample(m_Dimension, time);
        newSubSample.s_Statistic.add(measurement, time, count);
        m_Queue.insert(pos, newSubSample);
    }

    void resizeIfFull(void) {
        if (m_Queue.full()) {
            std::size_t currentSize = m_Queue.size();
            std::size_t newSize = static_cast<std::size_t>(static_cast<double>(currentSize) * (1.0 + m_GrowthFactor));
            m_Queue.set_capacity(std::max(newSize, currentSize + 1));
        }
    }

    void addAfterLatestStartTime(const TDouble1Vec& measurement, core_t::TTime time, unsigned int count, unsigned int sampleCount) {
        if (time >= m_Queue[0].s_End && this->shouldCreateNewSubSampleAfterLatest(time, sampleCount)) {
            this->pushFrontNewSubSample(measurement, time, count);
        } else {
            m_Queue[0].add(measurement, time, count);
        }
    }

    bool shouldCreateNewSubSampleAfterLatest(core_t::TTime time, unsigned int sampleCount) {
        if (m_Queue[0].s_Statistic.count() >= static_cast<double>(this->targetSubSampleCount(sampleCount))) {
            return true;
        }

        // If latency is non-zero, we also want to check whether the new measurement
        // is too far from the latest sub-sample or whether they belong in different buckets.
        if (m_Latency > 0) {
            if (!m_Queue[0].isClose(time, this->targetSubSampleSpan()) || !m_Queue[0].isInSameBucket(time, m_BucketLength)) {
                return true;
            }
        }
        return false;
    }

    core_t::TTime targetSubSampleSpan(void) const {
        return (m_BucketLength + static_cast<core_t::TTime>(m_SampleCountFactor) - 1) / static_cast<core_t::TTime>(m_SampleCountFactor);
    }

    std::size_t targetSubSampleCount(unsigned int sampleCount) const { return static_cast<std::size_t>(sampleCount) / m_SampleCountFactor; }

    void addHistorical(const TDouble1Vec& measurement, core_t::TTime time, unsigned int count, unsigned int sampleCount) {
        // We have to resize before we do the search of the upper bound. Otherwise,
        // a later resize will invalidate the upper bound iterator.
        this->resizeIfFull();

        reverse_iterator upperBound = std::upper_bound(m_Queue.rbegin(), m_Queue.rend(), time, timeEarlier);
        core_t::TTime targetSubSampleSpan = this->targetSubSampleSpan();

        if (upperBound == m_Queue.rbegin()) {
            if ((upperBound->s_Statistic.count() >= static_cast<double>(this->targetSubSampleCount(sampleCount))) ||
                !upperBound->isClose(time, targetSubSampleSpan) || !(*upperBound).isInSameBucket(time, m_BucketLength)) {
                this->pushBackNewSubSample(measurement, time, count);
            } else {
                upperBound->add(measurement, time, count);
            }
            return;
        }

        SSubSample& left = *(upperBound - 1);
        SSubSample& right = *upperBound;
        if (time <= left.s_End) {
            left.add(measurement, time, count);
            return;
        }
        bool sameBucketWithLeft = left.isInSameBucket(time, m_BucketLength);
        bool sameBucketWithRight = right.isInSameBucket(time, m_BucketLength);
        std::size_t spaceLimit = this->targetSubSampleCount(sampleCount);
        bool leftHasSpace = static_cast<std::size_t>(left.s_Statistic.count()) < spaceLimit;
        bool rightHasSpace = static_cast<std::size_t>(right.s_Statistic.count()) < spaceLimit;
        core_t::TTime leftDistance = time - left.s_End;
        core_t::TTime rightDistance = right.s_Start - time;
        SSubSample& candidate = maths::COrderings::lexicographical_compare(-static_cast<int>(sameBucketWithLeft),
                                                                           -static_cast<int>(leftHasSpace),
                                                                           leftDistance,
                                                                           -static_cast<int>(sameBucketWithRight),
                                                                           -static_cast<int>(rightHasSpace),
                                                                           rightDistance)
                                    ? left
                                    : right;

        if (candidate.isInSameBucket(time, m_BucketLength) &&
            (candidate.isClose(time, targetSubSampleSpan) || right.s_Start <= left.s_End + targetSubSampleSpan)) {
            candidate.add(measurement, time, count);
            return;
        }
        this->insertNewSubSample(upperBound.base(), measurement, time, count);
    }

    static bool timeEarlier(core_t::TTime time, const SSubSample& subSample) { return time < subSample.s_Start; }

    static bool timeLater(core_t::TTime time, const SSubSample& subSample) { return time > subSample.s_Start; }

private:
    std::size_t m_Dimension;
    TQueue m_Queue;
    std::size_t m_SampleCountFactor;
    double m_GrowthFactor;
    core_t::TTime m_BucketLength;
    core_t::TTime m_Latency;
};

template<typename STATISTIC>
const std::string CSampleQueue<STATISTIC>::SSubSample::SAMPLE_START_TAG("a");
template<typename STATISTIC>
const std::string CSampleQueue<STATISTIC>::SSubSample::SAMPLE_END_TAG("b");
template<typename STATISTIC>
const std::string CSampleQueue<STATISTIC>::SSubSample::SAMPLE_TAG("c");
template<typename STATISTIC>
const std::string CSampleQueue<STATISTIC>::SUB_SAMPLE_TAG("a");
}
}

#endif // INCLUDED_ml_model_CSampleQueue_h
