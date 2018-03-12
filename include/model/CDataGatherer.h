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

#ifndef INCLUDED_ml_model_CDataGatherer_h
#define INCLUDED_ml_model_CDataGatherer_h

#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CoreTypes.h>
#include <core/CStoredStringPtr.h>

#include <model/CBucketGatherer.h>
#include <model/CBucketQueue.h>
#include <model/CDynamicStringIdRegistry.h>
#include <model/CEventData.h>
#include <model/CModelParams.h>
#include <model/FunctionTypes.h>
#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <boost/any.hpp>
#include <boost/optional.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>

#include <string>
#include <utility>
#include <vector>

#include <stdint.h>


namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace model {
class CEventData;
class CMetricBucketGatherer;
class CResourceMonitor;
class CSampleCounts;
class CSearchKey;

//! \brief Time series data gathering interface and common functionality.
//!
//! DESCRIPTION:\n
//! This defines the common interface to gather the data which we model
//! in order to characterize a time series. The interface breaks down
//! in to fields, update with new event data, features, person, attribute,
//! metric and functionality to manage the passage of time.
//!
//! The features provide a way of customizing the data to model. For more
//! details see model_t::EFeature. Features are generally quantities that
//! are computed for the bucketing time intervals. There is a templated
//! accessor to retrieve all feature data which is the main interface used
//! by the model classes to retrieve data.
//!
//! The raw events can be partitioned by up to two categorical fields. These
//! map to the concept of person and attribute fields. The person interface
//! provides a mapping between people and their unique identifiers and also
//! manages retrieving people's counts and life cycle. The attribute interface
//! is only meaningful for population modeling and and provides a mapping
//! between attribute values and their unique identifiers and manages their
//! life cycle.
//!
//! Finally, the metric interface provides some custom functions which apply
//! specifically to metric valued time series. (See CMetricBucketGatherer
//! for more information.)
//!
//! IMPLEMENTATION:\n
//! This functionality has been separated from the CModel class hierarchy,
//! which own data gatherer objects because we want to avoid monolithic
//! model classes.
//!
//! This represents a natural division of the tasks of modeling a time
//! series and gathering the data to model. It is anticipated that the
//! data gathering could become reasonably involved if, for example, we
//! start doing regression to estimate gradient, curvature or even
//! arbitrary basis function coefficients, to describe the shape of the
//! time series or start estimating temporal correlation in the series
//! (auto-regression).
//!
//! All accessors for quantities which vary in time in this class *must*
//! take a time. This is so that they can sanity check the input to
//! ensure that data are available at the requested time. The intention
//! is that this should automatically detect if the gatherer is being
//! misused. The passage of time is managed in objects of this class
//! by addArrival (which invokes timeNow with the event time) and timeNow
//! which refreshes the current time. Data are only available for the
//! bucketing interval containing the current time so must be sampled
//! before the time is incremented passed the end of that bucketing
//! interval. For models, this is managed by the CModel::sample function
//! implementations.
//!
//! Time-based data gathering is handled by further classes derived from
//! CBucketGatherer, for Metrics and EventRates accordingly.
class MODEL_EXPORT CDataGatherer {
    public:
        typedef std::vector<double> TDoubleVec;
        typedef core::CSmallVector<double, 1> TDouble1Vec;
        typedef std::vector<std::size_t> TSizeVec;
        typedef std::vector<std::string> TStrVec;
        typedef TStrVec::const_iterator TStrVecCItr;
        typedef std::vector<const std::string*> TStrCPtrVec;
        typedef std::pair<std::size_t, uint64_t> TSizeUInt64Pr;
        typedef std::vector<TSizeUInt64Pr> TSizeUInt64PrVec;
        typedef model_t::TFeatureVec TFeatureVec;
        typedef TFeatureVec::const_iterator TFeatureVecCItr;
        typedef std::pair<std::size_t, std::size_t> TSizeSizePr;
        typedef std::pair<TSizeSizePr, uint64_t> TSizeSizePrUInt64Pr;
        typedef std::vector<TSizeSizePrUInt64Pr> TSizeSizePrUInt64PrVec;
        typedef boost::unordered_map<TSizeSizePr, uint64_t> TSizeSizePrUInt64UMap;
        typedef TSizeSizePrUInt64UMap::iterator TSizeSizePrUInt64UMapItr;
        typedef TSizeSizePrUInt64UMap::const_iterator TSizeSizePrUInt64UMapCItr;
        typedef CBucketQueue<TSizeSizePrUInt64UMap> TSizeSizePrUInt64UMapQueue;
        typedef TSizeSizePrUInt64UMapQueue::iterator TSizeSizePrUInt64UMapQueueItr;
        typedef TSizeSizePrUInt64UMapQueue::const_iterator TSizeSizePrUInt64UMapQueueCItr;
        typedef TSizeSizePrUInt64UMapQueue::const_reverse_iterator TSizeSizePrUInt64UMapQueueCRItr;
        typedef CBucketGatherer::TSizeSizePrStoredStringPtrPrUInt64UMap TSizeSizePrStoredStringPtrPrUInt64UMap;
        typedef TSizeSizePrStoredStringPtrPrUInt64UMap::const_iterator TSizeSizePrStoredStringPtrPrUInt64UMapCItr;
        typedef TSizeSizePrStoredStringPtrPrUInt64UMap::iterator TSizeSizePrStoredStringPtrPrUInt64UMapItr;
        typedef std::vector<TSizeSizePrStoredStringPtrPrUInt64UMap> TSizeSizePrStoredStringPtrPrUInt64UMapVec;
        typedef CBucketQueue<TSizeSizePrStoredStringPtrPrUInt64UMapVec> TSizeSizePrStoredStringPtrPrUInt64UMapVecQueue;
        typedef boost::reference_wrapper<const CSearchKey> TSearchKeyCRef;
        typedef std::vector<CBucketGatherer*> TBucketGathererPVec;
        typedef TBucketGathererPVec::iterator TBucketGathererPVecItr;
        typedef TBucketGathererPVec::const_iterator TBucketGathererPVecCItr;
        typedef std::pair<model_t::EFeature, boost::any> TFeatureAnyPr;
        typedef std::vector<TFeatureAnyPr> TFeatureAnyPrVec;
        typedef std::vector<model_t::EMetricCategory> TMetricCategoryVec;
        typedef boost::shared_ptr<CSampleCounts> TSampleCountsPtr;
        typedef std::vector<core_t::TTime> TTimeVec;
        typedef TTimeVec::const_iterator TTimeVecCItr;

    public:
        //! The summary count indicating an explicit null record.
        static const std::size_t EXPLICIT_NULL_SUMMARY_COUNT;

        //! The expected memory usage per by field
        static const std::size_t ESTIMATED_MEM_USAGE_PER_BY_FIELD;

        //! The expected memory usage per over field
        static const std::size_t ESTIMATED_MEM_USAGE_PER_OVER_FIELD;

    public:
        //! \name Life-cycle
        //@{
        //! Create a new data series gatherer.
        //!
        //! \param[in] gathererType Indicates what sort of bucket data to gather:
        //! EventRate/Metric, Population/Individual
        //! \param[in] summaryMode Indicates whether the data being gathered
        //! are already summarized by an external aggregation process.
        //! \param[in] modelParams The global configuration parameters.
        //! \param[in] summaryCountFieldName If \p summaryMode is E_Manual
        //! then this is the name of the field holding the summary count.
        //! \param[in] partitionFieldName The name of the field which splits
        //! the data.
        //! \param[in] partitionFieldValue The value of the field which splits
        //! the data.
        //! \param[in] personFieldName The name of the field which identifies
        //! people.
        //! \param[in] attributeFieldName The name of the field which defines
        //! the person attributes.
        //! \param[in] valueFieldName The name of the field which contains
        //! the metric values.
        //! \param[in] influenceFieldNames The field names for which we will
        //! compute influences.
        //! \param[in] useNull If true the gatherer will process missing
        //! person and attribute field values (assuming they are empty).
        //! \param[in] key The key of the search for which to gatherer data.
        //! \param[in] features The features of the data to model.
        //! \param[in] startTime The start of the time interval for which
        //! to gather data.
        //! \param[in] sampleCountOverride for the number of measurements
        //! in a statistic. (Note that this is intended for testing only.)
        //! A zero value means that the data gatherer class will determine
        //! an appropriate value for the bucket length and data rate.
        CDataGatherer(model_t::EAnalysisCategory gathererType,
                      model_t::ESummaryMode summaryMode,
                      const SModelParams &modelParams,
                      const std::string &summaryCountFieldName,
                      const std::string &partitionFieldName,
                      const std::string &partitionFieldValue,
                      const std::string &personFieldName,
                      const std::string &attributeFieldName,
                      const std::string &valueFieldName,
                      const TStrVec &influenceFieldNames,
                      bool useNull,
                      const CSearchKey &key,
                      const TFeatureVec &features,
                      core_t::TTime startTime,
                      int sampleCountOverride);

        //! Construct from a state document.
        CDataGatherer(model_t::EAnalysisCategory gathererType,
                      model_t::ESummaryMode summaryMode,
                      const SModelParams &modelParams,
                      const std::string &summaryCountFieldName,
                      const std::string &partitionFieldName,
                      const std::string &partitionFieldValue,
                      const std::string &personFieldName,
                      const std::string &attributeFieldName,
                      const std::string &valueFieldName,
                      const TStrVec &influenceFieldNames,
                      bool useNull,
                      const CSearchKey &key,
                      core::CStateRestoreTraverser &traverser);

        //! Create a copy that will result in the same persisted state as the
        //! original.  This is effectively a copy constructor that creates a
        //! copy that's only valid for a single purpose.  The boolean flag is
        //! redundant except to create a signature that will not be mistaken for
        //! a general purpose copy constructor.
        CDataGatherer(bool isForPersistence, const CDataGatherer &other);

        ~CDataGatherer(void);
        //@}

        //! \name Persistence
        //@{
        //! Persist state by passing information to the supplied inserter.
        void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! Create a clone of this data gatherer that will result in the same
        //! persisted state.  The clone may be incomplete in ways that do not
        //! affect the persisted representation, and must not be used for any
        //! other purpose.
        //!
        //! \warning The caller owns the object returned.
        CDataGatherer *cloneForPersistence(void) const;
        //@}

        //! Check if the data being gathered are already summarized by an
        //! external aggregation process.
        model_t::ESummaryMode summaryMode(void) const;

        //! Get the function.
        model::function_t::EFunction function(void) const;

        //! Get a description of the component searches.
        std::string description(void) const;

        //! Is this a population data gatherer?
        bool isPopulation(void) const;

        //! Get the maximum size of all the member containers.
        std::size_t maxDimension(void) const;

        //! \name Fields
        //@{
        //! Get the partition field name.
        //!
        //! The name of the partitioning field.
        const std::string &partitionFieldName(void) const;

        //! Get the partition field value.
        //!
        //! The value of the partitioning field.
        const std::string &partitionFieldValue(void) const;

        //! This is the common field in all searches "along" which the
        //! probabilities are aggregated, i.e. the "by" field name for
        //! individual models and the "over" field name for population
        //! models.
        const std::string &personFieldName(void) const;

        //! Get the attribute field name if one exists.
        const std::string &attributeFieldName(void) const;

        //! Get the name of the field containing the metric value.
        const std::string &valueFieldName(void) const;

        //! Get an iterator at the beginning the influencing field names.
        TStrVecCItr beginInfluencers(void) const;

        //! Get an iterator at the end of the influencing field names.
        TStrVecCItr endInfluencers(void) const;

        //! Return the search key for which this is gathering data.
        const CSearchKey &searchKey(void) const;

        //! Get the fields for which to gather data.
        //!
        //! This defines the fields to extract from a record. These include
        //! the fields which define the categories whose counts are being
        //! analyzed, the fields containing metric series names and values
        //! and the fields defining a population.
        const TStrVec &fieldsOfInterest(void) const;

        //! Get the number of by field values.  For a population model this will
        //! be equal to numberActiveAttributes(); for an individual model
        //! numberActivePeople().
        std::size_t numberByFieldValues(void) const;

        //! Get the number of over field values.  For a population model this
        //! will be equal to numberActivePeople(); for an individual model 0.
        std::size_t numberOverFieldValues(void) const;

        //! Have we been configured to use NULL values?
        bool useNull(void) const;
        //@}

        //! \name Update
        //@{
        //! Process the specified fields.
        //!
        //! This adds people and attributes as necessary and fills out the
        //! event data from \p fieldValues.
        bool processFields(const TStrCPtrVec &fieldValues,
                           CEventData &result,
                           CResourceMonitor &resourceMonitor);

        //! Record the arrival of \p data at \p time.
        bool addArrival(const TStrCPtrVec &fieldValues,
                        CEventData &data,
                        CResourceMonitor &resourceMonitor);

        //! Roll time to the end of the bucket that is latency after the sampled bucket.
        void sampleNow(core_t::TTime sampleBucketStart);

        //! Roll time to the end of the bucket that is latency after the sampled bucket
        //! without performing any updates that impact the model.
        void skipSampleNow(core_t::TTime sampleBucketStart);
        //@}

        //! \name Features
        //@{
        //! Get the number of features on which this is gathering data.
        std::size_t numberFeatures(void) const;

        //! Check if this is gathering data on \p feature.
        bool hasFeature(model_t::EFeature feature) const;

        //! Get the feature corresponding to \p i.
        //!
        //! \warning \p i must be in range for the features this gatherer
        //! is collecting, i.e. it must be less than numberFeatures.
        model_t::EFeature feature(std::size_t i) const;

        //! Get the collection of features for which data is being gathered.
        const TFeatureVec &features(void) const;

        //! Get the data for all features for the bucketing time interval
        //! containing \p time.
        //!
        //! \param[in] time The time of interest.
        //! \param[out] result Filled in with the feature data at \p time.
        //! \tparam T The type of the feature data.
        template<typename T>
        bool featureData(core_t::TTime time, core_t::TTime bucketLength,
                         std::vector<std::pair<model_t::EFeature, T> > &result) const {
            TFeatureAnyPrVec rawFeatureData;
            this->chooseBucketGatherer(time).featureData(time, bucketLength, rawFeatureData);

            bool succeeded = true;

            result.clear();
            result.reserve(rawFeatureData.size());
            for (std::size_t i = 0u; i < rawFeatureData.size(); ++i) {
                TFeatureAnyPr &feature = rawFeatureData[i];

                // Check the typeid before attempting the cast so we
                // don't use throw to handle failure, which is slow.
                if (feature.second.type() != typeid(T)) {
                    LOG_ERROR("Bad type for feature = " << model_t::print(feature.first)
                              << ", expected " << typeid(T).name()
                              << " got " << feature.second.type().name());
                    succeeded = false;
                    continue;
                }

                // We emulate move semantics here to avoid the expensive
                // copy if T is large (as we expect it might be sometimes).
                // We have to adopt the using std::swap idiom (contravening
                // coding guidelines) because T can be a built in type.
                // Unfortunately, this implementation requires T to be
                // default constructible.
                using std::swap;
                result.push_back(std::pair<model_t::EFeature, T>(feature.first, T()));
                T &tmp = boost::any_cast<T&>(feature.second);
                swap(result.back().second, tmp);
            }

            return succeeded;
        }
        //@}

        //! \name Person
        //@{
        //! Get the number of active people (not pruned).
        std::size_t numberActivePeople(void) const;

        //! Get the maximum person identifier seen so far
        //! (some of which might have been pruned).
        std::size_t numberPeople(void) const;

        //! Get the unique identifier of a person if it exists.
        //!
        //! \param[in] person The person of interest.
        //! \param[out] result Filled in with the identifier of \p person
        //! if they exist otherwise max std::size_t.
        //! \return True if the person exists and false otherwise.
        bool personId(const std::string &person, std::size_t &result) const;

        //! Get the unique identifier of an arbitrary known person.
        //! \param[out] result Filled in with the identifier of a person
        //! \return True if a person exists and false otherwise.
        bool anyPersonId(std::size_t &result) const;

        //! Get the name of the person identified by \p pid if they exist.
        //!
        //! \param[in] pid The unique identifier of the person of interest.
        //! \return The person name if they exist and a fallback otherwise.
        const std::string &personName(std::size_t pid) const;

        //! Get the name of the person identified by \p pid if they exist.
        //!
        //! \param[in] pid The unique identifier of the person of interest.
        //! \return The person name if they exist and a fallback otherwise.
        const core::CStoredStringPtr &personNamePtr(std::size_t pid) const;

        //! Get the name of the person identified by \p pid if they exist.
        //!
        //! \param[in] pid The unique identifier of the person of interest.
        //! \param[in] fallback The fall back name.
        //! \return The person name if they exist and \p fallback otherwise.
        const std::string &personName(std::size_t pid, const std::string &fallback) const;

        //! Get the non-zero counts by person for the bucketing interval
        //! containing \p time.
        //!
        //! \param[in] time The time of interest.
        //! \param[out] result Filled in with the non-zero counts by person.
        //! The first element is the person identifier and the second their
        //! count in the bucketing interval. The result is sorted by person.
        //! \note We expect the non-zero counts to be sparse on the space
        //! of people so use a sparse encoding:
        //! <pre class="fragment">
        //!   \f$ pid \leftarrow c\f$
        //! </pre>
        //! where,\n
        //!   \f$pid\f$ is the person identifier,\n
        //!   \f$c\f$ is the count for the person.
        void personNonZeroCounts(core_t::TTime time, TSizeUInt64PrVec &result) const;

        //! Stop gathering data on the people identified by \p peopleToRemove.
        void recyclePeople(const TSizeVec &peopleToRemove);

        //! Remove all traces of people whose identifiers are greater than
        //! or equal to \p lowestPersonToRemove.
        void removePeople(std::size_t lowestPersonToRemove);

        //! Get unique identifiers of any people that have been recycled.
        TSizeVec &recycledPersonIds(void);

        //! Check that the person is no longer being modeled.
        bool isPersonActive(std::size_t pid) const;

        //! Record a person called \p person.
        std::size_t addPerson(const std::string &person,
                              CResourceMonitor &resourceMonitor,
                              bool &addedPerson);
        //@}

        //! \name Attribute
        //@{
        //! Get the number of active attributes (not pruned).
        std::size_t numberActiveAttributes(void) const;

        //! Get the maximum attribute identifier seen so far
        //! (some of which might have been pruned).
        std::size_t numberAttributes(void) const;

        //! Get the unique identifier of an attribute if it exists.
        //!
        //! \param[in] attribute The person of interest.
        //! \param[out] result Filled in with the identifier of \p attribute
        //! if they exist otherwise max std::size_t.
        //! \return True if the attribute exists and false otherwise.
        bool attributeId(const std::string &attribute, std::size_t &result) const;

        //! Get the name of the attribute identified by \p cid if they exist.
        //!
        //! \param[in] cid The unique identifier of the attribute of interest.
        //! \return The attribute name if it exists anda fallback otherwise.
        const std::string &attributeName(std::size_t cid) const;

        //! Get the name of the attribute identified by \p pid if they exist.
        //!
        //! \param[in] cid The unique identifier of the attribute of interest.
        //! \return The attribute name if they exist and a fallback otherwise.
        const core::CStoredStringPtr &attributeNamePtr(std::size_t cid) const;

        //! Get the name of the attribute identified by \p cid if they exist.
        //!
        //! \param[in] cid The unique identifier of the attribute of interest.
        //! \param[in] fallback The fall back name.
        //! \return The attribute name if it exists and \p fallback otherwise.
        const std::string &attributeName(std::size_t cid, const std::string &fallback) const;

        //! Stop gathering data on the attributes identified by \p attributesToRemove.
        void recycleAttributes(const TSizeVec &attributesToRemove);

        //! Remove all traces of attributes whose identifiers are greater than
        //! or equal to \p lowestAttributeToRemove.
        void removeAttributes(std::size_t lowestAttributeToRemove);

        //! Get unique identifiers of any attributes that have been recycled.
        TSizeVec &recycledAttributeIds(void);

        //! Check that the person is no longer being modeled.
        bool isAttributeActive(std::size_t cid) const;
        //@}

        //! \name Metric
        //@{
        //! Get the current number of measurements in a sample for
        //! the model of the entity identified by \p id.
        //!
        //! If we are performing temporal analysis we have one sample
        //! count per person and if we are performing population analysis
        //! we have one sample count per attribute.
        double sampleCount(std::size_t id) const;

        //! Get the effective number of measurements in a sample for
        //! the model of the entity identified by \p id.
        //!
        //! If we are performing temporal analysis we have one sample
        //! count per person and if we are performing population analysis
        //! we have one sample count per attribute.
        double effectiveSampleCount(std::size_t id) const;

        //! Reset the number of measurements in a sample for the entity
        //! identified \p id.
        //!
        //! If we are performing individual analysis we have one sample
        //! count per person and if we are performing population analysis
        //! we have one sample count per attribute.
        void resetSampleCount(std::size_t id);

        //! Get the sample counts.
        TSampleCountsPtr sampleCounts(void) const;
        //@}

        //! \name Time
        //@{
        //! Get the start of the current bucketing time interval.
        core_t::TTime currentBucketStartTime(void) const;

        //! Reset the current bucketing interval start time.
        void currentBucketStartTime(core_t::TTime bucketStart);

        //! Get the length of the bucketing time interval.
        core_t::TTime bucketLength(void) const;

        //! Check if data is available at \p time.
        bool dataAvailable(core_t::TTime time) const;

        //! For each bucket in the interval [\p startTime, \p endTime],
        //! validate that it can be sampled and increment \p startTime
        //! to the first valid bucket or \p endTime if no valid buckets
        //! exist.
        //!
        //! \param[in,out] startTime The start of the interval to sample.
        //! \param[in] endTime The end of the interval to sample.
        bool validateSampleTimes(core_t::TTime &startTime,
                                 core_t::TTime endTime) const;

        //! Roll time forwards to \p time. Note this method is only supported
        //! for testing purposes and should not normally be called.
        void timeNow(core_t::TTime time);

        //! Print the current bucket.
        std::string printCurrentBucket(core_t::TTime time) const;

        //! Record a attribute called \p attribute.
        std::size_t addAttribute(const std::string &attribute,
                                 CResourceMonitor &resourceMonitor,
                                 bool &addedAttribute);
        //@}

        //! \name Counts
        //@{
        //! Get the non-zero (person, attribute) pair counts in the
        //! bucketing interval corresponding to the given time.
        const TSizeSizePrUInt64UMap &bucketCounts(core_t::TTime time) const;

        //! Get the non-zero (person, attribute) pair counts for each
        //! value of influencing field.
        const TSizeSizePrStoredStringPtrPrUInt64UMapVec &influencerCounts(core_t::TTime time) const;
        //@}

        //! Get the checksum of this gatherer.
        uint64_t checksum(void) const;

        //! Debug the memory used by this component.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this component.
        std::size_t memoryUsage(void) const;

        //! Clear this data gatherer.
        void clear(void);

        //! Reset bucket and return true if bucket was successfully
        //! reset or false otherwise.
        //! Note that this should not be used in conjunction with out-of-phase buckets
        //! where the concept of resetting a specific bucketed period of time is
        //! not valid.
        bool resetBucket(core_t::TTime bucketStart);

        //! Release memory that is no longer needed
        void releaseMemory(core_t::TTime samplingCutoffTime);

        //! Get the global configuration parameters.
        const SModelParams &params(void) const;

        // \name Tuple
        //@{
        //! Extract the person identifier from a tuple.
        template<typename T>
        static inline std::size_t extractPersonId(const std::pair<const TSizeSizePr, T> &tuple) {
            return tuple.first.first;
        }
        //! Extract the person identifier from a tuple.
        template<typename T>
        static inline std::size_t extractPersonId(const std::pair<TSizeSizePr, T> &tuple) {
            return tuple.first.first;
        }
        //! Extract the person identifier from a tuple.
        static inline std::size_t extractPersonId(const TSizeSizePr &tuple) {
            return tuple.first;
        }
        //! Extracts the person identifier from a tuple.
        struct SExtractPersonId {
            template<typename TUPLE>
            std::size_t operator()(const TUPLE &t) const {
                return CDataGatherer::extractPersonId(t);
            }
        };

        //! Extract the attribute identifier from a tuple.
        template<typename T>
        static inline std::size_t extractAttributeId(const std::pair<const TSizeSizePr, T> &tuple) {
            return tuple.first.second;
        }
        //! Extract the attribute identifier from a tuple.
        template<typename T>
        static inline std::size_t extractAttributeId(const std::pair<TSizeSizePr, T> &tuple) {
            return tuple.first.second;
        }
        //! Extract the attribute identifier from a tuple.
        static inline std::size_t extractAttributeId(const TSizeSizePr &tuple) {
            return tuple.second;
        }
        //! Extracts the attribute identifier from a tuple.
        struct SExtractAttributeId {
            template<typename TUPLE>
            std::size_t operator()(const TUPLE &t) const {
                return CDataGatherer::extractAttributeId(t);
            }
        };

        //! Extract the data from a tuple.
        template<typename T>
        static inline const T &extractData(const std::pair<const TSizeSizePr, T> &tuple) {
            return tuple.second;
        }
        //! Extract the data from a tuple.
        template<typename T>
        static inline const T &extractData(const std::pair<TSizeSizePr, T> &tuple) {
            return tuple.second;
        }
        //@}

        //! In the case of manually named summarized statistics, map the first
        //! feature to a metric category.
        bool determineMetricCategory(TMetricCategoryVec &fieldMetricCategories) const;

        //! Helper to avoid code duplication when getting a count from a
        //! field.  Logs different errors for missing value and invalid value.
        bool extractCountFromField(const std::string &fieldName,
                                   const std::string *fieldValue,
                                   std::size_t &count) const;

        //! Helper to avoid code duplication when getting a metric value from a
        //! field.  Logs different errors for missing value and invalid value.
        bool extractMetricFromField(const std::string &fieldName,
                                    std::string fieldValue,
                                    TDouble1Vec &metricValue) const;

        //! Returns the startTime of the earliest bucket for which data are still
        //! accepted.
        core_t::TTime earliestBucketStartTime(void) const;

        //! Check the class invariants.
        bool checkInvariants(void) const;

    private:
        //! The summary count field value to indicate that the record should
        //! be ignored.
        static const std::string EXPLICIT_NULL;

    private:
        typedef boost::reference_wrapper<const SModelParams> TModelParamsCRef;

    private:
        //! Select the correct bucket gatherer based on the time: if we have
        //! out-of-phase buckets, select either in-phase or out-of-phase.
        const CBucketGatherer &chooseBucketGatherer(core_t::TTime time) const;

        //! Select the correct bucket gatherer based on the time: if we have
        //! out-of-phase buckets, select either in-phase or out-of-phase.
        CBucketGatherer &chooseBucketGatherer(core_t::TTime time);

        //! Restore state from supplied traverser.
        bool acceptRestoreTraverser(const std::string &summaryCountFieldName,
                                    const std::string &personFieldName,
                                    const std::string &attributeFieldName,
                                    const std::string &valueFieldName,
                                    const TStrVec &influenceFieldNames,
                                    core::CStateRestoreTraverser &traverser);

        //! Restore a bucket gatherer from the supplied traverser.
        bool restoreBucketGatherer(const std::string &summaryCountFieldName,
                                   const std::string &personFieldName,
                                   const std::string &attributeFieldName,
                                   const std::string &valueFieldName,
                                   const TStrVec &influenceFieldNames,
                                   core::CStateRestoreTraverser &traverser);

        //! Persist a bucket gatherer by passing information to the supplied
        //! inserter.
        void persistBucketGatherers(core::CStatePersistInserter &inserter) const;

        //! Create the bucket specific data gatherer.
        void createBucketGatherer(model_t::EAnalysisCategory gathererType,
                                  const std::string &summaryCountFieldName,
                                  const std::string &personFieldName,
                                  const std::string &attributeFieldName,
                                  const std::string &valueFieldName,
                                  const TStrVec &influenceFieldNames,
                                  core_t::TTime startTime,
                                  unsigned int sampleCountOverride);

    private:
        //! The type of the bucket gatherer(s) used.
        model_t::EAnalysisCategory m_GathererType;

        //! The collection of features on which to gather data.
        TFeatureVec m_Features;

        //! The collection of bucket gatherers which contain the bucket-specific
        //! metrics and counts.
        TBucketGathererPVec m_Gatherers;

        //! Indicates whether the data being gathered are already summarized
        //! by an external aggregation process.
        model_t::ESummaryMode m_SummaryMode;

        //! The global configuration parameters.
        TModelParamsCRef m_Params;

        //! The partition field name or an empty string if there isn't one.
        std::string m_PartitionFieldName;

        //! The value of the partition field for this detector.
        core::CStoredStringPtr m_PartitionFieldValue;

        //! The key of the search for which data is being gathered.
        TSearchKeyCRef m_SearchKey;

        //! A registry where person names are mapped to unique IDs.
        CDynamicStringIdRegistry m_PeopleRegistry;

        //! A registry where attribute names are mapped to unique IDs.
        CDynamicStringIdRegistry m_AttributesRegistry;

        //! True if this is a population data gatherer and false otherwise.
        bool m_Population;

        //! If true the gatherer will process missing person field values.
        bool m_UseNull;

        //! The object responsible for managing sample counts.
        TSampleCountsPtr m_SampleCounts;
};

}
}

#endif // INCLUDED_ml_model_CDataGatherer_h
