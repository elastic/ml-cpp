/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CEventRateBucketGatherer_h
#define INCLUDED_ml_model_CEventRateBucketGatherer_h

#include <core/CCompressedDictionary.h>
#include <core/CMemory.h>
#include <core/CStoredStringPtr.h>
#include <core/CoreTypes.h>

#include <maths/CChecksum.h>

#include <model/CDataGatherer.h>
#include <model/CFeatureData.h>
#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <boost/any.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include <set>
#include <string>
#include <vector>

namespace ml {
namespace model {

//! \brief A structure to handle storing unique strings per person,
//! attribute and influencer, used for the analytic functions
//! "distinct_count" and "info_content"
class MODEL_EXPORT CUniqueStringFeatureData {
public:
    using TDictionary1 = core::CCompressedDictionary<1>;
    using TWord = TDictionary1::CWord;
    using TWordSet = TDictionary1::TWordSet;
    using TWordStringUMap = TDictionary1::TWordTUMap<std::string>;
    using TStoredStringPtrWordSetUMap = boost::unordered_map<core::CStoredStringPtr, TWordSet>;
    using TStoredStringPtrWordSetUMapVec = std::vector<TStoredStringPtrWordSetUMap>;
    using TStrCRef = SEventRateFeatureData::TStrCRef;
    using TDouble1Vec = SEventRateFeatureData::TDouble1Vec;
    using TDouble1VecDoublePr = SEventRateFeatureData::TDouble1VecDoublePr;
    using TStrCRefDouble1VecDoublePrPr = SEventRateFeatureData::TStrCRefDouble1VecDoublePrPr;
    using TStrCRefDouble1VecDoublePrPrVec = SEventRateFeatureData::TStrCRefDouble1VecDoublePrPrVec;
    using TStoredStringPtrVec = CBucketGatherer::TStoredStringPtrVec;

public:
    //! Add a string into the collection
    void insert(const std::string& value, const TStoredStringPtrVec& influences);

    //! Fill in a FeatureData structure with the influence strings and counts
    void populateDistinctCountFeatureData(SEventRateFeatureData& featureData) const;

    //! Fill in a FeatureData structure with the influence info_content
    void populateInfoContentFeatureData(SEventRateFeatureData& featureData) const;

    //! Persist state by passing information \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Initialize state reading from \p traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Get the checksum of this object.
    uint64_t checksum() const;

    //! Get the memory usage of this object in a tree structure.
    void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the memory usage of this object.
    std::size_t memoryUsage() const;

    //! Print the unique strings for debug.
    std::string print() const;

private:
    TDictionary1 m_Dictionary1;
    TWordStringUMap m_UniqueStrings;
    TStoredStringPtrWordSetUMapVec m_InfluencerUniqueStrings;
};

//! \brief Event rate data gathering class.
//!
//! DESCRIPTION:\n
//! This performs all pre-processing of the data, which we use in order
//! to model the event rate in an arbitrary time series.
//!
//! \sa CDataGatherer.
class MODEL_EXPORT CEventRateBucketGatherer final : public CBucketGatherer {
public:
    using TCategoryAnyMap = std::map<model_t::EEventRateCategory, boost::any>;
    using TStrCRef = SEventRateFeatureData::TStrCRef;
    using TDouble1Vec = SEventRateFeatureData::TDouble1Vec;
    using TDouble1VecDoublePr = SEventRateFeatureData::TDouble1VecDoublePr;
    using TStrCRefDouble1VecDoublePrPr = SEventRateFeatureData::TStrCRefDouble1VecDoublePrPr;
    using TStrCRefDouble1VecDoublePrPrVec = SEventRateFeatureData::TStrCRefDouble1VecDoublePrPrVec;
    using TStrCRefDouble1VecDoublePrPrVecVec = SEventRateFeatureData::TStrCRefDouble1VecDoublePrPrVecVec;
    using TSizeFeatureDataPr = std::pair<std::size_t, SEventRateFeatureData>;
    using TSizeFeatureDataPrVec = std::vector<TSizeFeatureDataPr>;
    using TSizeSizePrFeatureDataPr = std::pair<TSizeSizePr, SEventRateFeatureData>;
    using TSizeSizePrFeatureDataPrVec = std::vector<TSizeSizePrFeatureDataPr>;

public:
    //! \name Life-cycle
    //@{
    //! Create an event rate bucket gatherer.
    //!
    //! \param[in] dataGatherer The owning data gatherer.
    //! \param[in] summaryCountFieldName If summaryMode is E_Manual
    //! then this is the name of the field holding the summary count.
    //! \param[in] personFieldName The name of the field which identifies
    //! people.
    //! \param[in] attributeFieldName The name of the field which defines
    //! the person attributes.
    //! \param[in] valueFieldName The name of the field which contains
    //! the metric values.
    //! \param[in] influenceFieldNames The field names for which we will
    //! compute influences.
    //! \param[in] startTime The start of the time interval for which
    //! to gather data.
    CEventRateBucketGatherer(CDataGatherer& dataGatherer,
                             const std::string& summaryCountFieldName,
                             const std::string& personFieldName,
                             const std::string& attributeFieldName,
                             const std::string& valueFieldName,
                             const TStrVec& influenceFieldNames,
                             core_t::TTime startTime);

    //! Construct from a state document.
    CEventRateBucketGatherer(CDataGatherer& dataGatherer,
                             const std::string& summaryCountFieldName,
                             const std::string& personFieldName,
                             const std::string& attributeFieldName,
                             const std::string& valueFieldName,
                             const TStrVec& influenceFieldNames,
                             core::CStateRestoreTraverser& traverser);

    //! Create a copy that will result in the same persisted state as the
    //! original.  This is effectively a copy constructor that creates a
    //! copy that's only valid for a single purpose.  The boolean flag is
    //! redundant except to create a signature that will not be mistaken
    //! for a general purpose copy constructor.
    CEventRateBucketGatherer(bool isForPersistence, const CEventRateBucketGatherer& other);
    //@}

    //! \name Persistence
    //@{
    //! Fill in the state from \p traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Persist state by passing information to the supplied inserter
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Create a clone of this data gatherer that will result in the same
    //! persisted state.  The clone may be incomplete in ways that do not
    //! affect the persisted representation, and must not be used for any
    //! other purpose.
    //!
    //! \warning The caller owns the object returned.
    virtual CBucketGatherer* cloneForPersistence() const;

    //! The persistence tag name of this derived class.
    virtual const std::string& persistenceTag() const;
    //@}

    //! \name Fields
    //@{
    //! Get the person field name.
    //!
    //! This is the common field in all searches "along" which the
    //! probabilities are aggregated, i.e. the "over" field name for
    //! population searches and the "by" field name for individual
    //! searches.
    virtual const std::string& personFieldName() const;

    //! Get the attribute field name if one exists, i.e. the "by" for
    //! population searches, field name and returns empty otherwise.
    virtual const std::string& attributeFieldName() const;

    //! Returns an empty string.
    virtual const std::string& valueFieldName() const;

    //! Get an iterator at the beginning the influencing field names.
    virtual TStrVecCItr beginInfluencers() const;

    //! Get an iterator at the end of the influencing field names.
    virtual TStrVecCItr endInfluencers() const;

    //! Get the fields for which to gather data.
    //!
    //! For individual searches this gets the field which defines the
    //! categories whose counts are being analyzed. For population
    //! searches this gets the fields identifying the people and person
    //! attributes which are being analyzed. An empty string acts like
    //! a wild card and matches all records. This is used for analysis
    //! which is attribute independent such as total count.
    virtual const TStrVec& fieldsOfInterest() const;
    //@}

    //! Get a description of the search.
    virtual std::string description() const;

    //! \name Update
    //@{
    //! Process the specified fields.
    //!
    //! \note For individual searches \p fieldValues should contain one
    //! field containing the by clause field value or a generic name if
    //! none was specified. For population searches \p fieldValues should
    //! contain two fields. The first field should contain the over clause
    //! field value. The second field should the by clause field value
    //! or a generic name if none was specified.
    virtual bool processFields(const TStrCPtrVec& fieldValues,
                               CEventData& result,
                               CResourceMonitor& resourceMonitor);
    //@}

    //! \name Person
    //@{
    //! Stop gathering data on the people identified by \p peopleToRemove.
    virtual void recyclePeople(const TSizeVec& peopleToRemove);

    //! Remove all traces of people whose identifiers are greater than
    //! or equal to \p lowestPersonToRemove.
    virtual void removePeople(std::size_t lowestPersonToRemove);
    //@}

    //! \name Attribute
    //@{
    //! Stop gathering data on the attributes identified by \p attributesToRemove.
    virtual void recycleAttributes(const TSizeVec& attributesToRemove);

    //! Remove all traces of attributes whose identifiers are greater than
    //! or equal to \p lowestAttributeToRemove.
    virtual void removeAttributes(std::size_t lowestAttributeToRemove);
    //@}

    //! Get the checksum of this gatherer.
    virtual uint64_t checksum() const;

    //! Get the memory used by this object.
    virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the memory used by this object.
    virtual std::size_t memoryUsage() const;

    //! Get the static size of this object - used for virtual hierarchies
    virtual std::size_t staticSize() const;

    //! Clear this data gatherer.
    virtual void clear();

    //! Reset bucket and return true if bucket was successfully reset or false otherwise.
    virtual bool resetBucket(core_t::TTime bucketStart);

    //! Release memory that is no longer needed
    virtual void releaseMemory(core_t::TTime samplingCutoffTime);

    //! \name Features
    //@{
    //! Get the raw data for all features for the bucketing time interval
    //! containing \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[out] result Filled in with the feature data at \p time.
    virtual void featureData(core_t::TTime time,
                             core_t::TTime bucketLength,
                             TFeatureAnyPrVec& result) const;
    //@}

private:
    //! No-op.
    virtual void sample(core_t::TTime time);

    //! Append the counts by person for the bucketing interval containing
    //! \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[in,out] result Append (person identifier, count) for each
    //! person. The collection is sorted by person.
    void personCounts(model_t::EFeature feature, core_t::TTime time, TFeatureAnyPrVec& result) const;

    //! Append the non-zero counts by person for bucketing interval
    //! containing \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[in,out] result Append (person identifier, count) for each
    //! person present in the bucketing interval containing \p time. The
    //! collection is sorted by person.
    void nonZeroPersonCounts(model_t::EFeature feature,
                             core_t::TTime time,
                             TFeatureAnyPrVec& result) const;

    //! Append an indicator function for people present in the bucketing
    //! interval containing \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[in,out] result Append (person identifier, 1) for each person
    //! present in the bucketing interval containing \p time. The collection
    //! is sorted by person identifier.
    void personIndicator(model_t::EFeature feature, core_t::TTime time, TFeatureAnyPrVec& result) const;

    //! Append the mean arrival times for people present in the current
    //! bucketing interval.
    //!
    //! \param[in] time The time of interest.
    //! \param[in,out] result Append (person identifier, mean arrival time)
    //! for each person present in the bucketing interval containing \p time.
    //! The collection is sorted by person identifier.
    void personArrivalTimes(model_t::EFeature feature,
                            core_t::TTime time,
                            TFeatureAnyPrVec& result) const;

    //! Append the non-zero counts for each attribute by person for the
    //! bucketing interval containing \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[in,out] result Append the non-zero attribute counts by
    //! person. The first element of the key is person and the second
    //! attribute. The collection is sorted lexicographically by key.
    //! \note We expect the pairs present to be sparse on the full outer
    //! product space of attribute and person so use a sparse encoding.
    void nonZeroAttributeCounts(model_t::EFeature feature,
                                core_t::TTime time,
                                TFeatureAnyPrVec& result) const;

    //! Append the number of unique people hitting each attribute.
    //!
    //! \param[in,out] result Append the count of people per attribute.
    //! The person identifier is dummied to zero so that the result
    //! type matches other population features.
    void peoplePerAttribute(model_t::EFeature feature, TFeatureAnyPrVec& result) const;

    //! Append an indicator function for (person, attribute) pairs
    //! present in the bucketing interval containing \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[in,out] result Append one for each (person, attribute)
    //! pair present in the bucketing interval containing \p time. The
    //! first element of the key is person and the second attribute. The
    //! collection is sorted lexicographically by key.
    //! \note We expect the pairs present to be sparse on the full outer
    //! product space of attribute and person so use a sparse encoding.
    void attributeIndicator(model_t::EFeature feature,
                            core_t::TTime time,
                            TFeatureAnyPrVec& result) const;

    //! Append the number of unique values for each person
    //! in the bucketing interval containing \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[out] result Filled in with the unique value counts
    //! by person
    void bucketUniqueValuesPerPerson(model_t::EFeature feature,
                                     core_t::TTime time,
                                     TFeatureAnyPrVec& result) const;

    //! Append the number of unique values for each person and attribute
    //! in the bucketing interval containing \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[out] result Filled in with the unique value counts
    //! by person and attribute
    void bucketUniqueValuesPerPersonAttribute(model_t::EFeature feature,
                                              core_t::TTime time,
                                              TFeatureAnyPrVec& result) const;

    //! Append the compressed length of the unique attributes each person
    //! hits in the bucketing interval containing \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[out] result Filled in with the compressed length of the
    //! unique values by person and attribute
    void bucketCompressedLengthPerPerson(model_t::EFeature feature,
                                         core_t::TTime time,
                                         TFeatureAnyPrVec& result) const;

    //! Append the compressed length of the unique attributes each person
    //! hits in the bucketing interval containing \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[out] result Filled in with the compressed length of the
    //! unique values by person and attribute
    void bucketCompressedLengthPerPersonAttribute(model_t::EFeature feature,
                                                  core_t::TTime time,
                                                  TFeatureAnyPrVec& result) const;

    //! Append the time-of-day/week values for each person in the
    //! bucketing interval \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[out] result Filled in with the arrival time values
    //! by person.
    void bucketMeanTimesPerPerson(model_t::EFeature feature,
                                  core_t::TTime time,
                                  TFeatureAnyPrVec& result) const;

    //! Append the time-of-day/week values of each attribute and person
    //! in the bucketing interval \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[out] result Filled in with the arrival time values
    //! by attribute and person
    void bucketMeanTimesPerPersonAttribute(model_t::EFeature feature,
                                           core_t::TTime time,
                                           TFeatureAnyPrVec& result) const;

    //! Resize the necessary data structures so they can accommodate
    //! the person and attribute identified by \p pid and \p cid,
    //! respectively.
    //!
    //! \param[in] pid The identifier of the person to accommodate.
    //! \param[in] cid The identifier of the attribute to accommodate.
    virtual void resize(std::size_t pid, std::size_t cid);

    //! Record the arrival of the \p values for the person identified
    //! by \p pid.
    //!
    //! \param[in] pid The identifier of the person who generated the
    //! record(s).
    //! \param[in] cid The identifier of the attribute who generated
    //! the record(s).
    //! \param[in] time The end time of the record(s).
    //! \param[in] values Ignored.
    //! \param[in] count The number of records.
    //! \param[in] stringValue The value for the function string argument
    //! if there is one or null.
    //! \param[in] influences The influencing field values which label
    //! the value.
    virtual void addValue(std::size_t pid,
                          std::size_t cid,
                          core_t::TTime time,
                          const CEventData::TDouble1VecArray& values,
                          std::size_t count,
                          const CEventData::TOptionalStr& stringValue,
                          const TStoredStringPtrVec& influences);

    //! Start a new bucket.
    virtual void startNewBucket(core_t::TTime time, bool skipUpdates);

    //! Initialize the field names collection.
    void initializeFieldNames(const std::string& personFieldName,
                              const std::string& attributeFieldName,
                              const std::string& valueFieldName,
                              const std::string& summaryCountFieldName,
                              const TStrVec& influenceFieldNames);

    //! Initialize the feature data gatherers.
    void initializeFeatureData();

    //! Copy the influencer person counts to \p results.
    //!
    //! \warning This assumes that \p result is sorted by person
    //! identifier.
    void addInfluencerCounts(core_t::TTime time, TSizeFeatureDataPrVec& result) const;

    //! Copy the influencer person and attribute counts to \p results.
    //!
    //! \warning This assumes that \p result is sorted by person
    //! and attribute identifier.
    void addInfluencerCounts(core_t::TTime time, TSizeSizePrFeatureDataPrVec& result) const;

private:
    //! The name of the field value of interest for keyed functions
    std::string m_ValueFieldName;

    //! The names of the fields  of interest.
    //!
    //! This is of the form:
    //!   -# The name of the field which identifies people,
    //!   -# [The name of the field which identifies people's attributes],
    //!   -# [The names of the influencing fields],
    //!   -# [The name of the field which identifies a function to key off],
    //!   -# [The name of the field containing the person(/attribute) count
    //!       if summarized data are being gathered]
    TStrVec m_FieldNames;

    //! The position of the first influencer field
    std::size_t m_BeginInfluencingFields;

    //! The position of the first count/value field.
    std::size_t m_BeginValueField;

    //! The position of the field holding the summarised count.
    std::size_t m_BeginSummaryFields;

    //! The data features we are gathering.
    TCategoryAnyMap m_FeatureData;
};
}
}

namespace std {

//! Overload pair swap so that we use efficient swap of the feature data
//! when sorting.
inline void swap(ml::model::CEventRateBucketGatherer::TSizeFeatureDataPr& lhs,
                 ml::model::CEventRateBucketGatherer::TSizeFeatureDataPr& rhs) {
    swap(lhs.first, rhs.first);
    lhs.second.swap(rhs.second);
}

//! Overload pair swap so that we use efficient swap of the feature data
//! when sorting.
inline void swap(ml::model::CEventRateBucketGatherer::TSizeSizePrFeatureDataPr& lhs,
                 ml::model::CEventRateBucketGatherer::TSizeSizePrFeatureDataPr& rhs) {
    swap(lhs.first, rhs.first);
    lhs.second.swap(rhs.second);
}
}

#endif // INCLUDED_ml_model_CEventRateBucketGatherer_h
