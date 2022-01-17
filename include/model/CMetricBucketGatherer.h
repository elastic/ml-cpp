/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#ifndef INCLUDED_ml_model_CMetricBucketGatherer_h
#define INCLUDED_ml_model_CMetricBucketGatherer_h

#include <core/CMemory.h>
#include <core/CoreTypes.h>

#include <maths/common/CBasicStatistics.h>

#include <model/CDataGatherer.h>
#include <model/ImportExport.h>

#include <boost/any.hpp>

#include <map>
#include <string>
#include <vector>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace model {
class CDataGatherer;
class CResourceMonitor;

//! \brief Metric series data gathering class.
//!
//! DESCRIPTION:\n
//! This performs all pre-processing of the data which we model in order
//! to characterize metric time series.
//!
//! \sa CDataGatherer.
class MODEL_EXPORT CMetricBucketGatherer final : public CBucketGatherer {
public:
    using TCategorySizePr = std::pair<model_t::EMetricCategory, std::size_t>;
    using TCategorySizePrAnyMap = std::map<TCategorySizePr, boost::any>;
    using TCategorySizePrAnyMapItr = TCategorySizePrAnyMap::iterator;
    using TCategorySizePrAnyMapCItr = TCategorySizePrAnyMap::const_iterator;

public:
    //! \name Life-cycle
    //@{
    //! Create a new population metric data gatherer.
    //!
    //! \param[in] dataGatherer The owning data gatherer.
    //! \param[in] summaryCountFieldName If \p summaryMode is E_Manual
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
    CMetricBucketGatherer(CDataGatherer& dataGatherer,
                          const std::string& summaryCountFieldName,
                          const std::string& personFieldName,
                          const std::string& attributeFieldName,
                          const std::string& valueFieldName,
                          const TStrVec& influenceFieldNames,
                          core_t::TTime startTime);

    //! Construct from a state document.
    CMetricBucketGatherer(CDataGatherer& dataGatherer,
                          const std::string& summaryCountFieldName,
                          const std::string& personFieldName,
                          const std::string& attributeFieldName,
                          const std::string& valueFieldName,
                          const TStrVec& influenceFieldNames,
                          core::CStateRestoreTraverser& traverser);

    //! Create a copy that will result in the same persisted state as the
    //! original.  This is effectively a copy constructor that creates a
    //! copy that's only valid for a single purpose.  The boolean flag is
    //! redundant except to create a signature that will not be mistaken for
    //! a general purpose copy constructor.
    CMetricBucketGatherer(bool isForPersistence, const CMetricBucketGatherer& other);
    //@}

    //! \name Persistence
    //@{
    //! Persist state by passing information to the supplied inserter
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const override;

    //! Fill in the state from \p traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Create a clone of this data gatherer that will result in the same
    //! persisted state.  The clone may be incomplete in ways that do not
    //! affect the persisted representation, and must not be used for any
    //! other purpose.
    //! \warning The caller owns the object returned.
    CBucketGatherer* cloneForPersistence() const override;

    //! The persistence tag name of this derived class.
    const std::string& persistenceTag() const override;

private:
    //! Internal restore function.
    bool acceptRestoreTraverserInternal(core::CStateRestoreTraverser& traverser,
                                        bool isCurrentVersion);
    //@}

public:
    //! \name Fields
    //@{
    //! Get the person field name.
    //!
    //! This is the common field in all searches "along" which the
    //! probabilities are aggregated, i.e. the "over" field name for
    //! population searches and the "by" field name for individual
    //! searches.
    const std::string& personFieldName() const override;

    //! Get the attribute field name if one exists, i.e. the "by" for
    //! population searches, field name and returns empty otherwise.
    const std::string& attributeFieldName() const override;

    //! Returns an empty string.
    const std::string& valueFieldName() const override;

    //! Get an iterator at the beginning the influencing field names.
    TStrVecCItr beginInfluencers() const override;

    //! Get an iterator at the end of the influencing field names.
    TStrVecCItr endInfluencers() const override;

    //! Get the fields for which to gather data.
    //!
    //! For individual searches this gets the field which defines the
    //! categories whose counts are being analyzed. For population
    //! searches this gets the fields identifying the people and person
    //! attributes which are being analyzed. An empty string acts like
    //! a wild card and matches all records. This is used for analysis
    //! which is attribute independent such as total count.
    const TStrVec& fieldsOfInterest() const override;
    //@}

    //! Get a description of the search.
    std::string description() const override;

    //! \name Update
    //@{
    //! Process the specified fields.
    //!
    //! \note For individual searches \p fieldValues should contain two
    //! fields. The first field should contain the by clause field value
    //! or a generic name if none was specified. The second field should
    //! contain a number corresponding to the metric value. For population
    //! searches \p fieldValues should contain three fields. The first
    //! field should contain the over clause field value. The second field
    //! should the by clause field value or a generic name if none was
    //! specified. The third field should contain a number corresponding
    //! to the metric value.
    bool processFields(const TStrCPtrVec& fieldValues,
                       CEventData& result,
                       CResourceMonitor& resourceMonitor) override;
    //@}

    //! \name Person
    //@{
    //! Stop gathering data on the people identified by \p peopleToRemove.
    void recyclePeople(const TSizeVec& peopleToRemove) override;

    //! Remove all traces of people whose identifiers are greater than
    //! or equal to \p lowestPersonToRemove.
    void removePeople(std::size_t lowestPersonToRemove) override;
    //@}

    //! \name Attribute
    //@{
    //! Stop gathering data on the attributes identified by \p attributesToRemove.
    void recycleAttributes(const TSizeVec& attributesToRemove) override;

    //! Remove all traces of attributes whose identifiers are greater
    //! than or equal to \p lowestAttributeToRemove.
    void removeAttributes(std::size_t lowestAttributeToRemove) override;
    //@}

    //! Get the checksum of this gatherer.
    uint64_t checksum() const override;

    //! Debug the memory used by this object.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const override;

    //! Get the memory used by this object.
    std::size_t memoryUsage() const override;

    //! Get the static size of this object - used for virtual hierarchies
    std::size_t staticSize() const override;

    //! Clear this data gatherer.
    void clear() override;

    //! Reset bucket and return true if bucket was successfully reset or false otherwise.
    bool resetBucket(core_t::TTime bucketStart) override;

    //! Release memory that is no longer needed
    void releaseMemory(core_t::TTime samplingCutoffTime) override;

    //! \name Features
    //@{
    //! Get the raw data for all features for the bucketing time interval
    //! containing \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[out] result Filled in with the feature data at \p time.
    void featureData(core_t::TTime time,
                     core_t::TTime bucketLength,
                     TFeatureAnyPrVec& result) const override;
    //@}

private:
    //! Create samples if possible for the bucket pointed out by \p time.
    void sample(core_t::TTime time) override;

    //! Resize the necessary data structures so they can accommodate
    //! the person and attribute identified by \p pid and \p cid,
    //! respectively.
    //!
    //! \param[in] pid The identifier of the person to accommodate.
    //! \param[in] cid The identifier of the attribute to accommodate.
    void resize(std::size_t pid, std::size_t cid) override;

    //! Record the arrival of \p values for attribute identified by
    //! \p cid and person identified by \p pid.
    //!
    //! \param[in] pid The identifier of the person who generated
    //! the value.
    //! \param[in] cid The identifier of the value's attribute.
    //! \param[in] time The time of the \p values.
    //! \param[in] values The metric statistic value(s)
    //! \param[in] count The number of measurements in the metric
    //! statistic.
    //! \param[in] stringValue Ignored.
    //! \param[in] influences The influencing field values which
    //! label the value.
    void addValue(std::size_t pid,
                  std::size_t cid,
                  core_t::TTime time,
                  const CEventData::TDouble1VecArray& values,
                  std::size_t count,
                  const CEventData::TOptionalStr& stringValue,
                  const TStoredStringPtrVec& influences) override;

    //! Start a new bucket.
    void startNewBucket(core_t::TTime time, bool skipUpdates) override;

    //! Initialize the field names collection.
    //! initializeFieldNamesPart2() must be called after this.
    //! In the event that the data gatherer is being restored from persisted
    //! state, the sequence must be:
    //! 1) initializeFieldNamesPart1()
    //! 2) restore state
    //! 3) initializeFieldNamesPart2()
    void initializeFieldNamesPart1(const std::string& personFieldName,
                                   const std::string& attributeFieldName,
                                   const TStrVec& influenceFieldNames);

    //! Initialize the field names collection.
    //! initializeFieldNamesPart1() must be called before this.
    //! In the event that the data gatherer is being restored from persisted
    //! state, the sequence must be:
    //! 1) initializeFieldNamesPart1()
    //! 2) restore state
    //! 3) initializeFieldNamesPart2()
    void initializeFieldNamesPart2(const std::string& valueFieldName,
                                   const std::string& summaryCountFieldName);

    //! Initialize the feature data gatherers.
    void initializeFeatureData();

private:
    //! The metric value field name.  This is held separately to
    //! m_FieldNames because in the case of summarization the field
    //! names holding the summarized values will be mangled.
    std::string m_ValueFieldName;

    //! The names of the fields  of interest.
    //!
    //! The entries in order are:
    //!   -# The name of the field which identifies people,
    //!   -# For population models only, the name of the field which
    //!      identifies people's attributes,
    //!   -# The name of zero or more influencing fields,
    //!   -# The name of the field holding the count followed by the
    //!      field name(s) of the field(s) which hold the statistics
    //!      themselves, which must (for those that are present) be
    //!      ordered mean, min, max, sum.
    //!   -# For the API with user defined pre-summarisation, the name
    //!      of the field which holds the count then the name of the field
    //!      which holds the statistic value,
    //!   -# Otherwise the name of the field which holds the metric value.
    TStrVec m_FieldNames;

    //! The position of the first influencing field.
    std::size_t m_BeginInfluencingFields;

    //! The position of the first count/value field.
    std::size_t m_BeginValueFields;

    //! For summarized values, this stores the metric categories
    //! corresponding to the summarized field names in m_FieldNames;
    //! for non-summarized input this will be empty
    TMetricCategoryVec m_FieldMetricCategories;

    //! The data features we are gathering.
    TCategorySizePrAnyMap m_FeatureData;
};
}
}

#endif // INCLUDED_ml_model_CMetricBucketGatherer_h
