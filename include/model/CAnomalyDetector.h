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
#ifndef INCLUDED_ml_model_CAnomalyDetector_h
#define INCLUDED_ml_model_CAnomalyDetector_h

#include <core/CNonCopyable.h>
#include <core/CSmallVector.h>
#include <core/CoreTypes.h>

#include <model/CAnomalyDetectorModel.h>
#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CEventData.h>
#include <model/CForecastDataSink.h>
#include <model/CHierarchicalResults.h>
#include <model/CLimits.h>
#include <model/CModelFactory.h>
#include <model/CModelPlotData.h>
#include <model/FunctionTypes.h>
#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <boost/shared_ptr.hpp>

#include <functional>
#include <map>
#include <string>
#include <vector>

#include <stdint.h>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace model {
class CDataGatherer;
class CModel;
class CSearchKey;

//! \brief
//! Interface for detecting and reporting anomalies in different
//! types of unstructured data.
//!
//! DESCRIPTION:\n
//! Given a stream of data categorised by a particular field name,
//! this reports on anomalies in that data.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Data must be received in increasing time order.
//!
//! If the field values mean that more than a configurable amount
//! of memory is consumed by the models, for example if the number
//! of by field values is too high such as using "by _cd". (The
//! limit is controlled by mllimits.conf.)
//!
//! The different methods used for anomaly detection are largely
//! encapsulated by the CModel class hierarchy.  This means it is
//! possible to implement the function to output anomalies in terms
//! of that interface.
//!
//! We use the 'person' terminology throughout because we can
//! choose to analyse certain field values either individually or as
//! a population.

class MODEL_EXPORT CAnomalyDetector : private core::CNonCopyable {
public:
    using TStrVec = std::vector<std::string>;
    using TStrCPtrVec = std::vector<const std::string*>;
    using TModelPlotDataVec = std::vector<CModelPlotData>;

    using TDataGathererPtr = boost::shared_ptr<CDataGatherer>;
    using TModelFactoryCPtr = boost::shared_ptr<const CModelFactory>;
    using TModelPtr = boost::shared_ptr<CAnomalyDetectorModel>;

    //! A shared pointer to an instance of this class
    using TAnomalyDetectorPtr = boost::shared_ptr<CAnomalyDetector>;

    using TOutputModelPlotDataFunc =
        std::function<void(const std::string&, const std::string&, const std::string&, const std::string&, const CModelPlotData&)>;
    using TStrSet = CAnomalyDetectorModelConfig::TStrSet;

public:
    //! State version.  This must be incremented every time a change to the
    //! state is made that requires existing state to be discarded
    static const std::string STATE_VERSION;

    //! Name of the count field
    static const std::string COUNT_NAME;

    //! Name of a time field (for the GUI to make a distinction between a counter and a time)
    static const std::string TIME_NAME;

    //! Indicator that the GUI should expect a field name but no field value
    //! (because for a distinct count we're only interested in the number of
    //! different values, not the values themselves)
    static const std::string DISTINCT_COUNT_NAME;

    //! Indicator that the GUI should use a description template based on
    //! rare events rather than numerous events
    static const std::string RARE_NAME;

    //! Indicator that the GUI should use a description template based on
    //! information content of events
    static const std::string INFO_CONTENT_NAME;

    //! Output function names for metric anomalies
    static const std::string MEAN_NAME;
    static const std::string MEDIAN_NAME;
    static const std::string MIN_NAME;
    static const std::string MAX_NAME;
    static const std::string VARIANCE_NAME;
    static const std::string SUM_NAME;
    static const std::string LAT_LONG_NAME;
    static const std::string EMPTY_STRING;

public:
    CAnomalyDetector(int detectorIndex,
                     CLimits& limits,
                     const CAnomalyDetectorModelConfig& modelConfig,
                     const std::string& partitionFieldValue,
                     core_t::TTime firstTime,
                     const TModelFactoryCPtr& modelFactory);

    //! Create a copy that will result in the same persisted state as the
    //! original.  This is effectively a copy constructor that creates a
    //! copy that's only valid for a single purpose.  The boolean flag is
    //! redundant except to create a signature that will not be mistaken for
    //! a general purpose copy constructor.
    CAnomalyDetector(bool isForPersistence, const CAnomalyDetector& other);

    virtual ~CAnomalyDetector();

    //! Get the total number of people which this is modeling.
    size_t numberActivePeople() const;

    //! Get the total number of attributes which this is modeling.
    size_t numberActiveAttributes() const;

    //! Get the maximum size of all the member containers.
    size_t maxDimension() const;

    //! For the operationalised version of the product, we may create models
    //! that need to reflect the fact that no data of a particular type was
    //! seen for a period before the creation of the models, but WITHOUT
    //! reporting any results for the majority of that period.  This method
    //! provides that facility.
    void zeroModelsToTime(core_t::TTime time);

    //! Populate the object from a state document
    bool acceptRestoreTraverser(const std::string& partitionFieldValue,
                                core::CStateRestoreTraverser& traverser);

    //! Restore state for statics - this is only called from the
    //! simple count detector to ensure singleton behaviour
    bool staticsAcceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Find the partition field value given part of an state document.
    //!
    //! \note This is static so it can be called before the state is fully
    //! deserialised, because we need this value before to restoring the
    //! detector.
    static bool partitionFieldAcceptRestoreTraverser(core::CStateRestoreTraverser& traverser,
                                                     std::string& partitionFieldValue);

    //! Find the detector keys given part of an state document.
    //!
    //! \note This is static so it can be called before the state is fully
    //! deserialised, because we need these before to restoring the detector.
    static bool keyAcceptRestoreTraverser(core::CStateRestoreTraverser& traverser,
                                          CSearchKey& key);

    //! Persist the detector keys separately to the rest of the state.
    //! This must be done for a 100% streaming state restoration because
    //! the key must be known before a detector object is created into
    //! which other state can be restored.
    void keyAcceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Persist the partition field separately to the rest of the state.
    //! This must be done for a 100% streaming state restoration because
    //! the partition field must be known before a detector object is
    //! created into which other state can be restored.
    void partitionFieldAcceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Persist state for statics - this is only called from the
    //! simple count detector to ensure singleton behaviour
    void staticsAcceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Persist state by passing information to the supplied inserter
    //!
    //! \note Some information is duplicated in keyAcceptPersistInserter()
    //! and partitionFieldAcceptPersistInserter() due to historical reasons.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Get the cue for this detector.  This consists of the search key cue
    //! with the partition field value appended.
    std::string toCue() const;

    //! Debug representation.  Note that operator<<() is more efficient than
    //! generating this debug string and immediately outputting it to a
    //! stream.
    std::string debug() const;

    //! Check if this is a simple count detector.
    virtual bool isSimpleCount() const;

    //! Get the fields to extract from a record for processing by this detector.
    const TStrVec& fieldsOfInterest() const;

    //! Extract and add the necessary details of an event record.
    void addRecord(core_t::TTime time, const TStrCPtrVec& fieldValues);

    //! Update the results with this detector model's results.
    void buildResults(core_t::TTime bucketStartTime,
                      core_t::TTime bucketEndTime,
                      CHierarchicalResults& results);

    //! Update the results with this detector model's results.
    void buildInterimResults(core_t::TTime bucketStartTime,
                             core_t::TTime bucketEndTime,
                             CHierarchicalResults& results);

    //! Generate the model plot data for the time series identified
    //! by \p terms.
    void generateModelPlot(core_t::TTime bucketStartTime,
                           core_t::TTime bucketEndTime,
                           double boundsPercentile,
                           const TStrSet& terms,
                           TModelPlotDataVec& modelPlots) const;

    //! Generate ForecastPrerequistes, e.g. memory requirements
    CForecastDataSink::SForecastModelPrerequisites getForecastPrerequisites() const;

    //! Generate maths models for forecasting
    CForecastDataSink::SForecastResultSeries getForecastModels() const;

    //! Remove dead models, i.e. those models that have more-or-less
    //! reverted back to their non-informative state.  BE CAREFUL WHEN
    //! CALLING THIS METHOD that you do not hold pointers to any models
    //! that may be deleted as a result of this call.
    virtual void pruneModels();

    //! Reset bucket.
    void resetBucket(core_t::TTime bucketStart);

    //! Release memory that is no longer needed
    void releaseMemory(core_t::TTime samplingCutoffTime);

    //! Print the detector memory usage to the given stream
    void showMemoryUsage(std::ostream& stream) const;

    //! Get the memory used by this detector
    void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Return the total memory usage
    std::size_t memoryUsage() const;

    //! Get end of the last complete bucket we've observed.
    const core_t::TTime& lastBucketEndTime() const;

    //! Get writable end of the last complete bucket we've observed.
    core_t::TTime& lastBucketEndTime();

    //! Access to the bucket length being used in the current models.  This
    //! can be used to detect discrepancies between the model config and
    //! existing models.
    core_t::TTime modelBucketLength() const;

    //! Get a description of this anomaly detector.
    std::string description() const;

    //! Roll time forwards to \p time.
    void timeNow(core_t::TTime time);

    //! Rolls time to \p endTime while skipping sampling the models for buckets within the gap
    //! \param[in] endTime The end of the time interval to skip sampling.
    void skipSampling(core_t::TTime endTime);

    const TModelPtr& model() const;
    TModelPtr& model();

protected:
    //! This function is called before adding a record allowing
    //! for varied preprocessing.
    virtual const TStrCPtrVec& preprocessFieldValues(const TStrCPtrVec& fieldValues);

    //! Initializes simple counting by adding a person called "count".
    void initSimpleCounting();

private:
    // Shared code for building results
    template<typename SAMPLE_FUNC, typename LAST_SAMPLED_BUCKET_UPDATE_FUNC>
    void buildResultsHelper(core_t::TTime bucketStartTime,
                            core_t::TTime bucketEndTime,
                            SAMPLE_FUNC sampleFunc,
                            LAST_SAMPLED_BUCKET_UPDATE_FUNC lastSampledBucketUpdateFunc,
                            CHierarchicalResults& results);

    //! Updates the last sampled bucket
    void updateLastSampledBucket(core_t::TTime bucketEndTime);

    //! Does not update the last sampled bucket. To be used
    //! when interim results are calculated.
    void noUpdateLastSampledBucket(core_t::TTime bucketEndTime) const;

    //! Sample the model in the interval [\p startTime, \p endTime].
    void sample(core_t::TTime startTime, core_t::TTime endTime, CResourceMonitor& resourceMonitor);

    //! Sample bucket statistics and any other state needed to compute
    //! probabilities in the interval [\p startTime, \p endTime], but
    //! does not update the model.
    void sampleBucketStatistics(core_t::TTime startTime,
                                core_t::TTime endTime,
                                CResourceMonitor& resourceMonitor);

    //! Restores the state that was formerly part of the model ensemble class.
    //! This includes the data gatherer and the model.
    bool legacyModelEnsembleAcceptRestoreTraverser(const std::string& partitionFieldValue,
                                                   core::CStateRestoreTraverser& traverser);

    //! Restores the state that was formerly part of the live models
    //! in the model ensemble class.
    bool legacyModelsAcceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Persists the state that was formerly part of the model ensemble class.
    //! This includes the data gatherer and the model.
    void legacyModelEnsembleAcceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Persists the state that was formerly part of the live models
    //! in the model ensemble class.
    void legacyModelsAcceptPersistInserter(core::CStatePersistInserter& inserter) const;

protected:
    //! Configurable limits
    CLimits& m_Limits;

private:
    //! An identifier for the search for which this is detecting anomalies.
    int m_DetectorIndex;

    //! Configurable behaviour
    const CAnomalyDetectorModelConfig& m_ModelConfig;

    //! The end of the last complete bucket we've observed.  This is an OPEN
    //! endpoint, i.e. this time is the lowest time NOT in the last bucket.
    core_t::TTime m_LastBucketEndTime;

    //! The data gatherers.
    TDataGathererPtr m_DataGatherer;

    //! The factory for new data gatherers and models.
    TModelFactoryCPtr m_ModelFactory;

    // The model of the data in which we are detecting anomalies.
    TModelPtr m_Model;

    //! Is this a cloned detector containing the bare minimum information
    //! necessary to create a valid persisted state?
    bool m_IsForPersistence;

    friend MODEL_EXPORT std::ostream& operator<<(std::ostream&, const CAnomalyDetector&);
};

MODEL_EXPORT
std::ostream& operator<<(std::ostream& strm, const CAnomalyDetector& detector);
}
}

#endif // INCLUDED_ml_model_CAnomalyDetector_h
