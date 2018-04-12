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

#ifndef INCLUDED_ml_model_CAnomalyDetectorModel_h
#define INCLUDED_ml_model_CAnomalyDetectorModel_h

#include <core/CMemory.h>
#include <core/CNonCopyable.h>
#include <core/CSmallVector.h>
#include <core/CStatistics.h>
#include <core/CoreTypes.h>

#include <maths/CTimeSeriesModel.h>
#include <maths/MathsTypes.h>

#include <model/CMemoryUsageEstimator.h>
#include <model/CModelParams.h>
#include <model/CPartitioningFields.h>
#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <boost/optional.hpp>
#include <boost/ref.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>

#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <stdint.h>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}

namespace maths {
class CMultivariatePrior;
}

namespace model {

class CAttributeFrequencyGreaterThan;
class CInterimBucketCorrector;
class CDataGatherer;
class CHierarchicalResults;
class CModelDetailsView;
class CPersonFrequencyGreaterThan;
class CResourceMonitor;
struct SAnnotatedProbability;
struct SAttributeProbability;

//! \brief The model interface.
//!
//! DESCRIPTION:\n
//! This defines the interface common to all (statistical) models of
//! the (random) processes which describe system state. It declares
//! core functions used by the anomaly detection code to:
//!   -# Retrieve information about the categories and people of the
//!      processes being modeled.
//!   -# Sample the processes in a specified time interval and update
//!      the model.
//!   -# Manage the model life-cycle.
//!   -# Compute the probability of the samples of the process in a
//!      specified time interval.
//!
//! The raw events can be partitioned by attribute and/or person (for
//! population analysis). These are just two labels which can be
//! annotated on the events and induce equivalence relations on the
//! set of all events. The events in subsets comprise (some of)
//! the raw events for (one of) the processes we model. For example,
//! in temporal analysis we would model the history of all events
//! for which the labels are equal for each distinct value of the
//! label.
//!
//! There are three main types of analysis:
//!   -# Individual analysis.
//!   -# Population analysis.
//!   -# Peer group analysis.
//!
//! Individual analysis looks at the historical values of various
//! features on a single time series' events and detects significant
//! changes in those values. Population analysis looks at similar
//! features, but on a whole collection of processes in conjunction
//! (induced by the person label equivalence relation). Peer group
//! analysis is similar to population analysis, but assigns each
//! person to a peer group and looks for unusual behaviour w.r.t.
//! its peer group and not the population as a whole. The concrete
//! implementations of this class include more detailed descriptions.
//! This object also maintains the state to find the most correlated
//! pairs of time series.
//!
//! The extraction of the features from the raw process events is
//! managed by a separate object. These include a number of simple
//! statistics such as the count of events in a time interval, the
//! mean of a certain number of event values, the minimum of a
//! certain number of event values and so on. (See model::CDataGatherer
//! for more details.)
//!
//! The model hierarchy is also able to compare two time intervals
//! in which case a model really comprises two distinct models of the
//! underlying random process one for each time interval: see the
//! computeProbability for more details.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The model hierarchy has been abstracted to allow the code to detect
//! anomalies to be reused for different types of data, log messages,
//! metrics, etc, to perform different types of analysis on that data,
//! and to handle the case that data are continuously streamed to the
//! object or the case that two different data sets are to be compared.
//!
//! All models can be serialized to/from text representation.
//!
//! The hierarchy is non-copyable because we don't currently need to be
//! able to copy models and the "correct" copy semantics are not obvious.
class MODEL_EXPORT CAnomalyDetectorModel : private core::CNonCopyable {
    friend class CModelDetailsView;

public:
    using TSizeVec = std::vector<std::size_t>;
    using TDoubleVec = std::vector<double>;
    using TDouble1Vec = core::CSmallVector<double, 1>;
    using TDouble4Vec = core::CSmallVector<double, 4>;
    using TDouble10Vec = core::CSmallVector<double, 10>;
    using TDouble4Vec1Vec = core::CSmallVector<TDouble4Vec, 1>;
    using TDouble10Vec1Vec = core::CSmallVector<TDouble10Vec, 1>;
    using TDouble10Vec4Vec = core::CSmallVector<TDouble10Vec, 4>;
    using TDouble10Vec4Vec1Vec = core::CSmallVector<TDouble10Vec4Vec, 1>;
    using TDouble1VecDoublePr = std::pair<TDouble1Vec, double>;
    using TDouble1VecDouble1VecPr = std::pair<TDouble1Vec, TDouble1Vec>;
    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TSizeDoublePr1Vec = core::CSmallVector<TSizeDoublePr, 1>;
    using TSize1Vec = core::CSmallVector<std::size_t, 1>;
    using TSize2Vec = core::CSmallVector<std::size_t, 2>;
    using TSize2Vec1Vec = core::CSmallVector<TSize2Vec, 1>;
    using TDoubleDoublePr = std::pair<double, double>;
    using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
    using TSizeSizePr = std::pair<std::size_t, std::size_t>;
    using TStr1Vec = core::CSmallVector<std::string, 1>;
    using TOptionalDouble = boost::optional<double>;
    using TOptionalDoubleVec = std::vector<TOptionalDouble>;
    using TOptionalUInt64 = boost::optional<uint64_t>;
    using TOptionalSize = boost::optional<std::size_t>;
    using TAttributeProbability1Vec = core::CSmallVector<SAttributeProbability, 1>;
    using TInfluenceCalculatorCPtr = boost::shared_ptr<const CInfluenceCalculator>;
    using TFeatureInfluenceCalculatorCPtrPr =
        std::pair<model_t::EFeature, TInfluenceCalculatorCPtr>;
    using TFeatureInfluenceCalculatorCPtrPrVec = std::vector<TFeatureInfluenceCalculatorCPtrPr>;
    using TFeatureInfluenceCalculatorCPtrPrVecVec =
        std::vector<TFeatureInfluenceCalculatorCPtrPrVec>;
    using TMultivariatePriorPtr = boost::shared_ptr<maths::CMultivariatePrior>;
    using TFeatureMultivariatePriorPtrPr = std::pair<model_t::EFeature, TMultivariatePriorPtr>;
    using TFeatureMultivariatePriorPtrPrVec = std::vector<TFeatureMultivariatePriorPtrPr>;
    using TMathsModelPtr = boost::shared_ptr<maths::CModel>;
    using TMathsModelPtrVec = std::vector<TMathsModelPtr>;
    using TDataGathererPtr = boost::shared_ptr<CDataGatherer>;
    using TModelPtr = boost::shared_ptr<CAnomalyDetectorModel>;
    using TModelCPtr = boost::shared_ptr<const CAnomalyDetectorModel>;
    using TCorrelationsPtr = boost::shared_ptr<maths::CTimeSeriesCorrelations>;
    using CModelDetailsViewPtr = std::auto_ptr<CModelDetailsView>;

public:
    //! A value used to indicate a time variable is unset
    static const core_t::TTime TIME_UNSET;

public:
    //! \name Life-cycle.
    //@{
    //! \param[in] params The global configuration parameters.
    //! \param[in] dataGatherer The object that gathers time series data.
    //! \param[in] influenceCalculators The influence calculators to use
    //! for each feature.
    CAnomalyDetectorModel(const SModelParams& params,
                          const TDataGathererPtr& dataGatherer,
                          const TFeatureInfluenceCalculatorCPtrPrVecVec& influenceCalculators);

    //! Create a copy that will result in the same persisted state as the
    //! original.  This is effectively a copy constructor that creates a
    //! copy that's only valid for a single purpose.  The boolean flag is
    //! redundant except to create a signature that will not be mistaken for
    //! a general purpose copy constructor.
    CAnomalyDetectorModel(bool isForPersistence, const CAnomalyDetectorModel& other);

    virtual ~CAnomalyDetectorModel() = default;
    //@}

    //! Get a human understandable description of the model for debugging.
    std::string description() const;

    //! \name Persistence
    //@{
    //! Persist state by passing information to the supplied inserter.
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const = 0;

    //! Restore the model reading state from the supplied traverser.
    virtual bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) = 0;

    //! Create a clone of this model that will result in the same persisted
    //! state.  The clone may be incomplete in ways that do not affect the
    //! persisted representation, and must not be used for any other
    //! purpose.
    //! \warning The caller owns the object returned.
    virtual CAnomalyDetectorModel* cloneForPersistence() const = 0;
    //@}

    //! Get the model category.
    virtual model_t::EModelType category() const = 0;

    //! True if this is a population model.
    virtual bool isPopulation() const = 0;

    //! Check if this is an event rate model.
    virtual bool isEventRate() const = 0;

    //! Check if this is a metric model.
    virtual bool isMetric() const = 0;

    //! \name Bucket Statistics
    //!@{
    //! Get the count of the bucketing interval containing \p time
    //! for the person identified by \p pid.
    //!
    //! \param[in] pid The identifier of the person of interest.
    //! \param[in] time The time of interest.
    //! \return The count in the bucketing interval at \p time for the
    //! person identified by \p pid if available and null otherwise.
    virtual TOptionalUInt64 currentBucketCount(std::size_t pid, core_t::TTime time) const = 0;

    //! Get the mean count of the person identified by \p pid in the
    //! reference data set (for comparison).
    //!
    //! \param[in] pid The identifier of the person of interest.
    virtual TOptionalDouble baselineBucketCount(std::size_t pid) const = 0;

    //! Get the bucket value of \p feature for the person identified
    //! by \p pid and the attribute identified by \p cid in the
    //! bucketing interval including \p time.
    //!
    //! \param[in] feature The feature of interest.
    //! \param[in] pid The identifier of the person of interest.
    //! \param[in] cid The identifier of the attribute of interest.
    //! \param[in] time The time of interest.
    //! \return The value of \p feature in the bucket containing
    //! \p time if available and empty otherwise.
    virtual TDouble1Vec currentBucketValue(model_t::EFeature feature,
                                           std::size_t pid,
                                           std::size_t cid,
                                           core_t::TTime time) const = 0;

    //! Get the appropriate baseline bucket value of \p feature for
    //! the person identified by \p pid and the attribute identified
    //! by \p cid as of the start of the current bucketing interval.
    //! This has subtly different meanings dependent on the model.
    //!
    //! \param[in] feature The feature of interest.
    //! \param[in] pid The identifier of the person of interest.
    //! \param[in] cid The identifier of the attribute of interest.
    //! \param[in] type A description of the type of result for which
    //! to get the baseline. See CResultType for more details.
    //! \param[in] correlated The correlated series' identifiers and
    //! their values if any.
    //! \param[in] time The time of interest.
    //! \return The baseline mean value of \p feature if available
    //! and empty otherwise.
    virtual TDouble1Vec baselineBucketMean(model_t::EFeature feature,
                                           std::size_t pid,
                                           std::size_t cid,
                                           model_t::CResultType type,
                                           const TSizeDoublePr1Vec& correlated,
                                           core_t::TTime time) const = 0;

    //! Check if bucket statistics are available for the specified time.
    virtual bool bucketStatsAvailable(core_t::TTime time) const = 0;
    //@}

    //! \name Person
    //@{
    //! Get the name of the person identified by \p pid. This returns
    //! a default fallback string if the person doesn't exist.
    const std::string& personName(std::size_t pid) const;

    //! As above but with a specified fallback.
    const std::string& personName(std::size_t pid, const std::string& fallback) const;

    //! Print the people identified by \p pids.
    //! Optionally, this may be limited to return a string of the form:
    //! A B C and n others
    std::string printPeople(const TSizeVec& pids,
                            size_t limit = std::numeric_limits<size_t>::max()) const;

    //! Get the person unique identifiers which have a feature value
    //! in the bucketing time interval including \p time.
    //!
    //! \param[in] time The time of interest.
    //! \param[out] result Filled in with the person identifiers
    //! in the bucketing time interval of interest.
    virtual void currentBucketPersonIds(core_t::TTime time, TSizeVec& result) const = 0;

    // TODO this needs to be renamed to numberOfActivePeople, and
    // the places where it is used carefully checked
    // (currently only CModelInspector)
    //! Get the total number of people currently being modeled.
    std::size_t numberOfPeople() const;
    //@}

    //! \name Attribute
    //@{
    //! Get the name of the attribute identified by \p cid. This returns
    //! a default fallback string if the attribute doesn't exist.
    //!
    //! \param[in] cid The identifier of the attribute of interest.
    const std::string& attributeName(std::size_t cid) const;

    //! As above but with a specified fallback.
    const std::string& attributeName(std::size_t cid, const std::string& fallback) const;

    //! Print the attributes identified by \p cids.
    //! Optionally, this may be limited to return a string of the form:
    //! A B C and n others
    std::string printAttributes(const TSizeVec& cids,
                                size_t limit = std::numeric_limits<size_t>::max()) const;
    //@}

    //! \name Update
    //@{
    //! This samples the bucket statistics, and any state needed
    //! by computeProbablity, in the time interval [\p startTime,
    //! \p endTime], but does not update the model. This is needed
    //! by the results preview.
    //!
    //! \param[in] startTime The start of the time interval to sample.
    //! \param[in] endTime The end of the time interval to sample.
    virtual void sampleBucketStatistics(core_t::TTime startTime,
                                        core_t::TTime endTime,
                                        CResourceMonitor& resourceMonitor) = 0;

    //! Update the model with the samples of the process in the
    //! time interval [\p startTime, \p endTime].
    //!
    //! \param[in] startTime The start of the time interval to sample.
    //! \param[in] endTime The end of the time interval to sample.
    //! \param[in] resourceMonitor The resourceMonitor.
    virtual void sample(core_t::TTime startTime,
                        core_t::TTime endTime,
                        CResourceMonitor& resourceMonitor) = 0;

    //! This samples the bucket statistics, and any state needed
    //! by computeProbablity, in the time interval [\p startTime,
    //! \p endTime], but does not update the model. This is needed
    //! by the results preview.
    //!
    //! \param[in] startTime The start of the time interval to sample.
    //! \param[in] endTime The end of the time interval to sample.
    virtual void sampleOutOfPhase(core_t::TTime startTime,
                                  core_t::TTime endTime,
                                  CResourceMonitor& resourceMonitor) = 0;

    //! Rolls time to \p endTime while skipping sampling the models for
    //! buckets within the gap.
    //!
    //! \param[in] endTime The end of the time interval to skip sampling.
    void skipSampling(core_t::TTime endTime);

    //! Prune any person models which haven't been updated for a
    //! specified period.
    virtual void prune(std::size_t maximumAge) = 0;

    //! Prune any person models which haven't been updated for a
    //! sufficiently long period, based on the prior decay rates.
    void prune();

    //! Calculate the maximum permitted prune window for this model
    std::size_t defaultPruneWindow() const;

    //! Calculate the minimum permitted prune window for this model
    std::size_t minimumPruneWindow() const;
    //@}

    //! \name Probability
    //@{
    //! Compute the probability of seeing the samples of the process
    //! for the person identified by \p pid in the time interval
    //! [\p startTime, \p endTime].
    //!
    //! \param[in] pid The unique identifier of the person of interest.
    //! \param[in] startTime The start of the time interval of interest.
    //! \param[in] endTime The end of the time interval of interest.
    //! \param[in] partitioningFields The partitioning field (name, value)
    //! pairs for which to compute the the probability.
    //! \param[in] numberAttributeProbabilities The maximum number of
    //! attribute probabilities to retrieve.
    //! \param[out] result A structure containing the probability,
    //! the smallest \p numberAttributeProbabilities attribute
    //! probabilities, the influences and any extra descriptive data.
    virtual bool computeProbability(std::size_t pid,
                                    core_t::TTime startTime,
                                    core_t::TTime endTime,
                                    CPartitioningFields& partitioningFields,
                                    std::size_t numberAttributeProbabilities,
                                    SAnnotatedProbability& result) const = 0;

    //! Update the results with this model's probability.
    //!
    //! \param[in] detector An identifier of the detector generating this
    //! result.
    //! \param[in] startTime The start of the time interval of interest.
    //! \param[in] endTime The end of the time interval of interest.
    //! \param[in] numberAttributeProbabilities The maximum number of
    //! attribute probabilities to retrieve.
    //! \param[in,out] results The model results are added.
    bool addResults(int detector,
                    core_t::TTime startTime,
                    core_t::TTime endTime,
                    std::size_t numberAttributeProbabilities,
                    CHierarchicalResults& results) const;

    //! Compute the probability of seeing \p person's attribute processes
    //! so far given the population distributions.
    //!
    //! \param[in] person The person of interest.
    //! \param[in] numberAttributeProbabilities The maximum number of
    //! attribute probabilities to retrieve.
    //! \param[out] probability Filled in with the probability of seeing
    //! the person's processes given the population processes.
    //! \param[out] attributeProbabilities Filled in with the smallest
    //! \p numberAttributeProbabilities attribute probabilities and
    //! associated data describing the calculation.
    virtual bool
    computeTotalProbability(const std::string& person,
                            std::size_t numberAttributeProbabilities,
                            TOptionalDouble& probability,
                            TAttributeProbability1Vec& attributeProbabilities) const = 0;
    //@}

    //! Get the checksum of this model.
    //!
    //! \param[in] includeCurrentBucketStats If true then include
    //! the current bucket statistics. (This is designed to handle
    //! serialization, for which we don't serialize the current
    //! bucket statistics.)
    virtual uint64_t checksum(bool includeCurrentBucketStats = true) const = 0;

    //! Get the memory used by this model
    virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const = 0;

    //! Get the memory used by this model
    virtual std::size_t memoryUsage() const = 0;

    //! Estimate the memory usage of the model based on number of people,
    //! attributes and correlations. Returns empty when the estimator
    //! is unable to produce an estimate.
    TOptionalSize estimateMemoryUsage(std::size_t numberPeople,
                                      std::size_t numberAttributes,
                                      std::size_t numberCorrelations) const;

    //! Estimate the memory usage of the model based on number of people,
    //! attributes and correlations. When an estimate cannot be produced,
    //! the memory usage is computed and the estimator is updated.
    std::size_t estimateMemoryUsageOrComputeAndUpdate(std::size_t numberPeople,
                                                      std::size_t numberAttributes,
                                                      std::size_t numberCorrelations);

    //! Get the static size of this object - used for virtual hierarchies
    virtual std::size_t staticSize() const = 0;

    //! Get the time series data gatherer.
    const CDataGatherer& dataGatherer() const;
    //! Get the time series data gatherer.
    CDataGatherer& dataGatherer();

    //! Get the length of the time interval used to aggregate data.
    core_t::TTime bucketLength() const;

    //! Get a view of the internals of the model for visualization.
    virtual CModelDetailsViewPtr details() const = 0;

    //! Get the frequency of the person identified by \p pid.
    double personFrequency(std::size_t pid) const;
    //! Get the frequency of the attribute identified by \p cid.
    virtual double attributeFrequency(std::size_t cid) const = 0;

    //! Returns true if the the \p is an unset first bucket time
    static bool isTimeUnset(core_t::TTime);

    //! Get the descriptions of any occurring scheduled event descriptions for the bucket time
    virtual const TStr1Vec& scheduledEventDescriptions(core_t::TTime time) const;

protected:
    using TStrCRef = boost::reference_wrapper<const std::string>;
    using TSizeSize1VecUMap = boost::unordered_map<std::size_t, TSize1Vec>;
    using TFeatureSizeSize1VecUMapPr = std::pair<model_t::EFeature, TSizeSize1VecUMap>;
    using TFeatureSizeSize1VecUMapPrVec = std::vector<TFeatureSizeSize1VecUMapPr>;

    //! \brief The feature models.
    struct MODEL_EXPORT SFeatureModels {
        SFeatureModels(model_t::EFeature feature, TMathsModelPtr newModel);

        //! Restore the models reading state from \p traverser.
        bool acceptRestoreTraverser(const SModelParams& params,
                                    core::CStateRestoreTraverser& traverser);
        //! Persist the models passing state to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

        //! Debug the memory used by this model.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;
        //! Get the memory used by this model.
        std::size_t memoryUsage() const;

        //! The feature.
        model_t::EFeature s_Feature;
        //! A prototype model.
        TMathsModelPtr s_NewModel;
        //! The person models.
        TMathsModelPtrVec s_Models;
    };
    using TFeatureModelsVec = std::vector<SFeatureModels>;

    //! \brief The feature correlate models.
    struct MODEL_EXPORT SFeatureCorrelateModels {
        SFeatureCorrelateModels(model_t::EFeature feature,
                                TMultivariatePriorPtr modelPrior,
                                TCorrelationsPtr model);

        //! Restore the models reading state from \p traverser.
        bool acceptRestoreTraverser(const SModelParams& params,
                                    core::CStateRestoreTraverser& traverser);
        //! Persist the models passing state to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

        //! Debug the memory used by this model.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;
        //! Get the memory used by this model.
        std::size_t memoryUsage() const;

        //! The feature.
        model_t::EFeature s_Feature;
        //! The prototype prior for a correlate model.
        TMultivariatePriorPtr s_ModelPrior;
        //! The correlate models.
        TCorrelationsPtr s_Models;
    };
    using TFeatureCorrelateModelsVec = std::vector<SFeatureCorrelateModels>;

    //! \brief Implements the allocator for new correlate priors.
    class CTimeSeriesCorrelateModelAllocator : public maths::CTimeSeriesCorrelateModelAllocator {
    public:
        using TMemoryUsage = std::function<std::size_t(std::size_t)>;

    public:
        CTimeSeriesCorrelateModelAllocator(CResourceMonitor& resourceMonitor,
                                           TMemoryUsage memoryUsage,
                                           std::size_t resourceLimit,
                                           std::size_t maxNumberCorrelations);

        //! Check if we can still allocate any correlations.
        virtual bool areAllocationsAllowed() const;

        //! Check if \p correlations exceeds the memory limit.
        virtual bool exceedsLimit(std::size_t correlations) const;

        //! Get the maximum number of correlations we should model.
        virtual std::size_t maxNumberCorrelations() const;

        //! Get the chunk size in which to allocate correlations.
        virtual std::size_t chunkSize() const;

        //! Create a new prior for a correlation model.
        virtual TMultivariatePriorPtr newPrior() const;

        //! Set the prototype prior.
        void prototypePrior(const TMultivariatePriorPtr& prior);

    private:
        //! The global resource monitor.
        CResourceMonitor* m_ResourceMonitor;
        //! Computes the current memory usage.
        TMemoryUsage m_MemoryUsage;
        //! The number of correlations which can still be modeled.
        std::size_t m_ResourceLimit;
        //! The maximum permitted number of correlations which can be modeled.
        std::size_t m_MaxNumberCorrelations;
        //! The prototype correlate prior.
        TMultivariatePriorPtr m_PrototypePrior;
    };

protected:
    //! The maximum time a person or attribute is allowed to live
    //! without update.
    static const std::size_t MAXIMUM_PERMITTED_AGE;

    //! Convenience for persistence.
    static const std::string EMPTY_STRING;

protected:
    //! Remove heavy hitting people from the \p data if necessary.
    template<typename T, typename FILTER>
    void applyFilter(model_t::EExcludeFrequent exclude,
                     bool updateStatistics,
                     const FILTER& filter,
                     T& data) const {
        if (this->params().s_ExcludeFrequent & exclude) {
            std::size_t initialSize = data.size();
            data.erase(std::remove_if(data.begin(), data.end(), filter), data.end());
            if (updateStatistics && data.size() != initialSize) {
                core::CStatistics::stat(stat_t::E_NumberExcludedFrequentInvocations)
                    .increment(1);
            }
        }
    }

    //! Get the predicate used for removing heavy hitting people.
    CPersonFrequencyGreaterThan personFilter() const;

    //! Get the predicate used for removing heavy hitting attributes.
    CAttributeFrequencyGreaterThan attributeFilter() const;

    //! Get the global configuration parameters.
    const SModelParams& params() const;

    //! Get the LearnRate parameter from the model configuration -
    //! this may be affected by the current feature being used
    virtual double learnRate(model_t::EFeature feature) const;

    //! Get the start time of the current bucket.
    virtual core_t::TTime currentBucketStartTime() const = 0;

    //! Set the start time of the current bucket.
    virtual void currentBucketStartTime(core_t::TTime time) = 0;

    //! Get the influence calculator for the influencer field identified
    //! by \p iid and the \p feature.
    const CInfluenceCalculator* influenceCalculator(model_t::EFeature feature,
                                                    std::size_t iid) const;

    //! Get the person bucket counts.
    const TDoubleVec& personBucketCounts() const;
    //! Writable access to the person bucket counts.
    TDoubleVec& personBucketCounts();
    //! Set the total count of buckets in the window.
    void windowBucketCount(double windowBucketCount);
    //! Get the total count of buckets in the window.
    double windowBucketCount() const;

    //! Create the time series models for "n" newly observed people
    //! and "m" newly observed attributes.
    virtual void createNewModels(std::size_t n, std::size_t m) = 0;

    //! Reinitialize the time series models for recycled people and/or
    //! attributes.
    virtual void updateRecycledModels() = 0;

    //! Clear out large state objects for people/attributes that are pruned
    virtual void clearPrunedResources(const TSizeVec& people, const TSizeVec& attributes) = 0;

    //! Get the objects which calculates corrections for interim buckets.
    const CInterimBucketCorrector& interimValueCorrector() const;

    //! Check if any of the sample-filtering detection rules apply to this series.
    bool shouldIgnoreSample(model_t::EFeature feature,
                            std::size_t pid,
                            std::size_t cid,
                            core_t::TTime time) const;

    //! Check if any of the result-filtering detection rules apply to this series.
    bool shouldIgnoreResult(model_t::EFeature feature,
                            const model_t::CResultType& resultType,
                            std::size_t pid,
                            std::size_t cid,
                            core_t::TTime time) const;

    //! Get the non-estimated value of the the memory used by this model.
    virtual std::size_t computeMemoryUsage() const = 0;

    //! Restore interim bucket corrector.
    bool interimBucketCorrectorAcceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Persist the interim bucket corrector.
    void interimBucketCorrectorAcceptPersistInserter(const std::string& tag,
                                                     core::CStatePersistInserter& inserter) const;

    //! Create a stub version of maths::CModel for use when pruning people
    //! or attributes to free memory resource.
    static maths::CModel* tinyModel();

private:
    using TModelParamsCRef = boost::reference_wrapper<const SModelParams>;
    using TInterimBucketCorrectorPtr = boost::shared_ptr<CInterimBucketCorrector>;

private:
    //! Set the current bucket total count.
    virtual void currentBucketTotalCount(uint64_t totalCount) = 0;

    //! Skip sampling the interval \p endTime - \p startTime.
    virtual void doSkipSampling(core_t::TTime startTime, core_t::TTime endTime) = 0;

    //! Get the model memory usage estimator
    virtual CMemoryUsageEstimator* memoryUsageEstimator() const = 0;

private:
    //! The global configuration parameters.
    TModelParamsCRef m_Params;

    //! The data gatherer. (This is not persisted by the model hierarchy.)
    TDataGathererPtr m_DataGatherer;

    //! The bucket count of each person in the exponentially decaying
    //! window with decay rate equal to m_DecayRate.
    TDoubleVec m_PersonBucketCounts;

    //! The total number of buckets in the exponentially decaying window
    //! with decay rate equal to m_DecayRate.
    double m_BucketCount;

    //! The influence calculators to use for each feature which is being
    //! modeled.
    TFeatureInfluenceCalculatorCPtrPrVecVec m_InfluenceCalculators;

    //! A corrector that calculates adjustments for values of interim buckets.
    TInterimBucketCorrectorPtr m_InterimBucketCorrector;
};
}
}

#endif // INCLUDED_ml_model_CAnomalyDetectorModel_h
