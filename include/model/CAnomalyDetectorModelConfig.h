/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CAnomalyDetectorModelConfig_h
#define INCLUDED_ml_model_CAnomalyDetectorModelConfig_h

#include <core/CoreTypes.h>

#include <model/CSearchKey.h>
#include <model/FunctionTypes.h>
#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <boost/array.hpp>
#include <boost/property_tree/ptree_fwd.hpp>
#include <boost/unordered_map.hpp>

#include <cstddef>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace ml {
namespace model {
class CDetectionRule;
class CInterimBucketCorrector;
class CSearchKey;
class CModelAutoConfigurer;
class CModelFactory;

//! \brief Responsible for configuring anomaly detection models.
//!
//! DESCRIPTION:\n
//! Responsible for configuring classes for performing anomaly detection.
//! It also defines all parameter defaults.
//!
//! IMPLEMENTATION:\n
//! This wraps up the configuration of anomaly detection to encapsulate
//! the details from calling code. It is anticipated that:
//!   -# Some of this information will be exposed to the user via a
//!      configuration file,
//!   -# Some may be calculated from data characteristics and so on.
class MODEL_EXPORT CAnomalyDetectorModelConfig {
public:
    //! The possible factory types.
    enum EFactoryType {
        E_EventRateFactory = 0,
        E_MetricFactory = 1,
        E_EventRatePopulationFactory = 2,
        E_MetricPopulationFactory = 3,
        E_CountingFactory = 4,
        E_EventRatePeersFactory = 5,
        E_UnknownFactory,
        E_BadFactory
    };

    using TStrSet = std::set<std::string>;
    using TSizeVec = std::vector<std::size_t>;
    using TTimeVec = std::vector<core_t::TTime>;
    using TTimeVecCItr = TTimeVec::const_iterator;
    using TDoubleDoublePr = std::pair<double, double>;
    using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
    using TFeatureVec = model_t::TFeatureVec;
    using TStrVec = std::vector<std::string>;
    using TStrVecCItr = TStrVec::const_iterator;
    using TInterimBucketCorrectorPtr = std::shared_ptr<CInterimBucketCorrector>;
    using TModelFactoryPtr = std::shared_ptr<CModelFactory>;
    using TModelFactoryCPtr = std::shared_ptr<const CModelFactory>;
    using TFactoryTypeFactoryPtrMap = std::map<EFactoryType, TModelFactoryPtr>;
    using TFactoryTypeFactoryPtrMapItr = TFactoryTypeFactoryPtrMap::iterator;
    using TFactoryTypeFactoryPtrMapCItr = TFactoryTypeFactoryPtrMap::const_iterator;
    using TSearchKeyFactoryCPtrMap = std::map<CSearchKey, TModelFactoryCPtr>;

    // Const ref to detection rules map
    using TDetectionRuleVec = std::vector<CDetectionRule>;
    using TDetectionRuleVecCRef = std::reference_wrapper<const TDetectionRuleVec>;
    using TIntDetectionRuleVecUMap = boost::unordered_map<int, TDetectionRuleVec>;
    using TIntDetectionRuleVecUMapCRef = std::reference_wrapper<const TIntDetectionRuleVecUMap>;
    using TIntDetectionRuleVecUMapCItr = TIntDetectionRuleVecUMap::const_iterator;

    using TStrDetectionRulePr = std::pair<std::string, model::CDetectionRule>;
    using TStrDetectionRulePrVec = std::vector<TStrDetectionRulePr>;
    using TStrDetectionRulePrVecCRef = std::reference_wrapper<const TStrDetectionRulePrVec>;

public:
    //! \name Data Gathering
    //@{
    //! The default value used to separate components of a multivariate feature
    //! in its string value.
    static const std::string DEFAULT_MULTIVARIATE_COMPONENT_DELIMITER;

    //! Bucket length if none is specified on the command line.
    static const core_t::TTime DEFAULT_BUCKET_LENGTH;

    //! Default maximum number of buckets for receiving out of order records.
    static const std::size_t DEFAULT_LATENCY_BUCKETS;

    //! Default amount by which metric sample count is reduced for fine-grained
    //! sampling when there is no latency.
    static const std::size_t DEFAULT_SAMPLE_COUNT_FACTOR_NO_LATENCY;

    //! Default amount by which metric sample count is reduced for fine-grained
    //! sampling when there is latency.
    static const std::size_t DEFAULT_SAMPLE_COUNT_FACTOR_WITH_LATENCY;

    //! Default amount by which the metric sample queue expands when it is full.
    static const double DEFAULT_SAMPLE_QUEUE_GROWTH_FACTOR;

    //! Bucket length corresponding to the default decay and learn rates.
    static const core_t::TTime STANDARD_BUCKET_LENGTH;
    //@}

    //! \name Modelling
    //@{
    //! The default rate at which the model priors decay to non-informative
    //! per standard bucket length.
    static const double DEFAULT_DECAY_RATE;

    //! The initial rate, as a multiple of the default decay rate, at which
    //! the model priors decay to non-informative per standard bucket length.
    static const double DEFAULT_INITIAL_DECAY_RATE_MULTIPLIER;

    //! The rate at which information accrues in the model per standard
    //! bucket length elapsed.
    static const double DEFAULT_LEARN_RATE;

    //! The default minimum permitted fraction of points in a distribution
    //! mode for individual modeling.
    static const double DEFAULT_INDIVIDUAL_MINIMUM_MODE_FRACTION;

    //! The default minimum permitted fraction of points in a distribution
    //! mode for population modeling.
    static const double DEFAULT_POPULATION_MINIMUM_MODE_FRACTION;

    //! The default minimum count in a cluster we'll permit in a cluster.
    static const double DEFAULT_MINIMUM_CLUSTER_SPLIT_COUNT;

    //! The default proportion of initial count at which we'll delete a
    //! category from the sketch to cluster.
    static const double DEFAULT_CATEGORY_DELETE_FRACTION;

    //! The default size of the seasonal components we will model.
    static const std::size_t DEFAULT_COMPONENT_SIZE;

    //! The default minimum time to detect a change point in a time series.
    static const core_t::TTime DEFAULT_MINIMUM_TIME_TO_DETECT_CHANGE;

    //! The default maximum time to test for a change point in a time series.
    static const core_t::TTime DEFAULT_MAXIMUM_TIME_TO_TEST_FOR_CHANGE;

    //! The default number of time buckets used to generate multibucket features
    //! for anomaly detection.
    static const std::size_t MULTIBUCKET_FEATURES_WINDOW_LENGTH;

    //! The maximum value that the multi_bucket_impact can take
    static const double MAXIMUM_MULTI_BUCKET_IMPACT_MAGNITUDE;

    //! The maximum number of times we'll update a model in a bucketing
    //! interval. This only applies to our metric statistics, which are
    //! computed on a fixed number of measurements rather than a fixed
    //! time interval. A value of zero implies no constraint.
    static const double DEFAULT_MAXIMUM_UPDATES_PER_BUCKET;

    //! The default minimum value for the influence for which an influencing
    //! field value is judged to have any influence on a feature value.
    static const double DEFAULT_INFLUENCE_CUTOFF;

    //! The default scale factor of the decayRate that determines the minimum
    //! size of the sliding prune window for purging older entries from the
    //! model.
    static const double DEFAULT_PRUNE_WINDOW_SCALE_MINIMUM;

    //! The default scale factor of the decayRate that determines the maximum
    //! size of the sliding prune window for purging older entries from the
    //! model.
    static const double DEFAULT_PRUNE_WINDOW_SCALE_MAXIMUM;

    //! The default factor increase in priors used to model correlations.
    static const double DEFAULT_CORRELATION_MODELS_OVERHEAD;

    //! The default threshold for the Pearson correlation coefficient at
    //! which a correlate will be modeled.
    static const double DEFAULT_MINIMUM_SIGNIFICANT_CORRELATION;
    //@}

    //! \name Anomaly Score Calculation
    //@{
    //! The default values for the aggregation styles' parameters.
    static const double DEFAULT_AGGREGATION_STYLE_PARAMS[model_t::NUMBER_AGGREGATION_STYLES][model_t::NUMBER_AGGREGATION_PARAMS];

    //! The default maximum probability which is deemed to be anomalous.
    static const double DEFAULT_MAXIMUM_ANOMALOUS_PROBABILITY;
    //@}

    //! \name Anomaly Score Normalization
    //@{
    //! The default historic anomaly score percentile for which lower
    //! values are classified as noise.
    static const double DEFAULT_NOISE_PERCENTILE;

    //! The default multiplier applied to the noise level score in
    //! order to be classified as anomalous.
    static const double DEFAULT_NOISE_MULTIPLIER;

    //! We use a piecewise linear mapping between the raw anomaly score
    //! and the normalized anomaly score with these default knot points.
    //! In particular, if we define the percentile of a raw score \f$s\f$
    //! as \f$f_q(s)\f$ and \f$a = \max\{x \le f_q(s)\}\f$ and
    //! \f$b = \min{x \ge f_q(s)}\f$ where \f$x\f$ ranges over the knot point
    //! X- values then the normalized score would be:\n
    //! <pre class="fragment">
    //!   \f$\displaystyle \bar{s} = \frac{(y(b) - y(a))(f_q(s) - a)}{b - a}\f$
    //! </pre>
    //! Here, \f$y(.)\f$ denote the corresponding knot point Y- values.
    static const TDoubleDoublePr DEFAULT_NORMALIZED_SCORE_KNOT_POINTS[9];
    //@}

public:
    //! Create the default configuration.
    //!
    //! \param[in] bucketLength The bucketing interval length.
    //! \param[in] summaryMode Indicates whether the data being gathered
    //! are already summarized by an external aggregation process.
    //! \param[in] summaryCountFieldName If \p summaryMode is E_Manual
    //! then this is the name of the field holding the summary count.
    //! \param[in] latency The amount of time records are buffered for, to
    //! allow out-of-order records to be seen by the models in order.
    //! \param[in] multivariateByFields Should multivariate analysis of
    //! correlated 'by' fields be performed?
    static CAnomalyDetectorModelConfig defaultConfig(core_t::TTime bucketLength,
                                                     model_t::ESummaryMode summaryMode,
                                                     const std::string& summaryCountFieldName,
                                                     core_t::TTime latency,
                                                     bool multivariateByFields);

    //! Overload using defaults.
    static CAnomalyDetectorModelConfig
    defaultConfig(core_t::TTime bucketLength = DEFAULT_BUCKET_LENGTH,
                  model_t::ESummaryMode summaryMode = model_t::E_None,
                  const std::string& summaryCountFieldName = "") {
        return defaultConfig(bucketLength, summaryMode, summaryCountFieldName,
                             DEFAULT_LATENCY_BUCKETS * bucketLength, false);
    }

    //! Get the factor to normalize all bucket lengths to the default
    //! bucket length.
    static double bucketNormalizationFactor(core_t::TTime bucketLength);

    //! Get the decay rate to use for the time series decomposition given
    //! the model decay rate \p modelDecayRate.
    static double trendDecayRate(double modelDecayRate, core_t::TTime bucketLength);

public:
    CAnomalyDetectorModelConfig();

    //! Set the data bucketing interval.
    void bucketLength(core_t::TTime length);
    //! Set the single interim bucket correction calculator.
    void interimBucketCorrector(const TInterimBucketCorrectorPtr& interimBucketCorrector);
    //! Set whether to model multibucket features.
    void useMultibucketFeatures(bool enabled);
    //! Set whether multivariate analysis of correlated 'by' fields should
    //! be performed.
    void multivariateByFields(bool enabled);
    //! Set the model factories.
    void factories(const TFactoryTypeFactoryPtrMap& factories);
    //! Set the style and parameter value for raw score aggregation.
    bool aggregationStyleParams(model_t::EAggregationStyle style,
                                model_t::EAggregationParam param,
                                double value);
    //! Set the maximum anomalous probability.
    void maximumAnomalousProbability(double probability);
    //! Set the noise level as a percentile of historic raw anomaly scores.
    bool noisePercentile(double percentile);
    //! Set the noise multiplier to use when derating normalized scores
    //! based on the noise score level.
    bool noiseMultiplier(double multiplier);
    //! Set the normalized score knot points for the piecewise linear curve
    //! between historic raw score percentiles and normalized scores.
    bool normalizedScoreKnotPoints(const TDoubleDoublePrVec& points);

    //! Populate the parameters from a configuration file.
    bool init(const std::string& configFile);

    //! Populate the parameters from a configuration file, also retrieving
    //! the raw property tree created from the config file.  (The raw
    //! property tree is only valid if the method returns true.)
    bool init(const std::string& configFile, boost::property_tree::ptree& propTree);

    //! Populate the parameters from a property tree.
    bool init(const boost::property_tree::ptree& propTree);

    //! Get the factory for new models.
    //!
    //! \param[in] key The key of the detector for which the factory will be
    //! used.
    TModelFactoryCPtr factory(const CSearchKey& key) const;

    //! Get the factory for new models.
    //!
    //! \param[in] identifier The identifier of the search for which to get a model
    //! factory.
    //! \param[in] function The function being invoked.
    //! \param[in] useNull If true then we will process missing fields as if their
    //! value is equal to the empty string where possible.
    //! \param[in] excludeFrequent Whether to discard frequent results
    //! \param[in] personFieldName The name of the over field.
    //! \param[in] attributeFieldName The name of the by field.
    //! \param[in] valueFieldName The name of the field containing metric values.
    //! \param[in] influenceFieldNames The list of influence field names.
    TModelFactoryCPtr
    factory(int identifier,
            function_t::EFunction function,
            bool useNull = false,
            model_t::EExcludeFrequent excludeFrequent = model_t::E_XF_None,
            const std::string& partitionFieldName = std::string(),
            const std::string& personFieldName = std::string(),
            const std::string& attributeFieldName = std::string(),
            const std::string& valueFieldName = std::string(),
            const CSearchKey::TStoredStringPtrVec& influenceFieldNames =
                CSearchKey::TStoredStringPtrVec()) const;

    //! Set the rate at which the models lose information.
    void decayRate(double value);
    //! Get the rate at which the models lose information.
    double decayRate() const;

    //! Get the length of the baseline.
    core_t::TTime baselineLength() const;

    //! Get the bucket length.
    core_t::TTime bucketLength() const;

    //! Get the maximum latency in the arrival of out of order data.
    core_t::TTime latency() const;

    //! Get the maximum latency in the arrival of out of order data in
    //! numbers of buckets.
    std::size_t latencyBuckets() const;

    //! Get the single interim bucket correction calculator.
    const CInterimBucketCorrector& interimBucketCorrector() const;

    //! Should multivariate analysis of correlated 'by' fields be performed?
    bool multivariateByFields() const;

    //! \name Model Plot
    //@{
    //! Configure modelPlotConfig params from file
    bool configureModelPlot(const std::string& modelPlotConfigFile);

    //! Configure modelPlotConfig params from a property tree
    //! expected to contain three properties: 'boundsPercentile', 'annotationsEnabled'
    //! and 'terms'
    bool configureModelPlot(const boost::property_tree::ptree& propTree);

    //! Configure modelPlotConfig params directly, from the three properties
    //! 'modelPlotEnabled', 'annotationPlotEnabled' and 'terms'.
    //! This initialisation method does not allow setting the value of the
    //! 'boundsPercentile' property, instead a default value is used when 'modelPlotEnabled'
    //! is true and a value of -1.0 is used otherwise.
    void configureModelPlot(bool modelPlotEnabled,
                            bool annotationsEnabled,
                            const std::string& terms);

    //! Set the central confidence interval for the model debug plot
    //! to \p percentage.
    //!
    //! This controls upper and lower confidence interval error bars
    //! returned by the model debug plot.
    //! \note \p percentile should be in the range [0.0, 100.0).
    void modelPlotBoundsPercentile(double percentile);

    //! Get the central confidence interval for the model debug plot.
    double modelPlotBoundsPercentile() const;

    //! Is model plot enabled?
    bool modelPlotEnabled() const;

    //! Are annotations enabled for each of the models?
    bool modelPlotAnnotationsEnabled() const;

    //! Set terms (by, over, or partition field values) to filter
    //! model debug data. When empty, no filtering is applied.
    void modelPlotTerms(TStrSet terms);

    //! Get the terms (by, over, or partition field values)
    //! used to filter model debug data. Empty when no filtering applies.
    const TStrSet& modelPlotTerms() const;
    //@}

    //! \name Anomaly Score Calculation
    //@{
    //! Get the value of the aggregation style parameter identified by
    //! \p style and \p param.
    double aggregationStyleParam(model_t::EAggregationStyle style,
                                 model_t::EAggregationParam param) const;

    //! Get the maximum anomalous probability.
    double maximumAnomalousProbability() const;
    //@}

    //! \name Anomaly Score Normalization
    //@{
    //! Get the historic anomaly score percentile for which lower
    //! values are classified as noise.
    double noisePercentile() const;

    //! Get the multiplier applied to the noise level score in order
    //! to be classified as anomalous.
    double noiseMultiplier() const;

    //! Get the normalized anomaly score knot points.
    const TDoubleDoublePrVec& normalizedScoreKnotPoints() const;
    //@}

    //! Sets the reference to the detection rules map
    void detectionRules(TIntDetectionRuleVecUMapCRef detectionRules);

    //! Sets the reference to the scheduled events vector
    void scheduledEvents(TStrDetectionRulePrVecCRef scheduledEvents);

    //! Process the stanza properties corresponding \p stanzaName.
    //!
    //! \param[in] propertyTree The properties of the stanza called
    //! \p stanzaName.
    bool processStanza(const boost::property_tree::ptree& propertyTree);

    //! Get the factor to normalize all bucket lengths to the default
    //! bucket length.
    double bucketNormalizationFactor() const;

    //! The time window during which samples are accepted.
    core_t::TTime samplingAgeCutoff() const;

private:
    //! Bucket length.
    core_t::TTime m_BucketLength;

    //! Should multivariate analysis of correlated 'by' fields be performed?
    bool m_MultivariateByFields;

    //! The single interim bucket correction calculator.
    TInterimBucketCorrectorPtr m_InterimBucketCorrector;

    //! The new model factories for each data type.
    TFactoryTypeFactoryPtrMap m_Factories;

    //! A cache of customized factories requested from this config.
    mutable TSearchKeyFactoryCPtrMap m_FactoryCache;

    //! Is model plot enabled?
    bool m_ModelPlotEnabled{false};

    //! Are annotations enabled for each of the models?
    bool m_ModelPlotAnnotationsEnabled{false};

    //! The central confidence interval for the model debug plot.
    double m_ModelPlotBoundsPercentile;

    //! Terms (by, over, or partition field values) used to filter model
    //! debug data. Empty when no filtering applies.
    TStrSet m_ModelPlotTerms;
    //@}

    //! \name Anomaly Score Calculation
    //@{
    //! The values for the aggregation styles' parameters.
    double m_AggregationStyleParams[model_t::NUMBER_AGGREGATION_STYLES][model_t::NUMBER_AGGREGATION_PARAMS];

    //! The maximum probability which is deemed to be anomalous.
    double m_MaximumAnomalousProbability;
    //@}

    //! \name Anomaly Score Normalization
    //@{
    //! The historic anomaly score percentile for which lower values
    //! are classified as noise.
    double m_NoisePercentile;

    //! The multiplier applied to the noise level score in order to
    //! be classified as anomalous.
    double m_NoiseMultiplier;

    //! We use a piecewise linear mapping between the raw anomaly score
    //! and the normalized anomaly score with these knot points.
    //! \see DEFAULT_NORMALIZED_SCORE_KNOT_POINTS for details.
    TDoubleDoublePrVec m_NormalizedScoreKnotPoints;
    //@}

    //! A reference to the map containing detection rules per
    //! detector key. Note that the owner of the map is CAnomalyJobConfig::CAnalysisConfig.
    TIntDetectionRuleVecUMapCRef m_DetectionRules;

    //! A reference to the vector of scheduled events.
    //! The owner of the vector is CAnomalyJobConfig::CAnalysisConfig.
    TStrDetectionRulePrVecCRef m_ScheduledEvents;
};
}
}

#endif // INCLUDED_ml_model_CAnomalyDetectorModelConfig_h
