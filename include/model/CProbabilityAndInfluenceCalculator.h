/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CProbabilityAndInfluenceCalculator_h
#define INCLUDED_ml_model_CProbabilityAndInfluenceCalculator_h

#include <core/CStoredStringPtr.h>

#include <maths/CModel.h>

#include <model/CModelTools.h>
#include <model/CPartitioningFields.h>
#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <boost/ref.hpp>

#include <string>
#include <utility>
#include <vector>

namespace ml
{
namespace model
{
class CAnnotatedProbabilityBuilder;
class CInfluenceCalculator;

//! \brief Computes the overall probability and influences on a collection
//! of feature values and corresponding influencer field value feature
//! values.
//!
//! DESCRIPTION:\n
//! This implements the calculation of the aggregate probability of a
//! collection of feature values. Each feature value is supplied to the
//! addProbability function. In addition, for each feature value, a
//! collection of influencer field values, together with the corresponding
//! feature values for the restriction to the records labeled with each
//! influencer field value, can be added via addInfluencers. In this case
//! the influences on the value are found and this then updates the overall
//! influence probabilities. Ultimately, the overall, or aggregate, probability
//! and influences on this can be computed, after all probabilities and
//! influencer field values have been added, using the calculate function.
//!
//! IMPLEMENTATION:\n
//! This uses a plug-in pattern to support different influence calculations
//! which are selected at runtime. This is necessary because different features
//! use different influence calculations, but the features are selected based
//! on the commands the user runs.
class MODEL_EXPORT CProbabilityAndInfluenceCalculator
{
    public:
        using TDouble1Vec = core::CSmallVector<double, 1>;
        using TDouble2Vec = core::CSmallVector<double, 2>;
        using TDouble4Vec = core::CSmallVector<double, 4>;
        using TDouble2Vec1Vec = core::CSmallVector<TDouble2Vec, 1>;
        using TDouble4Vec1Vec = core::CSmallVector<TDouble4Vec, 1>;
        using TDouble1VecDoublePr = std::pair<TDouble1Vec, double>;
        using TDouble1VecDouble1VecPr = std::pair<TDouble1Vec, TDouble1Vec>;
        using TBool2Vec = core::CSmallVector<bool, 2>;
        using TSize1Vec = core::CSmallVector<std::size_t, 1>;
        using TSize2Vec = core::CSmallVector<std::size_t, 2>;
        using TSize2Vec1Vec = core::CSmallVector<TSize2Vec, 1>;
        using TTime2Vec = core::CSmallVector<core_t::TTime, 2>;
        using TTime2Vec1Vec = core::CSmallVector<TTime2Vec, 1>;
        using TTail2Vec = core::CSmallVector<maths_t::ETail, 2>;
        using TStrCRef = boost::reference_wrapper<const std::string>;
        using TStrCRefDouble1VecDoublePrPr = std::pair<TStrCRef, TDouble1VecDoublePr>;
        using TStrCRefDouble1VecDoublePrPrVec = std::vector<TStrCRefDouble1VecDoublePrPr>;
        using TStrCRefDouble1VecDouble1VecPrPr = std::pair<TStrCRef, TDouble1VecDouble1VecPr>;
        using TStrCRefDouble1VecDouble1VecPrPrVec = std::vector<TStrCRefDouble1VecDouble1VecPrPr>;
        using TStrCRefDouble1VecDouble1VecPrPrVecVec = std::vector<TStrCRefDouble1VecDouble1VecPrPrVec>;
        using TStoredStringPtrStoredStringPtrPr = std::pair<core::CStoredStringPtr, core::CStoredStringPtr>;
        using TStoredStringPtrStoredStringPtrPrVec = std::vector<TStoredStringPtrStoredStringPtrPr>;
        using TStoredStringPtrStoredStringPtrPrDoublePr = std::pair<TStoredStringPtrStoredStringPtrPr, double>;
        using TStoredStringPtrStoredStringPtrPrDoublePrVec = std::vector<TStoredStringPtrStoredStringPtrPrDoublePr>;
        using TStoredStringPtr1Vec = core::CSmallVector<core::CStoredStringPtr, 1>;

        //! \brief Wraps up the parameters to the influence calculation.
        struct MODEL_EXPORT SParams : private core::CNonCopyable
        {
            SParams(const CPartitioningFields &partitioningFields);

            //! Helper to print a description of the parameters.
            std::string describe() const;

            //! The feature of s_Value.
            model_t::EFeature s_Feature;
            //! The model of s_Value.
            const maths::CModel *s_Model;
            //! The time after the creation of the prior.
            core_t::TTime s_ElapsedTime;
            //! The time of s_Value.
            TTime2Vec1Vec s_Time;
            //! The feature value.
            TDouble2Vec1Vec s_Value;
            //! The count of measurements in s_Value.
            double s_Count;
            //! The parameters needed to compute probabilities.
            maths::CModelProbabilityParams s_ComputeProbabilityParams;
            //! The probability of the s_Value.
            double s_Probability;
            //! The tail that the s_Value is in.
            TTail2Vec s_Tail;
            //! The name of the field for which to compute and influences.
            core::CStoredStringPtr s_InfluencerName;
            //! The influencer field values, and corresponding feature
            //! value and count of measurements in the restrictions of
            //! records to those influencer field values.
            TStrCRefDouble1VecDoublePrPrVec s_InfluencerValues;
            //! The partitioning field (name, value) pairs.
            const CPartitioningFields &s_PartitioningFields;
            //! The level at which influence occurs.
            double s_Cutoff;
            //! If true then add in influences greater than the cutoff.
            bool s_IncludeCutoff;
            //! Filled in with the influences of s_Value if any.
            TStoredStringPtrStoredStringPtrPrDoublePrVec s_Influences;
        };

        //! \brief Wraps up the parameters to the influence calculation
        //! for correlates.
        struct MODEL_EXPORT SCorrelateParams : private core::CNonCopyable
        {
            SCorrelateParams(const CPartitioningFields &partitioningFields);

            //! Helper to print a description of the parameters.
            std::string describe() const;

            //! The feature of s_Values.
            model_t::EFeature s_Feature;
            //! The model of s_Values.
            const maths::CModel *s_Model;
            //! The time after the creation of the s_Priors.
            core_t::TTime s_ElapsedTime;
            //! The times of s_Values.
            TTime2Vec1Vec s_Times;
            //! The feature values.
            TDouble2Vec1Vec s_Values;
            //! The count of measurements in s_Values.
            TDouble2Vec1Vec s_Counts;
            //! The variable identifiers for the primary and correlated
            //! time series.
            TSize2Vec1Vec s_Variables;
            //! The correlated time series labels.
            TStoredStringPtr1Vec s_CorrelatedLabels;
            //! The correlated time series identifiers.
            TSize1Vec s_Correlated;
            //! The parameters needed to compute probabilities.
            maths::CModelProbabilityParams s_ComputeProbabilityParams;
            //! The probability of the s_Value.
            double s_Probability;
            //! The tail that the s_Value is in.
            TTail2Vec s_Tail;
            //! The index of the most anomalous correlate.
            TSize1Vec s_MostAnomalousCorrelate;
            //! The name of the field for which to compute and influences.
            core::CStoredStringPtr s_InfluencerName;
            //! The influencer field values, and corresponding feature
            //! value and count of measurements in the restrictions of
            //! records to those influencer field values.
            TStrCRefDouble1VecDouble1VecPrPrVec s_InfluencerValues;
            //! The partitioning field (name, value) pairs.
            const CPartitioningFields &s_PartitioningFields;
            //! The level at which influence occurs.
            double s_Cutoff;
            //! If true then add in influences greater than the cutoff.
            bool s_IncludeCutoff;
            //! Filled in with the influences of s_Values if any.
            TStoredStringPtrStoredStringPtrPrDoublePrVec s_Influences;
        };

    public:
        explicit CProbabilityAndInfluenceCalculator(double cutoff);

        //! Check if any probabilities have been added to the calculator.
        bool empty() const;

        //! Get the minimum value for the influence for which an influencing
        //! field value is judged to have any influence on a feature value.
        double cutoff() const;

        //! Plug-in the influence calculation to use.
        void plugin(const CInfluenceCalculator &influence);

        //! Add the joint probability aggregation style.
        void addAggregator(const maths::CJointProbabilityOfLessLikelySamples &aggregator);

        //! Add the extreme probability aggregation style.
        void addAggregator(const maths::CProbabilityOfExtremeSample &aggregator);

        //! Add a cache for the two probability calculations.
        void addCache(CModelTools::CProbabilityCache &cache);

        //! Add the probabilities and influences from \p other.
        void add(const CProbabilityAndInfluenceCalculator &other, double weight = 1.0);

        //! Add an attribute probability for \p value of the univariate
        //! feature \p feature.
        //!
        //! This is a wrapper around addProbability which fills in an attribute
        //! probability on \p builder.
        //!
        //! \param[in] attribute The attribute.
        //! \param[in] cid The attribute identifier.
        //! \param[in] pAttribute The probability of attribute.
        //! \param[in,out] params The parameters used in the probability calculation.
        //! \param[out] builder An attribute probability for \p attribute and
        //! \p value is added to this builder if it can be computed.
        //! \param[in] weight The weight to use when updating the aggregate
        //! probabilities.
        bool addAttributeProbability(const core::CStoredStringPtr &attribute,
                                     std::size_t cid,
                                     double pAttribute,
                                     SParams &params,
                                     CAnnotatedProbabilityBuilder &builder,
                                     double weight = 1.0);

        //! Add an attribute probability for \p values of the correlates
        //! of the univariate feature \p feature.
        bool addAttributeProbability(const core::CStoredStringPtr &attribute,
                                     std::size_t cid,
                                     double pAttribute,
                                     SCorrelateParams &params,
                                     CAnnotatedProbabilityBuilder &builder,
                                     double weight = 1.0);

        //! Compute and add the probability for \p value of the univariate
        //! feature \p feature.
        //!
        //! \param[in] feature The value's feature.
        //! \param[in] id A unique identifier of the value's model.
        //! \param[in] model The value's model.
        //! \param[in] elapsedTime The time that has elapsed since the
        //! model was created.
        //! \param[in] params Extra parameters needed by \p model to compute
        //! the probability.
        //! \param[in] time The value's time.
        //! \param[in] value The value for which to compute the probability.
        //! \param[out] probability Set to the probability of \p value
        //! if one could be calculated.
        //! \param[out] tail Set to the tail that \p value is in.
        //! \param[out] type Filled in with the type of anomaly, i.e. is the
        //! value anomalous in its own right or as a result of conditioning
        //! on a correlated variable.
        //! \param[out] mostAnomalousCorrelate Filled in with the index of the
        //! most anomalous correlated time series.
        //! \param[in] weight The weight to use when updating the aggregate
        //! probabilities.
        bool addProbability(model_t::EFeature feature,
                            std::size_t id,
                            const maths::CModel &model,
                            core_t::TTime elapsedTime,
                            const maths::CModelProbabilityParams &params,
                            const TTime2Vec1Vec &time,
                            const TDouble2Vec1Vec &value,
                            double &probability,
                            TTail2Vec &tail,
                            model_t::CResultType &type,
                            TSize1Vec &mostAnomalousCorrelate,
                            double weight = 1.0);

        //! Add the probability to the overall aggregate probability and
        //! all influencer aggregate probabilities.
        //!
        //! \param[in] probability The probability to add.
        //! \param[in] weight The weight to use when updating the aggregate
        //! probabilities.
        void addProbability(double probability, double weight = 1.0);

        //! Compute and add the influences from \p influencerValues.
        //!
        //! \param[in] influencerName The name of the field for which
        //! to compute and influences.
        //! \param[in] influencerValues The influencer field values and
        //! feature values and counts of measurements in the restrictions
        //! of records to the corresponding influencer field values.
        //! \param[in,out] params The parameters used in the probability calculation.
        //! \param[in] weight The weight to use when updating the aggregate
        //! probabilities.
        void addInfluences(const std::string &influencerName,
                           const TStrCRefDouble1VecDoublePrPrVec &influencerValues,
                           SParams &params,
                           double weight = 1.0);

        //! Compute and add the influences from \p influencerValues for
        //! the correlates of a univariate feature.
        void addInfluences(const std::string &influencerName,
                           const TStrCRefDouble1VecDouble1VecPrPrVecVec &influencerValues,
                           SCorrelateParams &params,
                           double weight = 1.0);

        //! Calculate the overall probability of all values added.
        //!
        //! \param[out] probability Filled in with the overall probability
        //! of all values added via addProbability.
        bool calculate(double &probability) const;

        //! Calculate the overall probability of all values added and
        //! any influences and their weights.
        //!
        //! \param[out] probability Filled in with the overall probability
        //! of all values added via addProbability.
        //! \param[out] influences Filled in with all influences of the
        //! overall probability.
        bool calculate(double &probability,
                       TStoredStringPtrStoredStringPtrPrDoublePrVec &influences) const;

    private:
        //! Actually commit any influences we've found.
        void commitInfluences(model_t::EFeature feature, double logp, double weight);

    private:
        //! The minimum value for the influence for which an influencing
        //! field value is judged to have any influence on a feature value.
        double m_Cutoff;

        //! The plug-in used to adapt the influence calculation for
        //! different features.
        const CInfluenceCalculator *m_InfluenceCalculator;

        //! The template probability calculator.
        CModelTools::CProbabilityAggregator m_ProbabilityTemplate;

        //! The probability calculator.
        CModelTools::CProbabilityAggregator m_Probability;

        //! The probability calculation cache if there is one.
        CModelTools::CProbabilityCache *m_ProbabilityCache;

        //! The influence probability calculator.
        CModelTools::TStoredStringPtrStoredStringPtrPrProbabilityAggregatorUMap m_InfluencerProbabilities;

        //! Placeholder for the influence weights so that it isn't
        //! allocated in a loop.
        TStoredStringPtrStoredStringPtrPrDoublePrVec m_Influences;
};

//! \brief Interface for influence calculations.
//!
//! DESCRIPTION:\n
//! The influence calculation tries to determine the extent to which
//! an anomaly is affected by records labeled with a particular
//! influencing field value. What we'd ideally like to determine is
//! whether an anomaly is reasonably considered to be caused by the
//! values of the fields labeled by an influencing fields value. To
//! quantify this, we like to say that if none of the records in the
//! anomaly are labeled with an influencing field value then it has
//! no influence on the anomaly, i.e. its influence is 0. Conversely,
//! if all the records in the anomaly are labeled with the field
//! value it has complete influence, i.e. its influence is 1. All
//! other cases are an interpolation between these two cases.
//!
//! The exact form of the influence calculation depends on the way
//! the feature is extracted from the records. All calculations
//! proceed along the lines of asking either 1) "would there still
//! be an anomaly if only the records labeled with the field value
//! were present", or 2) "would there still be an anomaly if none
//! of the records labeled with the field value were present".
class MODEL_EXPORT CInfluenceCalculator : private core::CNonCopyable
{
    public:
        //! See core::CMemory.
        static bool dynamicSizeAlwaysZero()
        {
            return true;
        }
        using TParams = CProbabilityAndInfluenceCalculator::SParams;
        using TCorrelateParams = CProbabilityAndInfluenceCalculator::SCorrelateParams;

    public:
        virtual ~CInfluenceCalculator();

        //! Compute the influence from the probability of set difference
        //! statistics.
        static double intersectionInfluence(double logp, double logpi);

        //! Compute the influence from the probability of set intersection
        //! statistics.
        static double complementInfluence(double logp, double logpi);

        //! Compute the influences on a univariate value.
        virtual void computeInfluences(TParams &params) const = 0;

        //! Compute the influences on a correlate value.
        virtual void computeInfluences(TCorrelateParams &params) const = 0;
};

//! \brief A stub implementation for the case that the influence
//! can't be calculated.
class MODEL_EXPORT CInfluenceUnavailableCalculator : public CInfluenceCalculator
{
    public:
        virtual void computeInfluences(TParams &params) const;
        virtual void computeInfluences(TCorrelateParams &params) const;
};

//! \brief A stub implementation for the case that every influence
//! is 1, irrespective of the feature value and influence values.
class MODEL_EXPORT CIndicatorInfluenceCalculator : public CInfluenceCalculator
{
    public:
        virtual void computeInfluences(TParams &params) const;
        virtual void computeInfluences(TCorrelateParams &params) const;
};

//! \brief Computes the influences for sum like features.
//!
//! DESCRIPTION:\n
//! This computes the probability \f$P_{A \ B} = 1.0 - F(g(A \ B))\f$
//! where \f$A\f$ are the bucket records, \f$B\f$ are the influencer
//! records, \f$g(\cdot)\f$ is the feature function and \f$F(\cdot)\f$
//! is the cumulative density function of the feature. The influence
//! is large if \f$P_{A \ B} >> P_A\f$. In particular, it is defined
//! as:
//! <pre class="fragment">
//!   \f$\displaystyle \left(1.0 - \frac{\log(P_{A \ B})}{\log(P_A)}\right)_0^1\f$
//! </pre>
//! where \f$(\cdot)_0^1\f$ denotes the restriction to the interval
//! \f$[0,1]\f$.
//!
//! This covers the case for all counts, sums, etc. Note, we can only
//! determine whether there is influence in this case if the count
//! anomaly is in the right tail, i.e. the count is unusually high.
//! Otherwise, the anomaly is likely due to the absence of counts for
//! one of the influencing field values, in which case we'd need to
//! know what its typical count is and we don't have this information.
class MODEL_EXPORT CLogProbabilityComplementInfluenceCalculator : public CInfluenceCalculator
{
    public:
        virtual void computeInfluences(TParams &params) const;
        virtual void computeInfluences(TCorrelateParams &params) const;
};

//! \brief Computes the influences for minimum like features.
//!
//! DESCRIPTION:\n
//! This computes the probability as follows:
//! <pre class="fragment">
//!   \f$\displaystyle P_{A \cap B} = \min\left\{ F(g(A \cap B)), 1-F(g(A \cap B)) \right\}\f$
//! </pre>
//! where \f$A\f$ are the bucket records, \f$B\f$ are the influencer
//! records, \f$g(\cdot)\f$ is the feature function and \f$F(\cdot)\f$
//! is the cumulative density function of the feature. The influence
//! is large if \f$P_{A \cap B} \lte P_A\f$. In particular, it is
//! defined as:
//! <pre class="fragment">
//!   \f$\displaystyle \left(\frac{\log(P_{A \cap B})}{\log(P_A)}\right)_0^1\f$
//! </pre>
//! where \f$(\cdot)_0^1\f$ denotes the restriction to the interval
//! \f$[0,1]\f$.
//!
//! This covers the case for distinct count, minimum, maximum, etc;
//! basically, any feature whose value is determined by a small
//! subset of the records. In this case we ask "would there still
//! be an anomaly if only the records labeled with the influencer
//! field value were present". Note, we can determine whether there
//! is influence in this case if the anomalous value is in either
//! the left or right tail.
class MODEL_EXPORT CLogProbabilityInfluenceCalculator : public CInfluenceCalculator
{
    public:
        virtual void computeInfluences(TParams &params) const;
        virtual void computeInfluences(TCorrelateParams &params) const;
};

//! \brief Computes the influences for the mean feature.
//!
//! DESCRIPTION:\n
//! This is the same as log-probability complement calculation except
//! for the following two differences:
//!   -# The mean can be influenced to be either less or more than
//!      its typical value, so we can compute influence in the case
//!      that the value is in either the left or right tail,
//!   -# In order to compute the mean on the difference set, we need
//!      to use the standard mean difference, i.e. weight the subtraction
//!      by the value counts. We must also account for the of the value
//!      count on the sample variance.
//!
//! \see CLogProbabilityComplementInfluenceCalculator for more details
//! on the calculation.
class MODEL_EXPORT CMeanInfluenceCalculator : public CInfluenceCalculator
{
    public:
        virtual void computeInfluences(TParams &params) const;
        virtual void computeInfluences(TCorrelateParams &params) const;
};

//! \brief Computes the influences for the mean feature.
//!
//! DESCRIPTION:\n
//! This is the same as log-probability complement calculation except
//! for the following two differences:
//!   -# The mean can be influenced to be either less or more than
//!      its typical value, so we can compute influence in the case
//!      that the value is in either the left or right tail,
//!   -# In order to compute the mean on the difference set, we need
//!      to use the standard mean difference, i.e. weight the subtraction
//!      by the value counts. We must also account for the of the value
//!      count on the sample variance.
//!
//! \see CLogProbabilityComplementInfluenceCalculator for more details
//! on the calculation.
class MODEL_EXPORT CVarianceInfluenceCalculator : public CInfluenceCalculator
{
    public:
        virtual void computeInfluences(TParams &params) const;
        virtual void computeInfluences(TCorrelateParams &params) const;
};

}
}

#endif // INCLUDED_ml_model_CProbabilityAndInfluenceCalculator_h
