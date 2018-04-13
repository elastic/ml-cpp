/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_config_CDetectorSpecification_h
#define INCLUDED_ml_config_CDetectorSpecification_h

#include <core/CoreTypes.h>

#include <config/ConfigTypes.h>
#include <config/Constants.h>
#include <config/ImportExport.h>

#include <boost/array.hpp>
#include <boost/operators.hpp>
#include <boost/optional.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/ref.hpp>

#include <string>
#include <vector>

namespace ml
{
namespace config
{
class CAutoconfigurerParams;
class CDataCountStatistics;
class CFieldStatistics;
class CPenalty;

//! \brief A detector specification.
//!
//! DESCRIPTION:\n
//! A full description of an anomaly detector. These are built by
//! automatic configuration as one of its main outputs. Sensible
//! values for the various components of a detector specification
//! are selected based on data characteristics.
//!
//! IMPLEMENTATION:\n
//! This is essentially a builder pattern where the final output
//! is a JSON document which can be used to configure the our
//! autodetect API or Splunk.
class CONFIG_EXPORT CDetectorSpecification : boost::equality_comparable< CDetectorSpecification,
                                             boost::less_than_comparable< CDetectorSpecification > >
{
    public:
        using TDoubleVec = std::vector<double>;
        using TDoubleVecVec = std::vector<TDoubleVec>;
        using TSizeVec = std::vector<std::size_t>;
        using TTimeVec = std::vector<core_t::TTime>;
        using TStrVec = std::vector<std::string>;
        using TOptionalStr = boost::optional<std::string>;
        using TFieldStatisticsVec = std::vector<CFieldStatistics>;
        using TPenaltyPtr = boost::shared_ptr<CPenalty>;

        //! Ternary boolean type which supports unknown.
        enum EFuzzyBool
        {
            E_True,
            E_False,
            E_Maybe
        };

        //! \brief The score for a given set of parameters.
        struct CONFIG_EXPORT SParamScores
        {
            SParamScores(core_t::TTime bucketLength,
                         const std::string &ignoreEmpty,
                         double score,
                         const TStrVec &descriptions);

            //! The bucket length.
            core_t::TTime s_BucketLength;

            //! The name of the ignore empty version.
            std::string s_IgnoreEmpty;

            //! The parameters score.
            double s_Score;

            //! The descriptions associated with this score.
            TStrVec s_Descriptions;
        };

        using TParamScoresVec = std::vector<SParamScores>;

    public:
        CDetectorSpecification(const CAutoconfigurerParams &params,
                               config_t::EFunctionCategory function,
                               std::size_t id);
        CDetectorSpecification(const CAutoconfigurerParams &params,
                               config_t::EFunctionCategory function,
                               const std::string &argument,
                               std::size_t id);

        //! Efficiently exchange the contents of two detectors.
        void swap(CDetectorSpecification &other);

        //! \name Builder Interface
        //@{
        //! Set the side the detector is sensitive to.
        void side(config_t::ESide side);

        //! Set whether the detector ignore empty buckets.
        void ignoreEmpty(bool ignoreEmpty);

        //! Check if we can add the partitioning field \p value.
        bool canAddPartitioning(std::size_t index, const std::string &value) const;

        //! Set the field identified by \p index which must be one
        //! of ARGUMENT_INDEX, BY_INDEX, OVER_INDEX or PARTITION_INDEX.
        void addPartitioning(std::size_t index, const std::string &value);

        //! Add \p influence as an influencer.
        void addInfluencer(const std::string &influence);

        //! Set the shortest bucket length.
        void bucketLength(core_t::TTime bucketLength);

        //! Add the statistics for the detector's fields.
        void addFieldStatistics(const TFieldStatisticsVec &stats);

        //! Set the detector's count statistics.
        void setCountStatistics(const CDataCountStatistics &stats);

        //! Set the penalty function for this detector.
        void setPenalty(const TPenaltyPtr &penalty);
        //@}

        //! \name Detector Scoring
        //@{
        //! Get the current detector score.
        double score() const;

        //! The penalties that apply to the various
        void scores(TParamScoresVec &result) const;

        //! Apply the penalty \p penalty.
        void applyPenalty(double penalty, const std::string &description);

        //! Apply the penalty for the bucket length \p bucketLength.
        void applyPenalties(const TSizeVec &indices,
                            const TDoubleVec &penalty,
                            const TStrVec &description);

        //! Refresh all scores.
        void refreshScores();
        //@}

        //! \name Detector Attributes
        //@{
        //! Is the detector one of the count functions?
        config_t::EFunctionCategory function() const;

        //! Get the field which is the argument of the function.
        const TOptionalStr &argumentField() const;

        //! Get the by field name. Null if there isn't one.
        const TOptionalStr &byField() const;

        //! Get the over field name. Null if there isn't one.
        const TOptionalStr &overField() const;

        //! Get the partition field name. Null if there isn't one.
        const TOptionalStr &partitionField() const;

        //! Get the influences which have been configured.
        const TStrVec &influences() const;

        //! Get the bucket lengths.
        void candidateBucketLengths(TTimeVec &result) const;

        //! Check if this detector is for population analysis.
        bool isPopulation() const;
        //@}

        //! A total order of two detector specifications.
        bool operator<(const CDetectorSpecification &rhs) const;

        //! Equality comparison for two detector specifications.
        bool operator==(const CDetectorSpecification &rhs) const;

        //! Get the identifier.
        std::size_t id() const;

        //! Set the identifier.
        void id(std::size_t id);

        //! Get the argument field statistics if there is one.
        const CFieldStatistics *argumentFieldStatistics() const;

        //! Get the by field statistics if there is one.
        const CFieldStatistics *byFieldStatistics() const;

        //! Get the over field statistics if there is one.
        const CFieldStatistics *overFieldStatistics() const;

        //! Get the partition field statistics if there is one.
        const CFieldStatistics *partitionFieldStatistics() const;

        //! Get the count statistics for this detector.
        const CDataCountStatistics *countStatistics() const;

        //! Write a configuration which can be interpreted by our
        //! autodetect processes.
        std::string detectorConfig() const;

        //! Get a description of the detector.
        std::string description() const;

    private:
        using TStrVecVec = std::vector<TStrVec>;
        using TOptionalTime = boost::optional<core_t::TTime>;
        using TSizeVecCPtrAry = boost::array<const TSizeVec*, 2>;
        using TAutoconfigurerParamsCRef = boost::reference_wrapper<const CAutoconfigurerParams>;

    private:
        //! Get the parameters.
        const CAutoconfigurerParams &params() const;

        //! Get the highest index of any non-null function field.
        int highestFieldIndex() const;

        //! Get the indices of the penalties used for this detector.
        TSizeVecCPtrAry penaltyIndicesInUse() const;

        //! Initialize the penalties.
        void initializePenalties();

        //! Update the value of the ignore empty flag based on the
        //! current scores.
        void refreshIgnoreEmpty();

    private:
        //! The parameters.
        TAutoconfigurerParamsCRef m_Params;

        //! \name Specification
        //@{
        //! The detector function category.
        config_t::EFunctionCategory m_Function;

        //! The side of the anomaly the detector is sensitive, i.e.
        //! high, low or both.
        config_t::ESide m_Side;

        //! Set to true if the function can ignore empty buckets and
        //! should, false if it can ignore empty buckets and shouldn't
        //! and left as maybe otherwise.
        EFuzzyBool m_IgnoreEmpty;

        //! The argument and partitioning fields (i.e. by, over and
        //! partition) for the detector. A null value means the it
        //! doesn't have one.
        TOptionalStr m_FunctionFields[constants::NUMBER_FIELD_INDICES];

        //! The influences configured for the detector.
        TStrVec m_Influencers;

        //! The shortest (in the context of multi-) bucket length to
        //! use for this detector.
        TOptionalTime m_BucketLength;

        // TODO exclude frequent + frequency

        // TODO include/exclude list
        //@}

        //! \name Scoring
        //@{
        //! The current assessment of the quality of the detector for its
        //! candidate parameters, such as bucket length, ignore empty, etc.
        //! These are in the range [0, 1.0] and are updated during the
        //! configuration process. A higher penalty indicates a better
        //! detector.
        TDoubleVec m_Penalties;

        //! The function for computing the penalties to apply to this detector.
        TPenaltyPtr m_Penalty;

        //! Descriptions of the penalties that apply.
        TStrVecVec m_PenaltyDescriptions;

        //! A unique identifier of this detector.
        std::size_t m_Id;

        //! The statistics for each of the detectors fields.
        const CFieldStatistics *m_FieldStatistics[constants::NUMBER_FIELD_INDICES];

        //! The count statistics for the detector.
        const CDataCountStatistics *m_CountStatistics;
        //@}
};

}
}

#endif // INCLUDED_ml_config_CDetectorSpecification_h
