/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CHierarchicalResultsNormalizer_h
#define INCLUDED_ml_model_CHierarchicalResultsNormalizer_h

#include <core/CNonCopyable.h>

#include <model/CAnomalyScore.h>
#include <model/CHierarchicalResultsLevelSet.h>
#include <model/ImportExport.h>

#include <string>
#include <utility>
#include <vector>


namespace ml
{
namespace model
{
class CAnomalyDetectorModelConfig;

namespace hierarchical_results_normalizer_detail
{

using TNormalizerPtr = boost::shared_ptr<CAnomalyScore::CNormalizer>;

//! \brief A normalizer instance and a descriptive string.
struct MODEL_EXPORT SNormalizer
{
    SNormalizer(const std::string &description, const TNormalizerPtr &normalizer);

    //! Clear the normalizer.
    void clear();

    //! Age the normalizer to account for \p time passing.
    void propagateForwardByTime(double time);

    //! Compute a checksum for this object.
    uint64_t checksum() const;

    std::string s_Description;
    TNormalizerPtr s_Normalizer;
};

}

//! \brief A collection of normalizers for some hierarchical results.
//!
//! DESCRIPTION:\n
//! Manages the normalization of a collection of hierarchical results.
//! Each distinct partition and "named system" or population has its own
//! normalizer in addition the unique "simple search" results, belonging
//! to the same named partition and population, have a normalizer.
//! Finally the bucket results have a normalizer. For example, if someone
//! were to configure the following searches:\n
//! <pre>
//!   partitionfield=region sum(bytes) by host
//!   partitionfield=region count by host
//!   partitionfield=region sum(bytes) over client
//!   partitionfield=host rare by process
//! </pre>
//! There would be nine normalizers in total:
//!   -# One normalizer for the bucket,
//!   -# One normalizer for count and sum(bytes) probabilities for system
//!      host and partition region,
//!   -# One normalizer for system host and partition region,
//!   -# Two (identical) normalizers for client and partition region,
//!   -# One normalizer for partition region,
//!   -# Two (identical) normalizers for process and partition host
//!   -# One normalizer for partition host.
//!
//! Note that the redundancy, for client and process fields, is because
//! each of these has one search; however, we still create normalizers
//! at both the "simple search" and "named system" or population level.
//! It is impossible for us to determine there is redundancy observing
//! results objects because we don't know if some results are not
//! present for the bucket, because a field value wasn't present in the
//! bucket. The wasted memory in creating a small number of redundant
//! normalizers is negligible.
class MODEL_EXPORT CHierarchicalResultsNormalizer :
          public CHierarchicalResultsLevelSet<hierarchical_results_normalizer_detail::SNormalizer>,
          private core::CNonCopyable
{
    public:
        using TBase = CHierarchicalResultsLevelSet<hierarchical_results_normalizer_detail::SNormalizer>;
        using TNormalizer = TBase::Type;
        using TNormalizerPtrVec = TBase::TTypePtrVec;
        using TWordNormalizerPr = TBase::TWordTypePr;
        using TWordNormalizerPrVec = TBase::TWordTypePrVec;
        using TStrVec = std::vector<std::string>;

        //! Enumeration of the possible jobs that the normalizer can
        //! perform when invoked.
        enum EJob
        {
            E_Update,
            E_Normalize,
            E_NoOp
        };

        //! Enumeration of possible outcomes of restoring from XML.
        enum ERestoreOutcome
        {
            E_Ok         = 0,
            E_Corrupt    = 1,
            E_Incomplete = 2
        };

    public:
        CHierarchicalResultsNormalizer(const CAnomalyDetectorModelConfig &modelConfig);

        //! Add a job for the subsequent invocations of the normalizer.
        void setJob(EJob job);

        //! Clear all state such that all normalizers restart from scratch.
        void clear();

        //! Reset the hasLastUpdateCausedBigChange() flag
        void resetBigChange();

        //! Update the normalizer with the node's anomaly score.
        virtual void visit(const CHierarchicalResults &results, const TNode &node, bool pivot);

        //! Age the maximum scores and quantile summaries.
        void propagateForwardByTime(double time);

        //! Returns true if the last update caused a big change to the quantiles
        //! or false otherwise.
        bool hasLastUpdateCausedBigChange() const;

        //! Convert each normalizer to a JSON document and store these as an
        //! array to the string provided.
        void toJson(core_t::TTime time, const std::string &key, std::string &json, bool makeArray) const;

        //! Replace the state of this object with normalizers restored from
        //! the JSON documents in the stream.
        ERestoreOutcome fromJsonStream(std::istream &inputStream);

        //! Access to the root normalizer.
        const CAnomalyScore::CNormalizer &bucketNormalizer() const;

        //! Get the influencer bucket normalizer for \p influencerFieldName.
        //!
        //! \note Returns NULL if there isn't a matching one.
        const CAnomalyScore::CNormalizer *
            influencerBucketNormalizer(const std::string &influencerFieldName) const;

        //! Get the influencer normalizer for \p influencerFieldName.
        //!
        //! \note Returns NULL if there isn't a matching one.
        const CAnomalyScore::CNormalizer *
            influencerNormalizer(const std::string &influencerFieldName) const;

        //! Get a partition normalizer.
        //!
        //! \note Returns NULL if there isn't a matching one.
        const CAnomalyScore::CNormalizer *
            partitionNormalizer(const std::string &partitionFieldName) const;

        //! Get a person normalizer.
        //!
        //! \note Returns NULL if there isn't a matching one.
        const CAnomalyScore::CNormalizer *
            personNormalizer(const std::string &partitionFieldName,
                             const std::string &personFieldName) const;

        //! Get a leaf normalizer.
        //!
        //! \note Returns NULL if there isn't a matching one.
        const CAnomalyScore::CNormalizer *
            leafNormalizer(const std::string &partitionFieldName,
                           const std::string &personFieldName,
                           const std::string &functionName,
                           const std::string &valueFieldName) const;

    private:
        //! Get the normalizer corresponding to \p cue if they exist
        //! and return NULL if it doesn't have an appropriate prefix.
        //! Also, extract the hash value.
        bool parseCue(const std::string &cue,
                      TWordNormalizerPrVec *&normalizers,
                      TDictionary::TUInt64Array &hashArray);

        //! Get the persistence cue for the root normalizer.
        static const std::string &bucketCue();

        //! Get the persistence cue for a influencer bucket normalizer.
        static std::string influencerBucketCue(const TWord &word);

        //! Get the persistence cue for an influencer normalizer.
        static std::string influencerCue(const TWord &word);

        //! Get the persistence cue for a partition normalizer.
        static std::string partitionCue(const TWord &word);

        //! Get the persistence cue for a person normalizer.
        static std::string personCue(const TWord &word);

        //! Get the persistence cue for a leaf normalizer.
        static std::string leafCue(const TWord &word);

    private:
        //! The jobs that the normalizer will perform when invoked
        //! can be: update, normalize or update + normalize.
        EJob m_Job;

        //! The model configuration file.
        const CAnomalyDetectorModelConfig &m_ModelConfig;

        //! Whether the last update of the quantiles has caused a big change.
        bool m_HasLastUpdateCausedBigChange;

};

}
}

#endif // INCLUDED_ml_model_CHierarchicalResultsNormalizer_h
