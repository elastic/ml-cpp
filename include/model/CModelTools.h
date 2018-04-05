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

#ifndef INCLUDED_ml_model_CModelTools_h
#define INCLUDED_ml_model_CModelTools_h

#include <core/CAllocationStrategy.h>
#include <core/CLogger.h>
#include <core/CSmallVector.h>

#include <maths/CMultivariatePrior.h>
#include <maths/COrderings.h>
#include <maths/CPRNG.h>
#include <maths/CPrior.h>
#include <maths/ProbabilityAggregators.h>

#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <boost/container/flat_map.hpp>
#include <boost/optional.hpp>
#include <boost/unordered_map.hpp>
#include <boost/variant.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace ml {
namespace maths {
class CModel;
class CMultinomialConjugate;
}
namespace model {
class CSample;
struct SModelParams;

//! \brief A collection of utility functionality for the CModel hierarchy.
//!
//! DESCRIPTION:\n
//! A collection of utility functions primarily intended for use by the
//! CModel class hierarchy.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This class is really just a proxy for a namespace, but a class has
//! been intentionally used to provide a single point for the declaration
//! and definition of utility functions within the model library. As such
//! all member functions should be static and it should be state-less.
//! If your functionality doesn't fit this pattern just make it a nested
//! class.
class MODEL_EXPORT CModelTools {
public:
    using TDoubleVec = std::vector<double>;
    using TDouble2Vec = core::CSmallVector<double, 2>;
    using TDouble2Vec1Vec = core::CSmallVector<TDouble2Vec, 1>;
    using TTimeDouble2VecPr = std::pair<core_t::TTime, TDouble2Vec>;
    using TSizeSizePr = std::pair<std::size_t, std::size_t>;
    using TStoredStringPtrStoredStringPtrPr = std::pair<core::CStoredStringPtr, core::CStoredStringPtr>;
    using TSampleVec = std::vector<CSample>;

    //! \brief De-duplicates nearly equal values.
    class MODEL_EXPORT CFuzzyDeduplicate {
    public:
        //! Add a value.
        void add(TDouble2Vec value);
        //! Compute quantization epsilons.
        void computeEpsilons(core_t::TTime bucketLength, std::size_t desiredNumberSamples);
        //! Check if we've got a near duplicate of \p value.
        std::size_t duplicate(core_t::TTime time, TDouble2Vec value);

    private:
        using TDouble2VecVec = std::vector<TDouble2Vec>;
        struct MODEL_EXPORT SDuplicateValueHash {
            std::size_t operator()(const TTimeDouble2VecPr& value) const;
        };
        using TTimeDouble2VecPrSizeUMap = boost::unordered_map<TTimeDouble2VecPr, std::size_t, SDuplicateValueHash>;

    private:
        //! Quantize \p value.
        TDouble2Vec quantize(TDouble2Vec value) const;
        //! Quantize \p time.
        core_t::TTime quantize(core_t::TTime time) const;

    private:
        //! If false then quantization is disabled.
        bool m_Quantize = true;
        //! The count of values added.
        std::size_t m_Count = 0;
        //! The time quantization interval.
        core_t::TTime m_TimeEps = 0;
        //! The value quantization interval.
        TDouble2Vec m_ValueEps;
        //! A random number generator used to sample added values.
        maths::CPRNG::CXorOShiro128Plus m_Rng;
        //! A random sample of the added values.
        TDouble2VecVec m_RandomSample;
        //! A collection of quantized values and their unique ids.
        TTimeDouble2VecPrSizeUMap m_QuantizedValues;
    };

    //! \brief Hashes a string pointer pair.
    struct MODEL_EXPORT SStoredStringPtrStoredStringPtrPrHash {
        std::size_t operator()(const TStoredStringPtrStoredStringPtrPr& target) const {
            return static_cast<std::size_t>(core::CHashing::hashCombine(static_cast<uint64_t>(s_Hasher(*target.first)),
                                                                        static_cast<uint64_t>(s_Hasher(*target.second))));
        }
        core::CHashing::CMurmurHash2String s_Hasher;
    };

    //! \brief Compares two string pointer pairs.
    struct MODEL_EXPORT SStoredStringPtrStoredStringPtrPrEqual {
        std::size_t operator()(const TStoredStringPtrStoredStringPtrPr& lhs, const TStoredStringPtrStoredStringPtrPr& rhs) const {
            return *lhs.first == *rhs.first && *lhs.second == *rhs.second;
        }
    };

    //! \brief Manages the aggregation of probabilities.
    //!
    //! DESCRIPTION:\n
    //! This allows one to register either one of or both the joint
    //! probability and extreme aggregation styles. The resulting
    //! aggregate probability is the minimum of the aggregates of
    //! the probabilities added so far for any of the registered
    //! aggregation styles.
    class MODEL_EXPORT CProbabilityAggregator {
    public:
        using TAggregator = boost::variant<maths::CJointProbabilityOfLessLikelySamples, maths::CProbabilityOfExtremeSample>;
        using TAggregatorDoublePr = std::pair<TAggregator, double>;
        using TAggregatorDoublePrVec = std::vector<TAggregatorDoublePr>;

        enum EStyle { E_Sum, E_Min };

    public:
        CProbabilityAggregator(EStyle style);

        //! Check if any probabilities have been added.
        bool empty() const;

        //! Add an aggregation style \p aggregator with weight \p weight.
        void add(const TAggregator& aggregator, double weight = 1.0);

        //! Add \p probability.
        void add(double probability, double weight = 1.0);

        //! Calculate the probability if possible.
        bool calculate(double& result) const;

    private:
        //! The style of aggregation to use.
        EStyle m_Style;

        //! The total weight of all samples.
        double m_TotalWeight;

        //! The collection of objects for computing "joint" probabilities.
        TAggregatorDoublePrVec m_Aggregators;
    };

    using TStoredStringPtrStoredStringPtrPrProbabilityAggregatorUMap = boost::unordered_map<TStoredStringPtrStoredStringPtrPr,
                                                                                            CProbabilityAggregator,
                                                                                            SStoredStringPtrStoredStringPtrPrHash,
                                                                                            SStoredStringPtrStoredStringPtrPrEqual>;

    //! Wraps up the calculation of less likely probabilities for a
    //! multinomial distribution.
    //!
    //! DESCRIPTION:\n
    //! This caches the probabilities for each category, in the multinomial
    //! distribution, since they can't be computed independently and for
    //! a large number of categories it is very wasteful to repeatedly
    //! compute them all.
    class MODEL_EXPORT CCategoryProbabilityCache {
    public:
        CCategoryProbabilityCache();
        CCategoryProbabilityCache(const maths::CMultinomialConjugate& prior);

        //! Calculate the probability of less likely categories than
        //! \p attribute.
        bool lookup(std::size_t category, double& result) const;

        //! Get the memory usage of the component
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory usage of the component
        std::size_t memoryUsage() const;

    private:
        //! The prior.
        const maths::CMultinomialConjugate* m_Prior;
        //! The cached probabilities.
        mutable TDoubleVec m_Cache;
        //! The smallest possible category probability.
        mutable double m_SmallestProbability;
    };

    //! \brief A cache of the probability calculation to use in cases that many
    //! probabilities are being computed from the same model.
    //!
    //! DESCRIPTION:\n
    //! This caches the probabilities for each feature and attribute since they
    //! can be expensive to compute and for large populations we can repeatedly
    //! calculate probabilities for the same model and similar parameters.
    //!
    //! This bounds the maximum relative error it'll introduce by only interpolating
    //! an interval if the difference in the probability at its end points satisfy
    //! \f$|P(b) - P(a)| < threshold \times min(P(A), P(b))\f$.
    class MODEL_EXPORT CProbabilityCache {
    public:
        using TTail2Vec = core::CSmallVector<maths_t::ETail, 2>;
        using TSize1Vec = core::CSmallVector<std::size_t, 1>;

    public:
        explicit CProbabilityCache(double maximumError);

        //! Clear the cache.
        void clear();

        //! Maybe add the modes of \p model.
        void addModes(model_t::EFeature feature, std::size_t id, const maths::CModel& model);

        //! Add a new ("value", "probability") result.
        //!
        //! \param[in] id The unique model identifier.
        //! \param[in] value The value.
        //! \param[in] probability The result of running the probability
        //! calculation for \p value.
        //! \param[in] tail The tail of \p value.
        //! \param[in] conditional True if the probability depends on the
        //! correlation structure and false otherwise.
        //! \param[in] mostAnomalousCorrelate The identifier of the most
        //! anomalous correlate (or empty if there isn't one).
        void addProbability(model_t::EFeature feature,
                            std::size_t id,
                            const TDouble2Vec1Vec& value,
                            double probability,
                            const TTail2Vec& tail,
                            bool conditional,
                            const TSize1Vec& mostAnomalousCorrelate);

        //! Try to lookup the probability of \p value in cache.
        //!
        //! \param[in] id The unique model identifier.
        //! \param[in] value The value.
        //! \param[out] probability An estimate of the probability
        //! corresponding to \p likelihood.
        //! \param[out] tail The tail of \p value.
        //! \param[out] conditional True if the probability depends on the
        //! correlation structure and false otherwise.
        //! \param[out] mostAnomalousCorrelate The identifier of the most
        //! anomalous correlate (or empty if there isn't one).
        //! \return True if the probability can be estimated within an
        //! acceptable error and false otherwise.
        bool lookup(model_t::EFeature feature,
                    std::size_t id,
                    const TDouble2Vec1Vec& value,
                    double& probability,
                    TTail2Vec& tail,
                    bool& conditional,
                    TSize1Vec& mostAnomalousCorrelate) const;

    private:
        using TDouble1Vec = core::CSmallVector<double, 1>;

        //! \brief A cache of the result of a probability calculation.
        struct MODEL_EXPORT SProbability {
            //! The value's probability.
            double s_Probability;
            //! The tail the value is in.
            TTail2Vec s_Tail;
            //! True if the probability depends on correlation structure.
            bool s_Conditional;
            //! The pairwise correlation with lowest probability.
            TSize1Vec s_MostAnomalousCorrelate;
        };

        using TDoubleProbabilityFMap = boost::container::flat_map<double, SProbability>;

        //! \brief A cache of all the results of a probability calculation
        //! for a specific model.
        struct MODEL_EXPORT SProbabilityCache {
            //! The modes of the model's residual distribution for which
            //! this is caching the result of the probability calculation.
            TDouble1Vec s_Modes;

            //! The probability cache.
            TDoubleProbabilityFMap s_Probabilities;
        };

        using TFeatureSizePr = std::pair<model_t::EFeature, std::size_t>;
        using TFeatureSizePrProbabilityCacheUMap = boost::unordered_map<TFeatureSizePr, SProbabilityCache>;

    private:
        //! The maximum relative error we'll tolerate in the probability.
        double m_MaximumError;

        //! The univariate probability cache.
        TFeatureSizePrProbabilityCacheUMap m_Caches;
    };
};
}
}

#endif // INCLUDED_ml_model_CModelTools_h
