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

#ifndef INCLUDED_ml_maths_CMultivariateMultimodalPrior_h
#define INCLUDED_ml_maths_CMultivariateMultimodalPrior_h

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStringUtils.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/CChecksum.h>
#include <maths/CClusterer.h>
#include <maths/CClustererStateSerialiser.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraTools.h>
#include <maths/CMathsFuncs.h>
#include <maths/CMathsFuncsForMatrixAndVectorTypes.h>
#include <maths/CMultimodalPrior.h>
#include <maths/CMultimodalPriorMode.h>
#include <maths/CMultivariatePrior.h>
#include <maths/CPriorStateSerialiser.h>
#include <maths/CRestoreParams.h>
#include <maths/CSetTools.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <boost/bind.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/ref.hpp>
#include <boost/shared_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace ml {
namespace maths {
namespace multivariate_multimodal_prior_detail {

using TSizeDoublePr = std::pair<size_t, double>;
using TSizeDoublePr3Vec = core::CSmallVector<TSizeDoublePr, 3>;
using TPriorPtr = boost::shared_ptr<CMultivariatePrior>;
using TDouble10Vec1Vec = CMultivariatePrior::TDouble10Vec1Vec;
using TDouble10Vec4Vec1Vec = CMultivariatePrior::TDouble10Vec4Vec1Vec;
using TMode = SMultimodalPriorMode<boost::shared_ptr<CMultivariatePrior>>;
using TModeVec = std::vector<TMode>;

//! Implementation of a sample joint log marginal likelihood calculation.
MATHS_EXPORT
maths_t::EFloatingPointErrorStatus
jointLogMarginalLikelihood(const TModeVec& modes,
                           const maths_t::TWeightStyleVec& weightStyles,
                           const TDouble10Vec1Vec& sample,
                           const TDouble10Vec4Vec1Vec& weights,
                           TSizeDoublePr3Vec& modeLogLikelihoods,
                           double& result);

//! Implementation of marginal likelihood sample.
MATHS_EXPORT
void sampleMarginalLikelihood(const TModeVec& modes,
                              std::size_t numberSamples,
                              TDouble10Vec1Vec& samples);

//! Implementation of mode printing.
MATHS_EXPORT
void print(const TModeVec& modes, const std::string& separator, std::string& result);

//! Implementation of mode merge callback.
MATHS_EXPORT
void modeMergeCallback(std::size_t dimension,
                       TModeVec& modes,
                       const TPriorPtr& seedPrior,
                       std::size_t numberSamples,
                       std::size_t leftMergeIndex,
                       std::size_t rightMergeIndex,
                       std::size_t targetIndex);

//! Implementation of a full debug dump of the mode weights.
MATHS_EXPORT
std::string debugWeights(const TModeVec& modes);

} // multivariate_multimodal_prior_detail::

//! \brief Implementation for a multimodal multivariate prior distribution.
//!
//! DESCRIPTION:\n
//! This is used to model a stationary process for which we expect there to be
//! distinct modes, which can be modeled accurately by any of our basic single
//! mode multivariate distributions.
//!
//! A separate mechanism is provided to identify the clusters in the data
//! corresponding to distinct modes so that different methods for identifying
//! clusters can be used.
//!
//! All prior distributions implement a process whereby they relax back to the
//! non-informative over some period without update (see propagateForwardsByTime).
//! The rate at which they relax is controlled by the decay factor supplied to the
//! constructor.
//!
//! IMPORTANT: Other than for testing, this class should not be constructed
//! directly. Creation of objects is managed by CMultivariateMultimodalPriorFactory.
//!
//! IMPLEMENTATION DECISIONS:\n
//! All derived priors are templated on their size. In practice, it is very hard
//! to estimate densities in more than a small number of dimensions. We achieve
//! this by partitioning the matrix into block diagonal form with one prior used
//! to model each block. Therefore, we never need priors for more than a small
//! number of dimensions. This means that we template them on their size so we
//! can use stack (mathematical) vectors and matrices.
//!
//! All priors are derived from CMultivariatePrior which defines the contract that
//! is used by composite priors. This allows us to select the most appropriate model
//! for the data when using one-of-n composition (see CMultivariateOneOfNPrior).
//! From a design point of view this is the composite pattern.
template<std::size_t N>
class CMultivariateMultimodalPrior : public CMultivariatePrior {
public:
    using TDouble5Vec = core::CSmallVector<double, 5>;
    using TPoint = CVectorNx1<double, N>;
    using TFloatPoint = CVectorNx1<CFloatStorage, N>;
    using TPointVec = std::vector<TPoint>;
    using TPoint4Vec = core::CSmallVector<TPoint, 4>;
    using TMeanAccumulator = typename CBasicStatistics::SSampleMean<TPoint>::TAccumulator;
    using TMatrix = CSymmetricMatrixNxN<double, N>;
    using TMatrixVec = std::vector<TMatrix>;
    using TClusterer = CClusterer<TFloatPoint>;
    using TClustererPtr = boost::shared_ptr<TClusterer>;
    using TPriorPtrVec = std::vector<TPriorPtr>;
    using TWeights = CConstantWeights;

    // Lift all overloads of into scope.
    //{
    using CMultivariatePrior::addSamples;
    using CMultivariatePrior::dataType;
    using CMultivariatePrior::decayRate;
    using CMultivariatePrior::print;
    //}

public:
    //! \name Life-Cycle
    //@{
    //! Create a new (empty) multimodal prior.
    CMultivariateMultimodalPrior(maths_t::EDataType dataType,
                                 const TClusterer& clusterer,
                                 const CMultivariatePrior& seedPrior,
                                 double decayRate = 0.0)
        : CMultivariatePrior(dataType, decayRate),
          m_Clusterer(clusterer.clone()), m_SeedPrior(seedPrior.clone()) {
        // Register the split and merge callbacks.
        m_Clusterer->splitFunc(CModeSplitCallback(*this));
        m_Clusterer->mergeFunc(CModeMergeCallback(*this));
    }

    //! Create from a collection of priors.
    //!
    //! \note The priors are shallow copied.
    //! \note This constructor doesn't support subsequent update of the prior.
    CMultivariateMultimodalPrior(maths_t::EDataType dataType, TPriorPtrVec& priors)
        : CMultivariatePrior(dataType, 0.0) {
        m_Modes.reserve(priors.size());
        for (std::size_t i = 0u; i < priors.size(); ++i) {
            m_Modes.emplace_back(i, priors[i]);
        }
    }

    //! Construct from part of a state document.
    CMultivariateMultimodalPrior(const SDistributionRestoreParams& params,
                                 core::CStateRestoreTraverser& traverser)
        : CMultivariatePrior(params.s_DataType, params.s_DecayRate) {
        traverser.traverseSubLevel(boost::bind(&CMultivariateMultimodalPrior::acceptRestoreTraverser,
                                               this, boost::cref(params), _1));
    }

    //! Implements value semantics for copy construction.
    CMultivariateMultimodalPrior(const CMultivariateMultimodalPrior& other)
        : CMultivariatePrior(other.dataType(), other.decayRate()),
          m_Clusterer(other.m_Clusterer->clone()),
          m_SeedPrior(other.m_SeedPrior->clone()) {
        // Register the split and merge callbacks.
        m_Clusterer->splitFunc(CModeSplitCallback(*this));
        m_Clusterer->mergeFunc(CModeMergeCallback(*this));

        // Clone all the modes up front so we can implement strong exception safety.
        TModeVec modes;
        modes.reserve(other.m_Modes.size());
        for (const auto& mode : other.m_Modes) {
            modes.emplace_back(mode.s_Index, TPriorPtr(mode.s_Prior->clone()));
        }
        m_Modes.swap(modes);

        this->addSamples(other.numberSamples());
    }

    //! Implements value semantics for assignment.
    //!
    //! \param[in] rhs The multimodal model to copy.
    //! \return The newly copied model.
    CMultivariateMultimodalPrior& operator=(const CMultivariateMultimodalPrior& rhs) {
        if (this != &rhs) {
            CMultivariateMultimodalPrior copy(rhs);
            this->swap(copy);
        }
        return *this;
    }

    //! An efficient swap of the contents of this and \p other.
    void swap(CMultivariateMultimodalPrior& other) {
        this->CMultivariatePrior::swap(other);

        std::swap(m_Clusterer, other.m_Clusterer);
        // The call backs for split and merge should point to the
        // appropriate priors (we don't swap the "this" pointers
        // after all). So we need to refresh them after swapping.
        m_Clusterer->splitFunc(CModeSplitCallback(*this));
        m_Clusterer->mergeFunc(CModeMergeCallback(*this));
        other.m_Clusterer->splitFunc(CModeSplitCallback(other));
        other.m_Clusterer->mergeFunc(CModeMergeCallback(other));

        std::swap(m_SeedPrior, other.m_SeedPrior);
        m_Modes.swap(other.m_Modes);
    }
    //@}

    //! \name Prior Contract
    //@{
    //! Create a copy of the prior.
    //!
    //! \warning Caller owns returned object.
    virtual CMultivariatePrior* clone() const {
        return new CMultivariateMultimodalPrior(*this);
    }

    //! Get the dimension of the prior.
    virtual std::size_t dimension() const { return N; }

    //! Set the data type.
    virtual void dataType(maths_t::EDataType value) {
        this->CMultivariatePrior::dataType(value);
        m_Clusterer->dataType(value);
        for (const auto& mode : m_Modes) {
            mode.s_Prior->dataType(value);
        }
    }

    //! Set the rate at which the prior returns to non-informative.
    virtual void decayRate(double value) {
        this->CMultivariatePrior::decayRate(value);
        m_Clusterer->decayRate(this->decayRate());
        for (const auto& mode : m_Modes) {
            mode.s_Prior->decayRate(this->decayRate());
        }
        m_SeedPrior->decayRate(this->decayRate());
    }

    //! Reset the prior to non-informative.
    virtual void setToNonInformative(double /*offset*/, double decayRate) {
        m_Clusterer->clear();
        m_Modes.clear();
        this->decayRate(decayRate);
        this->numberSamples(0.0);
    }

    //! For priors with non-negative support this adjusts the offset used
    //! to extend the support to handle negative samples.
    //!
    //! \param[in] weightStyles Controls the interpretation of the weight(s)
    //! that are associated with each sample. See maths_t::ESampleWeightStyle
    //! for more details.
    //! \param[in] samples The samples from which to determine the offset.
    //! \param[in] weights The weights of each sample in \p samples.
    virtual void adjustOffset(const TWeightStyleVec& weightStyles,
                              const TDouble10Vec1Vec& samples,
                              const TDouble10Vec4Vec1Vec& weights) {
        // This has to adjust offsets for its modes because it must be
        // possible to call jointLogMarginalLikelihood before the samples
        // have been added to the prior in order for model selection to
        // work.
        for (const auto& mode : m_Modes) {
            mode.s_Prior->adjustOffset(weightStyles, samples, weights);
        }
    }

    //! Update the prior with a collection of independent samples from the
    //! process.
    //!
    //! \param[in] weightStyles_ Controls the interpretation of the weight(s)
    //! that are associated with each sample. See maths_t::ESampleWeightStyle
    //! for more details.
    //! \param[in] samples A collection of samples of the process.
    //! \param[in] weights The weights of each sample in \p samples.
    virtual void addSamples(const TWeightStyleVec& weightStyles_,
                            const TDouble10Vec1Vec& samples,
                            const TDouble10Vec4Vec1Vec& weights) {
        if (samples.empty()) {
            return;
        }
        if (!this->check(samples, weights)) {
            return;
        }

        // See CMultimodalPrior::addSamples for discussion.

        using TSizeDoublePr2Vec = core::CSmallVector<TSizeDoublePr, 2>;

        // Declared outside the loop to minimize the number of times it
        // is initialized.
        TWeightStyleVec weightStyles(weightStyles_);
        TDouble10Vec1Vec sample(1);
        TDouble10Vec4Vec1Vec weight(1);
        TSizeDoublePr2Vec clusters;

        std::size_t indices[maths_t::NUMBER_WEIGHT_STYLES];
        std::size_t missing = weightStyles.size() + 1;
        std::fill_n(indices, maths_t::NUMBER_WEIGHT_STYLES, missing);
        for (std::size_t i = 0u; i < weightStyles.size(); ++i) {
            indices[weightStyles[i]] = i;
        }
        std::size_t seasonal = indices[maths_t::E_SampleSeasonalVarianceScaleWeight];
        std::size_t count = indices[maths_t::E_SampleCountWeight];
        std::size_t winsorisation = indices[maths_t::E_SampleWinsorisationWeight];
        if (count == missing) {
            count = weightStyles.size();
            weightStyles.push_back(maths_t::E_SampleCountWeight);
        }

        try {
            bool hasSeasonalScale = !this->isNonInformative() && seasonal != missing;

            TPoint mean = hasSeasonalScale ? this->mean() : TPoint(0.0);

            for (std::size_t i = 0u; i < samples.size(); ++i) {
                TPoint x(samples[i]);
                if (!CMathsFuncs::isFinite(x)) {
                    LOG_ERROR(<< "Discarding " << x);
                    continue;
                }

                if (hasSeasonalScale) {
                    TPoint seasonalScale = sqrt(TPoint(maths_t::seasonalVarianceScale(
                        N, weightStyles_, weights[i])));
                    x = mean + (x - mean) / seasonalScale;
                }

                sample[0] = x.template toVector<TDouble10Vec>();
                weight[0] = weights[i];
                weight[0].resize(weightStyles.size(), TDouble10Vec(N, 1.0));
                if (seasonal != missing) {
                    weight[0][seasonal].assign(N, 1.0);
                }

                double smallestCountWeight = this->smallest(weight[0][count]);
                clusters.clear();
                m_Clusterer->add(x, clusters, smallestCountWeight);

                double Z = std::accumulate(m_Modes.begin(), m_Modes.end(), smallestCountWeight,
                                           [](double sum, const TMode& mode) {
                                               return sum + mode.weight();
                                           });

                double n = 0.0;
                for (const auto& cluster : clusters) {
                    auto k = std::find_if(m_Modes.begin(), m_Modes.end(),
                                          CSetTools::CIndexInSet(cluster.first));
                    if (k == m_Modes.end()) {
                        LOG_TRACE(<< "Creating mode with index " << cluster.first);
                        m_Modes.emplace_back(cluster.first, m_SeedPrior);
                        k = m_Modes.end() - 1;
                    }
                    weight[0][count].assign(N, cluster.second);
                    if (winsorisation != missing) {
                        TDouble10Vec& ww = weight[0][winsorisation];
                        double f = (k->weight() + cluster.second) / Z;
                        for (auto& w : ww) {
                            w = std::max(1.0 - (1.0 - w) / f, w * f);
                        }
                    }
                    k->s_Prior->addSamples(weightStyles, sample, weight);
                    n += this->smallest(maths_t::countForUpdate(N, weightStyles, weight[0]));
                }
                this->addSamples(n);
            }
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Failed to update likelihood: " << e.what());
        }
    }

    //! Update the prior for the specified elapsed time.
    virtual void propagateForwardsByTime(double time) {
        if (!CMathsFuncs::isFinite(time) || time < 0.0) {
            LOG_ERROR(<< "Bad propagation time " << time);
            return;
        }

        if (this->isNonInformative()) {
            // Nothing to be done.
            return;
        }

        // We want to hold the probabilities constant. Since the i'th
        // probability:
        //   p(i) = w(i) / Sum_j{ w(j) }
        //
        // where w(i) is its weight we can achieve this by multiplying
        // all weights by some factor f in the range [0, 1].

        if (!this->isForForecasting()) {
            m_Clusterer->propagateForwardsByTime(time);
        }

        for (const auto& mode : m_Modes) {
            mode.s_Prior->propagateForwardsByTime(time);
        }

        this->numberSamples(this->numberSamples() *
                            std::exp(-this->scaledDecayRate() * time));
        LOG_TRACE(<< "numberSamples = " << this->numberSamples());
    }

    //! Compute the univariate prior marginalizing over the variables
    //! \p marginalize and conditioning on the variables \p condition.
    //!
    //! \param[in] marginalize The variables to marginalize out.
    //! \param[in] condition The variables to condition on.
    //! \warning The caller owns the result.
    //! \note The variables are passed by the index of their dimension
    //! which must therefore be in range.
    //! \note The caller must specify dimension - 1 variables between
    //! \p marginalize and \p condition so the resulting distribution
    //! is univariate.
    virtual TUnivariatePriorPtrDoublePr
    univariate(const TSize10Vec& marginalize, const TSizeDoublePr10Vec& condition) const {
        std::size_t n = m_Modes.size();

        CMultimodalPrior::TPriorPtrVec modes;
        TDouble5Vec weights;
        CBasicStatistics::COrderStatisticsStack<double, 1, std::greater<double>> maxWeight;
        modes.reserve(n);
        weights.reserve(n);

        for (const auto& mode : m_Modes) {
            TUnivariatePriorPtrDoublePr prior(mode.s_Prior->univariate(marginalize, condition));
            if (prior.first == nullptr) {
                return TUnivariatePriorPtrDoublePr();
            }
            if (prior.first->isNonInformative()) {
                continue;
            }
            modes.push_back(prior.first);
            weights.push_back(prior.second);
            maxWeight.add(weights.back());
        }

        double Z = 0.0;
        for (auto& weight : weights) {
            weight = std::exp(weight - maxWeight[0]);
            Z += weight;
        }
        for (std::size_t i = 0u; i < weights.size(); ++i) {
            modes[i]->numberSamples(weights[i] / Z * modes[i]->numberSamples());
        }

        return {TUnivariatePriorPtr(new CMultimodalPrior(this->dataType(),
                                                         this->decayRate(), modes)),
                Z > 0.0 ? maxWeight[0] + std::log(Z) : 0.0};
    }

    //! Compute the bivariate prior marginalizing over the variables
    //! \p marginalize and conditioning on the variables \p condition.
    //!
    //! \param[in] marginalize The variables to marginalize out.
    //! \param[in] condition The variables to condition on.
    //! \warning The caller owns the result.
    //! \note The variables are passed by the index of their dimension
    //! which must therefore be in range.
    //! \note It is assumed that the variables are in sorted order.
    //! \note The caller must specify dimension - 2 variables between
    //! \p marginalize and \p condition so the resulting distribution
    //! is univariate.
    virtual TPriorPtrDoublePr bivariate(const TSize10Vec& marginalize,
                                        const TSizeDoublePr10Vec& condition) const {
        if (N == 2) {
            return TPriorPtrDoublePr(TPriorPtr(this->clone()), 0.0);
        }

        std::size_t n = m_Modes.size();

        TPriorPtrVec modes;
        TDouble5Vec weights;
        modes.reserve(n);
        weights.reserve(n);
        CBasicStatistics::COrderStatisticsStack<double, 1, std::greater<double>> maxWeight;

        for (const auto& mode : m_Modes) {
            TPriorPtrDoublePr prior(mode.s_Prior->bivariate(marginalize, condition));
            if (prior.first == nullptr) {
                return TPriorPtrDoublePr();
            }
            if (prior.first->isNonInformative()) {
                continue;
            }
            modes.push_back(prior.first);
            weights.push_back(prior.second);
            maxWeight.add(weights.back());
        }

        double Z = 0.0;
        for (auto& weight : weights) {
            weight = std::exp(weight - maxWeight[0]);
            Z += weight;
        }
        for (std::size_t i = 0u; i < weights.size(); ++i) {
            modes[i]->numberSamples(weights[i] / Z * modes[i]->numberSamples());
        }

        return {TPriorPtr(new CMultivariateMultimodalPrior<2>(this->dataType(), modes)),
                Z > 0.0 ? maxWeight[0] + std::log(Z) : 0.0};
    }

    //! Get the support for the marginal likelihood function.
    virtual TDouble10VecDouble10VecPr marginalLikelihoodSupport() const {
        if (m_Modes.size() == 0) {
            return {TPoint::smallest().template toVector<TDouble10Vec>(),
                    TPoint::largest().template toVector<TDouble10Vec>()};
        }
        if (m_Modes.size() == 1) {
            return m_Modes[0].s_Prior->marginalLikelihoodSupport();
        }

        TPoint lower = TPoint::largest();
        TPoint upper = TPoint::smallest();

        // We define this is as the union of the mode supports.
        for (const auto& mode : m_Modes) {
            TDouble10VecDouble10VecPr s = mode.s_Prior->marginalLikelihoodSupport();
            lower = min(lower, TPoint(s.first));
            upper = max(upper, TPoint(s.second));
        }

        return {lower.template toVector<TDouble10Vec>(),
                upper.template toVector<TDouble10Vec>()};
    }

    //! Get the mean of the marginal likelihood function.
    virtual TDouble10Vec marginalLikelihoodMean() const {
        if (m_Modes.size() == 0) {
            return TDouble10Vec(N, 0.0);
        }
        if (m_Modes.size() == 1) {
            return m_Modes[0].s_Prior->marginalLikelihoodMean();
        }

        return this->mean().template toVector<TDouble10Vec>();
    }

    //! Get the nearest mean of the multimodal prior marginal likelihood,
    //! otherwise the marginal likelihood mean.
    virtual TDouble10Vec nearestMarginalLikelihoodMean(const TDouble10Vec& value_) const {
        if (m_Modes.empty()) {
            return TDouble10Vec(N, 0.0);
        }
        if (m_Modes.size() == 1) {
            return m_Modes[0].s_Prior->marginalLikelihoodMean();
        }

        TPoint value(value_);

        TPoint result(m_Modes[0].s_Prior->marginalLikelihoodMean());
        double distance = (value - result).euclidean();
        for (std::size_t i = 1u; i < m_Modes.size(); ++i) {
            TPoint mean(m_Modes[i].s_Prior->marginalLikelihoodMean());
            double di = (value - mean).euclidean();
            if (di < distance) {
                distance = di;
                result = mean;
            }
        }

        return result.template toVector<TDouble10Vec>();
    }

    //! Get the mode of the marginal likelihood function.
    virtual TDouble10Vec marginalLikelihoodMode(const TWeightStyleVec& weightStyles,
                                                const TDouble10Vec4Vec& weight) const {
        if (m_Modes.size() == 0) {
            return TDouble10Vec(N, 0.0);
        }
        if (m_Modes.size() == 1) {
            return m_Modes[0].s_Prior->marginalLikelihoodMode(weightStyles, weight);
        }

        using TMaxAccumulator =
            CBasicStatistics::COrderStatisticsStack<double, 1, std::greater<double>>;

        // We'll approximate this as the mode with the maximum likelihood.
        TPoint result(0.0);

        TPoint seasonalScale(1.0);
        TDouble10Vec4Vec1Vec weight_(1, TDouble10Vec4Vec(1));
        try {
            seasonalScale =
                sqrt(TPoint(maths_t::seasonalVarianceScale(N, weightStyles, weight)));
            weight_[0][0] = maths_t::countVarianceScale(N, weightStyles, weight);
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Failed to get variance scale " << e.what());
        }

        // Declared outside the loop to minimize number of times it is created.
        TDouble10Vec1Vec mode(1);

        TMaxAccumulator modeLikelihood;
        for (const auto& mode_ : m_Modes) {
            double w = mode_.weight();
            const TPriorPtr& prior = mode_.s_Prior;
            mode[0] = prior->marginalLikelihoodMode(TWeights::COUNT_VARIANCE, weight_[0]);
            double likelihood;
            if (prior->jointLogMarginalLikelihood(TWeights::COUNT_VARIANCE,
                                                  mode, weight_, likelihood) &
                maths_t::E_FpAllErrors) {
                continue;
            }
            if (modeLikelihood.add(std::log(w) + likelihood)) {
                result = TPoint(mode[0]);
            }
        }

        TPoint mean = this->mean();
        result = mean + seasonalScale * (result - mean);
        return result.template toVector<TDouble10Vec>();
    }

    //! Get the local maxima of the marginal likelihood functions.
    TDouble10Vec1Vec marginalLikelihoodModes(const TWeightStyleVec& weightStyles,
                                             const TDouble10Vec4Vec& weights) const {
        TDouble10Vec1Vec result;
        result.reserve(m_Modes.size());
        for (const auto& mode : m_Modes) {
            result.push_back(mode.s_Prior->marginalLikelihoodMode(weightStyles, weights));
        }
        return result;
    }

    //! Get the covariance matrix for the marginal likelihood.
    virtual TDouble10Vec10Vec marginalLikelihoodCovariance() const {
        if (m_Modes.size() == 0) {
            return TPoint::largest().diagonal().template toVectors<TDouble10Vec10Vec>();
        }
        if (m_Modes.size() == 1) {
            return m_Modes[0].s_Prior->marginalLikelihoodCovariance();
        }
        return this->covarianceMatrix().template toVectors<TDouble10Vec10Vec>();
    }

    //! Get the diagonal of the covariance matrix for the marginal likelihood.
    virtual TDouble10Vec marginalLikelihoodVariances() const {
        if (m_Modes.size() == 0) {
            return TPoint::largest().template toVector<TDouble10Vec>();
        }
        if (m_Modes.size() == 1) {
            return m_Modes[0].s_Prior->marginalLikelihoodVariances();
        }
        return this->covarianceMatrix().template diagonal<TDouble10Vec>();
    }

    //! Calculate the log marginal likelihood function, integrating over the
    //! prior density function.
    //!
    //! \param[in] weightStyles Controls the interpretation of the weight(s)
    //! that are associated with each sample. See maths_t::ESampleWeightStyle
    //! for more details.
    //! \param[in] samples A collection of samples of the process.
    //! \param[in] weights The weights of each sample in \p samples.
    //! \param[out] result Filled in with the joint likelihood of \p samples.
    virtual maths_t::EFloatingPointErrorStatus
    jointLogMarginalLikelihood(const TWeightStyleVec& weightStyles,
                               const TDouble10Vec1Vec& samples,
                               const TDouble10Vec4Vec1Vec& weights,
                               double& result) const {
        result = 0.0;

        if (samples.empty()) {
            LOG_ERROR(<< "Can't compute likelihood for empty sample set");
            return maths_t::E_FpFailed;
        }
        if (!this->check(samples, weights)) {
            return maths_t::E_FpFailed;
        }

        if (this->isNonInformative()) {
            // The non-informative likelihood is improper and effectively
            // zero everywhere. We use minus max double because
            // log(0) = HUGE_VALUE, which causes problems for Windows.
            // Calling code is notified when the calculation overflows
            // and should avoid taking the exponential since this will
            // underflow and pollute the floating point environment. This
            // may cause issues for some library function implementations
            // (see fe*exceptflag for more details).
            result = boost::numeric::bounds<double>::lowest();
            return maths_t::E_FpOverflowed;
        }

        if (m_Modes.size() == 1) {
            // Apply a small penalty to kill off this model if the data are
            // single mode.
            maths_t::EFloatingPointErrorStatus status = m_Modes[0].s_Prior->jointLogMarginalLikelihood(
                weightStyles, samples, weights, result);
            result -= 10.0 * this->decayRate();
            return status;
        }

        // See CMultimodalPrior::jointLogMarginalLikelihood for discussion.

        namespace detail = multivariate_multimodal_prior_detail;

        // Declared outside the loop to minimize number of times it is created.
        TDouble10Vec1Vec sample(1);
        detail::TSizeDoublePr3Vec modeLogLikelihoods;
        modeLogLikelihoods.reserve(m_Modes.size());

        bool hasSeasonalScale = maths_t::hasSeasonalVarianceScale(weightStyles, weights);

        TPoint mean = hasSeasonalScale ? this->mean() : TPoint(0.0);
        TDouble10Vec4Vec1Vec weights_(1, TDouble10Vec4Vec(1, TDouble10Vec(N, 1.0)));
        try {
            for (std::size_t i = 0u; i < samples.size(); ++i) {
                double n = this->smallest(
                    maths_t::countForUpdate(N, weightStyles, weights[i]));
                TPoint seasonalScale = sqrt(TPoint(
                    maths_t::seasonalVarianceScale(N, weightStyles, weights[i])));
                double logSeasonalScale = 0.0;
                for (std::size_t j = 0u; j < seasonalScale.dimension(); ++j) {
                    logSeasonalScale += std::log(seasonalScale(j));
                }

                TPoint x(samples[i]);
                if (hasSeasonalScale) {
                    x = mean + (x - mean) / seasonalScale;
                }
                sample[0] = x.template toVector<TDouble10Vec>();
                weights_[0][0] = maths_t::countVarianceScale(N, weightStyles, weights[i]);

                double sampleLogLikelihood;
                maths_t::EFloatingPointErrorStatus status = detail::jointLogMarginalLikelihood(
                    m_Modes, TWeights::COUNT_VARIANCE, sample, weights_,
                    modeLogLikelihoods, sampleLogLikelihood);
                if (status & maths_t::E_FpOverflowed) {
                    result = boost::numeric::bounds<double>::lowest();
                    return status;
                }
                if (status & maths_t::E_FpFailed) {
                    return status;
                }
                result += n * (sampleLogLikelihood - logSeasonalScale);
            }
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Failed to compute likelihood: " << e.what());
            return maths_t::E_FpFailed;
        }

        LOG_TRACE(<< "Joint log likelihood = " << result);

        maths_t::EFloatingPointErrorStatus status = CMathsFuncs::fpStatus(result);
        if (status & maths_t::E_FpFailed) {
            LOG_ERROR(<< "Failed to compute likelihood (" << this->debugWeights() << ")");
            LOG_ERROR(<< "samples = " << core::CContainerPrinter::print(samples));
            LOG_ERROR(<< "weights = " << core::CContainerPrinter::print(weights));
        }
        return status;
    }

    //! Sample the marginal likelihood function.
    //!
    //! The marginal likelihood functions are sampled in quantile intervals
    //! of the generalized cumulative density function, specifically intervals
    //! between contours of constant probability density.
    //!
    //! The idea is to capture a set of samples that accurately and efficiently
    //! represent the information in the prior. Random sampling (although it
    //! has nice asymptotic properties) doesn't fulfill the second requirement:
    //! typically requiring many more samples than sampling in quantile intervals
    //! to capture the same amount of information.
    //!
    //! This is to allow us to transform one prior distribution into another
    //! completely generically and relatively efficiently, by updating the target
    //! prior with these samples. As such the prior needs to maintain a count of
    //! the number of samples to date so that it isn't over sampled.
    //!
    //! \param[in] numberSamples The number of samples required.
    //! \param[out] samples Filled in with samples from the prior.
    //! \note \p numberSamples is truncated to the number of samples received.
    virtual void sampleMarginalLikelihood(std::size_t numberSamples,
                                          TDouble10Vec1Vec& samples) const {
        namespace detail = multivariate_multimodal_prior_detail;

        samples.clear();

        if (numberSamples == 0 || this->numberSamples() == 0.0) {
            return;
        }

        detail::sampleMarginalLikelihood(m_Modes, numberSamples, samples);
    }

    //! Check if this is a non-informative prior.
    virtual bool isNonInformative() const {
        return m_Modes.empty() ||
               (m_Modes.size() == 1 && m_Modes[0].s_Prior->isNonInformative());
    }

    //! Get a human readable description of the prior.
    //!
    //! \param[in] separator String used to separate priors.
    //! \param[in,out] result Filled in with the description.
    virtual void print(const std::string& separator, std::string& result) const {
        namespace detail = multivariate_multimodal_prior_detail;
        result += "\n" + separator + " multivariate multimodal";
        if (this->isNonInformative()) {
            result += " non-informative";
            return;
        }
        detail::print(m_Modes, separator, result);
        result += "\n" + separator;
    }

    //! Get a checksum for this object.
    virtual uint64_t checksum(uint64_t seed = 0) const {
        seed = this->CMultivariatePrior::checksum(seed);
        seed = CChecksum::calculate(seed, m_Clusterer);
        seed = CChecksum::calculate(seed, m_SeedPrior);
        return CChecksum::calculate(seed, m_Modes);
    }

    //! Get the memory used by this component
    virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
        mem->setName("CMultivariateMultimodalPrior");
        core::CMemoryDebug::dynamicSize("m_Clusterer", m_Clusterer, mem);
        core::CMemoryDebug::dynamicSize("m_SeedPrior", m_SeedPrior, mem);
        core::CMemoryDebug::dynamicSize("m_Modes", m_Modes, mem);
    }

    //! Get the memory used by this component
    virtual std::size_t memoryUsage() const {
        std::size_t mem = core::CMemory::dynamicSize(m_Clusterer);
        mem += core::CMemory::dynamicSize(m_SeedPrior);
        mem += core::CMemory::dynamicSize(m_Modes);
        return mem;
    }

    //! Get the static size of this object - used for virtual hierarchies
    virtual std::size_t staticSize() const { return sizeof(*this); }

    //! Get the tag name for this prior.
    virtual std::string persistenceTag() const {
        return MULTIMODAL_TAG + core::CStringUtils::typeToString(N);
    }

    //! Persist state by passing information to the supplied inserter
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const {
        inserter.insertLevel(CLUSTERER_TAG,
                             boost::bind<void>(CClustererStateSerialiser(),
                                               boost::cref(*m_Clusterer), _1));
        inserter.insertLevel(SEED_PRIOR_TAG,
                             boost::bind<void>(CPriorStateSerialiser(),
                                               boost::cref(*m_SeedPrior), _1));
        for (std::size_t i = 0u; i < m_Modes.size(); ++i) {
            inserter.insertLevel(MODE_TAG, boost::bind(&TMode::acceptPersistInserter,
                                                       &m_Modes[i], _1));
        }
        inserter.insertValue(DECAY_RATE_TAG, this->decayRate(), core::CIEEE754::E_SinglePrecision);
        inserter.insertValue(NUMBER_SAMPLES_TAG, this->numberSamples(),
                             core::CIEEE754::E_SinglePrecision);
    }
    //@}

    //! Get the current number of modes.
    std::size_t numberModes() const { return m_Modes.size(); }

    //! Get the expected mean of the marginal likelihood.
    TPoint mean() const {
        // By linearity we have that:
        //   Integral{ x * Sum_i{ w(i) * f(x | i) } }
        //     = Sum_i{ w(i) * Integral{ x * f(x | i) } }
        //     = Sum_i{ w(i) * mean(i) }

        TMeanAccumulator result;
        for (const auto& mode : m_Modes) {
            double weight = mode.weight();
            result.add(TPoint(mode.s_Prior->marginalLikelihoodMean()), weight);
        }
        return CBasicStatistics::mean(result);
    }

protected:
    using TMode = multivariate_multimodal_prior_detail::TMode;
    using TModeVec = multivariate_multimodal_prior_detail::TModeVec;

protected:
    //! Get the modes.
    const TModeVec& modes() const { return m_Modes; }

private:
    //! The callback invoked when a mode is split.
    class CModeSplitCallback {
    public:
        static const std::size_t MODE_SPLIT_NUMBER_SAMPLES;

    public:
        CModeSplitCallback(CMultivariateMultimodalPrior& prior)
            : m_Prior(&prior) {}

        void operator()(std::size_t sourceIndex,
                        std::size_t leftSplitIndex,
                        std::size_t rightSplitIndex) const {
            LOG_TRACE(<< "Splitting mode with index " << sourceIndex);

            TModeVec& modes = m_Prior->m_Modes;

            // Remove the split mode.
            auto mode = std::find_if(modes.begin(), modes.end(),
                                     CSetTools::CIndexInSet(sourceIndex));
            double numberSamples = mode != modes.end() ? mode->weight() : 0.0;
            modes.erase(mode);

            double pLeft = m_Prior->m_Clusterer->probability(leftSplitIndex);
            double pRight = m_Prior->m_Clusterer->probability(rightSplitIndex);
            double Z = pLeft + pRight;
            if (Z > 0.0) {
                pLeft /= Z;
                pRight /= Z;
            }
            LOG_TRACE(<< "# samples = " << numberSamples
                      << ", pLeft = " << pLeft << ", pRight = " << pRight);

            // Create the child modes.
            LOG_TRACE(<< "Creating mode with index " << leftSplitIndex);
            modes.emplace_back(leftSplitIndex, m_Prior->m_SeedPrior);
            {
                TPointVec samples;
                if (!m_Prior->m_Clusterer->sample(
                        leftSplitIndex, MODE_SPLIT_NUMBER_SAMPLES, samples)) {
                    LOG_ERROR(<< "Couldn't find cluster for " << leftSplitIndex);
                }
                LOG_TRACE(<< "samples = " << core::CContainerPrinter::print(samples));

                double nl = pLeft * numberSamples;
                double ns = std::min(nl, static_cast<double>(N + 2));
                double s = static_cast<double>(samples.size());
                LOG_TRACE(<< "# left = " << nl);

                TDouble10Vec1Vec samples_;
                samples_.reserve(samples.size());
                for (const auto& sample : samples) {
                    samples_.push_back(sample.template toVector<TDouble10Vec>());
                }
                TDouble10Vec seedWeight(N, ns / s);
                TDouble10Vec4Vec1Vec weights(samples_.size(), TDouble10Vec4Vec(1, seedWeight));
                modes.back().s_Prior->addSamples(TWeights::COUNT, samples_, weights);
                double weight = (nl - ns) / s;
                if (weight > 0.0) {
                    weights.assign(weights.size(),
                                   TDouble10Vec4Vec(1, TDouble10Vec(N, weight)));
                    modes.back().s_Prior->addSamples(TWeights::COUNT, samples_, weights);
                    LOG_TRACE(<< modes.back().s_Prior->print());
                }
            }

            LOG_TRACE(<< "Creating mode with index " << rightSplitIndex);
            modes.emplace_back(rightSplitIndex, m_Prior->m_SeedPrior);
            {
                TPointVec samples;
                if (!m_Prior->m_Clusterer->sample(
                        rightSplitIndex, MODE_SPLIT_NUMBER_SAMPLES, samples)) {
                    LOG_ERROR(<< "Couldn't find cluster for " << rightSplitIndex)
                }
                LOG_TRACE(<< "samples = " << core::CContainerPrinter::print(samples));

                double nr = pRight * numberSamples;
                double ns = std::min(nr, static_cast<double>(N + 2));
                double s = static_cast<double>(samples.size());
                LOG_TRACE(<< "# right = " << nr);

                TDouble10Vec1Vec samples_;
                samples_.reserve(samples.size());
                for (const auto& sample : samples) {
                    samples_.push_back(sample.template toVector<TDouble10Vec>());
                }
                TDouble10Vec seedWeight(N, ns / s);
                TDouble10Vec4Vec1Vec weights(samples_.size(), TDouble10Vec4Vec(1, seedWeight));
                modes.back().s_Prior->addSamples(TWeights::COUNT, samples_, weights);
                double weight = (nr - ns) / s;
                if (weight > 0.0) {
                    weights.assign(weights.size(),
                                   TDouble10Vec4Vec(1, TDouble10Vec(N, weight)));
                    modes.back().s_Prior->addSamples(TWeights::COUNT, samples_, weights);
                    LOG_TRACE(<< modes.back().s_Prior->print());
                }
            }

            LOG_TRACE(<< m_Prior->print());
            LOG_TRACE(<< "Split mode");
        }

    private:
        CMultivariateMultimodalPrior* m_Prior;
    };

    //! The callback invoked when two modes are merged.
    class CModeMergeCallback {
    public:
        static const std::size_t MODE_MERGE_NUMBER_SAMPLES;

    public:
        CModeMergeCallback(CMultivariateMultimodalPrior& prior)
            : m_Prior(&prior) {}

        void operator()(std::size_t leftMergeIndex,
                        std::size_t rightMergeIndex,
                        std::size_t targetIndex) const {
            namespace detail = multivariate_multimodal_prior_detail;
            detail::modeMergeCallback(N, m_Prior->m_Modes, m_Prior->m_SeedPrior,
                                      MODE_MERGE_NUMBER_SAMPLES, leftMergeIndex,
                                      rightMergeIndex, targetIndex);
        }

    private:
        CMultivariateMultimodalPrior* m_Prior;
    };

private:
    //! \name State tags for model persistence.
    //@{
    static const std::string CLUSTERER_TAG;
    static const std::string SEED_PRIOR_TAG;
    static const std::string MODE_TAG;
    static const std::string NUMBER_SAMPLES_TAG;
    static const std::string MINIMUM_TAG;
    static const std::string MAXIMUM_TAG;
    static const std::string DECAY_RATE_TAG;
    //@}

private:
    //! Read parameters from \p traverser.
    bool acceptRestoreTraverser(const SDistributionRestoreParams& params,
                                core::CStateRestoreTraverser& traverser) {
        do {
            const std::string& name = traverser.name();
            RESTORE_SETUP_TEARDOWN(DECAY_RATE_TAG, double decayRate,
                                   core::CStringUtils::stringToType(traverser.value(), decayRate),
                                   this->decayRate(decayRate))
            RESTORE(CLUSTERER_TAG, traverser.traverseSubLevel(boost::bind<bool>(
                                       CClustererStateSerialiser(), boost::cref(params),
                                       boost::ref(m_Clusterer), _1)))
            RESTORE(SEED_PRIOR_TAG, traverser.traverseSubLevel(boost::bind<bool>(
                                        CPriorStateSerialiser(), boost::cref(params),
                                        boost::ref(m_SeedPrior), _1)))
            RESTORE_SETUP_TEARDOWN(
                MODE_TAG, TMode mode,
                traverser.traverseSubLevel(boost::bind(
                    &TMode::acceptRestoreTraverser, &mode, boost::cref(params), _1)),
                m_Modes.push_back(mode))
            RESTORE_SETUP_TEARDOWN(
                NUMBER_SAMPLES_TAG, double numberSamples,
                core::CStringUtils::stringToType(traverser.value(), numberSamples),
                this->numberSamples(numberSamples))
        } while (traverser.next());

        if (m_Clusterer) {
            // Register the split and merge callbacks.
            m_Clusterer->splitFunc(CModeSplitCallback(*this));
            m_Clusterer->mergeFunc(CModeMergeCallback(*this));
        }

        return true;
    }

    //! We should only use this prior when it has multiple modes.
    virtual bool participatesInModelSelection() const {
        return m_Modes.size() > 1;
    }

    //! Get the number of nuisance parameters in the marginal likelihood.
    //!
    //! This is just number modes - 1 due to the normalization constraint.
    virtual double unmarginalizedParameters() const {
        return std::max(static_cast<double>(m_Modes.size()), 1.0) - 1.0;
    }

    //! Get the convariance matrix for the marginal likelihood.
    TMatrix covarianceMatrix() const {
        // By linearity we have that:
        //   Integral{ (x - m)' * (x - m) * Sum_i{ w(i) * f(x | i) } }
        //     = Sum_i{ w(i) * (Integral{ x' * x * f(x | i) } - m' * m) }
        //     = Sum_i{ w(i) * ((mi' * mi + Ci) - m' * m) }

        using TMatrixMeanAccumulator =
            typename CBasicStatistics::SSampleMean<TMatrix>::TAccumulator;

        TMatrix mean2 = TPoint(this->marginalLikelihoodMean()).outer();

        TMatrixMeanAccumulator result;
        for (const auto& mode : m_Modes) {
            double weight = mode.weight();
            TPoint modeMean(mode.s_Prior->marginalLikelihoodMean());
            TMatrix modeVariance(mode.s_Prior->marginalLikelihoodCovariance());
            result.add(modeMean.outer() - mean2 + modeVariance, weight);
        }

        return CBasicStatistics::mean(result);
    }

    //! Full debug dump of the mode weights.
    std::string debugWeights() const {
        return multivariate_multimodal_prior_detail::debugWeights(m_Modes);
    }

private:
    //! The object which partitions the data into clusters.
    TClustererPtr m_Clusterer;

    //! The object used to initialize new cluster priors.
    TPriorPtr m_SeedPrior;

    //! The modes of the distribution.
    TModeVec m_Modes;
};

template<std::size_t N>
const std::string CMultivariateMultimodalPrior<N>::CLUSTERER_TAG("a");
template<std::size_t N>
const std::string CMultivariateMultimodalPrior<N>::SEED_PRIOR_TAG("b");
template<std::size_t N>
const std::string CMultivariateMultimodalPrior<N>::MODE_TAG("c");
template<std::size_t N>
const std::string CMultivariateMultimodalPrior<N>::NUMBER_SAMPLES_TAG("d");
template<std::size_t N>
const std::string CMultivariateMultimodalPrior<N>::MINIMUM_TAG("e");
template<std::size_t N>
const std::string CMultivariateMultimodalPrior<N>::MAXIMUM_TAG("f");
template<std::size_t N>
const std::string CMultivariateMultimodalPrior<N>::DECAY_RATE_TAG("g");
template<std::size_t N>
const std::size_t
    CMultivariateMultimodalPrior<N>::CModeSplitCallback::MODE_SPLIT_NUMBER_SAMPLES(50 * N);
template<std::size_t N>
const std::size_t
    CMultivariateMultimodalPrior<N>::CModeMergeCallback::MODE_MERGE_NUMBER_SAMPLES(25 * N);
}
}

#endif // INCLUDED_ml_maths_CMultivariateMultimodalPrior_h
