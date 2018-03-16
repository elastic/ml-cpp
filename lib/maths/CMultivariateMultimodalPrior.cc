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

#include <maths/CMultivariateMultimodalPrior.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CSampling.h>

namespace ml {
namespace maths {
namespace multivariate_multimodal_prior_detail {

typedef std::vector<double> TDoubleVec;
typedef CMultivariatePrior::TDouble10Vec TDouble10Vec;
typedef CMultivariatePrior::TDouble10Vec4Vec TDouble10Vec4Vec;

namespace {

//! Print the set of mode indices.
std::string printIndices(const TModeVec& modes) {
    std::ostringstream result;
    result << "{";
    if (!modes.empty()) {
        result << modes[0].s_Index;
        for (std::size_t i = 1u; i < modes.size(); ++i) {
            result << ", " << modes[i].s_Index;
        }
    }
    result << "}";
    return result.str();
}
}

maths_t::EFloatingPointErrorStatus jointLogMarginalLikelihood(const TModeVec& modes,
                                                              const maths_t::TWeightStyleVec& weightStyles,
                                                              const TDouble10Vec1Vec& sample,
                                                              const TDouble10Vec4Vec1Vec& weights,
                                                              TSizeDoublePr3Vec& modeLogLikelihoods,
                                                              double& result) {
    try {
        // We re-normalize so that the maximum log likelihood is one
        // to avoid underflow.
        modeLogLikelihoods.clear();
        double maxLogLikelihood = boost::numeric::bounds<double>::lowest();

        for (std::size_t i = 0u; i < modes.size(); ++i) {
            double modeLogLikelihood;
            maths_t::EFloatingPointErrorStatus status =
                modes[i].s_Prior->jointLogMarginalLikelihood(weightStyles, sample, weights, modeLogLikelihood);
            if (status & maths_t::E_FpFailed) {
                // Logging handled at a lower level.
                return status;
            }
            if (!(status & maths_t::E_FpOverflowed)) {
                modeLogLikelihoods.push_back({i, modeLogLikelihood});
                maxLogLikelihood = std::max(maxLogLikelihood, modeLogLikelihood);
            }
        }

        if (modeLogLikelihoods.empty()) {
            // Technically, the marginal likelihood is zero here
            // so the log would be infinite. We use minus max
            // double because log(0) = HUGE_VALUE, which causes
            // problems for Windows. Calling code is notified
            // when the calculation overflows and should avoid
            // taking the exponential since this will underflow
            // and pollute the floating point environment. This
            // may cause issues for some library function
            // implementations (see fe*exceptflag for more details).
            result = boost::numeric::bounds<double>::lowest();
            return maths_t::E_FpOverflowed;
        }

        LOG_TRACE("modeLogLikelihoods = " << core::CContainerPrinter::print(modeLogLikelihoods));

        double sampleLikelihood = 0.0;
        double Z = 0.0;

        for (const auto& likelihood : modeLogLikelihoods) {
            double w = modes[likelihood.first].weight();
            // Divide through by the largest value to avoid underflow.
            sampleLikelihood += w * ::exp(likelihood.second - maxLogLikelihood);
            Z += w;
        }

        sampleLikelihood /= Z;
        result = (::log(sampleLikelihood) + maxLogLikelihood);

        LOG_TRACE("sample = " << core::CContainerPrinter::print(sample) << ", maxLogLikelihood = " << maxLogLikelihood
                              << ", sampleLogLikelihood = " << result);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to compute likelihood: " << e.what());
        return maths_t::E_FpFailed;
    }

    return maths_t::E_FpNoErrors;
}

void sampleMarginalLikelihood(const TModeVec& modes, std::size_t numberSamples, TDouble10Vec1Vec& samples) {
    samples.clear();

    if (modes.size() == 1) {
        modes[0].s_Prior->sampleMarginalLikelihood(numberSamples, samples);
        return;
    }

    // We sample each mode according to its weight.

    TDoubleVec normalizedWeights;
    normalizedWeights.reserve(modes.size());
    double Z = 0.0;

    for (const auto& mode : modes) {
        double weight = mode.weight();
        normalizedWeights.push_back(weight);
        Z += weight;
    }
    for (auto&& weight : normalizedWeights) {
        weight /= Z;
    }

    CSampling::TSizeVec sampling;
    CSampling::weightedSample(numberSamples, normalizedWeights, sampling);
    LOG_TRACE("normalizedWeights = " << core::CContainerPrinter::print(normalizedWeights)
                                     << ", sampling = " << core::CContainerPrinter::print(sampling));

    if (sampling.size() != modes.size()) {
        LOG_ERROR("Failed to sample marginal likelihood");
        return;
    }

    samples.reserve(numberSamples);
    TDouble10Vec1Vec modeSamples;
    for (std::size_t i = 0u; i < modes.size(); ++i) {
        modes[i].s_Prior->sampleMarginalLikelihood(sampling[i], modeSamples);
        LOG_TRACE("# modeSamples = " << modeSamples.size());
        LOG_TRACE("modeSamples = " << core::CContainerPrinter::print(modeSamples));
        std::copy(modeSamples.begin(), modeSamples.end(), std::back_inserter(samples));
    }
    LOG_TRACE("samples = " << core::CContainerPrinter::print(samples));
}

void print(const TModeVec& modes, const std::string& separator, std::string& result) {
    double Z = std::accumulate(
        modes.begin(), modes.end(), 0.0, [](double sum, const TMode& mode) { return sum + mode.weight(); });

    std::string separator_ = separator + separator;

    result += ":";
    for (const auto& mode : modes) {
        double weight = mode.weight() / Z;
        result += core_t::LINE_ENDING + separator_ + " weight " + core::CStringUtils::typeToStringPretty(weight);
        mode.s_Prior->print(separator_, result);
    }
}

void modeMergeCallback(std::size_t dimension,
                       TModeVec& modes,
                       const TPriorPtr& seedPrior,
                       std::size_t numberSamples,
                       std::size_t leftMergeIndex,
                       std::size_t rightMergeIndex,
                       std::size_t targetIndex) {
    LOG_TRACE("Merging modes with indices " << leftMergeIndex << " " << rightMergeIndex);

    typedef std::set<std::size_t> TSizeSet;

    // Create the new mode.
    TMode newMode(targetIndex, seedPrior);

    double wl = 0.0;
    double wr = 0.0;
    double n = 0.0;
    std::size_t nl = 0;
    std::size_t nr = 0;
    TDouble10Vec1Vec samples;

    auto leftMode = std::find_if(modes.begin(), modes.end(), CSetTools::CIndexInSet(leftMergeIndex));
    if (leftMode != modes.end()) {
        wl = leftMode->s_Prior->numberSamples();
        n += wl;
        TDouble10Vec1Vec leftSamples;
        leftMode->s_Prior->sampleMarginalLikelihood(numberSamples, leftSamples);
        nl = leftSamples.size();
        samples.insert(samples.end(), leftSamples.begin(), leftSamples.end());
    } else {
        LOG_ERROR("Couldn't find mode for " << leftMergeIndex << " in " << printIndices(modes) << ", other index = "
                                            << rightMergeIndex << ", merged index = " << targetIndex);
    }

    auto rightMode = std::find_if(modes.begin(), modes.end(), CSetTools::CIndexInSet(rightMergeIndex));
    if (rightMode != modes.end()) {
        wr = rightMode->s_Prior->numberSamples();
        n += wr;
        TDouble10Vec1Vec rightSamples;
        rightMode->s_Prior->sampleMarginalLikelihood(numberSamples, rightSamples);
        nr = rightSamples.size();
        samples.insert(samples.end(), rightSamples.begin(), rightSamples.end());
    } else {
        LOG_ERROR("Couldn't find mode for " << rightMergeIndex << " in " << printIndices(modes) << ", other index = "
                                            << leftMergeIndex << ", merged index = " << targetIndex);
    }

    if (n > 0.0) {
        double nl_ = static_cast<double>(nl);
        double nr_ = static_cast<double>(nr);
        double Z = (nl_ * wl + nr_ * wr) / (nl_ + nr_);
        wl /= Z;
        wr /= Z;
    }

    LOG_TRACE("samples = " << core::CContainerPrinter::print(samples));
    LOG_TRACE("n = " << n << ", wl = " << wl << ", wr = " << wr);

    double ns = std::min(n, 4.0);
    double s = static_cast<double>(samples.size());

    TDouble10Vec leftSeedWeight(dimension, wl * ns / s);
    TDouble10Vec rightSeedWeight(dimension, wl * ns / s);
    TDouble10Vec4Vec1Vec weights;
    weights.reserve(samples.size());
    weights.resize(nl, TDouble10Vec1Vec(1, leftSeedWeight));
    weights.resize(nl + nr, TDouble10Vec1Vec(1, rightSeedWeight));
    newMode.s_Prior->addSamples(CConstantWeights::COUNT, samples, weights);

    double weight = (n - ns) / s;
    if (weight > 0.0) {
        for (std::size_t i = 0u; i < dimension; ++i) {
            leftSeedWeight[i] = wl * weight;
            rightSeedWeight[i] = wr * weight;
        }
        weights.clear();
        weights.resize(nl, TDouble10Vec1Vec(1, leftSeedWeight));
        weights.resize(nl + nr, TDouble10Vec1Vec(1, rightSeedWeight));
        newMode.s_Prior->addSamples(CConstantWeights::COUNT, samples, weights);
    }

    // Remove the merged modes.
    TSizeSet mergedIndices;
    mergedIndices.insert(leftMergeIndex);
    mergedIndices.insert(rightMergeIndex);
    modes.erase(std::remove_if(modes.begin(), modes.end(), CSetTools::CIndexInSet(mergedIndices)), modes.end());

    // Add the new mode.
    LOG_TRACE("Creating mode with index " << targetIndex);
    modes.push_back(newMode);

    LOG_TRACE("Merged modes");
}

std::string debugWeights(const TModeVec& modes) {
    return TMode::debugWeights(modes);
}
}
}
}
