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

#include <maths/CTimeSeriesDecompositionInterface.h>

#include <core/CLogger.h>
#include <core/Constants.h>
#include <core/CSmallVector.h>
#include <core/CHashing.h>

#include <maths/CMultivariatePrior.h>
#include <maths/CPrior.h>
#include <maths/CPRNG.h>
#include <maths/CSampling.h>

#include <cmath>
#include <vector>

namespace ml
{
namespace maths
{
namespace
{

using TDoubleVec = std::vector<double>;
using TDouble4Vec = core::CSmallVector<double, 4>;
const std::size_t NUMBER_SAMPLES{100u};

}

bool initializePrior(core_t::TTime bucketLength,
                     double learnRate,
                     const CTimeSeriesDecompositionInterface &decomposition,
                     CPrior &prior)
{
    using TDouble4Vec1Vec = core::CSmallVector<TDouble4Vec, 1>;

    if (!decomposition.initialized())
    {
        return false;
    }

    // Note our estimate of variance is approximate so we use a
    // small tolerance when generating the samples with which to
    // initialize the prior.

    double variance{2.25 * decomposition.meanVariance()};
    double sd{std::sqrt(variance)};

    TDoubleVec samples;
    CPRNG::CXorOShiro128Plus rng;
    CSampling::normalSample(rng, 0.0, variance, NUMBER_SAMPLES, samples);
    std::sort(samples.begin(), samples.end());

    double weight{0.2 * static_cast<double>(core::constants::DAY)
                      / static_cast<double>(std::min(bucketLength, core::constants::HOUR))
                      / static_cast<double>(NUMBER_SAMPLES) * learnRate};
    TDouble4Vec1Vec weights(samples.size(), TDouble4Vec(1, weight));

    // Use an offset which ensures that the samples lie in the
    // supports of all priors.
    double offset{std::max(-samples[0] + 0.01 * sd, prior.offsetMargin())};
    prior.setToNonInformative(offset, prior.decayRate());
    prior.removeModels(CPrior::CModelFilter().remove(CPrior::E_Poisson));

    LOG_DEBUG("sd = " << sd << ", offset = " << offset << ", weight = " << weight);

    prior.addSamples(CConstantWeights::COUNT, samples, weights);

    return true;
}

bool initializePrior(core_t::TTime bucketLength,
                     double learnRate,
                     const TDecompositionPtr10Vec &decomposition,
                     CMultivariatePrior &prior)
{
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TDouble10Vec = core::CSmallVector<double, 10>;
    using TDouble10Vec4Vec = core::CSmallVector<TDouble10Vec, 4>;
    using TDouble10Vec4Vec1Vec = core::CSmallVector<TDouble10Vec4Vec, 1>;

    if (decomposition.empty())
    {
        return false;
    }

    // Note our estimate of variance is approximate so we use a
    // small tolerance when generating the samples with which to
    // initialize the prior.

    std::size_t dimension{decomposition.size()};

    TDoubleVec mean(dimension, 0.0);
    TDoubleVecVec covariance(dimension, TDoubleVec(dimension, 0.0));
    double sd{0.0};
    for (std::size_t i = 0u; i < dimension; ++i)
    {
        if (!decomposition[i]->initialized())
        {
            return false;
        }
        covariance[i][i] = 2.25 * decomposition[i]->meanVariance();
        sd += covariance[i][i];
    }
    sd /= static_cast<double>(dimension);
    sd = std::sqrt(sd);

    TDoubleVecVec samples;
    samples.reserve(NUMBER_SAMPLES);
    CPRNG::CXorOShiro128Plus rng;
    CSampling::multivariateNormalSample(rng, mean, covariance, NUMBER_SAMPLES, samples);
    std::sort(samples.begin(), samples.end());

    double weight{0.2 * static_cast<double>(core::constants::DAY)
                      / static_cast<double>(std::min(bucketLength, core::constants::HOUR))
                      / static_cast<double>(NUMBER_SAMPLES) * learnRate};
    TDouble10Vec4Vec1Vec weights(samples.size(), TDouble10Vec4Vec(1, TDouble10Vec(dimension, weight)));

    // Use an offset which ensures that the samples lie in the
    // supports of all priors.

    double offset{prior.offsetMargin()};
    for (auto &&sample : samples)
    {
        double min = *std::min_element(sample.begin(), sample.end());
        offset = std::max(offset, -min + 0.01 * sd);
    }
    prior.setToNonInformative(offset, prior.decayRate());

    LOG_DEBUG("sd = " << sd << ", offset = " << offset << ", weight = " << weight);

    prior.addSamples(CConstantWeights::COUNT, samples, weights);

    return true;
}

}
}
