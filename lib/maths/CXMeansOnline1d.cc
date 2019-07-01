/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CXMeansOnline1d.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CSmallVector.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/CPrior.h>
#include <maths/CRestoreParams.h>
#include <maths/CSolvers.h>
#include <maths/CTools.h>
#include <maths/Constants.h>
#include <maths/MathsTypes.h>

#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/digamma.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ml {
namespace maths {

namespace {

using TDouble1Vec = core::CSmallVector<double, 1>;
using TDoubleDoublePr = std::pair<double, double>;
using TSizeVec = std::vector<std::size_t>;
using TTuple = CNaturalBreaksClassifier::TTuple;
using TTupleVec = CNaturalBreaksClassifier::TTupleVec;

namespace detail {

using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

//! \brief Orders two clusters by their centres.
struct SClusterCentreLess {
    bool operator()(const CXMeansOnline1d::CCluster& lhs,
                    const CXMeansOnline1d::CCluster& rhs) const {
        return lhs.centre() < rhs.centre();
    }
    bool operator()(double lhs, const CXMeansOnline1d::CCluster& rhs) const {
        return lhs < rhs.centre();
    }
    bool operator()(const CXMeansOnline1d::CCluster& lhs, double rhs) const {
        return lhs.centre() < rhs;
    }
};

//! Get the minimum of \p x, \p y and \p z.
double min(double x, double y, double z) {
    return std::min(std::min(x, y), z);
}

//! Get the log of the likelihood that \p point is from the \p normal.
maths_t::EFloatingPointErrorStatus logLikelihoodFromCluster(double point,
                                                            const CNormalMeanPrecConjugate& normal,
                                                            double probability,
                                                            double& result) {
    result = core::constants::LOG_MIN_DOUBLE - 1.0;

    double likelihood;

    maths_t::EFloatingPointErrorStatus status = normal.jointLogMarginalLikelihood(
        {point}, maths_t::CUnitWeights::SINGLE_UNIT, likelihood);
    if (status & maths_t::E_FpFailed) {
        LOG_ERROR(<< "Unable to compute likelihood for: " << point);
        return status;
    }
    if (status & maths_t::E_FpOverflowed) {
        result = likelihood;
        return status;
    }

    result = likelihood + std::log(probability);
    return status;
}

//! Get the moments of \p categories and the splits into
//! [\p start, \p split) and [\p split, \p end).
void candidates(const TTupleVec& categories,
                std::size_t start,
                std::size_t split,
                std::size_t end,
                TMeanVarAccumulator& mv,
                TMeanVarAccumulator& mvl,
                TMeanVarAccumulator& mvr) {
    LOG_TRACE(<< "categories = "
              << core::CContainerPrinter::print(categories.begin() + start,
                                                categories.begin() + end));
    LOG_TRACE(<< "split at = " << split);

    for (std::size_t i = start; i < split; ++i) {
        mv += categories[i];
        mvl += categories[i];
    }
    for (std::size_t i = split; i < end; ++i) {
        mv += categories[i];
        mvr += categories[i];
    }

    LOG_TRACE(<< "mv = " << mv << ", mvl = " << mvl << ", mvr = " << mvr);
}

//! Compute the mean of \p category.
double mean(maths_t::EDataType dataType, const TTuple& category) {
    double result = CBasicStatistics::mean(category);
    switch (dataType) {
    case maths_t::E_DiscreteData:
        break;
    case maths_t::E_IntegerData:
        result += 0.5;
        break;
    case maths_t::E_ContinuousData:
        break;
    case maths_t::E_MixedData:
        break;
    }
    return result;
}

//! Compute the variance of \p category.
double variance(maths_t::EDataType dataType, const TTuple& category) {
    double n = CBasicStatistics::count(category);
    double result = (1.0 + 1.0 / n) * CBasicStatistics::maximumLikelihoodVariance(category);
    switch (dataType) {
    case maths_t::E_DiscreteData:
        break;
    case maths_t::E_IntegerData:
        result += 1.0 / 12.0;
        break;
    case maths_t::E_ContinuousData:
        break;
    case maths_t::E_MixedData:
        break;
    }
    return result;
}

//! Computes the Bayes Information Content of splitting verses
//! not splitting.
//!
//! This considers splitting the data at split such that the
//! first cluster is [\p start, \p split) and the second is
//! [\p split, \p end) and computes the information gain
//! (strictly \f$\max(-2\log(L(\{x_i\}|M_1)) + 2\log(L(\{x_i\}|M_2)) + (n_1 - n_2) \log(n), 0)\f$
//! where \f$M_1\f$ is the one cluster model, \f$M_2\f$ is the
//! two cluster model, \f$n_1\f$ is the number of parameters in
//! the one cluster model, \f$n_2\f$ is the number of parameters
//! in the two cluster model and \f$n\f$ is the total number of
//! points). Various models one and two cluster models are
//! considered.
void BICGain(maths_t::EDataType dataType,
             CAvailableModeDistributions distributions,
             double smallest,
             const TTupleVec& categories,
             std::size_t start,
             std::size_t split,
             std::size_t end,
             double& distance,
             double& nl,
             double& nr) {

    // The basic idea is to compute the difference between the
    // Bayes Information Content (BIC) for one and two clusters
    // for the sketch defined by the categories passed to this
    // function. The exact form of BIC for the mixture (i.e. the
    // two cluster case) depends on what one assumes about the
    // the model. For example, if one assumes that the categories
    // are labeled with their correct cluster then their likelihood
    // is their likelihood of coming from the corresponding cluster
    // multiplied by the cluster weight. Alternatively, if one
    // assumes that they have equal prior probability of coming
    // from either cluster, their likelihood is the weighted sum
    // of their likelihoods for both clusters. In this second case
    // the exact BIC (even modeling the modes as normals) can't be
    // computed in terms of the statistics we maintain in the sketch.
    // We could for example compute the expected BIC over a mixture
    // model of the sketch. However, we will assume the former case
    // (as they do in x-means). In this case, the BIC for the single
    // cluster and normal distribution is
    //
    //   \sum_i{ ni * (log(1/2/pi/v) + (vi + (mi - m)^2) / v) } + 2.0 * log(n)
    //
    // where ni is the count, mi is the mean and vi is the variance
    // of the i'th category, and n, m and v are the overall count
    // mean and variance. For the two cluster normal distribution
    // case it is
    //
    //   \sum_{i in l}{ ni * (log(1/2/pi/vl/wl/wl) + (vi + (mi - ml)^2) / vl)
    // + \sum_{i in r}{ ni * (log(1/2/pi/vl/wr/wr) + (vi + (mi - mr)^2) / vr)
    // + 5.0 * log(n)
    //
    // where wl, ml and vl are the weight, mean and variance of one
    // cluster and wr, mr and vr are the weight, mean and variance
    // of the other cluster.
    //
    // We also consider log-normal and mixture of log-normal and
    // gamma and mixture of gamma cases. We don't maintain the
    // sufficient statistics we need to estimate the exact BIC
    // for these distributions instead we compute the best
    // approximation with the statistics available. Specifically,
    // we estimate the distribution parameters using method of
    // moments rather than maximum likelihood and compute the
    // BIC using Taylor expansions of log(xi) and log(xi)^2.

    static const double MIN_RELATIVE_VARIANCE = 1e-10;
    static const double MIN_ABSOLUTE_VARIANCE = 1.0;

    TMeanVarAccumulator mv;
    TMeanVarAccumulator mvl;
    TMeanVarAccumulator mvr;
    candidates(categories, start, split, end, mv, mvl, mvr);
    double logNormalOffset = std::max(0.0, GAMMA_OFFSET_MARGIN - smallest);
    double gammaOffset = std::max(0.0, LOG_NORMAL_OFFSET_MARGIN - smallest);
    for (std::size_t i = start; i < end; ++i) {
        double x = mean(dataType, categories[i]);
        logNormalOffset = std::max(logNormalOffset, LOG_NORMAL_OFFSET_MARGIN - x);
        gammaOffset = std::max(gammaOffset, GAMMA_OFFSET_MARGIN - x);
    }
    LOG_TRACE(<< "offsets = [" << gammaOffset << "," << logNormalOffset << "]");

    distance = 0.0;
    nl = CBasicStatistics::count(mvl);
    nr = CBasicStatistics::count(mvr);

    // Compute the BIC gain for splitting the mode.

    double ll1n = 0.0;
    double ll1l = 0.0;
    double ll1g = 0.0;
    double ll2nl = 0.0;
    double ll2ll = 0.0;
    double ll2gl = 0.0;
    double ll2nr = 0.0;
    double ll2lr = 0.0;
    double ll2gr = 0.0;

    // Normal
    double n = CBasicStatistics::count(mv);
    double m = mean(dataType, mv);
    double v = variance(dataType, mv);
    if (v <= MINIMUM_COEFFICIENT_OF_VARIATION * std::fabs(m)) {
        return;
    }

    // Log-normal (method of moments)
    double s = std::log(1.0 + v / CTools::pow2(m + logNormalOffset));
    double l = std::log(m + logNormalOffset) - s / 2.0;
    // Gamma (method of moments)
    double a = CTools::pow2(m + gammaOffset) / v;
    double b = (m + gammaOffset) / v;

    double smin = std::max(logNormalOffset, gammaOffset);
    double vmin = std::min(MIN_RELATIVE_VARIANCE * std::max(v, CTools::pow2(smin)),
                           MIN_ABSOLUTE_VARIANCE);

    // Mixture of normals
    double wl = CBasicStatistics::count(mvl) / n;
    double ml = mean(dataType, mvl);
    double vl = std::max(variance(dataType, mvl), vmin);
    double wr = CBasicStatistics::count(mvr) / n;
    double mr = mean(dataType, mvr);
    double vr = std::max(variance(dataType, mvr), vmin);

    bool haveGamma = distributions.haveGamma();
    try {
        // Mixture of log-normals (method of moments)
        double sl = std::log(1.0 + vl / CTools::pow2(ml + logNormalOffset));
        double ll = std::log(ml + logNormalOffset) - sl / 2.0;
        double sr = std::log(1.0 + vr / CTools::pow2(mr + logNormalOffset));
        double lr = std::log(mr + logNormalOffset) - sr / 2.0;
        // Mixture of gammas (method of moments)
        double al = CTools::pow2(ml + gammaOffset) / vl;
        double bl = (ml + gammaOffset) / vl;
        double ar = CTools::pow2(mr + gammaOffset) / vr;
        double br = (mr + gammaOffset) / vr;

        double log2piv = std::log(boost::math::double_constants::two_pi * v);
        double log2pis = std::log(boost::math::double_constants::two_pi * s);

        double loggn = 0.0;
        double loggnl = 0.0;
        double loggnr = 0.0;

        if (haveGamma && maths::CTools::lgamma(a, loggn, true) &&
            maths::CTools::lgamma(al, loggnl, true) &&
            maths::CTools::lgamma(ar, loggnr, true)) {
            loggn -= a * std::log(b);
            loggnl -= al * std::log(bl) + std::log(wl);
            loggnr -= ar * std::log(br) + std::log(wr);
        } else {
            haveGamma = false;
        }

        double log2pivl =
            std::log(boost::math::double_constants::two_pi * vl / CTools::pow2(wl));
        double log2pivr =
            std::log(boost::math::double_constants::two_pi * vr / CTools::pow2(wr));
        double log2pisl =
            std::log(boost::math::double_constants::two_pi * sl / CTools::pow2(wl));
        double log2pisr =
            std::log(boost::math::double_constants::two_pi * sr / CTools::pow2(wr));

        for (std::size_t i = start; i < split; ++i) {
            double ni = CBasicStatistics::count(categories[i]);
            double mi = mean(dataType, categories[i]);
            double vi = variance(dataType, categories[i]);

            if (vi == 0.0) {
                double li = std::log(mi + logNormalOffset);
                ll1n += ni * ((vi + CTools::pow2(mi - m)) / v + log2piv);
                ll1l += ni * (CTools::pow2(li - l) / s + 2.0 * li + log2pis);
                ll1g += ni * 2.0 * (b * (mi + gammaOffset) - (a - 1.0) * li + loggn);
                ll2nl += ni * ((vi + CTools::pow2(mi - ml)) / vl + log2pivl);
                ll2ll += ni * (CTools::pow2(li - ll) / sl + 2.0 * li + log2pisl);
                ll2gl += ni * 2.0 * (bl * (mi + gammaOffset) - (al - 1.0) * li + loggnl);
            } else {
                double si = std::log(1.0 + vi / CTools::pow2(mi + logNormalOffset));
                double li = std::log(mi + logNormalOffset) - si / 2.0;
                ll1n += ni * ((vi + CTools::pow2(mi - m)) / v + log2piv);
                ll1l += ni * ((si + CTools::pow2(li - l)) / s + 2.0 * li + log2pis);
                ll1g += ni * 2.0 * (b * (mi + gammaOffset) - (a - 1.0) * li + loggn);
                ll2nl += ni * ((vi + CTools::pow2(mi - ml)) / vl + log2pivl);
                ll2ll += ni * ((si + CTools::pow2(li - ll)) / sl + 2.0 * li + log2pisl);
                ll2gl += ni * 2.0 * (bl * (mi + gammaOffset) - (al - 1.0) * li + loggnl);
            }
        }

        for (std::size_t i = split; i < end; ++i) {
            double ni = CBasicStatistics::count(categories[i]);
            double mi = mean(dataType, categories[i]);
            double vi = variance(dataType, categories[i]);

            if (vi == 0.0) {
                double li = std::log(mi + logNormalOffset);
                ll1n += ni * ((vi + CTools::pow2(mi - m)) / v + log2piv);
                ll1l += ni * (CTools::pow2(li - l) / s + 2.0 * li + log2pis);
                ll1g += ni * 2.0 * (b * (mi + gammaOffset) - (a - 1.0) * li + loggn);
                ll2nr += ni * ((vi + CTools::pow2(mi - mr)) / vr + log2pivr);
                ll2lr += ni * (CTools::pow2(li - lr) / sr + 2.0 * li + log2pisr);
                ll2gr += ni * 2.0 * (br * (mi + gammaOffset) - (ar - 1.0) * li + loggnr);
            } else {
                double si = std::log(1.0 + vi / CTools::pow2(mi + logNormalOffset));
                double li = std::log(mi + logNormalOffset) - si / 2.0;
                ll1n += ni * ((vi + CTools::pow2(mi - m)) / v + log2piv);
                ll1l += ni * ((si + CTools::pow2(li - l)) / s + 2.0 * li + log2pis);
                ll1g += ni * 2.0 * (b * (mi + gammaOffset) - (a - 1.0) * li + loggn);
                ll2nr += ni * ((vi + CTools::pow2(mi - mr)) / vr + log2pivr);
                ll2lr += ni * ((si + CTools::pow2(li - lr)) / sr + 2.0 * li + log2pisr);
                ll2gr += ni * 2.0 * (br * (mi + gammaOffset) - (ar - 1.0) * li + loggnr);
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to compute BIC gain: " << e.what() << ", n = " << n
                  << ", m = " << m << ", v = " << v << ", wl = " << wl
                  << ", ml = " << ml << ", vl = " << vl << ", wr = " << wr
                  << ", mr = " << mr << ", vr = " << vr);
        return;
    }

    double logn = std::log(n);
    double ll1 =
        min(distributions.haveNormal() ? ll1n : boost::numeric::bounds<double>::highest(),
            distributions.haveLogNormal() ? ll1l : boost::numeric::bounds<double>::highest(),
            haveGamma ? ll1g : boost::numeric::bounds<double>::highest()) +
        distributions.parameters() * logn;
    double ll2 =
        min(distributions.haveNormal() ? ll2nl : boost::numeric::bounds<double>::highest(),
            distributions.haveLogNormal() ? ll2ll : boost::numeric::bounds<double>::highest(),
            haveGamma ? ll2gl : boost::numeric::bounds<double>::highest()) +
        min(distributions.haveNormal() ? ll2nr : boost::numeric::bounds<double>::highest(),
            distributions.haveLogNormal() ? ll2lr : boost::numeric::bounds<double>::highest(),
            haveGamma ? ll2gr : boost::numeric::bounds<double>::highest()) +
        (2.0 * distributions.parameters() + 1.0) * logn;

    LOG_TRACE(<< "BIC(1) = " << ll1 << ", BIC(2) = " << ll2);

    distance = std::max(ll1 - ll2, 0.0);
}

//! Update the mean and variance of \p category to represent
//! truncating the values to the interval \p interval. This
//! is done by Winsorisation, i.e. rather than discarding values
//! outside the interval we restrict them to the closest interval
//! end point. To approximate the effect on the category mean and
//! variance we assume that the underlying distribution is normal
//! and calculate the restricted mean as:
//! <pre class="fragment">
//!    \f$m_{a,b} = a F(a) + \int_a^b{x f(x)}dx + b (1 - F(b))\f$
//! </pre>
//! and the variance as:
//! <pre class="fragment">
//!    \f$m_{a,b} = (a-m_{a,b})^2 F(a) + \int_a^b{(x-m_{a,b})^2 f(x)}dx + (b-m_{a,b})^2 (1 - F(b))\f$
//! </pre>
//!
//! \param[in] interval The Winsorisation interval.
//! \param[in,out] category The category to Winsorise.
void winsorise(const TDoubleDoublePr& interval, TTuple& category) {

    double a = interval.first;
    double b = interval.second;
    double m = CBasicStatistics::mean(category);
    double sigma = std::sqrt(CBasicStatistics::maximumLikelihoodVariance(category));
    double t = 3.0 * sigma;

    double xa = m - a;
    double xb = b - m;

    if (sigma == 0.0 || (xa > t && xb > t)) {
        return;
    }

    try {
        boost::math::normal_distribution<> normal(m, sigma);
        double pa = xa > t ? 0.0 : CTools::safeCdf(normal, a);
        double pb = xb > t ? 0.0 : CTools::safeCdfComplement(normal, b);

        xa /= sigma;
        xb /= sigma;

        double ea = xa > t ? 0.0 : std::exp(-xa * xa / 2.0);
        double eb = xb > t ? 0.0 : std::exp(-xb * xb / 2.0);

        double km = sigma / boost::math::double_constants::root_two_pi * (ea - eb);
        double kv = -sigma * sigma / boost::math::double_constants::root_two_pi *
                    (xa * ea + xb * eb);

        double wm = pa * a + pb * b + m * (1.0 - pb - pa) + km;

        xa = a - wm;
        xb = b - wm;
        double xm = wm - m;
        double wv = xa * xa * pa + xb * xb * pb +
                    (sigma * sigma + xm * xm) * (1.0 - pb - pa) + 2.0 * xm * km + kv;

        double n = CBasicStatistics::count(category);

        category.s_Moments[0] = wm;
        category.s_Moments[1] = std::max((n - 1.0) / n * wv, 0.0);
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Bad category = " << category << ": " << e.what());
    }
}

//! Search for a split of the data that satisfies the constraints
//! on both the BIC divergence and minimum count.
//!
//! In order to handle the constraint on minimum count, we do a
//! breadth first search of the binary tree of optimal 2-splits
//! of subsets of the data looking for splits which satisfy the
//! constraints on *both* BIC divergence and count. The search
//! terminates at any node which can't be split subject to BIC.
//!
//! The intention of this is to find "natural" splits of the data
//! which would be obscured when splitting into the optimal 2-split.
//! This can occur when a small number of points (less than minimum
//! count) are sufficiently far from the others that they split off
//! in preference to some other natural split of the data. Although
//! we can impose the count constraint when finding the optimal
//! 2-split this has associated problems. In particular, extreme
//! outliers then tend to rip sufficient points away from their
//! natural cluster in order to generate a new cluster.
bool splitSearch(double minimumCount,
                 double minimumDistance,
                 maths_t::EDataType dataType,
                 CAvailableModeDistributions distributions,
                 double smallest,
                 const TTupleVec& categories,
                 TSizeVec& result) {

    using TSizeSizePr = std::pair<std::size_t, std::size_t>;

    LOG_TRACE(<< "begin split search");

    result.clear();

    TSizeSizePr node(0, categories.size());
    TTupleVec nodeCategories;
    nodeCategories.reserve(categories.size());
    TSizeVec candidate;
    candidate.reserve(2);

    // The search effectively visits a binary tree of possible
    // 2-splits of the data, which is searched breadth first.
    // If a suitable split is found on a level of the tree then
    // the search terminates returning that split. Note that
    // if a subset of the data can be split we also check that
    // the corresponding full 2-split can be split subject to the
    // same constraints (to avoid merging the split straight away).

    for (;;) {
        LOG_TRACE(<< "node = " << core::CContainerPrinter::print(node));
        LOG_TRACE(<< "categories = " << core::CContainerPrinter::print(categories));

        nodeCategories.assign(categories.begin() + node.first,
                              categories.begin() + node.second);

        CNaturalBreaksClassifier::naturalBreaks(
            nodeCategories, 2, 0, CNaturalBreaksClassifier::E_TargetDeviation, candidate);
        LOG_TRACE(<< "candidate = " << core::CContainerPrinter::print(candidate));

        if (candidate.size() != 2) {
            LOG_ERROR(<< "Expected 2-split: " << core::CContainerPrinter::print(candidate));
            break;
        }
        if (candidate[0] == 0 || candidate[0] == nodeCategories.size()) {
            // This can happen if all the points are co-located,
            // in which case we can't split this node anyway.
            break;
        }

        candidate[0] += node.first;
        candidate[1] += node.first;

        double distance;
        double nl;
        double nr;
        BICGain(dataType, distributions, smallest, categories, node.first,
                candidate[0], node.second, distance, nl, nr);

        // Check the count constraint.
        bool satisfiesCount = (std::min(nl, nr) >= minimumCount);
        LOG_TRACE(<< "count = " << std::min(nl, nr) << " (to split " << minimumCount << ")");

        // Check the distance constraint.
        bool satisfiesDistance = (distance > minimumDistance);
        LOG_TRACE(<< "max(BIC(1) - BIC(2), 0) = " << distance << " (to split "
                  << minimumDistance << ")");

        if (!satisfiesCount) {
            // Recurse to the (one) node with sufficient count.
            if (nl > minimumCount && candidate[0] - node.first > 1) {
                node = {node.first, candidate[0]};
                continue;
            }
            if (nr > minimumCount && node.second - candidate[0] > 1) {
                node = {candidate[0], node.second};
                continue;
            }
        } else if (satisfiesDistance) {
            LOG_TRACE(<< "Checking full split");

            BICGain(dataType, distributions, smallest, categories, 0,
                    candidate[0], categories.size(), distance, nl, nr);

            LOG_TRACE(<< "max(BIC(1) - BIC(2), 0) = " << distance
                      << " (to split " << minimumDistance << ")");

            if (distance > minimumDistance) {
                result.push_back(candidate[0]);
                result.push_back(categories.size());
            }
        }
        break;
    }

    LOG_TRACE(<< "end split search");

    return !result.empty();
}

} // detail::

// 1 - "smallest hard assignment weight"
const double HARD_ASSIGNMENT_THRESHOLD = 0.01;

// CXMeansOnline1d
const ml::core::TPersistenceTag WEIGHT_CALC_TAG("a", "weight");
const ml::core::TPersistenceTag MINIMUM_CLUSTER_FRACTION_TAG("b", "cluster_fraction");
const ml::core::TPersistenceTag MINIMUM_CLUSTER_COUNT_TAG("c", "minimum_cluster_count");
const ml::core::TPersistenceTag
    WINSORISATION_CONFIDENCE_INTERVAL_TAG("d", "winsorisation_confidence_interval");
const ml::core::TPersistenceTag CLUSTER_INDEX_GENERATOR_TAG("e", "index_generator");
const ml::core::TPersistenceTag CLUSTER_TAG("f", "cluster");
const ml::core::TPersistenceTag AVAILABLE_DISTRIBUTIONS_TAG("g", "available_distributions");
const ml::core::TPersistenceTag SMALLEST_TAG("h", "smallest");
const ml::core::TPersistenceTag LARGEST_TAG("i", "largest");
const ml::core::TPersistenceTag DECAY_RATE_TAG("j", "decay_rate");
const ml::core::TPersistenceTag HISTORY_LENGTH_TAG("k", "history_length");

// CXMeansOnline1d::CCluster
const ml::core::TPersistenceTag INDEX_TAG("a", "index");
const ml::core::TPersistenceTag STRUCTURE_TAG("b", "structure");
const ml::core::TPersistenceTag PRIOR_TAG("c", "prior");

const std::string EMPTY_STRING;
}

CAvailableModeDistributions::CAvailableModeDistributions(int value)
    : m_Value(value) {
}

const CAvailableModeDistributions& CAvailableModeDistributions::
operator+(const CAvailableModeDistributions& rhs) {
    m_Value = m_Value | rhs.m_Value;
    return *this;
}

double CAvailableModeDistributions::parameters() const {
    return (this->haveNormal() ? 2.0 : 0.0) + (this->haveGamma() ? 2.0 : 0.0) +
           (this->haveLogNormal() ? 2.0 : 0.0);
}

bool CAvailableModeDistributions::haveNormal() const {
    return (m_Value & NORMAL) != 0;
}

bool CAvailableModeDistributions::haveGamma() const {
    return (m_Value & GAMMA) != 0;
}

bool CAvailableModeDistributions::haveLogNormal() const {
    return (m_Value & LOG_NORMAL) != 0;
}

std::string CAvailableModeDistributions::toString() const {
    return core::CStringUtils::typeToString(m_Value);
}

bool CAvailableModeDistributions::fromString(const std::string& value) {
    return core::CStringUtils::stringToType(value, m_Value);
}

CXMeansOnline1d::CXMeansOnline1d(maths_t::EDataType dataType,
                                 CAvailableModeDistributions availableDistributions,
                                 maths_t::EClusterWeightCalc weightCalc,
                                 double decayRate,
                                 double minimumClusterFraction,
                                 double minimumClusterCount,
                                 double minimumCategoryCount,
                                 double winsorisationConfidenceInterval,
                                 const TSplitFunc& splitFunc,
                                 const TMergeFunc& mergeFunc)
    : CClusterer1d(splitFunc, mergeFunc), m_DataType(dataType),
      m_WeightCalc(weightCalc), m_AvailableDistributions(availableDistributions),
      m_InitialDecayRate(decayRate), m_DecayRate(decayRate),
      m_HistoryLength(0.0), m_MinimumClusterFraction(minimumClusterFraction),
      m_MinimumClusterCount(minimumClusterCount),
      m_MinimumCategoryCount(minimumCategoryCount),
      m_WinsorisationConfidenceInterval(winsorisationConfidenceInterval),
      m_Clusters(1, CCluster(*this)) {
}

CXMeansOnline1d::CXMeansOnline1d(const SDistributionRestoreParams& params,
                                 core::CStateRestoreTraverser& traverser)
    : CClusterer1d(CDoNothing(), CDoNothing()), m_DataType(params.s_DataType),
      m_WeightCalc(maths_t::E_ClustersEqualWeight),
      m_AvailableDistributions(CAvailableModeDistributions::ALL),
      m_InitialDecayRate(params.s_DecayRate), m_DecayRate(params.s_DecayRate),
      m_HistoryLength(), m_MinimumClusterFraction(), m_MinimumClusterCount(),
      m_MinimumCategoryCount(params.s_MinimumCategoryCount),
      m_WinsorisationConfidenceInterval() {
    traverser.traverseSubLevel(std::bind(&CXMeansOnline1d::acceptRestoreTraverser, this,
                                         std::cref(params), std::placeholders::_1));
}

CXMeansOnline1d::CXMeansOnline1d(const SDistributionRestoreParams& params,
                                 const TSplitFunc& splitFunc,
                                 const TMergeFunc& mergeFunc,
                                 core::CStateRestoreTraverser& traverser)
    : CClusterer1d(splitFunc, mergeFunc), m_DataType(params.s_DataType),
      m_WeightCalc(maths_t::E_ClustersEqualWeight),
      m_AvailableDistributions(CAvailableModeDistributions::ALL),
      m_InitialDecayRate(params.s_DecayRate), m_DecayRate(params.s_DecayRate),
      m_HistoryLength(), m_MinimumClusterFraction(), m_MinimumClusterCount(),
      m_MinimumCategoryCount(params.s_MinimumCategoryCount),
      m_WinsorisationConfidenceInterval() {
    traverser.traverseSubLevel(std::bind(&CXMeansOnline1d::acceptRestoreTraverser, this,
                                         std::cref(params), std::placeholders::_1));
}

CXMeansOnline1d::CXMeansOnline1d(const CXMeansOnline1d& other)
    : CClusterer1d(other.splitFunc(), other.mergeFunc()),
      m_DataType(other.m_DataType), m_WeightCalc(other.m_WeightCalc),
      m_AvailableDistributions(other.m_AvailableDistributions),
      m_InitialDecayRate(other.m_InitialDecayRate),
      m_DecayRate(other.m_DecayRate), m_HistoryLength(other.m_HistoryLength),
      m_MinimumClusterFraction(other.m_MinimumClusterFraction),
      m_MinimumClusterCount(other.m_MinimumClusterCount),
      m_MinimumCategoryCount(other.m_MinimumCategoryCount),
      m_WinsorisationConfidenceInterval(other.m_WinsorisationConfidenceInterval),
      m_ClusterIndexGenerator(other.m_ClusterIndexGenerator.deepCopy()),
      m_Smallest(other.m_Smallest), m_Largest(other.m_Largest),
      m_Clusters(other.m_Clusters) {
}

CXMeansOnline1d& CXMeansOnline1d::operator=(const CXMeansOnline1d& other) {
    if (this != &other) {
        CXMeansOnline1d tmp(other);
        this->swap(tmp);
    }
    return *this;
}

void CXMeansOnline1d::swap(CXMeansOnline1d& other) {
    this->CClusterer1d::swap(other);
    std::swap(m_DataType, other.m_DataType);
    std::swap(m_WeightCalc, other.m_WeightCalc);
    std::swap(m_InitialDecayRate, other.m_InitialDecayRate);
    std::swap(m_DecayRate, other.m_DecayRate);
    std::swap(m_HistoryLength, other.m_HistoryLength);
    std::swap(m_MinimumClusterFraction, other.m_MinimumClusterFraction);
    std::swap(m_MinimumClusterCount, other.m_MinimumClusterCount);
    std::swap(m_MinimumCategoryCount, other.m_MinimumCategoryCount);
    std::swap(m_WinsorisationConfidenceInterval, other.m_WinsorisationConfidenceInterval);
    std::swap(m_AvailableDistributions, other.m_AvailableDistributions);
    std::swap(m_ClusterIndexGenerator, other.m_ClusterIndexGenerator);
    std::swap(m_Smallest, other.m_Smallest);
    std::swap(m_Largest, other.m_Largest);
    m_Clusters.swap(other.m_Clusters);
}

ml::core::TPersistenceTag CXMeansOnline1d::persistenceTag() const {
    return X_MEANS_ONLINE_1D_TAG;
}

void CXMeansOnline1d::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    for (const auto& cluster : m_Clusters) {
        inserter.insertLevel(CLUSTER_TAG, std::bind(&CCluster::acceptPersistInserter,
                                                    &cluster, std::placeholders::_1));
    }
    inserter.insertValue(AVAILABLE_DISTRIBUTIONS_TAG, m_AvailableDistributions.toString());
    inserter.insertValue(DECAY_RATE_TAG, m_DecayRate, core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(HISTORY_LENGTH_TAG, m_HistoryLength, core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(SMALLEST_TAG, m_Smallest.toDelimited());
    inserter.insertValue(LARGEST_TAG, m_Largest.toDelimited());
    inserter.insertValue(WEIGHT_CALC_TAG, static_cast<int>(m_WeightCalc));
    inserter.insertValue(MINIMUM_CLUSTER_FRACTION_TAG, m_MinimumClusterFraction.toString());
    inserter.insertValue(MINIMUM_CLUSTER_COUNT_TAG, m_MinimumClusterCount.toString());
    inserter.insertValue(WINSORISATION_CONFIDENCE_INTERVAL_TAG,
                         m_WinsorisationConfidenceInterval.toString());
    inserter.insertLevel(CLUSTER_INDEX_GENERATOR_TAG,
                         std::bind(&CIndexGenerator::acceptPersistInserter,
                                   &m_ClusterIndexGenerator, std::placeholders::_1));
}

CXMeansOnline1d* CXMeansOnline1d::clone() const {
    return new CXMeansOnline1d(*this);
}

void CXMeansOnline1d::clear() {
    *this = CXMeansOnline1d(
        m_DataType, m_AvailableDistributions, m_WeightCalc, m_InitialDecayRate,
        m_MinimumClusterFraction, m_MinimumClusterCount, m_MinimumCategoryCount,
        m_WinsorisationConfidenceInterval, this->splitFunc(), this->mergeFunc());
}

std::size_t CXMeansOnline1d::numberClusters() const {
    return m_Clusters.size();
}

void CXMeansOnline1d::dataType(maths_t::EDataType dataType) {
    m_DataType = dataType;
    for (auto& cluster : m_Clusters) {
        cluster.dataType(dataType);
    }
}

void CXMeansOnline1d::decayRate(double decayRate) {
    m_DecayRate = decayRate;
    for (auto& cluster : m_Clusters) {
        cluster.decayRate(decayRate);
    }
}

bool CXMeansOnline1d::hasCluster(std::size_t index) const {
    return this->cluster(index) != nullptr;
}

bool CXMeansOnline1d::clusterCentre(std::size_t index, double& result) const {
    const CCluster* cluster = this->cluster(index);
    if (!cluster) {
        LOG_ERROR(<< "Cluster " << index << " doesn't exist");
        return false;
    }
    result = cluster->centre();
    return true;
}

bool CXMeansOnline1d::clusterSpread(std::size_t index, double& result) const {
    const CCluster* cluster = this->cluster(index);
    if (!cluster) {
        LOG_ERROR(<< "Cluster " << index << " doesn't exist");
        return false;
    }
    result = cluster->spread();
    return true;
}

void CXMeansOnline1d::cluster(const double& point, TSizeDoublePr2Vec& result, double count) const {

    result.clear();

    if (m_Clusters.empty()) {
        LOG_ERROR(<< "No clusters");
        return;
    }

    auto rightCluster = std::lower_bound(m_Clusters.begin(), m_Clusters.end(),
                                         point, detail::SClusterCentreLess());

    if (rightCluster == m_Clusters.end()) {
        --rightCluster;
        result.emplace_back(rightCluster->index(), count);
    } else if (rightCluster == m_Clusters.begin()) {
        result.emplace_back(rightCluster->index(), count);
    } else {
        // This does a soft assignment. Given we are finding a
        // partitioning clustering (as a result of targeting
        // the k-means objective) we only consider the case that
        // the point comes from either the left or right cluster.
        // A-priori the probability a randomly selected point
        // comes from a cluster is proportional to its weight:
        //   P(i) = n(i) / Sum_j{ n(j) }
        //
        // Bayes theorem then immediately gives that the probability
        // that a given point is from the i'th cluster
        //   P(i | x) = L(x | i) * P(i) / Z
        //
        // where Z is the normalization constant:
        //   Z = Sum_i{ P(i | x) }
        //
        // Below we work with log likelihoods so the normalization
        // constant cancels on either side of the inequality. Note
        // also that we do not want to soft assign the point to a
        // cluster if its probability is close to zero.

        auto leftCluster = rightCluster;
        --leftCluster;
        double likelihoodLeft = leftCluster->logLikelihoodFromCluster(m_WeightCalc, point);
        double likelihoodRight = rightCluster->logLikelihoodFromCluster(m_WeightCalc, point);

        double renormalizer = std::max(likelihoodLeft, likelihoodRight);
        double pLeft = std::exp(likelihoodLeft - renormalizer);
        double pRight = std::exp(likelihoodRight - renormalizer);
        double normalizer = pLeft + pRight;
        pLeft /= normalizer;
        pRight /= normalizer;

        if (pLeft < HARD_ASSIGNMENT_THRESHOLD * pRight) {
            result.emplace_back(rightCluster->index(), count);
        } else if (pRight < HARD_ASSIGNMENT_THRESHOLD * pLeft) {
            result.emplace_back(leftCluster->index(), count);
        } else {
            result.emplace_back(leftCluster->index(), pLeft * count);
            result.emplace_back(rightCluster->index(), pRight * count);
        }
    }
}

void CXMeansOnline1d::add(const double& point, TSizeDoublePr2Vec& clusters, double count) {

    m_Smallest.add(point);
    m_Largest.add(point);

    clusters.clear();

    auto rightCluster = std::lower_bound(m_Clusters.begin(), m_Clusters.end(),
                                         point, detail::SClusterCentreLess());

    if (rightCluster == m_Clusters.end()) {
        --rightCluster;
        LOG_TRACE(<< "Adding " << point << " to " << rightCluster->centre());
        rightCluster->add(point, count);
        clusters.emplace_back(rightCluster->index(), count);
        if (this->maybeSplit(rightCluster)) {
            this->cluster(point, clusters, count);
        } else if (rightCluster != m_Clusters.begin()) {
            auto leftCluster = rightCluster;
            --leftCluster;
            if (this->maybeMerge(leftCluster, rightCluster)) {
                this->cluster(point, clusters, count);
            }
        }
    } else if (rightCluster == m_Clusters.begin()) {
        LOG_TRACE(<< "Adding " << point << " to " << rightCluster->centre());
        rightCluster->add(point, count);
        clusters.emplace_back(rightCluster->index(), count);
        if (this->maybeSplit(rightCluster)) {
            this->cluster(point, clusters, count);
        } else {
            auto leftCluster = rightCluster;
            ++rightCluster;
            if (this->maybeMerge(leftCluster, rightCluster)) {
                this->cluster(point, clusters, count);
            }
        }
    } else {
        // See the cluster member function for more details on
        // soft assignment.
        auto leftCluster = rightCluster;
        --leftCluster;
        double likelihoodLeft = leftCluster->logLikelihoodFromCluster(m_WeightCalc, point);
        double likelihoodRight = rightCluster->logLikelihoodFromCluster(m_WeightCalc, point);

        // Normalize the likelihood values.
        double renormalizer = std::max(likelihoodLeft, likelihoodRight);
        double pLeft = std::exp(likelihoodLeft - renormalizer);
        double pRight = std::exp(likelihoodRight - renormalizer);
        double pLeftPlusRight = pLeft + pRight;
        pLeft /= pLeftPlusRight;
        pRight /= pLeftPlusRight;

        if (pLeft < HARD_ASSIGNMENT_THRESHOLD * pRight) {
            LOG_TRACE(<< "Adding " << point << " to " << rightCluster->centre());
            rightCluster->add(point, count);
            clusters.emplace_back(rightCluster->index(), count);
            if (this->maybeSplit(rightCluster) || this->maybeMerge(leftCluster, rightCluster)) {
                this->cluster(point, clusters, count);
            }
        } else if (pRight < HARD_ASSIGNMENT_THRESHOLD * pLeft) {
            LOG_TRACE(<< "Adding " << point << " to " << leftCluster->centre());
            leftCluster->add(point, count);
            clusters.emplace_back(leftCluster->index(), count);
            if (this->maybeSplit(leftCluster) || this->maybeMerge(leftCluster, rightCluster)) {
                this->cluster(point, clusters, count);
            }
        } else {
            // Get the weighted counts.
            double countLeft = count * pLeft;
            double countRight = count * pRight;
            LOG_TRACE(<< "Soft adding " << point << " " << countLeft << " to "
                      << leftCluster->centre() << " and " << countRight
                      << " to " << rightCluster->centre());

            leftCluster->add(point, countLeft);
            rightCluster->add(point, countRight);
            clusters.emplace_back(leftCluster->index(), countLeft);
            clusters.emplace_back(rightCluster->index(), countRight);
            if (this->maybeSplit(leftCluster) || this->maybeSplit(rightCluster) ||
                this->maybeMerge(leftCluster, rightCluster)) {
                this->cluster(point, clusters, count);
            }
        }
    }

    if (this->prune()) {
        this->cluster(point, clusters, count);
    }
}

void CXMeansOnline1d::add(const TDoubleDoublePrVec& points) {
    if (m_Clusters.empty()) {
        m_Clusters.emplace_back(*this);
    }
    TSizeDoublePr2Vec dummy;
    for (const auto& point : points) {
        this->add(point.first, dummy, point.second);
    }
}

void CXMeansOnline1d::propagateForwardsByTime(double time) {
    if (time < 0.0) {
        LOG_ERROR(<< "Can't propagate backwards in time");
        return;
    }
    m_HistoryLength = (m_HistoryLength + time) * std::exp(-m_DecayRate * time);
    for (auto& cluster : m_Clusters) {
        cluster.propagateForwardsByTime(time);
    }
}

bool CXMeansOnline1d::sample(std::size_t index, std::size_t numberSamples, TDoubleVec& samples) const {
    const CCluster* cluster = this->cluster(index);
    if (!cluster) {
        LOG_ERROR(<< "Cluster " << index << " doesn't exist");
        return false;
    }
    cluster->sample(numberSamples, std::min(m_Smallest[0], 0.0), m_Largest[0], samples);
    return true;
}

double CXMeansOnline1d::probability(std::size_t index) const {
    double weight = 0.0;
    double Z = 0.0;
    for (const auto& cluster : m_Clusters) {
        if (cluster.index() == index) {
            weight = cluster.weight(maths_t::E_ClustersFractionWeight);
        }
        Z += cluster.weight(maths_t::E_ClustersFractionWeight);
    }
    return Z == 0.0 ? 0.0 : weight / Z;
}

void CXMeansOnline1d::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CXMeansOnline1d");
    core::CMemoryDebug::dynamicSize("m_ClusterIndexGenerator", m_ClusterIndexGenerator, mem);
    core::CMemoryDebug::dynamicSize("m_Clusters", m_Clusters, mem);
}

std::size_t CXMeansOnline1d::memoryUsage() const {
    std::size_t mem = core::CMemory::dynamicSize(m_ClusterIndexGenerator);
    mem += core::CMemory::dynamicSize(m_Clusters);
    return mem;
}

std::size_t CXMeansOnline1d::staticSize() const {
    return sizeof(*this);
}

uint64_t CXMeansOnline1d::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_DataType);
    seed = CChecksum::calculate(seed, m_DecayRate);
    seed = CChecksum::calculate(seed, m_HistoryLength);
    seed = CChecksum::calculate(seed, m_WeightCalc);
    return CChecksum::calculate(seed, m_Clusters);
}

double CXMeansOnline1d::count() const {
    return std::accumulate(m_Clusters.begin(), m_Clusters.end(), 0.0,
                           [](double count, const CCluster& cluster) {
                               return count + cluster.count();
                           });
}

const CXMeansOnline1d::TClusterVec& CXMeansOnline1d::clusters() const {
    return m_Clusters;
}

std::string CXMeansOnline1d::printClusters() const {

    if (m_Clusters.empty()) {
        return std::string();
    }

    std::ostringstream result;

    // We'll plot the marginal likelihood function over a range
    // where most of the mass is, i.e. the 99.9% confidence

    static const double RANGE = 99.9;
    static const unsigned int POINTS = 201;

    TDoubleDoublePr range(boost::numeric::bounds<double>::highest(),
                          boost::numeric::bounds<double>::lowest());

    for (const auto& cluster : m_Clusters) {
        const CPrior& prior = cluster.prior();
        TDoubleDoublePr clusterRange = prior.marginalLikelihoodConfidenceInterval(RANGE);
        range.first = std::min(range.first, clusterRange.first);
        range.second = std::max(range.second, clusterRange.second);
    }

    double Z = 0.0;
    for (const auto& cluster : m_Clusters) {
        Z += cluster.weight(m_WeightCalc);
    }

    TDouble1Vec x{range.first};
    double increment = (range.second - range.first) / (POINTS - 1.0);

    std::ostringstream coordinatesStr;
    std::ostringstream likelihoodStr;
    coordinatesStr << "x = [";
    likelihoodStr << "likelihood = [";
    for (unsigned int i = 0; i < POINTS; ++i, x[0] += increment) {
        double likelihood = 0.0;
        for (const auto& cluster : m_Clusters) {
            double logLikelihood;
            const CPrior& prior = cluster.prior();
            if (!(prior.jointLogMarginalLikelihood(x, maths_t::CUnitWeights::SINGLE_UNIT, logLikelihood) &
                  (maths_t::E_FpFailed | maths_t::E_FpOverflowed))) {
                likelihood += cluster.weight(m_WeightCalc) / Z * std::exp(logLikelihood);
            }
        }
        coordinatesStr << x[0] << " ";
        likelihoodStr << likelihood << " ";
    }
    coordinatesStr << "];" << core_t::LINE_ENDING;
    likelihoodStr << "];" << core_t::LINE_ENDING << "plot(x, likelihood);";

    return coordinatesStr.str() + likelihoodStr.str();
}

CXMeansOnline1d::CIndexGenerator& CXMeansOnline1d::indexGenerator() {
    return m_ClusterIndexGenerator;
}

bool CXMeansOnline1d::acceptRestoreTraverser(const SDistributionRestoreParams& params,
                                             core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE_SETUP_TEARDOWN(CLUSTER_TAG, CCluster cluster(*this),
                               traverser.traverseSubLevel(std::bind(
                                   &CCluster::acceptRestoreTraverser, &cluster,
                                   std::cref(params), std::placeholders::_1)),
                               m_Clusters.push_back(cluster))
        RESTORE(AVAILABLE_DISTRIBUTIONS_TAG,
                m_AvailableDistributions.fromString(traverser.value()))
        RESTORE_SETUP_TEARDOWN(DECAY_RATE_TAG, double decayRate,
                               core::CStringUtils::stringToType(traverser.value(), decayRate),
                               this->decayRate(decayRate))
        RESTORE(HISTORY_LENGTH_TAG, m_HistoryLength.fromString(traverser.value()));
        RESTORE(SMALLEST_TAG, m_Smallest.fromDelimited(traverser.value()))
        RESTORE(LARGEST_TAG, m_Largest.fromDelimited(traverser.value()))
        RESTORE(CLUSTER_INDEX_GENERATOR_TAG,
                traverser.traverseSubLevel(
                    std::bind(&CIndexGenerator::acceptRestoreTraverser,
                              &m_ClusterIndexGenerator, std::placeholders::_1)))
        RESTORE_ENUM(WEIGHT_CALC_TAG, m_WeightCalc, maths_t::EClusterWeightCalc)
        RESTORE(MINIMUM_CLUSTER_FRACTION_TAG,
                m_MinimumClusterFraction.fromString(traverser.value()))
        RESTORE(MINIMUM_CLUSTER_COUNT_TAG,
                m_MinimumClusterCount.fromString(traverser.value()))
        RESTORE(WINSORISATION_CONFIDENCE_INTERVAL_TAG,
                m_WinsorisationConfidenceInterval.fromString(traverser.value()))
    } while (traverser.next());

    return true;
}

const CXMeansOnline1d::CCluster* CXMeansOnline1d::cluster(std::size_t index) const {
    for (const auto& cluster : m_Clusters) {
        if (cluster.index() == index) {
            return &cluster;
        }
    }
    return nullptr;
}

double CXMeansOnline1d::minimumSplitCount() const {
    double result = m_MinimumClusterCount;
    if (m_MinimumClusterFraction > 0.0) {
        double count = this->count();
        double scale = m_HistoryLength * (1.0 - std::exp(-m_InitialDecayRate));
        count *= m_MinimumClusterFraction / std::max(scale, 1.0);
        result = std::max(result, count);
    }
    LOG_TRACE(<< "minimumSplitCount = " << result);
    return result;
}

bool CXMeansOnline1d::maybeSplit(TClusterVecItr cluster) {
    if (cluster == m_Clusters.end()) {
        return false;
    }
    TDoubleDoublePr interval = this->winsorisationInterval();
    if (TOptionalClusterClusterPr split =
            cluster->split(m_AvailableDistributions, this->minimumSplitCount(),
                           m_Smallest[0], interval, m_ClusterIndexGenerator)) {
        LOG_TRACE(<< "Splitting cluster " << cluster->index() << " at "
                  << cluster->centre());
        std::size_t index = cluster->index();
        *cluster = split->second;
        m_Clusters.insert(cluster, split->first);
        (this->splitFunc())(index, split->first.index(), split->second.index());
        return true;
    }
    return false;
}

bool CXMeansOnline1d::maybeMerge(TClusterVecItr cluster1, TClusterVecItr cluster2) {
    if (cluster1 == m_Clusters.end() || cluster2 == m_Clusters.end()) {
        return false;
    }
    TDoubleDoublePr interval = this->winsorisationInterval();
    if (cluster1->shouldMerge(*cluster2, m_AvailableDistributions, m_Smallest[0], interval)) {
        LOG_TRACE(<< "Merging cluster " << cluster1->index() << " at "
                  << cluster1->centre() << " and cluster " << cluster2->index()
                  << " at " << cluster2->centre());
        std::size_t index1 = cluster1->index();
        std::size_t index2 = cluster2->index();
        CCluster merged = cluster1->merge(*cluster2, m_ClusterIndexGenerator);
        *cluster1 = merged;
        m_Clusters.erase(cluster2);
        (this->mergeFunc())(index1, index2, merged.index());
        return true;
    }
    return false;
}

bool CXMeansOnline1d::prune() {

    if (m_Clusters.size() <= 1) {
        return false;
    }

    bool result = false;

    double minimumCount = this->minimumSplitCount() * CLUSTER_DELETE_FRACTION;
    for (std::size_t i = 1u; i < m_Clusters.size(); /**/) {
        CCluster& left = m_Clusters[i - 1];
        CCluster& right = m_Clusters[i];
        if (left.count() < minimumCount || right.count() < minimumCount) {
            std::size_t leftIndex = left.index();
            std::size_t rightIndex = right.index();
            LOG_TRACE(<< "Merging cluster " << leftIndex << " at " << left.centre()
                      << " and cluster " << rightIndex << " at " << right.centre());
            CCluster merge = left.merge(right, m_ClusterIndexGenerator);
            left = merge;
            m_Clusters.erase(m_Clusters.begin() + i);
            (this->mergeFunc())(leftIndex, rightIndex, merge.index());
            result = true;
        } else {
            ++i;
        }
    }

    return result;
}

TDoubleDoublePr CXMeansOnline1d::winsorisationInterval() const {

    double f = (1.0 - m_WinsorisationConfidenceInterval) / 2.0;

    if (f * this->count() < 1.0) {
        // Don't bother if we don't expect a sample outside the
        // Winsorisation interval.
        return {boost::numeric::bounds<double>::lowest() / 2.0,
                boost::numeric::bounds<double>::highest() / 2.0};
    }

    // The Winsorisation interval are the positions corresponding
    // to the f/2 and 1 - f/2 percentile counts where f is the
    // Winsorisation confidence interval, i.e. we truncate the
    // data to the 1 - f central confidence interval.

    double totalCount = this->count();
    double leftCount = f * totalCount;
    double rightCount = (1.0 - f) * totalCount;
    LOG_TRACE(<< "totalCount = " << totalCount << " interval = [" << leftCount
              << "," << rightCount << "]"
              << " # clusters = " << m_Clusters.size());

    TDoubleDoublePr result;

    double partialCount = 0.0;
    for (const auto& cluster : m_Clusters) {
        double count = cluster.count();
        if (partialCount < leftCount && partialCount + count >= leftCount) {
            double p = 100.0 * (leftCount - partialCount) / count;
            result.first = cluster.percentile(p);
        }
        if (partialCount < rightCount && partialCount + count >= rightCount) {
            double p = 100.0 * (rightCount - partialCount) / count;
            result.second = cluster.percentile(p);
            break;
        }
        partialCount += count;
    }

    LOG_TRACE(<< "Winsorisation interval = [" << result.first << "," << result.second << "]");

    return result;
}

//////////// CCluster Implementation ////////////

CXMeansOnline1d::CCluster::CCluster(const CXMeansOnline1d& clusterer)
    : m_Index(clusterer.m_ClusterIndexGenerator.next()),
      m_Prior(CNormalMeanPrecConjugate::nonInformativePrior(clusterer.m_DataType,
                                                            clusterer.m_DecayRate)),
      m_Structure(STRUCTURE_SIZE, clusterer.m_DecayRate, clusterer.m_MinimumCategoryCount) {
}

CXMeansOnline1d::CCluster::CCluster(std::size_t index,
                                    const CNormalMeanPrecConjugate& prior,
                                    const CNaturalBreaksClassifier& structure)
    : m_Index(index), m_Prior(prior), m_Structure(structure) {
}

bool CXMeansOnline1d::CCluster::acceptRestoreTraverser(const SDistributionRestoreParams& params,
                                                       core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE_BUILT_IN(INDEX_TAG, m_Index)
        RESTORE_NO_ERROR(PRIOR_TAG, m_Prior = CNormalMeanPrecConjugate(params, traverser))
        RESTORE(STRUCTURE_TAG, traverser.traverseSubLevel(std::bind(
                                   &CNaturalBreaksClassifier::acceptRestoreTraverser, &m_Structure,
                                   std::cref(params), std::placeholders::_1)))
    } while (traverser.next());

    return true;
}

void CXMeansOnline1d::CCluster::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(INDEX_TAG, m_Index);
    inserter.insertLevel(PRIOR_TAG, std::bind(&CNormalMeanPrecConjugate::acceptPersistInserter,
                                              &m_Prior, std::placeholders::_1));
    inserter.insertLevel(STRUCTURE_TAG,
                         std::bind(&CNaturalBreaksClassifier::acceptPersistInserter,
                                   &m_Structure, std::placeholders::_1));
}

void CXMeansOnline1d::CCluster::dataType(maths_t::EDataType dataType) {
    m_Prior.dataType(dataType);
}

void CXMeansOnline1d::CCluster::add(double point, double count) {
    m_Prior.addSamples({point}, {maths_t::countWeight(count)});
    m_Structure.add(point, count);
}

void CXMeansOnline1d::CCluster::decayRate(double decayRate) {
    m_Prior.decayRate(decayRate);
    m_Structure.decayRate(decayRate);
}

void CXMeansOnline1d::CCluster::propagateForwardsByTime(double time) {
    m_Prior.propagateForwardsByTime(time);
    m_Structure.propagateForwardsByTime(time);
}

std::size_t CXMeansOnline1d::CCluster::index() const {
    return m_Index;
}

double CXMeansOnline1d::CCluster::centre() const {
    return m_Prior.marginalLikelihoodMean();
}

double CXMeansOnline1d::CCluster::spread() const {
    return std::sqrt(m_Prior.marginalLikelihoodVariance());
}

double CXMeansOnline1d::CCluster::percentile(double p) const {
    return m_Structure.percentile(p);
}

double CXMeansOnline1d::CCluster::count() const {
    return m_Prior.numberSamples();
}

double CXMeansOnline1d::CCluster::weight(maths_t::EClusterWeightCalc calc) const {
    switch (calc) {
    case maths_t::E_ClustersEqualWeight:
        return 1.0;
    case maths_t::E_ClustersFractionWeight:
        return m_Prior.numberSamples();
    }
    LOG_ABORT(<< "Unexpected calculation style " << calc);
}

double CXMeansOnline1d::CCluster::logLikelihoodFromCluster(maths_t::EClusterWeightCalc calc,
                                                           double point) const {
    double result;
    if (detail::logLikelihoodFromCluster(point, m_Prior, this->weight(calc), result) &
        maths_t::E_FpFailed) {
        LOG_ERROR(<< "Unable to compute likelihood for: " << m_Index);
    }
    return result;
}

void CXMeansOnline1d::CCluster::sample(std::size_t numberSamples,
                                       double smallest,
                                       double largest,
                                       TDoubleVec& samples) const {
    m_Structure.sample(numberSamples, smallest, largest, samples);
}

CXMeansOnline1d::TOptionalClusterClusterPr
CXMeansOnline1d::CCluster::split(CAvailableModeDistributions distributions,
                                 double minimumCount,
                                 double smallest,
                                 const TDoubleDoublePr& interval,
                                 CIndexGenerator& indexGenerator) {

    // We do our clustering top down to minimize space and avoid
    // making splits before we are confident they exist. This is
    // important for anomaly detection because we do *not* want
    // to fit a cluster to an outlier and judge it to be not
    // anomalous as a result.
    //
    // By analogy to x-means we choose a candidate split of the
    // data by minimizing the total within class deviation of the
    // two classes. In order to decide whether or not to split we
    // 1) impose minimum count on the smaller cluster 2) use an
    // information theoretic criterion. Specifically, we threshold
    // the BIC gain of using the multi-mode distribution verses
    // the single mode distribution.

    LOG_TRACE(<< "split");

    if (m_Structure.buffering()) {
        return {};
    }

    maths_t::EDataType dataType = m_Prior.dataType();
    double decayRate = m_Prior.decayRate();

    std::size_t n = m_Structure.size();
    if (n < 2) {
        return {};
    }

    TSizeVec split;
    {
        TTupleVec categories;
        m_Structure.categories(n, 0, categories);
        for (auto& category : categories) {
            detail::winsorise(interval, category);
        }
        if (!detail::splitSearch(minimumCount, MINIMUM_SPLIT_DISTANCE, dataType,
                                 distributions, smallest, categories, split)) {
            return {};
        }
    }

    TTupleVec categories;
    m_Structure.categories(split, categories);

    CNaturalBreaksClassifier::TClassifierVec classifiers;
    m_Structure.split(split, classifiers);
    LOG_TRACE(<< "Splitting cluster " << this->index() << " at "
              << this->centre() << " left = " << classifiers[0].print()
              << ", right = " << classifiers[1].print());

    std::size_t index1 = indexGenerator.next();
    std::size_t index2 = indexGenerator.next();
    indexGenerator.recycle(m_Index);

    CNormalMeanPrecConjugate leftNormal(dataType, categories[0], decayRate);
    CNormalMeanPrecConjugate rightNormal(dataType, categories[1], decayRate);
    return TClusterClusterPr{CCluster(index1, leftNormal, classifiers[0]),
                             CCluster(index2, rightNormal, classifiers[1])};
}

bool CXMeansOnline1d::CCluster::shouldMerge(CCluster& other,
                                            CAvailableModeDistributions distributions,
                                            double smallest,
                                            const TDoubleDoublePr& interval) {

    if (m_Structure.buffering() || m_Structure.size() == 0 ||
        other.m_Structure.size() == 0) {
        return false;
    }

    maths_t::EDataType dataType = m_Prior.dataType();
    TTupleVec categories;
    if (!m_Structure.categories(m_Structure.size(), 0, categories)) {
        return false;
    }
    std::size_t split = categories.size();
    if (!other.m_Structure.categories(other.m_Structure.size(), 0, categories, true)) {
        return false;
    }

    for (auto& category : categories) {
        detail::winsorise(interval, category);
    }

    double distance;
    double nl;
    double nr;
    detail::BICGain(dataType, distributions, smallest, categories, 0, split,
                    categories.size(), distance, nl, nr);
    LOG_TRACE(<< "max(BIC(1) - BIC(2), 0) = " << distance << " (to merge "
              << MAXIMUM_MERGE_DISTANCE << ")");

    return distance <= MAXIMUM_MERGE_DISTANCE;
}

CXMeansOnline1d::CCluster
CXMeansOnline1d::CCluster::merge(CCluster& other, CIndexGenerator& indexGenerator) {

    TTupleVec left, right;
    m_Structure.categories(1, 0, left);
    other.m_Structure.categories(1, 0, right);

    std::size_t index = indexGenerator.next();

    CNormalMeanPrecConjugate::TMeanVarAccumulator mergedCategories;

    if (left.size() > 0) {
        LOG_TRACE(<< "left = " << left[0]);
        mergedCategories += left[0];
    }

    if (right.size() > 0) {
        LOG_TRACE(<< "right = " << right[0]);
        mergedCategories += right[0];
    }

    CNormalMeanPrecConjugate prior(m_Prior.dataType(), mergedCategories,
                                   m_Prior.decayRate());

    CNaturalBreaksClassifier structure(m_Structure);
    structure.merge(other.m_Structure);

    CCluster result(index, prior, structure);

    indexGenerator.recycle(m_Index);
    indexGenerator.recycle(other.m_Index);

    return result;
}

const CNormalMeanPrecConjugate& CXMeansOnline1d::CCluster::prior() const {
    return m_Prior;
}

uint64_t CXMeansOnline1d::CCluster::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_Index);
    seed = CChecksum::calculate(seed, m_Prior);
    return CChecksum::calculate(seed, m_Structure);
}

void CXMeansOnline1d::CCluster::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CXMeansOnline1d::CCluster");
    core::CMemoryDebug::dynamicSize("m_Prior", m_Prior, mem);
    core::CMemoryDebug::dynamicSize("m_Structure", m_Structure, mem);
}

std::size_t CXMeansOnline1d::CCluster::memoryUsage() const {
    std::size_t mem = core::CMemory::dynamicSize(m_Prior);
    mem += core::CMemory::dynamicSize(m_Structure);
    return mem;
}

const double CXMeansOnline1d::WINSORISATION_CONFIDENCE_INTERVAL(1.0);
const double CXMeansOnline1d::MINIMUM_SPLIT_DISTANCE(6.0);
const double CXMeansOnline1d::MAXIMUM_MERGE_DISTANCE(2.0);
const double CXMeansOnline1d::CLUSTER_DELETE_FRACTION(0.8);
const std::size_t CXMeansOnline1d::STRUCTURE_SIZE(12u);
}
}
