/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <maths/common/CBayesianOptimisation.h>

#include <core/CContainerPrinter.h>
#include <core/CIEEE754.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CMemory.h>
#include <core/CPersistUtils.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CChecksum.h>
#include <maths/common/CLbfgs.h>
#include <maths/common/CLinearAlgebraEigen.h>
#include <maths/common/CLinearAlgebraShims.h>
#include <maths/common/CMathsFuncs.h>
#include <maths/common/CSampling.h>
#include <maths/common/CTools.h>

#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/optional/optional_io.hpp>

#include <cmath>
#include <exception>
#include <limits>

namespace ml {
namespace maths {
namespace common {
namespace {
using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
using TMinAccumulator =
    CBasicStatistics::COrderStatisticsHeap<std::pair<double, CBayesianOptimisation::TVector>>;

const std::string VERSION_7_5_TAG{"7.5"};
const std::string MIN_BOUNDARY_TAG{"min_boundary"};
const std::string MAX_BOUNDARY_TAG{"max_boundary"};
const std::string ERROR_VARIANCES_TAG{"error_variances"};
const std::string KERNEL_PARAMETERS_TAG{"kernel_parameters"};
const std::string MIN_KERNEL_COORDINATE_DISTANCE_SCALES_TAG{"min_kernel_coordinate_distance_scales"};
const std::string FUNCTION_MEAN_VALUES_TAG{"function_mean_values"};
const std::string RANGE_SHIFT_TAG{"range_shift"};
const std::string RANGE_SCALE_TAG{"range_scale"};
const std::string RESTARTS_TAG{"restarts"};
const std::string RNG_TAG{"rng"};

// The kernel we use is v * I + a(0)^2 * O(I). We fall back to random search when
// a(0)^2 < eps * v since for small eps and a reasonable number of dimensions the
// expected improvement will be constant in the space we search. We don't terminate
// altogether because it is possible that the function we're interpolating has a
// narrow deep valley that the Gaussian Process hasn't sampled.
const double MINIMUM_KERNEL_SCALE_FOR_EXPECTATION_MAXIMISATION{1e-8};

//! Affine transform \p scale * (\p fx - \p shift).
double toScaled(double shift, double scale, double fx) {
    return scale * (fx - shift);
}

//! Affine transform \p shift + \p scale / \p fx.
double fromScaled(double shift, double scale, double fx) {
    return shift + fx / scale;
}

//! A version of the normal c.d.f. which is stable across our target platforms.
double stableNormCdf(double z) {
    return CTools::stable(CTools::safeCdf(boost::math::normal{0.0, 1.0}, z));
}

//! A version of the normal p.d.f. which is stable across our target platforms.
double stableNormPdf(double z) {
    return CTools::stable(CTools::safePdf(boost::math::normal{0.0, 1.0}, z));
}

double integrate1dKernel(double theta1, double x) {
    double c{std::sqrt(CTools::pow2(theta1) + MINIMUM_KERNEL_SCALE_FOR_EXPECTATION_MAXIMISATION)};
    return CTools::stable(boost::math::constants::root_pi<double>() *
                          (std::erf(c * (1 - x)) + std::erf(c * x)) / (2 * c));
}

double integrate1dKernelProduct(double theta1, double xit, double xjt) {
    double c{std::sqrt(CTools::pow2(theta1) + MINIMUM_KERNEL_SCALE_FOR_EXPECTATION_MAXIMISATION)};
    return CTools::stable(
        boost::math::constants::root_half_pi<double>() / (2 * c) *
        CTools::stableExp(-0.5 * CTools::pow2(c) * CTools::pow2(xit - xjt)) *
        (CTools::stable(std::erf(c / boost::math::constants::root_two<double>() * (xit + xjt))) -
         CTools::stable(std::erf(c / boost::math::constants::root_two<double>() *
                                 (xit + xjt - 2)))));
}
}

CBayesianOptimisation::CBayesianOptimisation(TDoubleDoublePrVec parameterBounds,
                                             std::size_t restarts)
    : m_Restarts{restarts}, m_MinBoundary(parameterBounds.size()),
      m_MaxBoundary(parameterBounds.size()),
      m_KernelParameters(parameterBounds.size() + 1),
      m_MinimumKernelCoordinateDistanceScale(parameterBounds.size()) {

    m_KernelParameters.setOnes();
    m_MinimumKernelCoordinateDistanceScale.setConstant(
        parameterBounds.size(), MINIMUM_KERNEL_COORDINATE_DISTANCE_SCALE);

    for (std::size_t i = 0; i < parameterBounds.size(); ++i) {
        m_MinBoundary(i) = parameterBounds[i].first;
        m_MaxBoundary(i) = parameterBounds[i].second;
    }
}

CBayesianOptimisation::CBayesianOptimisation(core::CStateRestoreTraverser& traverser) {
    if (traverser.traverseSubLevel([this](auto& traverser_) {
            return this->acceptRestoreTraverser(traverser_);
        }) == false) {
        throw std::runtime_error{"failed to restore Bayesian optimisation"};
    }
}

void CBayesianOptimisation::add(TVector x, double fx, double vx) {
    if (CMathsFuncs::isFinite(fx) == false || CMathsFuncs::isFinite(vx) == false) {
        LOG_ERROR(<< "Discarding point (" << x.transpose() << "," << fx << "," << vx << ")");
        return;
    }

    x = this->to01(std::move(x));
    fx = toScaled(m_RangeShift, m_RangeScale, fx);
    vx = CTools::pow2(m_RangeScale) * vx;

    std::size_t duplicate(std::find_if(m_FunctionMeanValues.begin(),
                                       m_FunctionMeanValues.end(),
                                       [&](const auto& value) {
                                           return (x - value.first).norm() == 0.0;
                                       }) -
                          m_FunctionMeanValues.begin());
    if (duplicate < m_FunctionMeanValues.size()) {
        auto& f = m_FunctionMeanValues[duplicate].second;
        auto& v = m_ErrorVariances[duplicate];
        auto moments = CBasicStatistics::momentsAccumulator(1.0, f, v) +
                       CBasicStatistics::momentsAccumulator(1.0, fx, vx);
        f = CBasicStatistics::mean(moments);
        v = CBasicStatistics::maximumLikelihoodVariance(moments);
    } else {
        m_FunctionMeanValues.emplace_back(std::move(x), fx);
        m_ErrorVariances.push_back(vx);
    }
}

void CBayesianOptimisation::reset() {
    m_FunctionMeanValues.clear();
    m_ErrorVariances.clear();
}

void CBayesianOptimisation::explainedErrorVariance(double vx) {
    m_ExplainedErrorVariance = CTools::pow2(m_RangeScale) * vx;
}

std::pair<CBayesianOptimisation::TVector, CBayesianOptimisation::TVector>
CBayesianOptimisation::boundingBox() const {
    return {m_MinBoundary, m_MaxBoundary};
}

std::pair<CBayesianOptimisation::TVector, CBayesianOptimisation::TOptionalDouble>
CBayesianOptimisation::maximumExpectedImprovement(double negligibleExpectedImprovement) {

    // Reapply conditioning and recompute the maximum likelihood kernel parameters.
    this->maximumLikelihoodKernel();

    TVector xmax;
    double fmax{-1.0};

    TEIFunc minusEI;
    TEIGradientFunc minusEIGradient;
    std::tie(minusEI, minusEIGradient) = this->minusExpectedImprovementAndGradient();

    // Use random restarts inside the constraint bounding box.
    TVector interpolate(m_MinBoundary.size());
    TDoubleVec interpolates;
    CSampling::uniformSample(m_Rng, 0.0, 1.0, 10 * m_Restarts * interpolate.size(), interpolates);

    TVector a{TVector::Zero(m_MinBoundary.size())};
    TVector b{TVector::Ones(m_MaxBoundary.size())};
    TVector x;
    TMeanAccumulator rho_;
    TMinAccumulator probes{m_Restarts};

    for (int i = 0; i < interpolate.size(); ++i) {
        interpolate(i) = interpolates[i];
    }
    xmax = a + interpolate.cwiseProduct(b - a);

    if (CTools::pow2(m_KernelParameters(0)) <
        MINIMUM_KERNEL_SCALE_FOR_EXPECTATION_MAXIMISATION * this->meanErrorVariance()) {

        for (std::size_t i = interpolate.size(); i < interpolates.size(); /**/) {
            for (int j = 0; j < interpolate.size(); ++i, ++j) {
                interpolate(j) = interpolates[i];
            }
            x = a + interpolate.cwiseProduct(b - a);
            if (this->dissimilarity(x) > this->dissimilarity(xmax)) {
                xmax = x;
            }
        }

    } else {

        for (std::size_t i = 0; i < interpolates.size(); /**/) {
            for (int j = 0; j < interpolate.size(); ++i, ++j) {
                interpolate(j) = interpolates[i];
            }
            x = a + interpolate.cwiseProduct(b - a);
            double fx{minusEI(x)};
            LOG_TRACE(<< "x = " << x.transpose() << " EI(x) = " << fx);

            if (-fx > fmax + negligibleExpectedImprovement ||
                this->dissimilarity(x) > this->dissimilarity(xmax)) {
                xmax = x;
                fmax = -fx;
            }
            rho_.add(std::fabs(fx));
            if (-fx > negligibleExpectedImprovement) {
                probes.add({fx, std::move(x)});
            }
        }

        // We set rho to give the constraint and objective approximately equal priority
        // in the following constrained optimisation problem.
        double rho{CBasicStatistics::mean(rho_)};
        LOG_TRACE(<< "rho = " << rho);

        CLbfgs<TVector> lbfgs{10};

        TVector xcand;
        double fcand;
        for (auto& x0 : probes) {
            LOG_TRACE(<< "x0 = " << x0.second.transpose());
            std::tie(xcand, fcand) = lbfgs.constrainedMinimize(
                minusEI, minusEIGradient, a, b, std::move(x0.second), rho);
            LOG_TRACE(<< "xcand = " << xcand.transpose() << " EI(cand) = " << fcand);
            if (-fcand > fmax + negligibleExpectedImprovement ||
                this->dissimilarity(xcand) > this->dissimilarity(xmax)) {
                std::tie(xmax, fmax) = std::make_pair(std::move(xcand), -fcand);
            }
        }
    }

    TOptionalDouble expectedImprovement;
    if (fmax >= 0.0 && CMathsFuncs::isFinite(fmax)) {
        expectedImprovement = fmax / m_RangeScale;
    }

    xmax = this->from01(std::move(xmax));
    LOG_TRACE(<< "best = " << xmax.transpose() << " EI(best) = " << expectedImprovement);

    return {std::move(xmax), expectedImprovement};
}

double CBayesianOptimisation::evaluate(const TVector& input) const {
    return this->evaluate(this->kinvf(), input);
}

double CBayesianOptimisation::evaluate(const TVector& Kinvf, const TVector& input) const {
    TVector Kxn;
    std::tie(Kxn, std::ignore) = this->kernelCovariates(
        m_KernelParameters, this->to01(input), this->meanErrorVariance());
    return fromScaled(m_RangeShift, m_RangeScale, Kxn.transpose() * Kinvf);
}

double CBayesianOptimisation::evaluate1D(const TVector& Kinvf, double input, int dimension) const {
    auto prodXt = [this](const TVector& x, int t) -> double {
        double prod{1.0};
        for (int d = 0; d < m_MinBoundary.size(); ++d) {
            if (d != t) {
                prod *= integrate1dKernel(m_KernelParameters(d + 1), x(d));
            }
        }
        return prod;
    };

    double sum{0.0};
    input = (input - m_MinBoundary(dimension)) /
            (m_MaxBoundary(dimension) - m_MinBoundary(dimension));
    for (std::size_t i = 0; i < m_FunctionMeanValues.size(); ++i) {
        const TVector& x{m_FunctionMeanValues[i].first};
        sum += Kinvf(static_cast<int>(i)) *
               CTools::stableExp(-(CTools::pow2(m_KernelParameters[dimension + 1]) +
                                   MINIMUM_KERNEL_COORDINATE_DISTANCE_SCALE) *
                                 CTools::pow2(input - x(dimension))) *
               prodXt(x, dimension);
    }
    double f2{this->anovaConstantFactor(Kinvf)};

    // We only get cancellation if the signs are the same (and we need also
    // to take the square root of both sum and f2 for which they need to be
    // positive).
    if (std::signbit(sum) == std::signbit(f2)) {
        // We rewrite theta_0^2 sum - f_0 as (theta + f) * (theta - f) where
        // theta = theta_0 sum^(1/2) and f = f_0^(1/2) because it has better
        // numerics.
        double theta{m_KernelParameters(0) * std::sqrt(std::fabs(sum))};
        double f{std::sqrt(std::fabs(f2))};
        return fromScaled(m_RangeShift, m_RangeScale,
                          std::copysign(1.0, sum) * (theta + f) * (theta - f));
    }
    return fromScaled(m_RangeShift, m_RangeScale,
                      CTools::pow2(m_KernelParameters(0)) * sum - f2);
}

double CBayesianOptimisation::evaluate1D(double input, int dimension) const {
    if (dimension < 0 || dimension > m_MinBoundary.size()) {
        LOG_ERROR(<< "Input error: dimension " << dimension << " is out of bounds. "
                  << "It should be between 0 and " << m_MinBoundary.size() << ".");
        return 0.0;
    }
    return this->evaluate1D(this->kinvf(), input, dimension);
}

double CBayesianOptimisation::anovaConstantFactor(const TVector& Kinvf) const {
    double sum{0.0};
    for (std::size_t i = 0; i < m_FunctionMeanValues.size(); ++i) {
        double prod{1.0};
        const TVector& x{m_FunctionMeanValues[i].first};
        for (int d = 0; d < x.size(); ++d) {
            prod *= integrate1dKernel(m_KernelParameters(d + 1), x(d));
        }
        sum += Kinvf(i) * prod;
    }
    return CTools::pow2(m_KernelParameters(0)) * sum;
}

double CBayesianOptimisation::anovaConstantFactor() const {
    return this->anovaConstantFactor(this->kinvf());
}

double CBayesianOptimisation::anovaTotalVariance(const TVector& Kinvf) const {
    auto prodIj = [&Kinvf, this](std::size_t i, std::size_t j) -> double {
        const TVector& xi{m_FunctionMeanValues[i].first};
        const TVector& xj{m_FunctionMeanValues[j].first};
        double prod{1.0};
        for (int d = 0; d < xi.size(); ++d) {
            prod *= integrate1dKernelProduct(m_KernelParameters[d + 1], xi(d), xj(d));
        }
        return Kinvf(static_cast<int>(i)) * Kinvf(static_cast<int>(j)) * prod;
    };

    double sum{0.0};
    for (std::size_t i = 0; i < m_FunctionMeanValues.size(); ++i) {
        sum += prodIj(i, i);
        for (std::size_t j = 0; j < i; ++j) {
            sum += 2.0 * prodIj(i, j);
        }
    }

    double theta2{CTools::pow2(m_KernelParameters(0))};
    double f0{this->anovaConstantFactor(Kinvf)};
    double scale2{CTools::pow2(m_RangeScale)};
    if (sum > 0.0) {
        // We rewrite theta_0^4 sum - f_0^2 as (theta^2 + f_0) * (theta^2 - f_0)
        // where theta^2 = theta_0^2 sum^(1/2) because it has better numerics.
        theta2 *= std::sqrt(sum);
        double variance{(theta2 + f0) * (theta2 - f0)};
        return std::max(0.0, variance / scale2);
    }
    return std::max(0.0, (theta2 * theta2 * sum - f0 * f0) / scale2);
}

double CBayesianOptimisation::anovaTotalCoefficientOfVariation() {
    this->precondition();
    return std::sqrt(this->anovaTotalVariance()) / m_RangeShift;
}

double CBayesianOptimisation::anovaTotalVariance() const {
    return this->anovaTotalVariance(this->kinvf());
}

double CBayesianOptimisation::anovaMainEffect(const TVector& Kinvf, int dimension) const {
    auto prodXt = [this](const TVector& x, int t) -> double {
        double prod{1.0};
        for (int d = 0; d < m_MinBoundary.size(); ++d) {
            if (d != t) {
                prod *= integrate1dKernel(m_KernelParameters(d + 1), x(d));
            }
        }
        return prod;
    };
    double sum1{0.0};
    double sum2{0.0};
    for (std::size_t i = 0; i < m_FunctionMeanValues.size(); ++i) {
        const TVector& xi{m_FunctionMeanValues[i].first};
        for (std::size_t j = 0; j < m_FunctionMeanValues.size(); ++j) {
            const TVector& xj{m_FunctionMeanValues[j].first};
            sum1 += Kinvf(static_cast<int>(i)) * Kinvf(static_cast<int>(j)) *
                    prodXt(xi, dimension) * prodXt(xj, dimension) *
                    integrate1dKernelProduct(m_KernelParameters(dimension + 1),
                                             xi(dimension), xj(dimension));
        }
        sum2 += Kinvf(static_cast<int>(i)) *
                integrate1dKernel(m_KernelParameters(dimension + 1), xi(dimension)) *
                prodXt(xi, dimension);
    }
    double scale2{CTools::pow2(m_RangeScale)};
    double theta02{CTools::pow2(m_KernelParameters(0))};
    double f0{this->anovaConstantFactor()};
    double f02{CTools::pow2(f0)};
    return (theta02 * (theta02 * sum1 - 2.0 * f0 * sum2) + f02) / scale2;
}

double CBayesianOptimisation::anovaMainEffect(int dimension) const {
    if (dimension < 0 || dimension > m_MinBoundary.size()) {
        LOG_ERROR(<< "Input error: dimension " << dimension << " is out of bounds. "
                  << "It should be between 0 and " << m_MinBoundary.size() << ".");
        return 0.0;
    }
    return this->anovaMainEffect(this->kinvf(), dimension);
}

CBayesianOptimisation::TDoubleDoublePrVec CBayesianOptimisation::anovaMainEffects() const {
    TDoubleDoublePrVec mainEffects;
    mainEffects.reserve(static_cast<std::size_t>(m_MinBoundary.size()));
    TVector Kinvf{this->kinvf()};
    double f0{this->anovaConstantFactor(Kinvf)};
    double totalVariance{this->anovaTotalVariance(Kinvf)};
    for (int i = 0; i < m_MinBoundary.size(); ++i) {
        double effect{this->anovaMainEffect(Kinvf, i)};
        mainEffects.emplace_back(effect, effect / totalVariance);
    }
    LOG_TRACE(<< "GP ANOVA constant " << f0 << " variance " << totalVariance
              << "\nmain effects " << core::CContainerPrinter::print(mainEffects)
              << "\nkernel parameters " << m_KernelParameters.transpose());
    return mainEffects;
}

void CBayesianOptimisation::kernelParameters(const TVector& parameters) {
    if (m_KernelParameters.size() == parameters.size()) {
        m_KernelParameters = parameters;
        m_RangeShift = 0.0;
        m_RangeScale = 1.0;
    }
}

std::pair<CBayesianOptimisation::TLikelihoodFunc, CBayesianOptimisation::TLikelihoodGradientFunc>
CBayesianOptimisation::minusLikelihoodAndGradient() const {

    TVector f{this->function()};
    double v{this->meanErrorVariance()};
    TVector ones;
    TVector gradient;
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> Kqr;
    TMatrix K;
    TVector Kinvf;
    TMatrix Kinv;
    TMatrix dKdai;
    double eps{1e-4};

    // We need to be careful when we compute the kernel decomposition. Basically,
    // if the kernel matrix is singular to working precision then if the function
    // value vector projection onto the null-space has non-zero length the likelihood
    // function is effectively -infinity. This follow from the fact that although
    // log(1 / lambda_i) -> +infinity, -1/2 sum_i{ ||f_i||^2 / lambda_i } -> -infinity
    // faster for all ||f_i|| > 0 and lambda_i sufficiently small. Here {lambda_i}
    // denote the Eigenvalues of the nullspace. We use a rank revealing decomposition
    // and compute the likelihood on the row space.

    auto minusLogLikelihood = [=](const TVector& a) mutable -> double {
        K = this->kernel(a, v + eps);
        Kqr.compute(K);
        Kinvf.noalias() = Kqr.solve(f);
        // Note that Kqr.logAbsDeterminant() = -infinity if K is singular.
        double logAbsDet{0.0};
        for (int i = 0; i < Kqr.rank(); ++i) {
            logAbsDet += std::log(std::fabs(Kqr.matrixR()(i, i)));
        }
        logAbsDet = CTools::stable(logAbsDet);
        return 0.5 * (f.transpose() * Kinvf + logAbsDet);
    };

    auto minusLogLikelihoodGradient = [=](const TVector& a) mutable -> TVector {
        K = this->kernel(a, v + eps);
        Kqr.compute(K);

        Kinvf.noalias() = Kqr.solve(f);

        ones = TVector::Ones(f.size());
        Kinv.noalias() = Kqr.solve(TMatrix::Identity(f.size(), f.size()));

        K.diagonal() -= (v + eps) * ones;

        gradient = TVector::Zero(a.size());
        for (int i = 0; i < Kinvf.size(); ++i) {
            double di{(Kinvf(i) * Kinvf.transpose() - Kinv.row(i)) * K.col(i)};
            gradient(0) -= di / a(0);
        }
        for (int i = 1; i < a.size(); ++i) {
            dKdai = this->dKerneld(a, i);
            for (int j = 0; j < Kinvf.size(); ++j) {
                double di{(Kinvf(j) * Kinvf.transpose() - Kinv.row(j)) * dKdai.col(j)};
                gradient(i) += 0.5 * di;
            }
        }

        return gradient;
    };

    return {std::move(minusLogLikelihood), std::move(minusLogLikelihoodGradient)};
}

std::pair<CBayesianOptimisation::TEIFunc, CBayesianOptimisation::TEIGradientFunc>
CBayesianOptimisation::minusExpectedImprovementAndGradient() const {

    TMatrix K{this->kernel(m_KernelParameters, this->meanErrorVariance())};
    Eigen::LDLT<Eigen::MatrixXd> Kldl{K};
    TVector Kinvf{Kldl.solve(this->function())};
    double vx{this->meanErrorVariance()};

    TVector Kxn;
    TVector KinvKxn;
    TVector muGradient;
    TVector sigmaGradient;
    TVector dKxndx;
    TVector zGradient;

    double fmin{
        std::min_element(m_FunctionMeanValues.begin(), m_FunctionMeanValues.end(),
                         [](const TVectorDoublePr& lhs, const TVectorDoublePr& rhs) {
                             return lhs.second < rhs.second;
                         })
            ->second};

    auto EI = [=](const TVector& x) mutable -> double {
        double Kxx;
        std::tie(Kxn, Kxx) = this->kernelCovariates(m_KernelParameters, x, vx);
        if (CMathsFuncs::isNan(Kxx)) {
            return 0.0;
        }

        KinvKxn = Kldl.solve(Kxn);
        double error{(K.lazyProduct(KinvKxn) - Kxn).norm()};
        if (CMathsFuncs::isNan(error) || error > 0.01 * Kxn.norm()) {
            return 0.0;
        }

        double sigma{Kxx - Kxn.transpose() * KinvKxn};
        if (sigma <= 0.0) {
            return 0.0;
        }

        double mu{Kxn.transpose() * Kinvf};
        sigma = std::sqrt(sigma);

        double z{(fmin - mu) / sigma};
        double cdfz{stableNormCdf(z)};
        double pdfz{stableNormPdf(z)};
        return -sigma * (z * cdfz + pdfz);
    };

    auto EIGradient = [=](const TVector& x) mutable -> TVector {
        double Kxx;
        std::tie(Kxn, Kxx) = this->kernelCovariates(m_KernelParameters, x, vx);
        if (CMathsFuncs::isNan(Kxx)) {
            return las::zero(x);
        }

        KinvKxn = Kldl.solve(Kxn);
        double error{(K.lazyProduct(KinvKxn) - Kxn).norm()};
        if (CMathsFuncs::isNan(error) || error > 0.01 * Kxn.norm()) {
            return las::zero(x);
        }

        double sigma{Kxx - Kxn.transpose() * KinvKxn};
        if (sigma <= 0.0) {
            return las::zero(x);
        }

        double mu{Kxn.transpose() * Kinvf};
        sigma = std::sqrt(sigma);

        double z{(fmin - mu) / sigma};
        double cdfz{stableNormCdf(z)};
        double pdfz{stableNormPdf(z)};

        muGradient.resize(x.size());
        sigmaGradient.resize(x.size());
        for (int i = 0; i < x.size(); ++i) {
            dKxndx.resize(Kxn.size());
            for (int j = 0; j < Kxn.size(); ++j) {
                const TVector& xj{m_FunctionMeanValues[j].first};
                dKxndx(j) = 2.0 *
                            (m_MinimumKernelCoordinateDistanceScale(0) +
                             CTools::pow2(m_KernelParameters(i + 1))) *
                            (xj(i) - x(i)) * Kxn(j);
            }
            muGradient(i) = Kinvf.transpose() * dKxndx;
            sigmaGradient(i) = -2.0 * KinvKxn.transpose() * dKxndx;
        }
        sigmaGradient /= 2.0 * sigma;

        zGradient = ((mu - fmin) / CTools::pow2(sigma)) * sigmaGradient - muGradient / sigma;

        return -(z * cdfz + pdfz) * sigmaGradient - sigma * cdfz * zGradient;
    };

    return {std::move(EI), std::move(EIGradient)};
}

const CBayesianOptimisation::TVector& CBayesianOptimisation::maximumLikelihoodKernel() {

    if (m_FunctionMeanValues.size() < 2) {
        return m_KernelParameters;
    }

    using TDoubleVecVec = std::vector<TDoubleVec>;

    this->precondition();

    TLikelihoodFunc l;
    TLikelihoodGradientFunc g;
    std::tie(l, g) = this->minusLikelihoodAndGradient();

    CLbfgs<TVector> lbfgs{10};

    double lmax{l(m_KernelParameters)};
    TVector amax{m_KernelParameters};

    // Try the current values first.
    double la;
    TVector a;
    std::tie(a, la) = lbfgs.minimize(l, g, m_KernelParameters, 1e-8, 75);
    if (COrderings::lexicographical_compare(la, a.norm(), lmax, amax.norm())) {
        lmax = la;
        amax = a;
    }

    TMinAccumulator probes{m_Restarts - 1};

    // We restart optimization with scales of the current values for global probing.
    std::size_t n(m_KernelParameters.size());
    TDoubleVecVec scales;
    scales.reserve(10 * (m_Restarts - 1));
    CSampling::sobolSequenceSample(n, 10 * (m_Restarts - 1), scales);

    for (const auto& scale : scales) {
        a.noalias() = m_KernelParameters;
        for (std::size_t j = 0; j < n; ++j) {
            a(j) *= CTools::stableExp(CTools::linearlyInterpolate(
                0.0, 1.0, std::log(0.2), std::log(2.0), scale[j]));
        }
        la = l(a);
        if (COrderings::lexicographical_compare(la, a.norm(), lmax, amax.norm())) {
            lmax = la;
            amax = a;
        }
        probes.add({la, std::move(a)});
    }

    for (auto& a0 : probes) {
        std::tie(a, la) = lbfgs.minimize(l, g, std::move(a0.second), 1e-8, 75);
        if (COrderings::lexicographical_compare(la, a.norm(), lmax, amax.norm())) {
            lmax = la;
            amax = std::move(a);
        }
    }

    // Ensure that kernel lengths are always positive. It shouldn't change the results
    // but improves traceability.
    m_KernelParameters = amax.cwiseAbs();
    LOG_TRACE(<< "kernel parameters = " << m_KernelParameters.transpose());
    LOG_TRACE(<< "likelihood = " << -lmax);

    return m_KernelParameters;
}

void CBayesianOptimisation::precondition() {

    // The Gaussian process expects the data to be centred. We also scale the variance.
    // This is useful if one wants to threshold values such as EI but the scale of the
    // function values is very different for example if we're modelling the loss surface
    // for different loss functions.

    for (auto& value : m_FunctionMeanValues) {
        value.second = fromScaled(m_RangeShift, m_RangeScale, value.second);
    }
    for (auto& variance : m_ErrorVariances) {
        variance /= CTools::pow2(m_RangeScale);
    }
    m_ExplainedErrorVariance /= CTools::pow2(m_RangeScale);

    TMeanVarAccumulator valueMoments;
    for (const auto& value : m_FunctionMeanValues) {
        valueMoments.add(value.second);
    }

    m_RangeShift = CBasicStatistics::mean(valueMoments);
    m_RangeScale = CBasicStatistics::variance(valueMoments) == 0.0
                       ? 1.0
                       : 1.0 / std::sqrt(CBasicStatistics::variance(valueMoments));

    for (auto& value : m_FunctionMeanValues) {
        value.second = toScaled(m_RangeShift, m_RangeScale, value.second);
    }
    for (auto& variance : m_ErrorVariances) {
        variance *= CTools::pow2(m_RangeScale);
    }
    m_ExplainedErrorVariance *= CTools::pow2(m_RangeScale);
}

CBayesianOptimisation::TVector CBayesianOptimisation::function() const {
    TVector result(m_FunctionMeanValues.size());
    for (std::size_t i = 0; i < m_FunctionMeanValues.size(); ++i) {
        result(i) = m_FunctionMeanValues[i].second;
    }
    return result;
}

double CBayesianOptimisation::meanErrorVariance() const {

    // So what are we doing here? When we supply function values we also supply their
    // error variance. Typically these might be the mean test loss function across
    // folds and their variance for a particular choice of hyperparameters. Sticking
    // with this example, the variance allows us to estimate the error w.r.t. the
    // true generalisation error due to finite sample size. We can think of the source
    // of this variance as being due to two effects: one which shifts the loss values
    // in each fold (this might be due to some folds simply having more hard examples)
    // and another which permutes the order of loss values. A shift in the loss function
    // is not something we wish to capture in the GP: it shouldn't materially affect
    // where to choose points to test since any sensible optimisation strategy should
    // only care about the difference in loss between points, which is unaffected by a
    // shift. More formally, if we assume the shift and permutation errors are independent
    // we have for losses l_i, mean loss per fold m_i and mean loss for a given set of
    // hyperparameters m that the variance is
    //
    //   sum_i{ (l_i - m)^2 } = sum_i{ (l_i - m_i + m_i - m)^2 }
    //                        = sum_i{ (l_i - m_i)^2 } + sum_i{ (m_i - m)^2 }
    //                        = "permutation variance" + "shift variance"          (1)
    //
    // with the cross-term expected to be small by independence. (Note, the independence
    // assumption is reasonable if one assumes that the shift is due to mismatch in hard
    // examples since the we choose folds independently at random.) We can estimate the
    // shift variance by looking at mean loss over all distinct hyperparameter settings
    // and we assume it is supplied as the parameter m_ExplainedErrorVariance. It should
    // also be smaller than the variance by construction although for numerical stability
    // we prevent the difference becoming too small. As discussed, here we wish return
    // the permutation variance which we get by rearranging (1).

    TMeanAccumulator variance;
    variance.add(m_ErrorVariances);
    return CBasicStatistics::mean(variance) -
           std::min(m_ExplainedErrorVariance, 0.99 * CBasicStatistics::mean(variance));
}

CBayesianOptimisation::TMatrix CBayesianOptimisation::dKerneld(const TVector& a, int k) const {
    TMatrix result{m_FunctionMeanValues.size(), m_FunctionMeanValues.size()};
    for (std::size_t i = 0; i < m_FunctionMeanValues.size(); ++i) {
        result(i, i) = 0.0;
        const TVector& xi{m_FunctionMeanValues[i].first};
        for (std::size_t j = 0; j < i; ++j) {
            const TVector& xj{m_FunctionMeanValues[j].first};
            result(i, j) = result(j, i) = 2.0 * a(k) *
                                          CTools::pow2(xi(k - 1) - xj(k - 1)) *
                                          this->kernel(a, xi, xj);
        }
    }
    return result;
}

CBayesianOptimisation::TMatrix CBayesianOptimisation::kernel(const TVector& a, double v) const {
    TMatrix result{m_FunctionMeanValues.size(), m_FunctionMeanValues.size()};
    for (std::size_t i = 0; i < m_FunctionMeanValues.size(); ++i) {
        result(i, i) = CTools::pow2(a(0)) + v;
        const TVector& xi{m_FunctionMeanValues[i].first};
        for (std::size_t j = 0; j < i; ++j) {
            const TVector& xj{m_FunctionMeanValues[j].first};
            result(i, j) = result(j, i) = this->kernel(a, xi, xj);
        }
    }
    return result;
}

CBayesianOptimisation::TVectorDoublePr
CBayesianOptimisation::kernelCovariates(const TVector& a, const TVector& x, double vx) const {
    double Kxx{CTools::pow2(a(0)) + vx};
    TVector Kxn(m_FunctionMeanValues.size());
    for (std::size_t i = 0; i < m_FunctionMeanValues.size(); ++i) {
        Kxn(i) = this->kernel(a, x, m_FunctionMeanValues[i].first);
    }
    return {std::move(Kxn), Kxx};
}

double CBayesianOptimisation::kernel(const TVector& a, const TVector& x, const TVector& y) const {
    return CTools::pow2(a(0)) *
           CTools::stableExp(-(x - y).transpose() * (m_MinimumKernelCoordinateDistanceScale +
                                                     a.tail(a.size() - 1).cwiseAbs2())
                                                        .cwiseProduct(x - y));
}

CBayesianOptimisation::TVector CBayesianOptimisation::kinvf() const {
    TVector Kinvf;
    TMatrix K{this->kernel(m_KernelParameters, this->meanErrorVariance())};
    Kinvf = K.ldlt().solve(this->function());
    return Kinvf;
}

double CBayesianOptimisation::dissimilarity(const TVector& x) const {
    // This is used as a fallback when GP is very unsure we can actually make progress,
    // i.e. EI is miniscule. In this case we fallback to a different strategy to break
    // ties at the probes we used for the GP. We use two criteria:
    //   1. The average distance to points we already tried: we prefer evaluation points
    //      where the density of points is low,
    //   2. The minimum distance to any point we've already tried: we assume the loss
    //      is fairly smooth (to bother trying to do better than random search) so any
    //      existing point tells us accurately what the loss will be in its immediate
    //      neighbourhood and running there again is duplicate work.
    double sum{0.0};
    double min{std::numeric_limits<double>::max()};
    for (const auto& y : m_FunctionMeanValues) {
        double dxy{las::distance(x, y.first)};
        sum += dxy;
        min += std::min(min, dxy);
    }
    return sum / static_cast<double>(m_FunctionMeanValues.size()) + min;
}

CBayesianOptimisation::TVector CBayesianOptimisation::to01(TVector x) const {
    // Self assign so operations are performed inplace.
    x = (x - m_MinBoundary).cwiseQuotient(m_MaxBoundary - m_MinBoundary);
    return x;
}

CBayesianOptimisation::TVector CBayesianOptimisation::from01(TVector x) const {
    x = m_MinBoundary + x.cwiseProduct(m_MaxBoundary - m_MinBoundary);
    return x;
}

void CBayesianOptimisation::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    try {
        core::CPersistUtils::persist(VERSION_7_5_TAG, "", inserter);
        inserter.insertValue(RNG_TAG, m_Rng.toString());
        core::CPersistUtils::persist(MIN_BOUNDARY_TAG, m_MinBoundary, inserter);
        core::CPersistUtils::persist(MAX_BOUNDARY_TAG, m_MaxBoundary, inserter);
        core::CPersistUtils::persist(ERROR_VARIANCES_TAG, m_ErrorVariances, inserter);
        core::CPersistUtils::persist(KERNEL_PARAMETERS_TAG, m_KernelParameters, inserter);
        core::CPersistUtils::persist(MIN_KERNEL_COORDINATE_DISTANCE_SCALES_TAG,
                                     m_MinimumKernelCoordinateDistanceScale, inserter);
        core::CPersistUtils::persist(FUNCTION_MEAN_VALUES_TAG, m_FunctionMeanValues, inserter);
        core::CPersistUtils::persist(RANGE_SCALE_TAG, m_RangeScale, inserter);
        core::CPersistUtils::persist(RANGE_SHIFT_TAG, m_RangeShift, inserter);
        core::CPersistUtils::persist(RESTARTS_TAG, m_Restarts, inserter);
    } catch (std::exception& e) {
        LOG_ERROR(<< "Failed to persist state! " << e.what());
    }
}

bool CBayesianOptimisation::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    if (traverser.name() == VERSION_7_5_TAG) {
        try {
            do {
                const std::string& name = traverser.name();
                RESTORE(RNG_TAG, m_Rng.fromString(traverser.value()))
                RESTORE(MIN_BOUNDARY_TAG,
                        core::CPersistUtils::restore(MIN_BOUNDARY_TAG, m_MinBoundary, traverser))
                RESTORE(MAX_BOUNDARY_TAG,
                        core::CPersistUtils::restore(MAX_BOUNDARY_TAG, m_MaxBoundary, traverser))
                RESTORE(ERROR_VARIANCES_TAG,
                        core::CPersistUtils::restore(ERROR_VARIANCES_TAG,
                                                     m_ErrorVariances, traverser))
                RESTORE(RANGE_SHIFT_TAG,
                        core::CPersistUtils::restore(RANGE_SHIFT_TAG, m_RangeShift, traverser))
                RESTORE(RANGE_SCALE_TAG,
                        core::CPersistUtils::restore(RANGE_SCALE_TAG, m_RangeScale, traverser))
                RESTORE(RESTARTS_TAG,
                        core::CPersistUtils::restore(RESTARTS_TAG, m_Restarts, traverser))
                RESTORE(KERNEL_PARAMETERS_TAG,
                        core::CPersistUtils::restore(KERNEL_PARAMETERS_TAG,
                                                     m_KernelParameters, traverser))
                RESTORE(MIN_KERNEL_COORDINATE_DISTANCE_SCALES_TAG,
                        core::CPersistUtils::restore(
                            MIN_KERNEL_COORDINATE_DISTANCE_SCALES_TAG,
                            m_MinimumKernelCoordinateDistanceScale, traverser))
                RESTORE(FUNCTION_MEAN_VALUES_TAG,
                        core::CPersistUtils::restore(FUNCTION_MEAN_VALUES_TAG,
                                                     m_FunctionMeanValues, traverser))
            } while (traverser.next());
        } catch (std::exception& e) {
            LOG_ERROR(<< "Failed to restore state! " << e.what());
            return false;
        }

        this->checkRestoredInvariants();

        return true;
    }
    LOG_ERROR(<< "Input error: unsupported state serialization version. Currently supported version: "
              << VERSION_7_5_TAG);
    return false;
}

void CBayesianOptimisation::checkRestoredInvariants() const {
    VIOLATES_INVARIANT(m_FunctionMeanValues.size(), !=, m_ErrorVariances.size());
    VIOLATES_INVARIANT(m_MinBoundary.size(), !=, m_MaxBoundary.size());
    VIOLATES_INVARIANT(m_KernelParameters.size(), !=, m_MinBoundary.size() + 1);
    VIOLATES_INVARIANT(m_MinimumKernelCoordinateDistanceScale.size(), !=,
                       m_MinBoundary.size());
    for (const auto& point : m_FunctionMeanValues) {
        VIOLATES_INVARIANT(point.first.size(), !=, m_MinBoundary.size());
    }
}

std::size_t CBayesianOptimisation::memoryUsage() const {
    std::size_t mem{core::CMemory::dynamicSize(m_MinBoundary)};
    mem += core::CMemory::dynamicSize(m_MaxBoundary);
    mem += core::CMemory::dynamicSize(m_FunctionMeanValues);
    mem += core::CMemory::dynamicSize(m_ErrorVariances);
    mem += core::CMemory::dynamicSize(m_KernelParameters);
    mem += core::CMemory::dynamicSize(m_MinimumKernelCoordinateDistanceScale);
    return mem;
}

std::size_t CBayesianOptimisation::estimateMemoryUsage(std::size_t numberParameters,
                                                       std::size_t numberRounds) {
    std::size_t boundaryMemoryUsage{2 * numberParameters * sizeof(double)};
    std::size_t functionMeanValuesMemoryUsage{numberRounds * sizeof(TVectorDoublePr)};
    std::size_t errorVariancesMemoryUsage{numberRounds * sizeof(double)};
    std::size_t kernelParametersMemoryUsage{(numberParameters + 1) * sizeof(double)};
    std::size_t minimumKernelCoordinateDistanceScale{numberParameters * sizeof(double)};
    return sizeof(CBayesianOptimisation) + boundaryMemoryUsage +
           functionMeanValuesMemoryUsage + errorVariancesMemoryUsage +
           kernelParametersMemoryUsage + minimumKernelCoordinateDistanceScale;
}

std::uint64_t CBayesianOptimisation::checksum(std::uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_Rng);
    seed = CChecksum::calculate(seed, m_Restarts);
    seed = CChecksum::calculate(seed, m_RangeShift);
    seed = CChecksum::calculate(seed, m_RangeScale);
    seed = CChecksum::calculate(seed, m_ExplainedErrorVariance);
    seed = CChecksum::calculate(seed, m_MinBoundary);
    seed = CChecksum::calculate(seed, m_MaxBoundary);
    seed = CChecksum::calculate(seed, m_FunctionMeanValues);
    seed = CChecksum::calculate(seed, m_ErrorVariances);
    seed = CChecksum::calculate(seed, m_KernelParameters);
    return CChecksum::calculate(seed, m_MinimumKernelCoordinateDistanceScale);
}

const std::size_t CBayesianOptimisation::RESTARTS{10};
const double CBayesianOptimisation::NEGLIGIBLE_EXPECTED_IMPROVEMENT{1e-12};
const double CBayesianOptimisation::MINIMUM_KERNEL_COORDINATE_DISTANCE_SCALE{1e-3};
}
}
}
