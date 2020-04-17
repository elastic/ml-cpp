/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CBayesianOptimisation.h>

#include <core/CIEEE754.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CMemory.h>
#include <core/CPersistUtils.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CLbfgs.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CLinearAlgebraShims.h>
#include <maths/CMathsFuncs.h>
#include <maths/CSampling.h>
#include <maths/CTools.h>

#include <boost/math/constants/constants.hpp>
#include <boost/optional/optional_io.hpp>

#include <exception>

namespace ml {
namespace maths {

namespace {
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

//! A version of the normal c.d.f. which is stable across our target platforms.
double stableNormCdf(double z) {
    return (1.0 + CTools::stable(std::erf(z / boost::math::constants::root_two<double>()))) / 2.0;
}

//! A version of the normal p.d.f. which is stable across our target platforms.
double stableNormPdf(double z) {
    return CTools::stableExp(-z * z / 2.0) / boost::math::constants::root_two_pi<double>();
}

// The kernel we use is v * I + a(0)^2 * O(I). We fall back to random search when
// a(0)^2 < eps * v since for small eps and a reasonable number of dimensions the
// expected improvement will be constant in the space we search. We don't terminate
// altogether because it is possible that the function we're interpolating has a
// narrow deep valley that the Gaussian Process hasn't sampled.
const double MINIMUM_KERNEL_SCALE_FOR_EXPECTATION_MAXIMISATION{1e-8};
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
    if (traverser.traverseSubLevel(std::bind(&CBayesianOptimisation::acceptRestoreTraverser,
                                             this, std::placeholders::_1)) == false) {
        throw std::runtime_error{"failed to restore Bayesian optimisation"};
    }
}

void CBayesianOptimisation::add(TVector x, double fx, double vx) {
    m_FunctionMeanValues.emplace_back(x.cwiseQuotient(m_MaxBoundary - m_MinBoundary),
                                      m_RangeScale * (fx - m_RangeShift));
    m_ErrorVariances.push_back(CTools::pow2(m_RangeScale) * vx);
}

std::pair<CBayesianOptimisation::TVector, CBayesianOptimisation::TVector>
CBayesianOptimisation::boundingBox() const {
    return {m_MinBoundary, m_MaxBoundary};
}

std::pair<CBayesianOptimisation::TVector, CBayesianOptimisation::TOptionalDouble>
CBayesianOptimisation::maximumExpectedImprovement() {

    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMinAccumulator =
        CBasicStatistics::COrderStatisticsHeap<std::pair<double, TVector>>;

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
    CSampling::uniformSample(m_Rng, 0.0, 1.0, 3 * m_Restarts * interpolate.size(), interpolates);

    TVector a{m_MinBoundary.cwiseQuotient(m_MaxBoundary - m_MinBoundary)};
    TVector b{m_MaxBoundary.cwiseQuotient(m_MaxBoundary - m_MinBoundary)};
    TMeanAccumulator rho_;
    TMinAccumulator seeds{m_Restarts};

    if (CTools::pow2(m_KernelParameters(0)) <
        MINIMUM_KERNEL_SCALE_FOR_EXPECTATION_MAXIMISATION * this->meanErrorVariance()) {

        for (int i = 0, j = 0; j < interpolate.size(); ++i, ++j) {
            interpolate(j) = interpolates[i];
        }
        xmax = a + interpolate.cwiseProduct(b - a);

    } else {

        TVector x;
        for (std::size_t i = 0; i < interpolates.size(); /**/) {

            for (int j = 0; j < interpolate.size(); ++i, ++j) {
                interpolate(j) = interpolates[i];
            }
            x = a + interpolate.cwiseProduct(b - a);
            double fx{minusEI(x)};
            LOG_TRACE(<< "x = " << x.transpose() << " EI(x) = " << fx);

            if (COrderings::lexicographical_compare(fmax, xmax, -fx, x)) {
                xmax = x;
                fmax = -fx;
            }
            rho_.add(std::fabs(fx));
            seeds.add({fx, std::move(x)});
        }

        // We set rho to give the constraint and objective approximately equal priority
        // in the following constrained optimisation problem.
        double rho{CBasicStatistics::mean(rho_)};
        LOG_TRACE(<< "rho = " << rho);

        CLbfgs<TVector> lbfgs{10};

        TVector xcand;
        double fcand;
        for (auto& x0 : seeds) {

            LOG_TRACE(<< "x0 = " << x0.second.transpose());
            std::tie(xcand, fcand) = lbfgs.constrainedMinimize(
                minusEI, minusEIGradient, a, b, std::move(x0.second), rho);
            LOG_TRACE(<< "xcand = " << xcand.transpose() << " EI(cand) = " << fcand);
            if (COrderings::lexicographical_compare(fmax, xmax, -fcand, xcand)) {
                std::tie(xmax, fmax) = std::make_pair(std::move(xcand), -fcand);
            }
        }
    }

    // fmax was probably NaN, in anycase xmax wasn't initialised so fallback to
    // random search.
    TOptionalDouble expectedImprovement;
    if (xmax.size() == 0) {
        xmax = a + interpolate.cwiseProduct(b - a);
        expectedImprovement = TOptionalDouble{};
    } else if (fmax < 0.0 || CMathsFuncs::isFinite(fmax) == false) {
        expectedImprovement = TOptionalDouble{};
    } else {
        expectedImprovement = fmax / m_RangeScale;
    }

    xmax = xmax.cwiseProduct(m_MaxBoundary - m_MinBoundary);
    LOG_TRACE(<< "best = " << xmax.transpose() << " EI(best) = " << expectedImprovement);

    return {std::move(xmax), expectedImprovement};
}

std::pair<CBayesianOptimisation::TLikelihoodFunc, CBayesianOptimisation::TLikelihoodGradientFunc>
CBayesianOptimisation::minusLikelihoodAndGradient() const {

    TVector f{this->function()};
    double v{this->meanErrorVariance()};
    TVector ones;
    TVector gradient;
    TMatrix K;
    TVector Kinvf;
    TMatrix Kinv;
    TMatrix dKdai;

    auto minusLogLikelihood = [=](const TVector& a) mutable {
        K = this->kernel(a, v);
        Eigen::LDLT<Eigen::MatrixXd> Kldl{K};
        Kinvf = Kldl.solve(f);
        // We can only determine values up to eps * "max diagonal". If the diagonal
        // has a zero it blows up the determinant term. In practice, we know the
        // kernel can't be singular by construction so we perturb the diagonal by
        // the numerical error in such a way as to recover a non-singular matrix.
        // (Note that the solve routine deals with the zero for us.)
        double eps{std::numeric_limits<double>::epsilon() * Kldl.vectorD().maxCoeff()};
        return 0.5 * (f.transpose() * Kinvf +
                      Kldl.vectorD().cwiseMax(eps).array().log().sum());
    };

    auto minusLogLikelihoodGradient = [=](const TVector& a) mutable {
        K = this->kernel(a, v);
        Eigen::LDLT<Eigen::MatrixXd> Kldl{K};

        Kinvf = Kldl.solve(f);

        ones = TVector::Ones(f.size());
        Kinv = Kldl.solve(TMatrix::Identity(f.size(), f.size()));

        K.diagonal() -= v * ones;

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

    auto EI = [=](const TVector& x) mutable {
        double Kxx;
        std::tie(Kxn, Kxx) = this->kernelCovariates(m_KernelParameters, x, vx);

        double sigma{Kxx - Kxn.transpose() * Kldl.solve(Kxn)};

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

    auto EIGradient = [=](const TVector& x) mutable {
        double Kxx;
        std::tie(Kxn, Kxx) = this->kernelCovariates(m_KernelParameters, x, vx);

        KinvKxn = Kldl.solve(Kxn);
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

        return TVector{-(z * cdfz + pdfz) * sigmaGradient - sigma * cdfz * zGradient};
    };

    return {std::move(EI), std::move(EIGradient)};
}

const CBayesianOptimisation::TVector& CBayesianOptimisation::maximumLikelihoodKernel() {

    // Use random restarts of L-BFGS to find maximum likelihood parameters.

    this->precondition();

    std::size_t n(m_KernelParameters.size());

    // We restart optimization with initial guess on different scales for global probing.
    TDoubleVec scales;
    scales.reserve((m_Restarts - 1) * n);
    CSampling::uniformSample(m_Rng, CTools::stableLog(0.2),
                             CTools::stableLog(5.0), (m_Restarts - 1) * n, scales);

    TLikelihoodFunc l;
    TLikelihoodGradientFunc g;
    std::tie(l, g) = this->minusLikelihoodAndGradient();

    CLbfgs<TVector> lbfgs{10};

    double lmax;
    TVector amax;
    std::tie(amax, lmax) = lbfgs.minimize(l, g, m_KernelParameters, 1e-8, 75);

    TVector scale{n};
    for (std::size_t i = 1; i < m_Restarts; ++i) {

        TVector a{m_KernelParameters};
        for (std::size_t j = 0; j < n; ++j) {
            scale(j) = scales[(i - 1) * n + j];
        }
        a.array() *= scale.array().exp();

        double la;
        std::tie(a, la) = lbfgs.minimize(l, g, std::move(a), 1e-8, 75);

        if (COrderings::lexicographical_compare(la, a, lmax, amax)) {
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

    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

    for (auto& value : m_FunctionMeanValues) {
        value.second = m_RangeShift + value.second / m_RangeScale;
    }
    for (auto& variance : m_ErrorVariances) {
        variance /= CTools::pow2(m_RangeScale);
    }

    TMeanVarAccumulator rangeMoments;
    for (const auto& value : m_FunctionMeanValues) {
        rangeMoments.add(value.second);
    }

    double eps{std::numeric_limits<double>::epsilon()};
    m_RangeShift = CBasicStatistics::mean(rangeMoments);
    m_RangeScale = 1.0 / std::sqrt(CBasicStatistics::variance(rangeMoments) + eps);

    for (auto& value : m_FunctionMeanValues) {
        value.second = m_RangeScale * (value.second - m_RangeShift);
    }
    for (auto& variance : m_ErrorVariances) {
        variance *= CTools::pow2(m_RangeScale);
    }
}

CBayesianOptimisation::TVector CBayesianOptimisation::function() const {
    TVector result(m_FunctionMeanValues.size());
    for (std::size_t i = 0; i < m_FunctionMeanValues.size(); ++i) {
        result(i) = m_FunctionMeanValues[i].second;
    }
    return result;
}

double CBayesianOptimisation::meanErrorVariance() const {
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
    TMeanAccumulator variance;
    variance.add(m_ErrorVariances);
    return CBasicStatistics::mean(variance);
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

        return true;
    }
    LOG_ERROR(<< "Input error: unsupported state serialization version. Currently supported version: "
              << VERSION_7_5_TAG);
    return false;
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

const std::size_t CBayesianOptimisation::RESTARTS{10};
const double CBayesianOptimisation::MINIMUM_KERNEL_COORDINATE_DISTANCE_SCALE{1e-3};
}
}
