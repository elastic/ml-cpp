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
#include <maths/CSampling.h>
#include <maths/CTools.h>

#include <boost/math/distributions/normal.hpp>

#include <exception>

namespace ml {
namespace maths {

namespace {
const std::string MIN_BOUNDARY_TAG{"min_boundary"};
const std::string MAX_BOUNDARY_TAG{"max_boundary"};
const std::string ERROR_VARIANCES_TAG{"error_variances"};
const std::string KERNEL_PARAMETERS_TAG{"kernel_parameters"};
const std::string MIN_KERNEL_COORDINATE_DISTANCE_SCALES_TAG{"min_kernel_coordinate_distance_scales"};
const std::string FUNCTION_MEAN_VALUES_TAG{"function_mean_values"};
}

CBayesianOptimisation::CBayesianOptimisation(TDoubleDoublePrVec parameterBounds)
    : m_MinBoundary(parameterBounds.size()), m_MaxBoundary(parameterBounds.size()),
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

CBayesianOptimisation::TVector CBayesianOptimisation::maximumExpectedImprovement() {

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

    for (std::size_t i = 0; i < interpolates.size(); /**/) {

        for (int j = 0; j < interpolate.size(); ++i, ++j) {
            interpolate(j) = interpolates[i];
        }
        TVector x{a + interpolate.asDiagonal() * (b - a)};
        double fx{minusEI(x)};
        LOG_TRACE(<< "x = " << x.transpose() << " EI(x) = " << fx);

        if (-fx > fmax) {
            xmax = std::move(x);
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

    for (auto& x0 : seeds) {

        TVector xcand;
        double fcand;
        std::tie(xcand, fcand) = lbfgs.constrainedMinimize(
            minusEI, minusEIGradient, a, b, std::move(x0.second), rho);
        LOG_TRACE(<< "xcand = " << xcand.transpose() << " EI(cand) = " << fcand);

        if (-fcand > fmax) {
            xmax = std::move(xcand);
            fmax = -fcand;
        }
    }

    LOG_TRACE(<< "best = " << xmax.cwiseProduct(m_MaxBoundary - m_MinBoundary).transpose()
              << " EI(best) = " << fmax);

    return xmax.cwiseProduct(m_MaxBoundary - m_MinBoundary);
}

std::pair<CBayesianOptimisation::TLikelihoodFunc, CBayesianOptimisation::TLikelihoodGradientFunc>
CBayesianOptimisation::minusLikelihoodAndGradient() const {

    TVector f{this->function()};
    double v{this->meanErrorVariance()};

    auto minusLogLikelihood = [f, v, this](const TVector& a) {
        Eigen::LDLT<Eigen::MatrixXd> Kldl{this->kernel(a, v)};
        TVector Kinvf{Kldl.solve(f)};
        // We can only determine values up to eps * "max diagonal". If the diagonal
        // has a zero it blows up the determinant term. In practice, we know the
        // kernel can't be singular by construction so we perturb the diagonal by
        // the numerical error in such a way as to recover a non-singular matrix.
        // (Note that the solve routine deals with the zero for us.)
        double eps{std::numeric_limits<double>::epsilon() * Kldl.vectorD().maxCoeff()};
        return 0.5 *
               (f.transpose() * Kinvf + (Kldl.vectorD().array() + eps).log().sum());
    };

    auto minusLogLikelihoodGradient = [f, v, this](const TVector& a) {
        TMatrix K{this->kernel(a, v)};
        Eigen::LDLT<Eigen::MatrixXd> Kldl{K};

        TVector Kinvf{Kldl.solve(f)};

        TVector ones(f.size());
        ones.setOnes();
        TMatrix KInv{Kldl.solve(TMatrix{ones.asDiagonal()})};

        K.diagonal() -= v * ones;

        TVector gradient{a.size()};
        gradient.setZero();

        for (int i = 0; i < Kinvf.size(); ++i) {
            gradient(0) -=
                1.0 / a(0) *
                double{(Kinvf(i) * Kinvf.transpose() - KInv.row(i)) * K.col(i)};
        }
        for (int i = 1; i < a.size(); ++i) {
            TMatrix dKdai{this->dKerneld(a, i)};
            for (int j = 0; j < Kinvf.size(); ++j) {
                gradient(i) += 0.5 * double{(Kinvf(j) * Kinvf.transpose() - KInv.row(j)) *
                                            dKdai.col(j)};
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

    double fmin{
        std::min_element(m_FunctionMeanValues.begin(), m_FunctionMeanValues.end(),
                         [](const TVectorDoublePr& lhs, const TVectorDoublePr& rhs) {
                             return lhs.second < rhs.second;
                         })
            ->second};

    auto EI = [Kldl, Kinvf, vx, fmin, this](const TVector& x) {
        double Kxx;
        TVector Kxn;
        std::tie(Kxn, Kxx) = this->kernelCovariates(m_KernelParameters, x, vx);

        double sigma{Kxx - Kxn.transpose() * Kldl.solve(Kxn)};

        if (sigma <= 0.0) {
            return 0.0;
        }

        double mu{Kxn.transpose() * Kinvf};
        sigma = std::sqrt(sigma);

        boost::math::normal normal{0.0, 1.0};
        double z{(fmin - mu) / sigma};
        return -sigma * (z * CTools::safeCdf(normal, z) + CTools::safePdf(normal, z));
    };

    auto EIGradient = [Kldl, Kinvf, vx, fmin, this](const TVector& x) {
        double Kxx;
        TVector Kxn;
        std::tie(Kxn, Kxx) = this->kernelCovariates(m_KernelParameters, x, vx);

        TVector KinvKxn{Kldl.solve(Kxn)};
        double sigma{Kxx - Kxn.transpose() * KinvKxn};

        if (sigma <= 0.0) {
            return las::zero(x);
        }

        double mu{Kxn.transpose() * Kinvf};
        sigma = std::sqrt(sigma);

        boost::math::normal normal{0.0, 1.0};
        double z{(fmin - mu) / sigma};
        double cdfz{CTools::safeCdf(normal, z)};
        double pdfz{CTools::safePdf(normal, z)};

        TVector muGradient{x.size()};
        TVector sigmaGradient{x.size()};
        for (int i = 0; i < x.size(); ++i) {
            TVector dKxndx{Kxn.size()};
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

        TVector zGradient{((mu - fmin) / CTools::pow2(sigma)) * sigmaGradient -
                          muGradient / sigma};

        return TVector{-(z * cdfz + pdfz) * sigmaGradient - sigma * cdfz * zGradient};
    };

    return {std::move(EI), std::move(EIGradient)};
}

const CBayesianOptimisation::TVector& CBayesianOptimisation::maximumLikelihoodKernel() {

    // Use random restarts of L-BFGS to find maximum likelihood parameters.

    this->precondition();

    std::size_t n(m_KernelParameters.size());

    TDoubleVec scales;
    scales.reserve((m_Restarts - 1) * n);
    CSampling::uniformSample(m_Rng, std::log(0.1), std::log(4.0),
                             (m_Restarts - 1) * n, scales);

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
        a = scale.array().exp() * a.array();

        double la;
        std::tie(a, la) = lbfgs.minimize(l, g, std::move(a), 1e-8, 75);

        if (la < lmax) {
            lmax = la;
            amax = std::move(a);
        }
    }

    m_KernelParameters = std::move(amax);
    LOG_TRACE(<< "kernel parameters = " << m_KernelParameters.transpose());

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
    return {Kxn, Kxx};
}

double CBayesianOptimisation::kernel(const TVector& a, const TVector& x, const TVector& y) const {
    return CTools::pow2(a(0)) * std::exp(-(x - y).transpose() *
                                         (m_MinimumKernelCoordinateDistanceScale +
                                          a.tail(a.size() - 1).cwiseAbs2().matrix())
                                             .asDiagonal() *
                                         (x - y));
}

void CBayesianOptimisation::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    try {
        core::CPersistUtils::persist(MIN_BOUNDARY_TAG, m_MinBoundary, inserter);

        core::CPersistUtils::persist(MAX_BOUNDARY_TAG, m_MaxBoundary, inserter);
        core::CPersistUtils::persist(ERROR_VARIANCES_TAG, m_ErrorVariances, inserter);
        core::CPersistUtils::persist(KERNEL_PARAMETERS_TAG, m_KernelParameters, inserter);
        core::CPersistUtils::persist(MIN_KERNEL_COORDINATE_DISTANCE_SCALES_TAG,
                                     m_MinimumKernelCoordinateDistanceScale, inserter);
        core::CPersistUtils::persist(FUNCTION_MEAN_VALUES_TAG, m_FunctionMeanValues, inserter);
    } catch (std::exception& e) {
        LOG_ERROR(<< "Failed to persist state! " << e.what());
    }
}

bool CBayesianOptimisation::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    try {
        do {
            const std::string& name = traverser.name();
            RESTORE(MIN_BOUNDARY_TAG,
                    core::CPersistUtils::restore(MIN_BOUNDARY_TAG, m_MinBoundary, traverser))
            RESTORE(MAX_BOUNDARY_TAG,
                    core::CPersistUtils::restore(MAX_BOUNDARY_TAG, m_MaxBoundary, traverser))
            RESTORE(ERROR_VARIANCES_TAG,
                    core::CPersistUtils::restore(ERROR_VARIANCES_TAG, m_ErrorVariances, traverser))
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

std::size_t CBayesianOptimisation::memoryUsage() const {
    std::size_t mem{core::CMemory::dynamicSize(m_MinBoundary)};
    mem += core::CMemory::dynamicSize(m_MaxBoundary);
    mem += core::CMemory::dynamicSize(m_FunctionMeanValues);
    mem += core::CMemory::dynamicSize(m_ErrorVariances);
    mem += core::CMemory::dynamicSize(m_KernelParameters);
    mem += core::CMemory::dynamicSize(m_MinimumKernelCoordinateDistanceScale);
    return mem;
}

const double CBayesianOptimisation::MINIMUM_KERNEL_COORDINATE_DISTANCE_SCALE{1e-3};
}
}
