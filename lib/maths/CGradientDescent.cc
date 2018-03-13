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

#include <maths/CGradientDescent.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CLinearAlgebraTools.h>

namespace ml {
namespace maths {

CGradientDescent::CGradientDescent(double learnRate, double momentum)
    : m_LearnRate(learnRate), m_Momentum(momentum) {}

void CGradientDescent::learnRate(double learnRate) { m_LearnRate = learnRate; }

void CGradientDescent::momentum(double momentum) { m_Momentum = momentum; }

bool CGradientDescent::run(std::size_t n,
                           const TVector &x0,
                           const CFunction &f,
                           const CGradient &gf,
                           TVector &xBest,
                           TDoubleVec &fi) {
    fi.clear();
    fi.reserve(n);

    m_PreviousStep = TVector(x0.dimension(), 0.0);
    TVector x(x0);
    TVector gfx(x.dimension(), 0.0);

    CBasicStatistics::COrderStatisticsStack<double, 1> min;
    CBasicStatistics::SSampleMean<double>::TAccumulator scale;

    for (std::size_t i = 0u; i < n; ++i) {
        double fx;
        if (!f(x, fx)) {
            LOG_ERROR("Bailing on iteration " << i);
            return false;
        }

        if (min.add(fx)) {
            xBest = x;
        }
        fi.push_back(fx);

        if (!gf(x, gfx)) {
            LOG_ERROR("Bailing on iteration " << i);
            return false;
        }
        double norm = gfx.euclidean();
        scale.add(norm);
        gfx *= (-m_LearnRate / CBasicStatistics::mean(scale));

        m_PreviousStep *= m_Momentum;
        m_PreviousStep += gfx;
        LOG_TRACE("gradient fx = " << gfx << ", step = " << m_PreviousStep);

        x += m_PreviousStep;
    }

    LOG_TRACE("fi = " << core::CContainerPrinter::print(fi));
    return true;
}

CGradientDescent::CFunction::~CFunction(void) {}

CGradientDescent::CGradient::~CGradient(void) {}

CGradientDescent::CEmpiricalCentralGradient::CEmpiricalCentralGradient(const CFunction &f,
                                                                       double eps)
    : m_Eps(eps), m_F(f) {}

bool CGradientDescent::CEmpiricalCentralGradient::operator()(const TVector &x,
                                                             TVector &result) const {
    if (x.dimension() != result.dimension()) {
        LOG_ERROR("Dimension mismatch");
        return false;
    }

    xShiftEps = x;
    for (std::size_t i = 0u; i < x.dimension(); ++i) {
        xShiftEps(i) -= m_Eps;
        double fMinusEps;
        if (!m_F(xShiftEps, fMinusEps)) {
            LOG_ERROR("Failed to evaluate function at x - eps");
            return false;
        }
        xShiftEps(i) += 2.0 * m_Eps;
        double fPlusEps;
        if (!m_F(xShiftEps, fPlusEps)) {
            LOG_ERROR("Failed to evaluate function at x + eps");
            return false;
        }
        xShiftEps(i) -= m_Eps;
        result(i) = (fPlusEps - fMinusEps) / (2.0 * m_Eps);
    }

    return true;
}
}
}
