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

#include <maths/CStatisticalTests.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CTools.h>

#include <boost/bind.hpp>
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/range.hpp>

#include <algorithm>

namespace ml {
namespace maths {

namespace {

//! Compute the significance of the Kolmogorov-Smirnov test
//! statistic, \p lambda. In particular, the significance is
//! approximately
//! <pre class="fragment">
//!     \f$P(D > observed) = Q_{KS}\left(\left(\sqrt{N_e}+0.12+\frac{0.11}{\sqrt{N_e}}\right) D\right)\f$
//! </pre>
//! where,
//! <pre class="fragment">
//!    \f$Q_{KS}(\lambda)=2\sum_{j=1}^{\infty}{(-1)^{j-1}e^{-2j^2\lambda^2}}\f$
//! </pre>
//! \see Journal of the Royal Statistical Society, ser. B,
//! vol. 32, pp 115-122 M.A. Stephens.
double significance(double lambda) {
    static const double EPS1 = 0.001;
    static const double EPS2 = 1e-8;
    double sum = 0.0;
    double fac = 2.0;
    double t2 = -2.0 * lambda * lambda;
    double termbf = 0.0;
    for (std::size_t j = 1; j <= 100; ++j) {
        double term = fac * ::exp(t2 * static_cast<double>(j * j));
        sum += term;
        if (::fabs(term) <= EPS1 * termbf || ::fabs(term) <= EPS2 * sum) {
            return sum;
        }
        fac = -fac;
        termbf = term;
    }
    return 1.0;
}

// CCramerVonMises
const std::string SIZE_TAG("a");
const std::string T_TAG("b");
const std::string F_TAG("c");
const std::string EMPTY_STRING;

}

double CStatisticalTests::leftTailFTest(double x, double d1, double d2) {
    if (x < 0.0) {
        return 0.0;
    }
    if (boost::math::isinf(x)) {
        return 1.0;
    }
    try {
        boost::math::fisher_f_distribution<> F(d1, d2);
        return boost::math::cdf(F, x);
    } catch (const std::exception &e)   {
        LOG_ERROR("Failed to compute significance " << e.what()
                  << " d1 = " << d1 << ", d2 = " << d2 << ", x = " << x);
    }
    return 1.0;
}

double CStatisticalTests::rightTailFTest(double x, double d1, double d2) {
    if (x < 0.0) {
        return 1.0;
    }
    if (boost::math::isinf(x)) {
        return 0.0;
    }
    try {
        boost::math::fisher_f_distribution<> F(d1, d2);
        return boost::math::cdf(boost::math::complement(F, x));
    } catch (const std::exception &e)   {
        LOG_ERROR("Failed to compute significance " << e.what()
                  << " d1 = " << d1 << ", d2 = " << d2 << ", x = " << x);
    }
    return 1.0;
}

double CStatisticalTests::twoSampleKS(TDoubleVec x, TDoubleVec y) {
    if (x.empty() || y.empty()) {
        return 1.0;
    }

    std::size_t nx = x.size();
    std::size_t ny = y.size();

    std::sort(x.begin(), x.end());
    x.push_back(boost::numeric::bounds<double>::highest());

    std::sort(y.begin(), y.end());
    y.push_back(boost::numeric::bounds<double>::highest());

    double D = 0.0;
    for (std::size_t i = 0, j = 0; i < nx && j < ny; /**/) {
        double xi = x[i];
        double yj = y[j];
        if (xi <= yj) {
            ++i;
        }
        if (yj <= xi) {
            ++j;
        }
        double Fx = static_cast<double>(i) / static_cast<double>(nx);
        double Fy = static_cast<double>(j) / static_cast<double>(ny);
        D = std::max(D, ::fabs(Fx - Fy));
    }

    double neff = ::sqrt(  static_cast<double>(nx)
                           * static_cast<double>(ny)
                           / static_cast<double>(nx + ny));
    double result = significance((neff + 0.12 + 0.11 / neff) * D);
    LOG_TRACE("nx = " << nx
              << ", ny = " << ny
              << ", D = " << D
              << ", significance = " << result);
    return result;
}

CStatisticalTests::CCramerVonMises::CCramerVonMises(std::size_t size) :
    m_Size(CTools::truncate(size, N[0] - 1, N[12] - 1)) {
    m_F.reserve(size);
}

CStatisticalTests::CCramerVonMises::CCramerVonMises(core::CStateRestoreTraverser &traverser) {
    traverser.traverseSubLevel(boost::bind(&CStatisticalTests::CCramerVonMises::acceptRestoreTraverser,
                                           this,
                                           _1));
}

bool CStatisticalTests::CCramerVonMises::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser) {
    do {
        const std::string &name = traverser.name();
        RESTORE_SETUP_TEARDOWN(SIZE_TAG,
                               /*no-op*/,
                               core::CStringUtils::stringToType(traverser.value(), m_Size),
                               m_F.reserve(m_Size))
        RESTORE(T_TAG, m_T.fromDelimited(traverser.value()))
        RESTORE_SETUP_TEARDOWN(F_TAG, int f,
                               core::CStringUtils::stringToType(traverser.value(), f),
                               m_F.push_back(static_cast<uint16_t>(f)))
    } while (traverser.next());

    return true;
}

void CStatisticalTests::CCramerVonMises::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertValue(SIZE_TAG, m_Size);
    inserter.insertValue(T_TAG, m_T.toDelimited());
    for (std::size_t i= 0u; i < m_F.size(); ++i) {
        inserter.insertValue(F_TAG, static_cast<int>(m_F[i]));
    }
}

void CStatisticalTests::CCramerVonMises::addF(double f) {
    typedef std::vector<double> TDoubleVec;

    if (m_F.size() == m_Size) {
        TDoubleVec ff;
        ff.reserve(m_F.size() + 1);
        for (std::size_t i = 0u; i < m_F.size(); ++i) {
            ff.push_back(static_cast<double>(m_F[i]) / SCALE);
        }
        ff.push_back(f);
        std::sort(ff.begin(), ff.end());
        m_F.clear();

        // Compute the test statistic.
        double n = static_cast<double>(ff.size());
        double t = 1.0 / (12.0 * n);
        for (std::size_t i = 0u; i < ff.size(); ++i) {
            double r = (2.0 * static_cast<double>(i) + 1.0) / (2.0 * n) - ff[i];
            t += r * r;
        }
        m_T.add(t);
    } else   {
        m_F.push_back(static_cast<uint16_t>(SCALE * f));
    }
}

double CStatisticalTests::CCramerVonMises::pValue(void) const
{
    if (CBasicStatistics::count(m_T) == 0.0) {
        return 1.0;
    }

    // Linearly interpolate between the rows of the T statistic
    // values.
    double tt[16];
    ptrdiff_t row = CTools::truncate(std::lower_bound(boost::begin(N),
                                                      boost::end(N),
                                                      m_Size + 1) - N,
                                     ptrdiff_t(1),
                                     ptrdiff_t(12));
    double alpha =   static_cast<double>(m_Size + 1 - N[row - 1])
                   / static_cast<double>(N[row] - N[row - 1]);
    double beta = 1.0 - alpha;
    for (std::size_t i = 0u; i < 16; ++i) {
        tt[i] = alpha * T_VALUES[row][i] + beta * T_VALUES[row - 1][i];
    }
    LOG_TRACE("n = " << m_Size + 1
              << ", tt = " << core::CContainerPrinter::print(tt));

    double t = CBasicStatistics::mean(m_T);
    LOG_TRACE("t = " << t);

    if (t == 0.0) {
        return 1.0;
    }

    ptrdiff_t col = CTools::truncate(std::lower_bound(boost::begin(tt),
                                                      boost::end(tt), t) - tt,
                                     ptrdiff_t(1),
                                     ptrdiff_t(15));
    double a = tt[col - 1];
    double b = tt[col];
    double fa = P_VALUES[col - 1];
    double fb = P_VALUES[col];

    if (fb <= 0.5) {
        // The following fits p(T) = exp(-m/T + c) to extrapolate
        // for t less than the first knot point. Solving for m and
        // c using the value at the first two knot points (a, p(a))
        // and (b, f(b)) gives:
        //   m = ab/(b-a) * log(f(b)/f(a))
        //   c = b/(b-a) * log(f(b)/f(a)) + log(f(a))

        double m = a * b / (b - a) * ::log((fb) / (fa));
        double c = b / (b - a) * ::log((fb) / (fa)) + ::log(fa);

        double p = 1.0 - ::exp(-m / t + c);
        LOG_TRACE("p = 1.0 - exp(" << -m << " / T + " << c << ") = " << p);
        return p;
    }

    // The following fits p(T) = 1 - exp(-mT + c) to interpolate
    // and extrapolate. Solving for m and c using the value of
    // p at the two nearest knot points (a, p(a)) and (b, p(b))
    // gives:
    //   m = 1/(a-b) * log((1-f(b))/(1-f(a)))
    //   c = a/(a-b) * log((1-f(b))/(1-f(a))) + log(1-f(a))

    double m = 1.0 / (a - b) * ::log((1.0 - fb) / (1.0 - fa));
    double c = a / (a - b) * ::log((1.0 - fb) / (1.0 - fa)) + ::log(1.0 - fa);
    double p = ::exp(-m * t + c);
    LOG_TRACE("p = exp(" << -m << " T + " << c << ") = " << p);

    return p;
}

void CStatisticalTests::CCramerVonMises::age(double factor) {
    m_T.age(factor);
}

uint64_t CStatisticalTests::CCramerVonMises::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, m_Size);
    seed = CChecksum::calculate(seed, m_T);
    return CChecksum::calculate(seed, m_F);
}

const double CStatisticalTests::CCramerVonMises::P_VALUES[16] =
{
    0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99, 0.999
};
const std::size_t CStatisticalTests::CCramerVonMises::N[13] =
{
    2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 200, 1000
};
const double CStatisticalTests::CCramerVonMises::T_VALUES[13][16] =
{
    {0.04326, 0.04565, 0.04962, 0.05758, 0.06554, 0.07350, 0.08146, 0.12659, 0.21522, 0.24743, 0.28854, 0.34343, 0.42480, 0.48901, 0.55058, 0.62858},
    {0.03319, 0.03774, 0.04360, 0.05205, 0.06091, 0.06887, 0.07683, 0.12542, 0.21338, 0.24167, 0.27960, 0.33785, 0.43939, 0.53318, 0.63980, 0.82240},
    {0.03002, 0.03536, 0.04149, 0.05093, 0.05896, 0.06681, 0.07493, 0.12406, 0.21171, 0.24260, 0.28336, 0.34184, 0.44206, 0.54200, 0.67017, 0.92970},
    {0.02869, 0.03422, 0.04036, 0.04969, 0.05800, 0.06610, 0.07427, 0.12250, 0.21164, 0.24237, 0.28305, 0.34238, 0.44697, 0.55056, 0.68352, 0.98730},
    {0.02796, 0.03344, 0.03959, 0.04911, 0.05747, 0.06548, 0.07351, 0.12200, 0.21110, 0.24198, 0.28331, 0.34352, 0.44911, 0.55572, 0.69443, 1.02000},
    {0.02741, 0.03292, 0.03914, 0.04869, 0.05698, 0.06492, 0.07297, 0.12158, 0.21087, 0.24197, 0.28345, 0.34397, 0.45100, 0.55935, 0.70154, 1.04250},
    {0.02702, 0.03257, 0.03875, 0.04823, 0.05650, 0.06448, 0.07254, 0.12113, 0.21065, 0.24186, 0.28356, 0.34458, 0.45240, 0.56220, 0.70720, 1.05910},
    {0.02679, 0.03230, 0.03850, 0.04798, 0.05625, 0.06423, 0.07228, 0.12088, 0.21051, 0.24179, 0.28361, 0.34487, 0.45367, 0.56493, 0.71233, 1.07220},
    {0.02657, 0.03209, 0.03830, 0.04778, 0.05605, 0.06403, 0.07208, 0.12068, 0.21040, 0.24173, 0.28365, 0.34510, 0.45441, 0.56643, 0.71531, 1.08220},
    {0.02564, 0.03120, 0.03742, 0.04689, 0.05515, 0.06312, 0.07117, 0.11978, 0.20989, 0.24148, 0.28384, 0.34617, 0.45778, 0.57331, 0.72895, 1.11898},
    {0.02512, 0.03068, 0.03690, 0.04636, 0.05462, 0.06258, 0.07062, 0.11924, 0.20958, 0.24132, 0.28396, 0.34682, 0.45986, 0.57754, 0.73728, 1.14507},
    {0.02488, 0.03043, 0.03665, 0.04610, 0.05435, 0.06231, 0.07035, 0.11897, 0.20943, 0.24125, 0.28402, 0.34715, 0.46091, 0.57968, 0.74149, 1.15783},
    {0.02481, 0.03037, 0.03658, 0.04603, 0.05428, 0.06224, 0.07027, 0.11889, 0.20938, 0.24123, 0.28403, 0.34724, 0.46119, 0.58026, 0.74262, 1.16120}
};
const double CStatisticalTests::CCramerVonMises::SCALE(65536.0);

}
}
