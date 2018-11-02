/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CLocalOutlierFactors.h>

#include <maths/CTools.h>

namespace ml {
namespace maths {

void CLocalOutlierFactors::normalize(TDoubleVec& scores) {
    using TMaxAccumulator =
        CBasicStatistics::COrderStatisticsHeap<double, std::greater<double>>;

    TMaxAccumulator max(std::max(scores.size() / 10, std::size_t(1)));
    max.add(scores);
    double cutoff{max.biggest()};

    TMeanVarAccumulator moments;
    for (auto score : scores) {
        moments.add(std::min(score, cutoff));
    }

    try {
        double mean{CBasicStatistics::mean(moments)};
        double variance{CBasicStatistics::variance(moments)};
        if (variance > 0.0) {
            boost::math::normal normal(mean, std::sqrt(variance));
            for (auto& score : scores) {
                score = cdfComplementToScore(CTools::safeCdfComplement(normal, score));
            }
        } else {
            scores.assign(scores.size(), 0.0);
        }
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to normalise scores: " << e.what());
    }
}

double CLocalOutlierFactors::cdfComplementToScore(double cdfComplement) {
    auto logInterpolate = [](double xa, double fa, double xb, double fb, double x) {
        x = CTools::truncate(x, std::min(xa, xb), std::max(xa, xb));
        return CTools::linearlyInterpolate(CTools::fastLog(xa), CTools::fastLog(xb),
                                           fa, fb, CTools::fastLog(x));
    };
    if (cdfComplement > 1e-4) {
        return 1e-4 / cdfComplement;
    }
    if (cdfComplement > 1e-20) {
        return logInterpolate(1e-4, 1.0, 1e-20, 50.0, cdfComplement);
    }
    return logInterpolate(1e-20, 50.0, std::numeric_limits<double>::min(), 100.0, cdfComplement);
}
}
}
