/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CLocalOutlierFactors.h>

#include <core/CDataFrame.h>

#include <maths/CLinearAlgebraEigen.h>
#include <maths/CTools.h>

#include <boost/math/distributions/normal.hpp>

#include <tuple>
#include <vector>

namespace ml {
namespace maths {

CLocalOutlierFactors::CLocalOutlierFactors(TProgressCallback recordProgress)
    : m_RecordProgress{recordProgress} {
}

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
            core::parallel_for_each(scores.begin(), scores.end(), [normal](double& score) {
                score = cdfComplementToScore(CTools::safeCdfComplement(normal, score));
            });
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
    if (cdfComplement > 1e-3) {
        return 1e-3 / cdfComplement;
    }
    if (cdfComplement > 1e-10) {
        return logInterpolate(1e-10, 50.0, 1e-3, 1.0, cdfComplement);
    }
    return logInterpolate(std::numeric_limits<double>::min(), 100.0, 1e-10, 50.0, cdfComplement);
}

void CLocalOutlierFactors::noop(double) {
}

namespace {
bool computeOutliersNoPartitions(std::size_t numberThreads,
                                 std::function<void(double)> recordProgress,
                                 core::CDataFrame& frame) {

    using TDoubleVec = std::vector<double>;
    using TVector = CMemoryMappedDenseVector<CFloatStorage>;
    using TVectorVec = std::vector<TVector>;
    using TRowItr = core::CDataFrame::TRowItr;

    // The points will be entirely overwritten by readRows so the initial
    // value is not important. This is presized so that rowsToPoints only
    // needs to access and write to each element. Since it does this once
    // per element it is thread safe.
    TVectorVec points(frame.numberRows(), TVector{nullptr, 1});

    auto rowsToPoints = [&points](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            points[row->index()] =
                TVector{row->data(), static_cast<long>(row->numberColumns())};
        }
    };

    std::uint64_t checksum{frame.checksum()};

    bool successful;
    std::tie(std::ignore, successful) = frame.readRows(numberThreads, rowsToPoints);

    if (successful == false) {
        LOG_ERROR(<< "Failed to read the data frame");
        return false;
    }

    CLocalOutlierFactors outliers{recordProgress};

    TDoubleVec scores;
    outliers.ensemble(std::move(points), scores);

    // This never happens now, but it is a sanity check against someone
    // changing CLocalOutlierFactors to accidentally write to the data
    // frame via one of the memory mapped vectors. All bets are off as to
    // if we generate anything meaningful if this happens.
    if (checksum != frame.checksum()) {
        LOG_ERROR(<< "Accidentally modified the data frame");
        return false;
    }

    auto writeScores = [&scores](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            row->writeColumn(row->numberColumns() - 1, scores[row->index()]);
        }
    };

    frame.resizeColumns(numberThreads, frame.numberColumns() + 1);

    if (frame.writeColumns(numberThreads, writeScores) == false) {
        LOG_ERROR(<< "Failed to write scores to the data frame");
        return false;
    }
    return true;
}
}

bool computeOutliers(std::size_t numberThreads,
                     std::function<void(double)> recordProgress,
                     core::CDataFrame& frame) {
    // TODO memory monitoring.
    // TODO different analysis types.

    if (frame.inMainMemory()) {
        if (computeOutliersNoPartitions(numberThreads, recordProgress, frame) == false) {
            HANDLE_FATAL(<< "Internal error: computing outliers for data frame. There "
                         << "may be more details in the logs. Please report this problem.");
        }
        return true;
    }

    // TODO this needs to work out the partitioning and run outlier detection
    // one partition at a time.

    return false;
}
}
}
