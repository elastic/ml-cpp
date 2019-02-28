/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CDataFrameUtils.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/Concurrency.h>

#include <maths/CBasicStatistics.h>
#include <maths/CMathsFuncs.h>
#include <maths/CQuantileSketch.h>

#include <vector>

namespace ml {
namespace maths {
using TRowItr = core::CDataFrame::TRowItr;

bool CDataFrameUtils::standardizeColumns(std::size_t numberThreads, core::CDataFrame& frame) {
    using TDoubleVec = std::vector<double>;
    using TMeanVarAccumulatorVec =
        std::vector<CBasicStatistics::SSampleMeanVar<double>::TAccumulator>;

    if (frame.numberRows() == 0 || frame.numberColumns() == 0) {
        return true;
    }

    auto computeColumnMoments = core::bindRetrievableState(
        [](TMeanVarAccumulatorVec& moments, TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                for (std::size_t i = 0; i < row->numberColumns(); ++i) {
                    if (isMissing((*row)[i]) == false) {
                        moments[i].add((*row)[i]);
                    }
                }
            }
        },
        TMeanVarAccumulatorVec{frame.numberColumns()});

    auto results = frame.readRows(numberThreads, computeColumnMoments);
    if (results.second == false) {
        LOG_ERROR(<< "Failed to standardise columns");
        return false;
    }

    TMeanVarAccumulatorVec moments{frame.numberColumns()};
    for (const auto& result : results.first) {
        for (std::size_t i = 0; i < moments.size(); ++i) {
            moments[i] += result.s_FunctionState[i];
        }
    }

    TDoubleVec mean(moments.size());
    TDoubleVec scale(moments.size());
    for (std::size_t i = 0; i < moments.size(); ++i) {
        double variance{CBasicStatistics::variance(moments[i])};
        mean[i] = CBasicStatistics::mean(moments[i]);
        scale[i] = variance == 0.0 ? 1.0 : 1.0 / std::sqrt(variance);
    }

    LOG_TRACE(<< "means = " << core::CContainerPrinter::print(mean));
    LOG_TRACE(<< "scales = " << core::CContainerPrinter::print(scale));

    auto standardiseColumns = [&mean, &scale](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            for (std::size_t i = 0; i < row->numberColumns(); ++i) {
                row->writeColumn(i, scale[i] * ((*row)[i] - mean[i]));
            }
        }
    };

    return frame.writeColumns(numberThreads, standardiseColumns);
}

bool CDataFrameUtils::columnQuantiles(std::size_t numberThreads,
                                      const core::CDataFrame& frame,
                                      const TSizeVec& columnMask,
                                      TQuantileSketchVec& result,
                                      TWeightFunction weight) {
    if (result.size() != columnMask.size()) {
        return false;
    }

    auto results = frame.readRows(
        numberThreads,
        core::bindRetrievableState(
            [&](TQuantileSketchVec& quantiles, TRowItr beginRows, TRowItr endRows) {
                for (auto row = beginRows; row != endRows; ++row) {
                    for (auto i : columnMask) {
                        if (isMissing((*row)[i]) == false) {
                            quantiles[i].add((*row)[i], weight(*row));
                        }
                    }
                }
            },
            std::move(result)));

    if (results.second == false) {
        return false;
    }

    result = std::move(results.first[0].s_FunctionState);
    for (std::size_t i = 1; i < results.first.size(); ++i) {
        for (std::size_t j = 0; j < result.size(); ++j) {
            result[j] += results.first[i].s_FunctionState[j];
        }
    }

    return true;
}

bool CDataFrameUtils::isMissing(double x) {
    return CMathsFuncs::isFinite(x) == false;
}
}
}
