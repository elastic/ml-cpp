/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CDataFrameUtils.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CPackedBitVector.h>
#include <core/Concurrency.h>

#include <maths/CBasicStatistics.h>
#include <maths/CMathsFuncs.h>
#include <maths/CMic.h>
#include <maths/COrderings.h>
#include <maths/CQuantileSketch.h>
#include <maths/CSampling.h>

#include <boost/unordered_set.hpp>

#include <vector>

namespace ml {
namespace maths {
namespace {
using TFloatVec = std::vector<CFloatStorage>;
using TFloatVecVec = std::vector<TFloatVec>;
using TFloatFloatPr = std::pair<CFloatStorage, CFloatStorage>;
using TFloatFloatPrVec = std::vector<TFloatFloatPr>;
using TFloatUSet = boost::unordered_set<CFloatStorage, std::hash<double>>;
using TRowItr = core::CDataFrame::TRowItr;
using TRowRef = core::CDataFrame::TRowRef;
using TRowSampler = CSampling::CRandomStreamSampler<TRowRef>;

//! Reduce the results of a call to core::CDataFrame::readRows using \p reduceFirst
//! for the first and \p reduce for the rest and writing the result \p reduction.
//!
//! \tparam REDUCER Must be a binary operator whose logical type is the function
//! void (typeof(READER.s_FunctionState[0]), REDUCTION&).
template<typename READER, typename REDUCER, typename FIRST_REDUCER, typename REDUCTION>
bool doReduce(std::pair<std::vector<READER>, bool> readResults,
              FIRST_REDUCER reduceFirst,
              REDUCER reduce,
              REDUCTION& reduction) {
    if (readResults.second == false) {
        return false;
    }
    reduceFirst(std::move(readResults.first[0].s_FunctionState), reduction);
    for (std::size_t i = 1; i < readResults.first.size(); ++i) {
        reduce(std::move(readResults.first[i].s_FunctionState), reduction);
    }
    return true;
}

//! Get a row feature sampler.
auto rowFeatureSampler(std::size_t i, std::size_t targetColumn, TFloatFloatPrVec& samples) {
    return [i, targetColumn, &samples](std::size_t slot, const TRowRef& row) {
        if (slot >= samples.size()) {
            samples.resize(slot + 1, {0.0, 0.0});
        }
        samples[slot].first = row[i];
        samples[slot].second = row[targetColumn];
    };
}

//! Get a row sampler.
auto rowSampler(TFloatVecVec& samples) {
    return [&samples](std::size_t slot, const TRowRef& row) {
        if (slot >= samples.size()) {
            samples.resize(slot + 1, TFloatVec(row.numberColumns()));
        }
        row.copyTo(samples[slot].begin());
    };
}

const std::size_t NUMBER_SAMPLES_TO_COMPUTE_MIC{10000};
}

bool CDataFrameUtils::standardizeColumns(std::size_t numberThreads, core::CDataFrame& frame) {

    using TMeanVarAccumulatorVec =
        std::vector<CBasicStatistics::SSampleMeanVar<double>::TAccumulator>;

    if (frame.numberRows() == 0 || frame.numberColumns() == 0) {
        return true;
    }

    auto readColumnMoments = core::bindRetrievableState(
        [](TMeanVarAccumulatorVec& moments_, TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                for (std::size_t i = 0; i < row->numberColumns(); ++i) {
                    if (isMissing((*row)[i]) == false) {
                        moments_[i].add((*row)[i]);
                    }
                }
            }
        },
        TMeanVarAccumulatorVec(frame.numberColumns()));
    auto copyColumnMoments = [](TMeanVarAccumulatorVec moments_,
                                TMeanVarAccumulatorVec& moments) {
        moments = std::move(moments_);
    };
    auto reduceColumnMoments = [](TMeanVarAccumulatorVec moments_,
                                  TMeanVarAccumulatorVec& moments) {
        for (std::size_t i = 0; i < moments.size(); ++i) {
            moments[i] += moments_[i];
        }
    };

    TMeanVarAccumulatorVec moments;
    if (doReduce(frame.readRows(numberThreads, readColumnMoments),
                 copyColumnMoments, reduceColumnMoments, moments) == false) {
        LOG_ERROR(<< "Failed to standardise columns");
        return false;
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

    return frame.writeColumns(numberThreads, standardiseColumns).second;
}

bool CDataFrameUtils::columnQuantiles(std::size_t numberThreads,
                                      const core::CDataFrame& frame,
                                      const core::CPackedBitVector& rowMask,
                                      const TSizeVec& columnMask,
                                      const CQuantileSketch& sketch,
                                      TQuantileSketchVec& result,
                                      TWeightFunction weight) {

    auto readQuantiles = core::bindRetrievableState(
        [&](TQuantileSketchVec& quantiles, TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                for (std::size_t i = 0; i < columnMask.size(); ++i) {
                    if (isMissing((*row)[columnMask[i]]) == false) {
                        quantiles[i].add((*row)[columnMask[i]], weight(*row));
                    }
                }
            }
        },
        TQuantileSketchVec(columnMask.size(), sketch));
    auto copyQuantiles = [](TQuantileSketchVec quantiles, TQuantileSketchVec& result_) {
        result_ = std::move(quantiles);
    };
    auto reduceQuantiles = [&](TQuantileSketchVec quantiles, TQuantileSketchVec& result_) {
        for (std::size_t i = 0; i < columnMask.size(); ++i) {
            result_[i] += quantiles[i];
        }
    };

    if (doReduce(frame.readRows(numberThreads, 0, frame.numberRows(), readQuantiles, &rowMask),
                 copyQuantiles, reduceQuantiles, result) == false) {
        LOG_ERROR(<< "Failed to compute column quantiles");
        return false;
    }

    return true;
}

CDataFrameUtils::TDoubleVecVec
CDataFrameUtils::categoryFrequencies(std::size_t numberThreads,
                                     const core::CDataFrame& frame,
                                     TSizeVec columnMask) {

    removeMetricColumns(frame, columnMask);
    if (frame.numberRows() == 0 || columnMask.empty()) {
        return TDoubleVecVec(frame.numberColumns());
    }

    auto readCategoryCounts = core::bindRetrievableState(
        [&](TDoubleVecVec& counts, TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                for (std::size_t i : columnMask) {
                    std::size_t id{static_cast<std::size_t>((*row)[i])};
                    counts[i].resize(std::max(counts[i].size(), id + 1), 0.0);
                    counts[i][id] += 1.0;
                }
            }
        },
        TDoubleVecVec(frame.numberColumns()));
    auto copyCategoryCounts = [](TDoubleVecVec counts, TDoubleVecVec& result) {
        result = std::move(counts);
    };
    auto reduceCategoryCounts = [](TDoubleVecVec counts, TDoubleVecVec& result) {
        for (std::size_t i = 0; i < counts.size(); ++i) {
            for (std::size_t j = 0; j < counts[i].size(); ++j) {
                result[i][j] += counts[i][j];
            }
        }
    };

    TDoubleVecVec result;
    if (doReduce(frame.readRows(numberThreads, readCategoryCounts),
                 copyCategoryCounts, reduceCategoryCounts, result) == false) {
        HANDLE_FATAL(<< "Internal error: failed to calculate category"
                     << " frequencies. Please report this problem.");
        return result;
    }
    for (std::size_t i = 0; i < result.size(); ++i) {
        for (std::size_t j = 0; j < result[i].size(); ++j) {
            result[i][j] /= static_cast<double>(frame.numberRows());
        }
    }

    return result;
}

CDataFrameUtils::TDoubleVecVec
CDataFrameUtils::meanValueOfTargetForCategories(std::size_t numberThreads,
                                                const core::CDataFrame& frame,
                                                TSizeVec columnMask,
                                                std::size_t targetColumn) {

    TDoubleVecVec result(frame.numberColumns());

    if (targetColumn >= frame.numberColumns()) {
        HANDLE_FATAL(<< "Internal error: target column out of bounds '"
                     << targetColumn << " >= " << frame.numberColumns()
                     << "'. Please report this problem.");
        return result;
    }

    removeMetricColumns(frame, columnMask);
    if (frame.numberRows() == 0 || columnMask.empty()) {
        return result;
    }

    using TMeanAccumulatorVec = std::vector<CBasicStatistics::SSampleMean<double>::TAccumulator>;
    using TMeanAccumulatorVecVec = std::vector<TMeanAccumulatorVec>;

    auto readColumnMeans = core::bindRetrievableState(
        [&](TMeanAccumulatorVecVec& means_, TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                for (std::size_t i : columnMask) {
                    std::size_t id{static_cast<std::size_t>((*row)[i])};
                    means_[i].resize(std::max(means_[i].size(), id + 1));
                    means_[i][id].add((*row)[targetColumn]);
                }
            }
        },
        TMeanAccumulatorVecVec(frame.numberColumns()));
    auto copyColumnMeans = [](TMeanAccumulatorVecVec means_, TMeanAccumulatorVecVec& means) {
        means = std::move(means_);
    };
    auto reduceColumnMeans = [](TMeanAccumulatorVecVec means_, TMeanAccumulatorVecVec& means) {
        for (std::size_t i = 0; i < means_.size(); ++i) {
            for (std::size_t j = 0; j < means_[i].size(); ++j) {
                means[i][j] += means_[i][j];
            }
        }
    };

    TMeanAccumulatorVecVec means;
    if (doReduce(frame.readRows(numberThreads, readColumnMeans),
                 copyColumnMeans, reduceColumnMeans, means) == false) {
        HANDLE_FATAL(<< "Internal error: failed to calculate mean target value"
                     << " for categories. Please report this problem.");
        return result;
    }
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i].resize(means[i].size());
        for (std::size_t j = 0; j < means[i].size(); ++j) {
            result[i][j] = CBasicStatistics::mean(means[i][j]);
        }
    }

    return result;
}

CDataFrameUtils::TSizeDoublePrVecVec
CDataFrameUtils::categoryMicWithColumn(std::size_t numberThreads,
                                       const core::CDataFrame& frame,
                                       TSizeVec columnMask,
                                       std::size_t targetColumn,
                                       double minimumFrequency) {

    TSizeDoublePrVecVec none(frame.numberColumns());

    if (targetColumn >= frame.numberColumns()) {
        HANDLE_FATAL(<< "Internal error: target column out of bounds '"
                     << targetColumn << " >= " << frame.numberColumns()
                     << "'. Please report this problem.");
        return none;
    }

    removeMetricColumns(frame, columnMask);
    if (frame.numberRows() == 0 || columnMask.empty()) {
        return none;
    }

    std::size_t numberSamples{std::min(NUMBER_SAMPLES_TO_COMPUTE_MIC, frame.numberRows())};

    auto method = frame.inMainMemory() ? categoryMicWithColumnDataFrameInMemory
                                       : categoryMicWithColumnDataFrameOnDisk;

    TSizeDoublePrVecVec mics(method(numberThreads, frame, columnMask, targetColumn,
                                    numberSamples, minimumFrequency));

    for (auto& categoryMics : mics) {
        std::sort(categoryMics.begin(), categoryMics.end(),
                  [](const TSizeDoublePr& lhs, const TSizeDoublePr& rhs) {
                      return COrderings::lexicographical_compare(
                          -lhs.second, lhs.first, -rhs.second, rhs.first);
                  });
    }

    return mics;
}

CDataFrameUtils::TDoubleVec CDataFrameUtils::micWithColumn(const core::CDataFrame& frame,
                                                           TSizeVec columnMask,
                                                           std::size_t targetColumn) {

    TDoubleVec zeros(frame.numberColumns(), 0.0);

    if (targetColumn >= frame.numberColumns()) {
        HANDLE_FATAL(<< "Internal error: target column out of bounds '"
                     << targetColumn << " >= " << frame.numberColumns()
                     << "'. Please report this problem.");
        return zeros;
    }

    removeCategoricalColumns(frame, columnMask);
    if (frame.numberRows() == 0 || columnMask.empty()) {
        return zeros;
    }

    std::size_t numberSamples{std::min(NUMBER_SAMPLES_TO_COMPUTE_MIC, frame.numberRows())};

    auto method = frame.inMainMemory() ? micWithColumnDataFrameInMemory
                                       : micWithColumnDataFrameOnDisk;

    return method(frame, columnMask, targetColumn, numberSamples);
}

bool CDataFrameUtils::isMissing(double x) {
    return CMathsFuncs::isFinite(x) == false;
}

CDataFrameUtils::TSizeDoublePrVecVec
CDataFrameUtils::categoryMicWithColumnDataFrameInMemory(std::size_t numberThreads,
                                                        const core::CDataFrame& frame,
                                                        const TSizeVec& columnMask,
                                                        std::size_t targetColumn,
                                                        std::size_t numberSamples,
                                                        double minimumFrequency) {

    TSizeDoublePrVecVec mics(frame.numberColumns());

    TDoubleVecVec frequencies(categoryFrequencies(numberThreads, frame, columnMask));

    TFloatFloatPrVec samples;
    TFloatUSet categories;
    CMic mic;

    samples.reserve(numberSamples);
    mic.reserve(numberSamples);

    for (auto i : columnMask) {

        // Sample

        TRowSampler sampler{numberSamples, rowFeatureSampler(i, targetColumn, samples)};
        frame.readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                std::size_t j{static_cast<std::size_t>((*row)[i])};
                if (frequencies[i][j] >= minimumFrequency &&
                    isMissing((*row)[targetColumn]) == false) {
                    sampler.sample(*row);
                }
            }
        });
        LOG_TRACE(<< "# samples = " << samples.size());

        // Compute MICe

        categories.clear();
        for (const auto& sample : samples) {
            categories.insert(sample.first);
        }

        TSizeDoublePrVec categoryMics;
        categoryMics.reserve(categories.size());
        for (auto category : categories) {
            mic.clear();
            for (const auto& sample : samples) {
                mic.add(sample.first != category ? 0.0 : 1.0, sample.second);
            }
            categoryMics.emplace_back(static_cast<std::size_t>(category), mic.compute());
        }
        mics[i] = std::move(categoryMics);

        samples.clear();
    }

    return mics;
}

CDataFrameUtils::TSizeDoublePrVecVec
CDataFrameUtils::categoryMicWithColumnDataFrameOnDisk(std::size_t numberThreads,
                                                      const core::CDataFrame& frame,
                                                      const TSizeVec& columnMask,
                                                      std::size_t targetColumn,
                                                      std::size_t numberSamples,
                                                      double minimumFrequency) {

    TSizeDoublePrVecVec mics(frame.numberColumns());

    TDoubleVecVec frequencies(categoryFrequencies(numberThreads, frame, columnMask));

    TFloatVecVec samples;
    TFloatUSet categories;
    CMic mic;

    samples.reserve(numberSamples);
    mic.reserve(numberSamples);

    // Sample
    //
    // The law of large numbers means we have a high probability of sampling
    // each category provided minimumFrequency * NUMBER_SAMPLES_TO_COMPUTE_MIC
    // is large (which we ensure it is).

    TRowSampler sampler{numberSamples, rowSampler(samples)};
    frame.readRows(1, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            if (isMissing((*row)[targetColumn]) == false) {
                sampler.sample(*row);
            }
        }
    });
    LOG_TRACE(<< "# samples = " << samples.size());

    for (auto i : columnMask) {
        categories.clear();
        for (const auto& sample : samples) {
            std::size_t category{static_cast<std::size_t>(sample[i])};
            if (frequencies[i][category] >= minimumFrequency) {
                categories.insert(sample[i]);
            }
        }

        TSizeDoublePrVec categoryMics;
        categoryMics.reserve(categories.size());
        for (auto category : categories) {
            mic.clear();
            for (const auto& sample : samples) {
                mic.add(sample[i] != category ? 0.0 : 1.0, sample[targetColumn]);
            }
            categoryMics.emplace_back(static_cast<std::size_t>(category), mic.compute());
        }
        mics[i] = std::move(categoryMics);
    }

    return mics;
}

CDataFrameUtils::TDoubleVec
CDataFrameUtils::micWithColumnDataFrameInMemory(const core::CDataFrame& frame,
                                                const TSizeVec& columnMask,
                                                std::size_t targetColumn,
                                                std::size_t numberSamples) {

    TDoubleVec mics(frame.numberColumns());

    TFloatFloatPrVec samples;
    samples.reserve(numberSamples);

    for (auto i : columnMask) {

        // Do sampling

        TRowSampler sampler{numberSamples, rowFeatureSampler(i, targetColumn, samples)};
        auto missingCount = frame.readRows(
            1, core::bindRetrievableState(
                   [&](std::size_t& missing, TRowItr beginRows, TRowItr endRows) {
                       for (auto row = beginRows; row != endRows; ++row) {
                           if (isMissing((*row)[i])) {
                               ++missing;
                           } else if (isMissing((*row)[targetColumn]) == false) {
                               sampler.sample(*row);
                           }
                       }
                   },
                   std::size_t{0}));
        LOG_TRACE(<< "# samples = " << samples.size());

        double fractionMissing{static_cast<double>(missingCount.first[0].s_FunctionState) /
                               static_cast<double>(frame.numberRows())};
        LOG_TRACE(<< "Fraction missing = " << fractionMissing);

        // Compute MICe

        CMic mic;
        mic.reserve(samples.size());
        for (const auto& sample : samples) {
            mic.add(sample.first, sample.second);
        }

        mics[i] = (1.0 - fractionMissing) * mic.compute();
        samples.clear();
    }

    return mics;
}

CDataFrameUtils::TDoubleVec
CDataFrameUtils::micWithColumnDataFrameOnDisk(const core::CDataFrame& frame,
                                              const TSizeVec& columnMask,
                                              std::size_t targetColumn,
                                              std::size_t numberSamples) {

    TDoubleVec mics(frame.numberColumns());

    TFloatVecVec samples;
    samples.reserve(numberSamples);

    // Do sampling

    TRowSampler sampler{numberSamples, rowSampler(samples)};
    auto missingCounts = frame.readRows(
        1, core::bindRetrievableState(
               [&](TSizeVec& missing, TRowItr beginRows, TRowItr endRows) {
                   for (auto row = beginRows; row != endRows; ++row) {
                       for (std::size_t i = 0; i < row->numberColumns(); ++i) {
                           missing[i] += isMissing((*row)[i]) ? 1 : 0;
                       }
                       if (isMissing((*row)[targetColumn]) == false) {
                           sampler.sample(*row);
                       }
                   }
               },
               TSizeVec(frame.numberColumns(), 0)));
    LOG_TRACE(<< "# samples = " << samples.size());

    TDoubleVec fractionMissing(frame.numberColumns());
    for (std::size_t i = 0; i < fractionMissing.size(); ++i) {
        for (const auto& missingCount : missingCounts.first) {
            fractionMissing[i] += static_cast<double>(missingCount.s_FunctionState[i]) /
                                  static_cast<double>(frame.numberRows());
        }
    }
    LOG_TRACE(<< "Fraction missing = " << core::CContainerPrinter::print(fractionMissing));

    // Compute MICe

    for (auto i : columnMask) {
        if (i != targetColumn) {
            CMic mic;
            mic.reserve(samples.size());
            for (const auto& sample : samples) {
                if (isMissing(sample[i]) == false) {
                    mic.add(sample[i], sample[targetColumn]);
                }
            }
            mics[i] = (1.0 - fractionMissing[i]) * mic.compute();
        }
    }

    return mics;
}

void CDataFrameUtils::removeMetricColumns(const core::CDataFrame& frame, TSizeVec& columnMask) {
    const auto& columnIsCategorical = frame.columnIsCategorical();
    columnMask.erase(std::remove_if(columnMask.begin(), columnMask.end(),
                                    [&columnIsCategorical](std::size_t i) {
                                        return columnIsCategorical[i] == false;
                                    }),
                     columnMask.end());
}

void CDataFrameUtils::removeCategoricalColumns(const core::CDataFrame& frame,
                                               TSizeVec& columnMask) {
    const auto& columnIsCategorical = frame.columnIsCategorical();
    columnMask.erase(std::remove_if(columnMask.begin(), columnMask.end(),
                                    [&columnIsCategorical](std::size_t i) {
                                        return columnIsCategorical[i];
                                    }),
                     columnMask.end());
}

double CDataFrameUtils::unitWeight(const TRowRef&) {
    return 1.0;
}
}
}
