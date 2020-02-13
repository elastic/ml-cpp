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
#include <maths/CDataFrameCategoryEncoder.h>
#include <maths/CMathsFuncs.h>
#include <maths/CMic.h>
#include <maths/COrderings.h>
#include <maths/CQuantileSketch.h>
#include <maths/CSampling.h>
#include <maths/CSolvers.h>
#include <maths/CTools.h>

#include <boost/unordered_map.hpp>

#include <memory>
#include <numeric>
#include <vector>

namespace ml {
namespace maths {
namespace {
using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TSizeVecVec = std::vector<TSizeVec>;
using TFloatVec = std::vector<CFloatStorage>;
using TFloatVecVec = std::vector<TFloatVec>;
using TRowItr = core::CDataFrame::TRowItr;
using TRowRef = core::CDataFrame::TRowRef;
using TRowSampler = CSampling::CRandomStreamSampler<TRowRef>;
using TRowSamplerVec = std::vector<TRowSampler>;
using TSizeEncoderPtrUMap =
    boost::unordered_map<std::size_t, std::unique_ptr<CDataFrameUtils::CColumnValue>>;
using TPackedBitVectorVec = CDataFrameUtils::TPackedBitVectorVec;

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

//! \brief Manages stratified sampling.
class CStratifiedSampler {
public:
    using TSamplerSelector = std::function<std::size_t(const TRowRef&)>;

public:
    CStratifiedSampler(std::size_t size) : m_SampledRowIndices(size) {
        m_DesiredCounts.reserve(size);
        m_Samplers.reserve(size);
    }

    void sample(const TRowRef& row) { m_Samplers[m_Selector(row)].sample(row); }

    //! Add one of the strata samplers.
    void addSampler(std::size_t count, CPRNG::CXorOShiro128Plus rng) {
        TSizeVec& samples{m_SampledRowIndices[m_Samplers.size()]};
        samples.reserve(count);
        auto sampler = [&](std::size_t slot, const TRowRef& row) {
            if (slot >= samples.size()) {
                samples.resize(slot + 1);
            }
            samples[slot] = row.index();
        };
        m_DesiredCounts.push_back(count);
        m_Samplers.emplace_back(count, sampler, rng);
    }

    //! Define the callback to select the sampler.
    void samplerSelector(TSamplerSelector selector) {
        m_Selector = std::move(selector);
    }

    //! This selects the final samples, writing to \p result, and resets the sampling
    //! state so this is ready to sample again.
    void finishSampling(CPRNG::CXorOShiro128Plus& rng, TSizeVec& result) {
        result.clear();
        for (std::size_t i = 0; i < m_SampledRowIndices.size(); ++i) {
            std::size_t sampleSize{m_Samplers[i].sampleSize()};
            std::size_t desiredCount{std::min(m_DesiredCounts[i], sampleSize)};
            CSampling::random_shuffle(rng, m_SampledRowIndices[i].begin(),
                                      m_SampledRowIndices[i].begin() + sampleSize);
            result.insert(result.end(), m_SampledRowIndices[i].begin(),
                          m_SampledRowIndices[i].begin() + desiredCount);
            m_SampledRowIndices[i].clear();
            m_Samplers[i].reset();
        }
    }

private:
    TSizeVec m_DesiredCounts;
    TSizeVecVec m_SampledRowIndices;
    TRowSamplerVec m_Samplers;
    TSamplerSelector m_Selector;
};

//! Get a classifier stratified row sampler for cross fold validation.
std::pair<std::unique_ptr<CStratifiedSampler>, TDoubleVec>
classifierStratifiedCrossValidationRowSampler(std::size_t numberThreads,
                                              const core::CDataFrame& frame,
                                              std::size_t targetColumn,
                                              CPRNG::CXorOShiro128Plus rng,
                                              std::size_t numberFolds,
                                              const core::CPackedBitVector& allTrainingRowsMask) {

    TDoubleVec categoryFrequencies{CDataFrameUtils::categoryFrequencies(
        numberThreads, frame, allTrainingRowsMask, {targetColumn})[targetColumn]};

    TSizeVec categoryCounts;
    double numberTrainingRows{allTrainingRowsMask.manhattan()};
    std::size_t desiredCount{
        (static_cast<std::size_t>(numberTrainingRows) + numberFolds / 2) / numberFolds};
    CSampling::weightedSample(desiredCount, categoryFrequencies, categoryCounts);
    LOG_TRACE(<< "desired category counts per test fold = "
              << core::CContainerPrinter::print(categoryCounts));

    auto sampler = std::make_unique<CStratifiedSampler>(categoryCounts.size());
    for (std::size_t i = 0; i < categoryCounts.size(); ++i) {
        sampler->addSampler(categoryCounts[i], rng);
    }
    sampler->samplerSelector([targetColumn](const TRowRef& row) mutable {
        return static_cast<std::size_t>(row[targetColumn]);
    });

    return {std::move(sampler), std::move(categoryFrequencies)};
}

//! Get a regression stratified row sampler for cross fold validation.
std::unique_ptr<CStratifiedSampler>
regressionStratifiedCrossValiationRowSampler(std::size_t numberThreads,
                                             const core::CDataFrame& frame,
                                             std::size_t targetColumn,
                                             CPRNG::CXorOShiro128Plus rng,
                                             std::size_t numberFolds,
                                             std::size_t numberBuckets,
                                             const core::CPackedBitVector& allTrainingRowsMask) {

    auto quantiles = CDataFrameUtils::columnQuantiles(
                         numberThreads, frame, allTrainingRowsMask, {targetColumn},
                         CQuantileSketch{CQuantileSketch::E_Linear, 50})
                         .first;

    TDoubleVec buckets;
    for (double step = 100.0 / static_cast<double>(numberBuckets), percentile = step;
         percentile < 100.0; percentile += step) {
        double xQuantile;
        quantiles[0].quantile(percentile, xQuantile);
        buckets.push_back(xQuantile);
    }
    buckets.erase(std::unique(buckets.begin(), buckets.end()), buckets.end());
    buckets.push_back(std::numeric_limits<double>::max());
    LOG_TRACE(<< "buckets = " << core::CContainerPrinter::print(buckets));

    auto bucketSelector = [buckets, targetColumn](const TRowRef& row) mutable {
        return static_cast<std::size_t>(
            std::upper_bound(buckets.begin(), buckets.end(), row[targetColumn]) -
            buckets.begin());
    };

    auto countBucketRows = core::bindRetrievableState(
        [&](TDoubleVec& bucketCounts, TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                bucketCounts[bucketSelector(*row)] += 1.0;
            }
        },
        TDoubleVec(buckets.size(), 0.0));
    auto copyBucketRowCounts = [](TDoubleVec counts_, TDoubleVec& counts) {
        counts = std::move(counts_);
    };
    auto reduceBucketRowCounts = [](TDoubleVec counts_, TDoubleVec& counts) {
        for (std::size_t i = 0; i < counts.size(); ++i) {
            counts[i] += counts_[i];
        }
    };

    TDoubleVec bucketFrequencies;
    doReduce(frame.readRows(numberThreads, 0, frame.numberRows(),
                            countBucketRows, &allTrainingRowsMask),
             copyBucketRowCounts, reduceBucketRowCounts, bucketFrequencies);
    double totalCount{std::accumulate(bucketFrequencies.begin(),
                                      bucketFrequencies.end(), 0.0)};
    for (auto& frequency : bucketFrequencies) {
        frequency /= totalCount;
    }

    TSizeVec bucketCounts;
    std::size_t desiredCount{
        (static_cast<std::size_t>(totalCount) + numberFolds / 2) / numberFolds};
    CSampling::weightedSample(desiredCount, bucketFrequencies, bucketCounts);
    LOG_TRACE(<< "desired bucket counts per fold = "
              << core::CContainerPrinter::print(bucketCounts));

    auto sampler = std::make_unique<CStratifiedSampler>(buckets.size());
    for (std::size_t i = 0; i < buckets.size(); ++i) {
        sampler->addSampler(bucketCounts[i], rng);
    }
    sampler->samplerSelector(bucketSelector);

    return sampler;
}

//! Get the test row masks corresponding to \p foldRowMasks.
TPackedBitVectorVec complementRowMasks(const TPackedBitVectorVec& foldRowMasks,
                                       core::CPackedBitVector allRowsMask) {
    TPackedBitVectorVec complementFoldRowMasks(foldRowMasks.size(), std::move(allRowsMask));
    for (std::size_t fold = 0; fold < foldRowMasks.size(); ++fold) {
        complementFoldRowMasks[fold] ^= foldRowMasks[fold];
    }
    return complementFoldRowMasks;
}

//! Get a row feature sampler.
template<typename TARGET>
auto rowFeatureSampler(std::size_t i, const TARGET& target, TFloatVecVec& samples) {
    return [i, &target, &samples](std::size_t slot, const TRowRef& row) {
        if (slot >= samples.size()) {
            samples.resize(slot + 1, {0.0, 0.0});
        }
        samples[slot][0] = row[i];
        samples[slot][1] = target(row);
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

template<typename TARGET>
auto computeEncodedCategory(CMic& mic,
                            const TARGET& target,
                            TSizeEncoderPtrUMap& encoders,
                            TFloatVecVec& samples) {

    CDataFrameUtils::TSizeDoublePrVec encodedMics;
    encodedMics.reserve(encoders.size());
    for (const auto& encoder : encoders) {
        std::size_t category{encoder.first};
        const auto& encode = *encoder.second;
        mic.clear();
        for (const auto& sample : samples) {
            mic.add(encode(sample), target(sample));
        }
        encodedMics.emplace_back(category, mic.compute());
    }
    return encodedMics;
}

const std::size_t NUMBER_SAMPLES_TO_COMPUTE_MIC{10000};
}

std::string CDataFrameUtils::SDataType::toDelimited() const {
    // clang-format off
    return core::CStringUtils::typeToString(static_cast<int>(s_IsInteger)) +
           INTERNAL_DELIMITER +
           core::CStringUtils::typeToStringPrecise(s_Min, core::CIEEE754::E_DoublePrecision) +
           INTERNAL_DELIMITER +
           core::CStringUtils::typeToStringPrecise(s_Max, core::CIEEE754::E_DoublePrecision) +
           INTERNAL_DELIMITER;
    // clang-format on
}

bool CDataFrameUtils::SDataType::fromDelimited(const std::string& delimited) {
    TDoubleVec state(3);
    int pos{0}, i{0};
    for (auto delimiter = delimited.find(INTERNAL_DELIMITER); delimiter != std::string::npos;
         delimiter = delimited.find(INTERNAL_DELIMITER, pos)) {
        if (core::CStringUtils::stringToType(delimited.substr(pos, delimiter - pos),
                                             state[i++]) == false) {
            return false;
        }
        pos = static_cast<int>(delimiter + 1);
    }
    std::tie(s_IsInteger, s_Min, s_Max) =
        std::make_tuple(state[0] == 1.0, state[1], state[2]);
    return true;
}

const char CDataFrameUtils::SDataType::INTERNAL_DELIMITER{':'};
const char CDataFrameUtils::SDataType::EXTERNAL_DELIMITER{';'};

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

CDataFrameUtils::TDataTypeVec
CDataFrameUtils::columnDataTypes(std::size_t numberThreads,
                                 const core::CDataFrame& frame,
                                 const core::CPackedBitVector& rowMask,
                                 const TSizeVec& columnMask,
                                 const CDataFrameCategoryEncoder* encoder) {

    if (frame.numberRows() == 0) {
        return {};
    }

    using TMinMax = CBasicStatistics::CMinMax<double>;
    using TMinMaxBoolPrVec = std::vector<std::pair<TMinMax, bool>>;

    auto readDataTypes = core::bindRetrievableState(
        [&](TMinMaxBoolPrVec& types, TRowItr beginRows, TRowItr endRows) {
            double integerPart;
            if (encoder != nullptr) {
                for (auto row = beginRows; row != endRows; ++row) {
                    CEncodedDataFrameRowRef encodedRow{encoder->encode(*row)};
                    for (auto i : columnMask) {
                        double value{encodedRow[i]};
                        if (isMissing(value) == false) {
                            types[i].first.add(value);
                            types[i].second = types[i].second &&
                                              (std::modf(value, &integerPart) == 0.0);
                        }
                    }
                }
            } else {
                for (auto row = beginRows; row != endRows; ++row) {
                    for (auto i : columnMask) {
                        double value{(*row)[i]};
                        if (isMissing(value) == false) {
                            types[i].first.add(value);
                            types[i].second = types[i].second &&
                                              (std::modf(value, &integerPart) == 0.0);
                        }
                    }
                }
            }
        },
        TMinMaxBoolPrVec(encoder != nullptr ? encoder->numberEncodedColumns()
                                            : frame.numberColumns(),
                         {TMinMax{}, true}));

    auto copyDataTypes = [](TMinMaxBoolPrVec types, TMinMaxBoolPrVec& result) {
        result = std::move(types);
    };
    auto reduceDataTypes = [&](TMinMaxBoolPrVec types, TMinMaxBoolPrVec& result) {
        for (auto i : columnMask) {
            result[i].first += types[i].first;
            result[i].second = result[i].second && types[i].second;
        }
    };

    TMinMaxBoolPrVec types;
    doReduce(frame.readRows(numberThreads, 0, frame.numberRows(), readDataTypes, &rowMask),
             copyDataTypes, reduceDataTypes, types);

    TDataTypeVec result(types.size());
    for (auto i : columnMask) {
        result[i] = SDataType{types[i].second, types[i].first.min(),
                              types[i].first.max()};
    }

    return result;
}

std::pair<CDataFrameUtils::TQuantileSketchVec, bool>
CDataFrameUtils::columnQuantiles(std::size_t numberThreads,
                                 const core::CDataFrame& frame,
                                 const core::CPackedBitVector& rowMask,
                                 const TSizeVec& columnMask,
                                 CQuantileSketch estimateQuantiles,
                                 const CDataFrameCategoryEncoder* encoder,
                                 TWeightFunction weight) {

    auto readQuantiles = core::bindRetrievableState(
        [&](TQuantileSketchVec& quantiles, TRowItr beginRows, TRowItr endRows) {
            if (encoder != nullptr) {
                for (auto row = beginRows; row != endRows; ++row) {
                    CEncodedDataFrameRowRef encodedRow{encoder->encode(*row)};
                    for (std::size_t i = 0; i < columnMask.size(); ++i) {
                        if (isMissing(encodedRow[columnMask[i]]) == false) {
                            quantiles[i].add(encodedRow[columnMask[i]], weight(*row));
                        }
                    }
                }
            } else {
                for (auto row = beginRows; row != endRows; ++row) {
                    for (std::size_t i = 0; i < columnMask.size(); ++i) {
                        if (isMissing((*row)[columnMask[i]]) == false) {
                            quantiles[i].add((*row)[columnMask[i]], weight(*row));
                        }
                    }
                }
            }
        },
        TQuantileSketchVec(columnMask.size(), std::move(estimateQuantiles)));
    auto copyQuantiles = [](TQuantileSketchVec quantiles, TQuantileSketchVec& result) {
        result = std::move(quantiles);
    };
    auto reduceQuantiles = [&](TQuantileSketchVec quantiles, TQuantileSketchVec& result) {
        for (std::size_t i = 0; i < columnMask.size(); ++i) {
            result[i] += quantiles[i];
        }
    };

    TQuantileSketchVec result;
    if (doReduce(frame.readRows(numberThreads, 0, frame.numberRows(), readQuantiles, &rowMask),
                 copyQuantiles, reduceQuantiles, result) == false) {
        LOG_ERROR(<< "Failed to compute column quantiles");
        return {std::move(result), false};
    }

    return {std::move(result), true};
}

std::tuple<TPackedBitVectorVec, TPackedBitVectorVec, TDoubleVec>
CDataFrameUtils::stratifiedCrossValidationRowMasks(std::size_t numberThreads,
                                                   const core::CDataFrame& frame,
                                                   std::size_t targetColumn,
                                                   CPRNG::CXorOShiro128Plus rng,
                                                   std::size_t numberFolds,
                                                   std::size_t numberBuckets,
                                                   const core::CPackedBitVector& allTrainingRowsMask) {

    TDoubleVec frequencies;
    std::unique_ptr<CStratifiedSampler> sampler;

    if (frame.columnIsCategorical()[targetColumn]) {
        std::tie(sampler, frequencies) = classifierStratifiedCrossValidationRowSampler(
            numberThreads, frame, targetColumn, rng, numberFolds, allTrainingRowsMask);
    } else {
        sampler = regressionStratifiedCrossValiationRowSampler(
            numberThreads, frame, targetColumn, rng, numberFolds, numberBuckets,
            allTrainingRowsMask);
    }

    LOG_TRACE(<< "number training rows = " << allTrainingRowsMask.manhattan());

    TPackedBitVectorVec testingRowMasks(numberFolds);

    TSizeVec rowIndices;
    core::CPackedBitVector candidateTestingRowsMask{allTrainingRowsMask};
    for (std::size_t fold = 0; fold < numberFolds - 1; ++fold) {
        frame.readRows(1, 0, frame.numberRows(),
                       [&](TRowItr beginRows, TRowItr endRows) {
                           for (auto row = beginRows; row != endRows; ++row) {
                               sampler->sample(*row);
                           }
                       },
                       &candidateTestingRowsMask);
        sampler->finishSampling(rng, rowIndices);
        std::sort(rowIndices.begin(), rowIndices.end());
        LOG_TRACE(<< "# row indices = " << rowIndices.size());

        for (auto row : rowIndices) {
            testingRowMasks[fold].extend(false, row - testingRowMasks[fold].size());
            testingRowMasks[fold].extend(true);
        }
        testingRowMasks[fold].extend(false, allTrainingRowsMask.size() -
                                                testingRowMasks[fold].size());

        // We exclusive or here to remove the rows we've selected for the current
        //test fold. This is equivalent to samplng without replacement
        candidateTestingRowsMask ^= testingRowMasks[fold];
    }

    // Everything which is left.
    testingRowMasks.back() = std::move(candidateTestingRowsMask);
    LOG_TRACE(<< "# remaining rows = " << testingRowMasks.back().manhattan());

    TPackedBitVectorVec trainingRowMasks{complementRowMasks(testingRowMasks, allTrainingRowsMask)};

    return {std::move(trainingRowMasks), std::move(testingRowMasks), std::move(frequencies)};
}

CDataFrameUtils::TDoubleVecVec
CDataFrameUtils::categoryFrequencies(std::size_t numberThreads,
                                     const core::CDataFrame& frame,
                                     const core::CPackedBitVector& rowMask,
                                     TSizeVec columnMask) {

    removeMetricColumns(frame, columnMask);
    if (frame.numberRows() == 0 || columnMask.empty()) {
        return TDoubleVecVec(frame.numberColumns());
    }

    // Note this can throw a length_error in resize hence the try block around read.
    auto readCategoryCounts = core::bindRetrievableState(
        [&](TDoubleVecVec& counts, TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                for (std::size_t i : columnMask) {
                    std::size_t category{static_cast<std::size_t>((*row)[i])};
                    counts[i].resize(std::max(counts[i].size(), category + 1), 0.0);
                    counts[i][category] += 1.0;
                }
            }
        },
        TDoubleVecVec(frame.numberColumns()));
    auto copyCategoryCounts = [](TDoubleVecVec counts, TDoubleVecVec& result) {
        result = std::move(counts);
    };
    auto reduceCategoryCounts = [](TDoubleVecVec counts, TDoubleVecVec& result) {
        for (std::size_t i = 0; i < counts.size(); ++i) {
            result[i].resize(std::max(result[i].size(), counts[i].size()), 0.0);
            for (std::size_t j = 0; j < counts[i].size(); ++j) {
                result[i][j] += counts[i][j];
            }
        }
    };

    TDoubleVecVec result;
    try {
        if (doReduce(frame.readRows(numberThreads, 0, frame.numberRows(),
                                    readCategoryCounts, &rowMask),
                     copyCategoryCounts, reduceCategoryCounts, result) == false) {
            HANDLE_FATAL(<< "Internal error: failed to calculate category"
                         << " frequencies. Please report this problem.");
            return result;
        }
    } catch (const std::exception& e) {
        HANDLE_FATAL(<< "Internal error: '" << e.what() << "' exception calculating"
                     << " category frequencies. Please report this problem.");
    }

    double Z{rowMask.manhattan()};
    for (std::size_t i = 0; i < result.size(); ++i) {
        for (std::size_t j = 0; j < result[i].size(); ++j) {
            result[i][j] /= Z;
        }
    }

    return result;
}

CDataFrameUtils::TDoubleVecVec
CDataFrameUtils::meanValueOfTargetForCategories(const CColumnValue& target,
                                                std::size_t numberThreads,
                                                const core::CDataFrame& frame,
                                                const core::CPackedBitVector& rowMask,
                                                TSizeVec columnMask) {

    TDoubleVecVec result(frame.numberColumns());

    removeMetricColumns(frame, columnMask);
    if (frame.numberRows() == 0 || columnMask.empty()) {
        return result;
    }

    using TMeanAccumulatorVec = std::vector<CBasicStatistics::SSampleMean<double>::TAccumulator>;
    using TMeanAccumulatorVecVec = std::vector<TMeanAccumulatorVec>;

    // Note this can throw a length_error in resize hence the try block around read.
    auto readColumnMeans = core::bindRetrievableState(
        [&](TMeanAccumulatorVecVec& means_, TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                for (std::size_t i : columnMask) {
                    if (isMissing(target(*row)) == false) {
                        std::size_t category{static_cast<std::size_t>((*row)[i])};
                        means_[i].resize(std::max(means_[i].size(), category + 1));
                        means_[i][category].add(target(*row));
                    }
                }
            }
        },
        TMeanAccumulatorVecVec(frame.numberColumns()));
    auto copyColumnMeans = [](TMeanAccumulatorVecVec means_, TMeanAccumulatorVecVec& means) {
        means = std::move(means_);
    };
    auto reduceColumnMeans = [](TMeanAccumulatorVecVec means_, TMeanAccumulatorVecVec& means) {
        for (std::size_t i = 0; i < means_.size(); ++i) {
            means[i].resize(std::max(means[i].size(), means_[i].size()));
            for (std::size_t j = 0; j < means_[i].size(); ++j) {
                means[i][j] += means_[i][j];
            }
        }
    };

    TMeanAccumulatorVecVec means;
    try {
        if (doReduce(frame.readRows(numberThreads, 0, frame.numberRows(), readColumnMeans, &rowMask),
                     copyColumnMeans, reduceColumnMeans, means) == false) {
            HANDLE_FATAL(<< "Internal error: failed to calculate mean target values"
                         << " for categories. Please report this problem.");
            return result;
        }
    } catch (const std::exception& e) {
        HANDLE_FATAL(<< "Internal error: '" << e.what() << "' exception calculating"
                     << " mean target values for categories. Please report this problem.");
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

CDataFrameUtils::TSizeDoublePrVecVecVec
CDataFrameUtils::categoricalMicWithColumn(const CColumnValue& target,
                                          std::size_t numberThreads,
                                          const core::CDataFrame& frame,
                                          const core::CPackedBitVector& rowMask,
                                          TSizeVec columnMask,
                                          const TEncoderFactoryVec& encoderFactories) {

    TSizeDoublePrVecVecVec none(encoderFactories.size(),
                                TSizeDoublePrVecVec(frame.numberColumns()));

    removeMetricColumns(frame, columnMask);
    if (frame.numberRows() == 0 || columnMask.empty()) {
        return none;
    }

    auto method = frame.inMainMemory() ? categoricalMicWithColumnDataFrameInMemory
                                       : categoricalMicWithColumnDataFrameOnDisk;

    TDoubleVecVec frequencies(categoryFrequencies(numberThreads, frame, rowMask, columnMask));
    LOG_TRACE(<< "frequencies = " << core::CContainerPrinter::print(frequencies));

    TSizeDoublePrVecVecVec mics(
        method(target, frame, rowMask, columnMask, encoderFactories, frequencies,
               std::min(NUMBER_SAMPLES_TO_COMPUTE_MIC, frame.numberRows())));

    for (auto& encoderMics : mics) {
        for (auto& categoryMics : encoderMics) {
            std::sort(categoryMics.begin(), categoryMics.end(),
                      [](const TSizeDoublePr& lhs, const TSizeDoublePr& rhs) {
                          return COrderings::lexicographical_compare(
                              -lhs.second, lhs.first, -rhs.second, rhs.first);
                      });
        }
    }

    return mics;
}

CDataFrameUtils::TDoubleVec
CDataFrameUtils::metricMicWithColumn(const CColumnValue& target,
                                     const core::CDataFrame& frame,
                                     const core::CPackedBitVector& rowMask,
                                     TSizeVec columnMask) {

    TDoubleVec zeros(frame.numberColumns(), 0.0);

    removeCategoricalColumns(frame, columnMask);
    if (frame.numberRows() == 0 || columnMask.empty()) {
        return zeros;
    }

    auto method = frame.inMainMemory() ? metricMicWithColumnDataFrameInMemory
                                       : metricMicWithColumnDataFrameOnDisk;

    return method(target, frame, rowMask, columnMask,
                  std::min(NUMBER_SAMPLES_TO_COMPUTE_MIC, frame.numberRows()));
}

double
CDataFrameUtils::maximumMinimumRecallDecisionThreshold(std::size_t numberThreads,
                                                       const core::CDataFrame& frame,
                                                       const core::CPackedBitVector& rowMask,
                                                       std::size_t targetColumn,
                                                       std::size_t predictionColumn) {

    auto readQuantiles = core::bindRetrievableState(
        [&](TQuantileSketchVec& quantiles, TRowItr beginRows, TRowItr endRows) {
            for (auto row = beginRows; row != endRows; ++row) {
                if (isMissing((*row)[targetColumn]) == false) {
                    quantiles[static_cast<std::size_t>((*row)[targetColumn])].add(
                        CTools::logisticFunction((*row)[predictionColumn]));
                }
            }
        },
        TQuantileSketchVec(2, CQuantileSketch{CQuantileSketch::E_Linear, 100}));
    auto copyQuantiles = [](TQuantileSketchVec quantiles, TQuantileSketchVec& result) {
        result = std::move(quantiles);
    };
    auto reduceQuantiles = [&](TQuantileSketchVec quantiles, TQuantileSketchVec& result) {
        for (std::size_t i = 0; i < 2; ++i) {
            result[i] += quantiles[i];
        }
    };

    TQuantileSketchVec classProbabilityClassOneQuantiles;
    if (doReduce(frame.readRows(numberThreads, 0, frame.numberRows(), readQuantiles, &rowMask),
                 copyQuantiles, reduceQuantiles, classProbabilityClassOneQuantiles) == false) {
        HANDLE_FATAL(<< "Failed to compute category quantiles");
        return 0.5;
    }

    auto minRecall = [&](double threshold) {
        double cdf[2];
        classProbabilityClassOneQuantiles[0].cdf(threshold, cdf[0]);
        classProbabilityClassOneQuantiles[1].cdf(threshold, cdf[1]);
        double recalls[]{cdf[0], 1.0 - cdf[1]};
        return std::min(recalls[0], recalls[1]);
    };

    double threshold;
    double minRecallAtThreshold;
    std::size_t maxIterations{20};
    CSolvers::maximize(0.0, 1.0, minRecall(0.0), minRecall(1.0), minRecall,
                       1e-3, maxIterations, threshold, minRecallAtThreshold);
    LOG_TRACE(<< "threshold = " << threshold
              << ", min recall at threshold = " << minRecallAtThreshold);
    return threshold;
}

bool CDataFrameUtils::isMissing(double x) {
    return CMathsFuncs::isFinite(x) == false;
}

CDataFrameUtils::TSizeDoublePrVecVecVec CDataFrameUtils::categoricalMicWithColumnDataFrameInMemory(
    const CColumnValue& target,
    const core::CDataFrame& frame,
    const core::CPackedBitVector& rowMask,
    const TSizeVec& columnMask,
    const TEncoderFactoryVec& encoderFactories,
    const TDoubleVecVec& frequencies,
    std::size_t numberSamples) {

    TSizeDoublePrVecVecVec encoderMics;
    encoderMics.reserve(encoderFactories.size());

    TFloatVecVec samples;
    TSizeEncoderPtrUMap encoders;
    CMic mic;
    samples.reserve(numberSamples);
    mic.reserve(numberSamples);

    for (const auto& encoderFactory : encoderFactories) {

        TEncoderFactory makeEncoder;
        double minimumFrequency;
        std::tie(makeEncoder, minimumFrequency) = encoderFactory;

        TSizeDoublePrVecVec mics(frame.numberColumns());

        for (auto i : columnMask) {

            // Sample

            samples.clear();
            TRowSampler sampler{numberSamples, rowFeatureSampler(i, target, samples)};
            frame.readRows(
                1, 0, frame.numberRows(),
                [&](TRowItr beginRows, TRowItr endRows) {
                    for (auto row = beginRows; row != endRows; ++row) {
                        std::size_t category{static_cast<std::size_t>((*row)[i])};
                        if (frequencies[i][category] >= minimumFrequency &&
                            isMissing(target(*row)) == false) {
                            sampler.sample(*row);
                        }
                    }
                },
                &rowMask);
            LOG_TRACE(<< "# samples = " << samples.size());

            // Setup encoders

            encoders.clear();
            for (const auto& sample : samples) {
                std::size_t category{static_cast<std::size_t>(sample[0])};
                auto encoder = makeEncoder(i, 0, category);
                std::size_t hash{encoder->hash()};
                encoders.emplace(hash, std::move(encoder));
            }

            auto target_ = [](const TFloatVec& sample) { return sample[1]; };
            mics[i] = computeEncodedCategory(mic, target_, encoders, samples);
        }

        encoderMics.push_back(std::move(mics));
    }

    return encoderMics;
}

CDataFrameUtils::TSizeDoublePrVecVecVec CDataFrameUtils::categoricalMicWithColumnDataFrameOnDisk(
    const CColumnValue& target,
    const core::CDataFrame& frame,
    const core::CPackedBitVector& rowMask,
    const TSizeVec& columnMask,
    const TEncoderFactoryVec& encoderFactories,
    const TDoubleVecVec& frequencies,
    std::size_t numberSamples) {

    TSizeDoublePrVecVecVec encoderMics;
    encoderMics.reserve(encoderFactories.size());

    TFloatVecVec samples;
    TSizeEncoderPtrUMap encoders;
    CMic mic;
    samples.reserve(numberSamples);
    mic.reserve(numberSamples);

    for (const auto& encoderFactory : encoderFactories) {

        TEncoderFactory makeEncoder;
        double minimumFrequency;
        std::tie(makeEncoder, minimumFrequency) = encoderFactory;

        TSizeDoublePrVecVec mics(frame.numberColumns());

        // Sample
        //
        // The law of large numbers means we have a high probability of sampling
        // each category provided minimumFrequency * NUMBER_SAMPLES_TO_COMPUTE_MIC
        // is large (which we ensure it is).

        samples.clear();
        TRowSampler sampler{numberSamples, rowSampler(samples)};
        frame.readRows(1, 0, frame.numberRows(),
                       [&](TRowItr beginRows, TRowItr endRows) {
                           for (auto row = beginRows; row != endRows; ++row) {
                               if (isMissing(target(*row)) == false) {
                                   sampler.sample(*row);
                               }
                           }
                       },
                       &rowMask);
        LOG_TRACE(<< "# samples = " << samples.size());

        for (auto i : columnMask) {

            // Setup encoders

            encoders.clear();
            for (const auto& sample : samples) {
                std::size_t category{static_cast<std::size_t>(sample[i])};
                if (frequencies[i][category] >= minimumFrequency) {
                    auto encoder = makeEncoder(i, i, category);
                    std::size_t hash{encoder->hash()};
                    encoders.emplace(hash, std::move(encoder));
                }
            }

            mics[i] = computeEncodedCategory(mic, target, encoders, samples);
        }

        encoderMics.push_back(std::move(mics));
    }

    return encoderMics;
}

CDataFrameUtils::TDoubleVec
CDataFrameUtils::metricMicWithColumnDataFrameInMemory(const CColumnValue& target,
                                                      const core::CDataFrame& frame,
                                                      const core::CPackedBitVector& rowMask,
                                                      const TSizeVec& columnMask,
                                                      std::size_t numberSamples) {

    TDoubleVec mics(frame.numberColumns(), 0.0);

    TFloatVecVec samples;
    samples.reserve(numberSamples);
    double numberMaskedRows{rowMask.manhattan()};

    for (auto i : columnMask) {

        // Do sampling

        TRowSampler sampler{numberSamples, rowFeatureSampler(i, target, samples)};
        auto missingCount = frame.readRows(
            1, 0, frame.numberRows(),
            core::bindRetrievableState(
                [&](std::size_t& missing, TRowItr beginRows, TRowItr endRows) {
                    for (auto row = beginRows; row != endRows; ++row) {
                        if (isMissing((*row)[i])) {
                            ++missing;
                        } else if (isMissing(target(*row)) == false) {
                            sampler.sample(*row);
                        }
                    }
                },
                std::size_t{0}),
            &rowMask);
        LOG_TRACE(<< "# samples = " << samples.size());

        double fractionMissing{static_cast<double>(missingCount.first[0].s_FunctionState) /
                               numberMaskedRows};
        LOG_TRACE(<< "feature = " << i << " fraction missing = " << fractionMissing);

        // Compute MICe

        CMic mic;
        mic.reserve(samples.size());
        for (const auto& sample : samples) {
            mic.add(sample[0], sample[1]);
        }

        mics[i] = (1.0 - fractionMissing) * mic.compute();
        samples.clear();
    }

    return mics;
}

CDataFrameUtils::TDoubleVec
CDataFrameUtils::metricMicWithColumnDataFrameOnDisk(const CColumnValue& target,
                                                    const core::CDataFrame& frame,
                                                    const core::CPackedBitVector& rowMask,
                                                    const TSizeVec& columnMask,
                                                    std::size_t numberSamples) {

    TDoubleVec mics(frame.numberColumns(), 0.0);

    TFloatVecVec samples;
    samples.reserve(numberSamples);
    double numberMaskedRows{rowMask.manhattan()};

    // Do sampling

    TRowSampler sampler{numberSamples, rowSampler(samples)};
    auto missingCounts = frame.readRows(
        1, 0, frame.numberRows(),
        core::bindRetrievableState(
            [&](TSizeVec& missing, TRowItr beginRows, TRowItr endRows) {
                for (auto row = beginRows; row != endRows; ++row) {
                    for (std::size_t i = 0; i < row->numberColumns(); ++i) {
                        missing[i] += isMissing((*row)[i]) ? 1 : 0;
                    }
                    if (isMissing(target(*row)) == false) {
                        sampler.sample(*row);
                    }
                }
            },
            TSizeVec(frame.numberColumns(), 0)),
        &rowMask);
    LOG_TRACE(<< "# samples = " << samples.size());

    TDoubleVec fractionMissing(frame.numberColumns());
    for (std::size_t i = 0; i < fractionMissing.size(); ++i) {
        for (const auto& missingCount : missingCounts.first) {
            fractionMissing[i] +=
                static_cast<double>(missingCount.s_FunctionState[i]) / numberMaskedRows;
        }
    }
    LOG_TRACE(<< "Fraction missing = " << core::CContainerPrinter::print(fractionMissing));

    // Compute MICe

    for (auto i : columnMask) {
        CMic mic;
        mic.reserve(samples.size());
        for (const auto& sample : samples) {
            if (isMissing(sample[i]) == false) {
                mic.add(sample[i], target(sample));
            }
        }
        mics[i] = (1.0 - fractionMissing[i]) * mic.compute();
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
