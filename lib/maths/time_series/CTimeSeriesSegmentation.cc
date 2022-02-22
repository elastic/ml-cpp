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

#include <maths/time_series/CTimeSeriesSegmentation.h>

#include <core/CContainerPrinter.h>
#include <core/CTriple.h>
#include <core/CVectorRange.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CLeastSquaresOnlineRegression.h>
#include <maths/common/CLeastSquaresOnlineRegressionDetail.h>
#include <maths/common/COrderings.h>
#include <maths/common/CStatisticalTests.h>
#include <maths/common/CTools.h>

#include <maths/time_series/CSignal.h>

#include <algorithm>
#include <limits>
#include <numeric>

namespace ml {
namespace maths {
namespace time_series {

CTimeSeriesSegmentation::TSizeVec
CTimeSeriesSegmentation::piecewiseLinear(const TFloatMeanAccumulatorVec& values,
                                         double pValueToSegment,
                                         double outlierFraction,
                                         std::size_t maxSegments) {

    std::size_t maxDepth{static_cast<std::size_t>(
        std::ceil(std::log2(static_cast<double>(maxSegments))))};
    LOG_TRACE(<< "max depth = " << maxDepth);

    TSizeVec segmentation{0, values.size()};
    TDoubleDoublePrVec depthAndPValue;
    TFloatMeanAccumulatorVec reweighted;
    fitTopDownPiecewiseLinear(values.cbegin(), values.cend(), 0, // depth
                              0, // offset of first value in range
                              maxDepth, pValueToSegment, outlierFraction,
                              segmentation, depthAndPValue, reweighted);

    selectSegmentation(maxSegments, segmentation, depthAndPValue);

    return segmentation;
}

CTimeSeriesSegmentation::TFloatMeanAccumulatorVec
CTimeSeriesSegmentation::removePiecewiseLinear(TFloatMeanAccumulatorVec values,
                                               const TSizeVec& segmentation,
                                               double outlierFraction,
                                               TDoubleVec& shifts) {
    TFloatMeanAccumulatorVec reweighted{values};
    auto predict = fitPiecewiseLinear(segmentation, outlierFraction, reweighted);
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (common::CBasicStatistics::count(values[i]) > 0.0) {
            common::CBasicStatistics::moment<0>(values[i]) -=
                predict(static_cast<double>(i));
        }
    }
    shifts.clear();
    shifts.reserve(segmentation.size() - 1);
    for (std::size_t i = 1; i < segmentation.size(); ++i) {
        TMeanAccumulator shift;
        for (std::size_t j = segmentation[i - 1]; j < segmentation[i]; ++j) {
            shift.add(predict(static_cast<double>(j)));
        }
        shifts.push_back(common::CBasicStatistics::mean(shift));
    }
    return values;
}

CTimeSeriesSegmentation::TFloatMeanAccumulatorVec
CTimeSeriesSegmentation::removePiecewiseLinearDiscontinuities(TFloatMeanAccumulatorVec values,
                                                              const TSizeVec& segmentation,
                                                              double outlierFraction) {

    TFloatMeanAccumulatorVec reweighted{values};
    auto predict = fitPiecewiseLinear(segmentation, outlierFraction, reweighted);
    TDoubleVec steps;
    steps.reserve(segmentation.size() - 1);
    for (std::size_t i = 1; i + 1 < segmentation.size(); ++i) {
        double j{static_cast<double>(segmentation[i])};
        steps.push_back(predict(j - 0.05) - predict(j + 0.05));
    }
    steps.push_back(0.0);
    LOG_TRACE(<< "steps = " << core::CContainerPrinter::print(steps));

    std::partial_sum(steps.rbegin(), steps.rend(), steps.rbegin());
    for (std::size_t i = segmentation.size() - 1; i > 0; --i) {
        LOG_TRACE(<< "Shifting [" << segmentation[i - 1] << ","
                  << segmentation[i] << ") by " << steps[i - 1]);
        for (std::size_t j = segmentation[i - 1]; j < segmentation[i]; ++j) {
            common::CBasicStatistics::moment<0>(values[j]) -= steps[i - 1];
        }
    }

    return values;
}

CTimeSeriesSegmentation::TSizeVec
CTimeSeriesSegmentation::piecewiseLinearScaledSeasonal(const TFloatMeanAccumulatorVec& values,
                                                       const TSeasonality& seasonality,
                                                       double pValueToSegment,
                                                       std::size_t maxSegments) {

    std::size_t maxDepth{static_cast<std::size_t>(
        std::ceil(std::log2(static_cast<double>(maxSegments))))};
    LOG_TRACE(<< "max depth = " << maxDepth);

    TSizeVec segmentation{0, values.size()};
    TDoubleDoublePrVec depthAndPValue;
    fitTopDownPiecewiseLinearScaledSeasonal(values.cbegin(), values.cend(),
                                            0, // depth
                                            0, // offset of first value in range
                                            seasonality, maxDepth, pValueToSegment,
                                            segmentation, depthAndPValue);

    selectSegmentation(maxSegments, segmentation, depthAndPValue);

    return segmentation;
}

CTimeSeriesSegmentation::TFloatMeanAccumulatorVec
CTimeSeriesSegmentation::removePiecewiseLinearScaledSeasonal(TFloatMeanAccumulatorVec values,
                                                             const TSeasonality& model,
                                                             const TSizeVec& segmentation,
                                                             double outlierFraction,
                                                             TDoubleVec& scales) {
    TFloatMeanAccumulatorVec reweighted{values};
    fitPiecewiseLinearScaledSeasonal(values, model, segmentation,
                                     outlierFraction, reweighted, scales);
    for (std::size_t i = 0; i < values.size(); ++i) {
        common::CBasicStatistics::moment<0>(values[i]) -=
            scaleAt(i, segmentation, scales) * model(i);
    }
    return values;
}

CTimeSeriesSegmentation::TFloatMeanAccumulatorVec
CTimeSeriesSegmentation::constantScalePiecewiseLinearScaledSeasonal(
    const TFloatMeanAccumulatorVec& values,
    const TSeasonalComponentVec& periods,
    const TSizeVec& segmentation,
    const TConstantScale& computeConstantScale,
    double outlierFraction,
    TMeanAccumulatorVecVec& models,
    TDoubleVec& scales) {

    TFloatMeanAccumulatorVec scaledValues;
    fitPiecewiseLinearScaledSeasonal(values, periods, segmentation, outlierFraction,
                                     scaledValues, models, scales);
    LOG_TRACE(<< "models = " << core::CContainerPrinter::print(models));
    LOG_TRACE(<< "segmentation = " << core::CContainerPrinter::print(segmentation));
    LOG_TRACE(<< "scales = " << core::CContainerPrinter::print(scales));

    double constantScale{computeConstantScale(segmentation, scales)};
    LOG_TRACE(<< "scale = " << constantScale);

    auto seasonality = [&](std::size_t i) {
        double result{0.0};
        for (std::size_t j = 0; j < periods.size(); ++j) {
            result += periods[j].value(models[j], i);
        }
        return result;
    };

    scaledValues = values;
    TDoubleVec weights;
    weights.reserve(scales.size());
    for (std::size_t i = 1; i < segmentation.size(); ++i) {
        TMeanVarAccumulator valueMoments;
        TMeanVarAccumulator residualMoments;
        double scale{std::max(scales[i - 1] - 1e-3, 0.0)};
        for (std::size_t j = segmentation[i - 1]; j < segmentation[i]; ++j) {
            double count{common::CBasicStatistics::count(scaledValues[j])};
            if (count > 0.0) {
                double value{common::CBasicStatistics::mean(values[j])};
                double residual{value - scale * seasonality(j)};
                common::CBasicStatistics::moment<0>(scaledValues[j]) = residual;
                valueMoments.add(value, count);
                residualMoments.add(residual, count);
            }
        }
        double valueRmse{std::sqrt(common::CBasicStatistics::variance(valueMoments))};
        double residualRmse{std::sqrt(common::CBasicStatistics::variance(residualMoments))};
        weights.push_back(common::CTools::linearlyInterpolate(
            residualRmse, 4.0 * residualRmse, 0.0, 1.0, valueRmse));
    }
    LOG_TRACE(<< "weights = " << core::CContainerPrinter::print(weights));

    for (std::size_t i = 0; i < scaledValues.size(); ++i) {
        if (std::any_of(periods.begin(), periods.end(),
                        [&](const auto& period) { return period.contains(i); })) {
            double weight{scaleAt(i, segmentation, weights)};
            if (weight <= 0.0) {
                // If the component is scaled to the noise level.
                scaledValues[i] = TFloatMeanAccumulator{};
            } else if (common::CBasicStatistics::count(scaledValues[i]) > 0.0) {
                auto& mean = common::CBasicStatistics::moment<0>(scaledValues[i]);
                double scale{scaleAt(i, segmentation, scales)};
                common::CBasicStatistics::count(scaledValues[i]) *= weight;
                mean *= constantScale / scale;
                mean += constantScale * seasonality(i);
            }
        }
    }

    return scaledValues;
}

double CTimeSeriesSegmentation::meanScale(const TSizeVec& segmentation,
                                          const TDoubleVec& scales,
                                          const TIndexWeight& indexWeight) {
    TMeanAccumulator result;
    for (std::size_t i = 1; i < segmentation.size(); ++i) {
        for (std::size_t j = segmentation[i - 1]; j < segmentation[i]; ++j) {
            result.add(scales[i - 1], indexWeight(j));
        }
    }
    return common::CBasicStatistics::mean(result);
}

double CTimeSeriesSegmentation::scaleAt(std::size_t index,
                                        const TSizeVec& segmentation,
                                        const TDoubleVec& scales) {
    return scales[std::upper_bound(segmentation.begin(), segmentation.end(), index) -
                  segmentation.begin() - 1];
}

CTimeSeriesSegmentation::TSizeVec
CTimeSeriesSegmentation::piecewiseTimeShifted(const TFloatMeanAccumulatorVec& values,
                                              core_t::TTime bucketLength,
                                              const TTimeVec& candidateShifts,
                                              const TModel& model,
                                              double pValueToSegment,
                                              std::size_t maxSegments,
                                              TTimeVec* shifts) {

    std::size_t maxDepth{static_cast<std::size_t>(
        std::ceil(std::log2(static_cast<double>(maxSegments))))};
    LOG_TRACE(<< "max depth = " << maxDepth);

    TDoubleVecVec predictions(candidateShifts.size() + 1, TDoubleVec(values.size()));
    for (std::size_t j = 0; j < values.size(); ++j) {
        predictions[0][j] = model(bucketLength * static_cast<core_t::TTime>(j));
    }
    for (std::size_t i = 0; i < candidateShifts.size(); ++i) {
        core_t::TTime time{candidateShifts[i]};
        for (std::size_t j = 0; j < values.size(); ++j, time += bucketLength) {
            predictions[i + 1][j] = model(time);
        }
    }

    TFloatMeanAccumulatorVec reweighted;
    TSizeVec segmentation{0, values.size()};
    TDoubleDoublePrVec depthAndPValue;
    fitTopDownPiecewiseTimeShifted(values.cbegin(), values.cend(),
                                   0, // depth
                                   0, // offset of first value in range
                                   predictions, maxDepth, pValueToSegment,
                                   segmentation, depthAndPValue);

    selectSegmentation(maxSegments, segmentation, depthAndPValue);

    if (shifts != nullptr) {
        shifts->clear();
        shifts->reserve(segmentation.size() - 1);
        for (std::size_t i = 1; i < segmentation.size(); ++i) {
            std::size_t shift;
            std::tie(std::ignore, shift) = residualMoments(
                values.begin() + segmentation[i - 1],
                values.begin() + segmentation[i], segmentation[i - 1], predictions);
            shifts->push_back(shift == 0 ? 0 : candidateShifts[shift - 1]);
        }
    }

    return segmentation;
}

core_t::TTime CTimeSeriesSegmentation::shiftAt(std::size_t index,
                                               const TSizeVec& segmentation,
                                               const TTimeVec& shifts) {
    return shifts[std::upper_bound(segmentation.begin(), segmentation.end(), index) -
                  segmentation.begin() - 1];
}

void CTimeSeriesSegmentation::selectSegmentation(std::size_t maxSegments,
                                                 TSizeVec& segmentation,
                                                 TDoubleDoublePrVec& depthAndPValue) {

    if (segmentation.size() - 1 > maxSegments) {
        // Prune by depth breaking ties by descending p-value.
        auto splits = core::make_range(segmentation, 2, segmentation.size());
        common::COrderings::simultaneousSort(
            depthAndPValue, splits,
            [](const TDoubleDoublePr& lhs, const TDoubleDoublePr& rhs) {
                return common::COrderings::lexicographical_compare(
                    lhs.first, -lhs.second, rhs.first, -rhs.second);
            });
        LOG_TRACE(<< "depth and p-values = "
                  << core::CContainerPrinter::print(depthAndPValue));
        LOG_TRACE(<< "splits = " << core::CContainerPrinter::print(splits));
        segmentation.resize(maxSegments + 1);
    }
    std::sort(segmentation.begin(), segmentation.end());
}

template<typename ITR>
void CTimeSeriesSegmentation::fitTopDownPiecewiseLinear(ITR begin,
                                                        ITR end,
                                                        std::size_t depth,
                                                        std::size_t offset,
                                                        std::size_t maxDepth,
                                                        double pValueToSegment,
                                                        double outlierFraction,
                                                        TSizeVec& segmentation,
                                                        TDoubleDoublePrVec& depthAndPValue,
                                                        TFloatMeanAccumulatorVec& reweighted) {

    // We want to find the partition of [begin, end) into linear models which
    // efficiently predicts the values. To this end we minimize residual variance
    // whilst maintaining a parsimonious model. We achieve the later objective by
    // doing model selection for each candidate segment using a significance test
    // of the explained variance.

    auto range = std::distance(begin, end);
    std::ptrdiff_t initialStep{4};
    if (range < 2 * initialStep || depth > maxDepth) {
        return;
    }

    LOG_TRACE(<< "Considering splitting [" << offset << "," << offset + range << ")");

    double startTime{static_cast<double>(offset)};
    TMeanVarAccumulator moments{[&] {
        reweighted.assign(begin, end);
        TRegression model{fitLinearModel(reweighted.cbegin(), reweighted.cend(), startTime)};
        auto predict = [&](std::size_t i) {
            double time{startTime + static_cast<double>(i)};
            return model.predict(time);
        };
        CSignal::reweightOutliers(predict, outlierFraction, reweighted);
        return centredResidualMoments(reweighted.cbegin(), reweighted.cend(), startTime);
    }()};

    if (common::CBasicStatistics::variance(moments) > 0.0) {
        using TDoubleItrPr = std::pair<double, ITR>;
        using TMinAccumulator = typename common::CBasicStatistics::SMin<TDoubleItrPr>::TAccumulator;

        auto variance = [](const TMeanVarAccumulator& l, const TMeanVarAccumulator& r) {
            double nl{common::CBasicStatistics::count(l)};
            double nr{common::CBasicStatistics::count(r)};
            return (nl * common::CBasicStatistics::maximumLikelihoodVariance(l) +
                    nr * common::CBasicStatistics::maximumLikelihoodVariance(r)) /
                   (nl + nr);
        };

        // We iterate through every possible binary split of the data into
        // contiguous ranges looking for the split which minimizes the total
        // variance of the values minus linear model predictions. Outliers
        // are removed based on an initial best split.

        TMinAccumulator minResidualVariance;
        reweighted.assign(begin, end);

        for (auto reweights : {0, 1}) {
            LOG_TRACE(<< "  reweights = " << reweights);

            TRegression leftModel;
            TRegression rightModel;
            TRegressionArray leftParameters;
            TRegressionArray rightParameters;

            auto minimumResidualVarianceSplit = [&](ITR begin_, ITR end_, std::ptrdiff_t j) {
                TMinAccumulator result;
                double splitTime{startTime +
                                 static_cast<double>(begin_ - reweighted.cbegin())};
                leftModel = fitLinearModel(reweighted.cbegin(), begin_, startTime);
                rightModel = fitLinearModel(begin_, reweighted.cend(), splitTime);
                for (ITR i = begin_; j < end_ - i;
                     i = i + j, splitTime += static_cast<double>(j)) {
                    TRegression deltaModel{fitLinearModel(i, i + j, splitTime)};
                    leftModel += deltaModel;
                    rightModel -= deltaModel;
                    double residualVariance{variance(
                        residualMoments(reweighted.cbegin(), i + j, startTime, leftModel),
                        residualMoments(i + j, reweighted.cend(),
                                        splitTime + static_cast<double>(j), rightModel))};
                    result.add({residualVariance, i + j});
                }
                return result;
            };

            minResidualVariance = minimumResidualVarianceSplit(
                reweighted.cbegin(), reweighted.cend(), initialStep);
            ITR split{minResidualVariance[0].second};
            minResidualVariance = minimumResidualVarianceSplit(
                split - std::min(split - reweighted.cbegin(), initialStep),
                split + std::min(reweighted.cend() - split, initialStep), 1);

            if (outlierFraction < 0.0 || outlierFraction >= 1.0) {
                LOG_ERROR(<< "Ignoring bad value for outlier fraction " << outlierFraction
                          << ". This must be in the range [0,1).");
                break;
            }
            if (reweights == 1 || outlierFraction == 0.0) {
                break;
            }
            split = minResidualVariance[0].second;
            double splitTime{static_cast<double>(std::distance(reweighted.cbegin(), split))};
            leftModel = fitLinearModel(reweighted.cbegin(), split, startTime);
            rightModel = fitLinearModel(split, reweighted.cend(), splitTime);
            leftModel.parameters(leftParameters);
            rightModel.parameters(rightParameters);
            auto predict = [&](std::size_t j) {
                double time{startTime + static_cast<double>(j)};
                return time < splitTime ? TRegression::predict(leftParameters, time)
                                        : TRegression::predict(rightParameters, time);
            };
            reweighted.assign(begin, end);
            CSignal::reweightOutliers(predict, outlierFraction, reweighted);
        }

        ITR split{minResidualVariance[0].second};
        std::size_t splitOffset(std::distance(reweighted.cbegin(), split));
        double splitTime{static_cast<double>(splitOffset)};
        LOG_TRACE(<< "  candidate = " << offset + splitOffset
                  << ", variance = " << minResidualVariance[0].first);

        TMeanVarAccumulator leftMoments{
            centredResidualMoments(reweighted.cbegin(), split, startTime)};
        TMeanVarAccumulator rightMoments{
            centredResidualMoments(split, reweighted.cend(), splitTime)};
        TMeanAccumulator meanAbs{std::accumulate(
            begin, end, TMeanAccumulator{}, [](auto partialMean, const auto& value) {
                partialMean.add(std::fabs(common::CBasicStatistics::mean(value)),
                                common::CBasicStatistics::count(value));
                return partialMean;
            })};
        LOG_TRACE(<< "  total moments = " << moments << ", left moments = " << leftMoments
                  << ", right moments = " << rightMoments << ", mean abs = " << meanAbs);

        double v[]{common::CBasicStatistics::maximumLikelihoodVariance(moments),
                   common::CBasicStatistics::maximumLikelihoodVariance(leftMoments + rightMoments) +
                       common::MINIMUM_COEFFICIENT_OF_VARIATION *
                           common::CBasicStatistics::mean(meanAbs)};
        double df[]{4.0 - 2.0, static_cast<double>(range - 4)};
        double pValue{common::CTools::oneMinusPowOneMinusX(
            common::CStatisticalTests::rightTailFTest(std::max(v[0] - v[1], 0.0),
                                                      v[1], df[0], df[1]),
            static_cast<double>(reweighted.size() / initialStep))};
        LOG_TRACE(<< "  p-value = " << pValue);

        if (pValue < pValueToSegment) {
            fitTopDownPiecewiseLinear(begin, begin + splitOffset, depth + 1, offset,
                                      maxDepth, pValueToSegment, outlierFraction,
                                      segmentation, depthAndPValue, reweighted);
            fitTopDownPiecewiseLinear(begin + splitOffset, end, depth + 1, offset + splitOffset,
                                      maxDepth, pValueToSegment, outlierFraction,
                                      segmentation, depthAndPValue, reweighted);
            segmentation.push_back(offset + splitOffset);
            depthAndPValue.emplace_back(static_cast<double>(depth), pValue);
        }
    }
}

CTimeSeriesSegmentation::TPredictor
CTimeSeriesSegmentation::fitPiecewiseLinear(const TSizeVec& segmentation,
                                            double outlierFraction,
                                            TFloatMeanAccumulatorVec& values) {
    using TRegressionVec = std::vector<TRegression>;
    using TRegressionArrayVec = std::vector<TRegressionArray>;

    TDoubleVec segmentEndTimes(segmentation.size() - 1,
                               static_cast<double>(values.size()));
    TRegressionVec models(segmentation.size() - 1);
    TRegressionArrayVec parameters(segmentation.size() - 1);

    for (std::size_t i = 1; i < segmentation.size(); ++i) {
        std::size_t a{segmentation[i - 1]};
        std::size_t b{segmentation[i]};
        segmentEndTimes[i - 1] = static_cast<double>(b);
        models[i - 1] = fitLinearModel(values.begin() + a, values.begin() + b,
                                       static_cast<double>(a));
        models[i - 1].parameters(parameters[i - 1]);
    }
    LOG_TRACE(<< "segmentation = " << core::CContainerPrinter::print(segmentation));

    auto predict = [&](std::size_t i) {
        double time{static_cast<double>(i)};
        auto index = std::upper_bound(segmentEndTimes.begin(), segmentEndTimes.end(), time) -
                     segmentEndTimes.begin();
        return TRegression::predict(parameters[index], time);
    };

    if (CSignal::reweightOutliers(predict, outlierFraction, values)) {
        for (std::size_t i = 1; i < segmentation.size(); ++i) {
            double a{static_cast<double>(segmentation[i - 1])};
            models[i - 1] = fitLinearModel(values.begin() + segmentation[i - 1],
                                           values.begin() + segmentation[i], a);
            models[i - 1].parameters(parameters[i - 1]);
            LOG_TRACE(<< "model = " << models[i - 1].print());
        }
    }

    return [=](double time) {
        auto index = std::upper_bound(segmentEndTimes.begin(), segmentEndTimes.end(), time) -
                     segmentEndTimes.begin();
        return TRegression::predict(parameters[index], time);
    };
}

template<typename ITR>
void CTimeSeriesSegmentation::fitTopDownPiecewiseLinearScaledSeasonal(
    ITR begin,
    ITR end,
    std::size_t depth,
    std::size_t offset,
    const TSeasonality& model,
    std::size_t maxDepth,
    double pValueToSegment,
    TSizeVec& segmentation,
    TDoubleDoublePrVec& depthAndPValue) {

    // We want to find the partition of [begin, end) into scaled seasonal
    // models which efficiently predicts the values. To this end we minimize
    // residual variance whilst maintaining a parsimonious model. We achieve
    // the later objective by doing model selection for each candidate segment
    // using a significance test of the explained variance.

    using TDoubleItrPr = std::pair<double, ITR>;
    using TMinAccumulator = typename common::CBasicStatistics::SMin<TDoubleItrPr>::TAccumulator;
    using TMeanAccumulatorAry = std::array<TMeanAccumulator, 3>;

    std::size_t range{static_cast<std::size_t>(std::distance(begin, end))};
    if (range < 4 || depth > maxDepth) {
        return;
    }

    LOG_TRACE(<< "Considering splitting [" << offset << "," << offset + range << ")");

    TMeanVarAccumulator moments{centredResidualMoments(begin, end, offset, model)};

    if (common::CBasicStatistics::variance(moments) > 0.0) {
        TMinAccumulator minResidualVariance;

        auto statistics = [offset, begin, &model](ITR begin_, ITR end_) {
            TMeanAccumulatorAry result;
            for (ITR i = begin_; i != end_; ++i) {
                double x{common::CBasicStatistics::mean(*i)};
                double w{common::CBasicStatistics::count(*i)};
                double p{model(offset + std::distance(begin, i))};
                result[0].add(x * x, w);
                result[1].add(x * p, w);
                result[2].add(p * p, w);
            }
            return result;
        };
        auto variance = [](const TMeanAccumulatorAry& l, const TMeanAccumulatorAry& r) {
            double nl{common::CBasicStatistics::count(l[0])};
            double nr{common::CBasicStatistics::count(r[0])};
            double vl{common::CBasicStatistics::mean(l[0]) -
                      (common::CBasicStatistics::mean(l[1]) < 0.0 ||
                               common::CBasicStatistics::mean(l[2]) == 0.0
                           ? 0.0
                           : common::CTools::pow2(common::CBasicStatistics::mean(l[1])) /
                                 common::CBasicStatistics::mean(l[2]))};
            double vr{common::CBasicStatistics::mean(r[0]) -
                      (common::CBasicStatistics::mean(r[1]) < 0.0 ||
                               common::CBasicStatistics::mean(r[2]) == 0.0
                           ? 0.0
                           : common::CTools::pow2(common::CBasicStatistics::mean(r[1])) /
                                 common::CBasicStatistics::mean(r[2]))};
            return (nl * vl + nr * vr) / (nl + nr);
        };

        ITR i{begin + 2};

        TMeanAccumulatorAry leftStatistics{statistics(begin, i)};
        TMeanAccumulatorAry rightStatistics{statistics(i, end)};
        minResidualVariance.add({variance(leftStatistics, rightStatistics), i});

        // The following loop is the interesting part. We compute (up to constants)
        //
        //   min_i{ min_{sl, sr}{ sum_{j<i}{(x_j - sl p_j)^2} + sum_{j>=i}{(x_j - sr p_j)^2} } }
        //
        // We use the fact that we can maintain sufficient statistics to which
        // we can both add and remove values to compute this in O(|S||T|) for
        // S candidate splits and T sufficient statistics.

        for (ITR end_ = end - 2; i != end_; ++i) {
            TMeanAccumulatorAry deltaStatistics{statistics(i, i + 1)};
            for (std::size_t k = 0; k < 3; ++k) {
                leftStatistics[k] += deltaStatistics[k];
                rightStatistics[k] -= deltaStatistics[k];
            }
            minResidualVariance.add({variance(leftStatistics, rightStatistics), i + 1});
        }

        ITR split{minResidualVariance[0].second};
        std::size_t splitOffset{offset + std::distance(begin, split)};
        LOG_TRACE(<< "  candidate = " << splitOffset
                  << ", variance = " << minResidualVariance[0].first);

        TMeanVarAccumulator leftMoments{centredResidualMoments(begin, split, offset, model)};
        TMeanVarAccumulator rightMoments{
            centredResidualMoments(split, end, splitOffset, model)};
        TMeanAccumulator meanAbs{std::accumulate(
            begin, end, TMeanAccumulator{}, [](auto partialMean, const auto& value) {
                partialMean.add(std::fabs(common::CBasicStatistics::mean(value)),
                                common::CBasicStatistics::count(value));
                return partialMean;
            })};
        LOG_TRACE(<< "  total moments = " << moments << ", left moments = " << leftMoments
                  << ", right moments = " << rightMoments << ", mean abs = " << meanAbs);

        double v[]{common::CBasicStatistics::maximumLikelihoodVariance(moments),
                   common::CBasicStatistics::maximumLikelihoodVariance(leftMoments + rightMoments) +
                       common::MINIMUM_COEFFICIENT_OF_VARIATION *
                           common::CBasicStatistics::mean(meanAbs)};
        double df[]{2.0 - 1.0, static_cast<double>(range - 2)};
        double pValue{common::CTools::oneMinusPowOneMinusX(
            common::CStatisticalTests::rightTailFTest(std::max(v[0] - v[1], 0.0),
                                                      v[1], df[0], df[1]),
            static_cast<double>(range - 4))};
        LOG_TRACE(<< "  p-value = " << pValue);

        if (pValue < pValueToSegment) {
            fitTopDownPiecewiseLinearScaledSeasonal(begin, split, depth + 1, offset,
                                                    model, maxDepth, pValueToSegment,
                                                    segmentation, depthAndPValue);
            fitTopDownPiecewiseLinearScaledSeasonal(split, end, depth + 1, splitOffset,
                                                    model, maxDepth, pValueToSegment,
                                                    segmentation, depthAndPValue);
            segmentation.push_back(splitOffset);
            depthAndPValue.emplace_back(static_cast<double>(depth), pValue);
        }
    }
}

void CTimeSeriesSegmentation::fitPiecewiseLinearScaledSeasonal(
    const TFloatMeanAccumulatorVec& values,
    const TSeasonality& model,
    const TSizeVec& segmentation,
    double outlierFraction,
    TFloatMeanAccumulatorVec& reweighted,
    TDoubleVec& scales) {

    scales.assign(segmentation.size() - 1, 1.0);

    auto scale = [&](std::size_t i) { return scaleAt(i, segmentation, scales); };

    auto scaledModel = [&](std::size_t i) { return scale(i) * model(i); };

    auto computeScales = [&] {
        for (std::size_t j = 1; j < segmentation.size(); ++j) {
            TMeanAccumulator projection;
            TMeanAccumulator Z;
            for (std::size_t i = segmentation[j - 1]; i < segmentation[j]; ++i) {
                double x{common::CBasicStatistics::mean(reweighted[i])};
                double w{common::CBasicStatistics::count(reweighted[i])};
                double p{model(i)};
                if (w > 0.0) {
                    projection.add(x * p, w);
                    Z.add(p * p, w);
                }
            }
            scales[j - 1] = common::CBasicStatistics::mean(Z) == 0.0
                                ? 1.0
                                : std::max(common::CBasicStatistics::mean(projection) /
                                               common::CBasicStatistics::mean(Z),
                                           0.0);
        }
    };

    LOG_TRACE(<< "segmentation = " << core::CContainerPrinter::print(segmentation));

    // First pass to re-weight any large outliers.
    reweighted = values;
    computeScales();
    CSignal::reweightOutliers(scaledModel, outlierFraction, reweighted);

    // Fit the model adjusting for scales.
    computeScales();
    LOG_TRACE(<< "scales = " << core::CContainerPrinter::print(scales));

    // Re-weight outliers based on the new model.
    reweighted = values;
    if (CSignal::reweightOutliers(scaledModel, outlierFraction, reweighted)) {
        // If any re-weighting happened fine tune.
        computeScales();
        LOG_TRACE(<< "scales = " << core::CContainerPrinter::print(scales));
    }
}

void CTimeSeriesSegmentation::fitPiecewiseLinearScaledSeasonal(
    const TFloatMeanAccumulatorVec& values,
    const TSeasonalComponentVec& periods,
    const TSizeVec& segmentation,
    double outlierFraction,
    TFloatMeanAccumulatorVec& reweighted,
    TMeanAccumulatorVecVec& models,
    TDoubleVec& scales) {

    models.assign(periods.size(), {});
    scales.assign(segmentation.size() - 1, 1.0);

    auto model = [&](std::size_t i) {
        double result{0.0};
        for (std::size_t j = 0; j < periods.size(); ++j) {
            result += periods[j].value(models[j], i);
        }
        return result;
    };

    auto scale = [&](std::size_t i) { return scaleAt(i, segmentation, scales); };

    auto scaledModel = [&](std::size_t i) { return scale(i) * model(i); };

    auto computeScale = [&](std::size_t begin, std::size_t end) {
        TMeanAccumulator projection;
        TMeanAccumulator Z;
        for (std::size_t i = begin; i < end; ++i) {
            double x{common::CBasicStatistics::mean(reweighted[i])};
            double w{common::CBasicStatistics::count(reweighted[i])};
            double p{model(i)};
            if (w > 0.0) {
                projection.add(x * p, w);
                Z.add(p * p, w);
            }
        }
        return common::CBasicStatistics::mean(Z) == 0.0
                   ? 1.0
                   : std::max(common::CBasicStatistics::mean(projection) /
                                  common::CBasicStatistics::mean(Z),
                              0.0);
    };

    auto fitSeasonalModels = [&](const TScale& scale_) {
        for (std::size_t i = 0; i < periods.size(); ++i) {
            models[i].assign(periods[i].period(), TMeanAccumulator{});
        }
        std::size_t iterations(periods.size() > 1 ? 2 : 1);
        for (std::size_t i = 0; i < iterations; ++i) {
            for (std::size_t j = 0; j < periods.size(); ++j) {
                models[j] = fitSeasonalModel(reweighted.begin(), reweighted.end(),
                                             periods[j], model, scale_);
            }
        }
    };

    auto fitScaledSeasonalModels = [&](const TScale& scale_) {
        for (std::size_t i = 0; i < 2; ++i) {
            fitSeasonalModels(scale_);
            for (std::size_t j = 1; j < segmentation.size(); ++j) {
                scales[j - 1] = computeScale(segmentation[j - 1], segmentation[j]);
            }
        }
    };

    LOG_TRACE(<< "segmentation = " << core::CContainerPrinter::print(segmentation));

    // First pass to re-weight any large outliers.
    reweighted = values;
    fitSeasonalModels([](std::size_t) { return 1.0; });
    CSignal::reweightOutliers(scaledModel, outlierFraction, reweighted);

    // Fit the model adjusting for scales.
    fitScaledSeasonalModels(scale);
    LOG_TRACE(<< "models = " << core::CContainerPrinter::print(models));
    LOG_TRACE(<< "scales = " << core::CContainerPrinter::print(scales));

    // Re-weight outliers based on the new model.
    reweighted = values;
    if (CSignal::reweightOutliers(scaledModel, outlierFraction, reweighted)) {
        // If any re-weighting happened fine tune.
        fitScaledSeasonalModels(scale);
        LOG_TRACE(<< "model = " << core::CContainerPrinter::print(models));
        LOG_TRACE(<< "scales = " << core::CContainerPrinter::print(scales));
    }
}

template<typename ITR>
void CTimeSeriesSegmentation::fitTopDownPiecewiseTimeShifted(ITR begin,
                                                             ITR end,
                                                             std::size_t depth,
                                                             std::size_t offset,
                                                             const TDoubleVecVec& predictions,
                                                             std::size_t maxDepth,
                                                             double pValueToSegment,
                                                             TSizeVec& segmentation,
                                                             TDoubleDoublePrVec& depthAndPValue) {

    using TMeanVarAccumulatorVec = std::vector<TMeanVarAccumulator>;

    // We want to find the partition of [begin, end) into shifted seasonal
    // models which efficiently predicts the values. To this end we maximise
    // residual variance whilst maintaining a parsimonious model. We achieve
    // the later objective by doing model selection for each candidate segment
    // using a significance test of the explained variance.

    std::size_t range{static_cast<std::size_t>(std::distance(begin, end))};
    if (range < 10 || depth > maxDepth) {
        return;
    }

    LOG_TRACE(<< "Considering splitting [" << offset << "," << offset + range << ")");

    TMeanVarAccumulator totalMoments;
    std::tie(totalMoments, std::ignore) = residualMoments(begin, end, offset, predictions);
    double totalVariance{common::CBasicStatistics::maximumLikelihoodVariance(totalMoments)};
    LOG_TRACE(<< "total moments = " << totalMoments);

    if (totalVariance > 0.0) {
        double bestSplitVariance{std::numeric_limits<double>::max()};
        std::size_t bestSplit{range};

        TMeanVarAccumulatorVec leftMoments(predictions.size());
        for (std::size_t j = 0; j < 4; ++j) {
            const auto& value = *(begin + j);
            for (std::size_t i = 0; i < predictions.size(); ++i) {
                double prediction{predictions[i][offset + j]};
                leftMoments[i].add(common::CBasicStatistics::mean(value) - prediction,
                                   common::CBasicStatistics::count(value));
            }
        }
        TMeanVarAccumulatorVec rightMoments(predictions.size());
        for (std::size_t j = 4; j < range; ++j) {
            const auto& value = *(begin + j);
            for (std::size_t i = 0; i < predictions.size(); ++i) {
                double prediction{predictions[i][offset + j]};
                rightMoments[i].add(common::CBasicStatistics::mean(value) - prediction,
                                    common::CBasicStatistics::count(value));
            }
        }

        // The following loop is the interesting part. We compute
        //
        //   min_i{ min_{sl, sr}{ sum_{j<i}{(x_j - p_{j+sl})^2} + sum_{j>=i}{(x_j - p_{j+sr})^2} } }
        //
        // We use the fact that we can remove values from the central moments
        // accumulator to compute this in O(|S||T|^2) for S candidate splits
        // and T candidate shifts.

        for (std::size_t i = 5; i + 5 < range; ++i) {
            const auto& valueLeftOfSplit = *(begin + i - 1);
            for (std::size_t j = 0; j < predictions.size(); ++j) {
                double prediction{predictions[j][offset + i - 1]};
                TMeanVarAccumulator delta;
                delta.add(common::CBasicStatistics::mean(valueLeftOfSplit) - prediction,
                          common::CBasicStatistics::count(valueLeftOfSplit));
                leftMoments[j].add(common::CBasicStatistics::mean(valueLeftOfSplit) - prediction,
                                   common::CBasicStatistics::count(valueLeftOfSplit));
                rightMoments[j] -= delta;
            }

            double variance{std::numeric_limits<double>::max()};
            for (std::size_t sl = 0; sl < leftMoments.size(); ++sl) {
                for (std::size_t sr = 0; sr < rightMoments.size(); ++sr) {
                    double shiftsVariance{common::CBasicStatistics::maximumLikelihoodVariance(
                        leftMoments[sl] + rightMoments[sr])};
                    if (shiftsVariance < variance) {
                        variance = shiftsVariance;
                    }
                }
            }
            LOG_TRACE(<< "  split = " << i << ", variance = " << variance);

            if (variance < bestSplitVariance) {
                std::tie(bestSplitVariance, bestSplit) = std::make_pair(variance, i);
            }
        }

        auto split = begin + bestSplit;
        std::size_t splitOffset{offset + bestSplit};

        TMeanAccumulator meanAbs{std::accumulate(
            begin, end, TMeanAccumulator{}, [](auto partialMean, const auto& value) {
                partialMean.add(std::fabs(common::CBasicStatistics::mean(value)),
                                common::CBasicStatistics::count(value));
                return partialMean;
            })};
        LOG_TRACE(<< "  total variance = " << totalVariance << ", best split = " << bestSplit
                  << ", best split variance = " << bestSplitVariance
                  << ", mean abs = " << meanAbs);

        double v[]{totalVariance,
                   bestSplitVariance + common::MINIMUM_COEFFICIENT_OF_VARIATION *
                                           common::CBasicStatistics::mean(meanAbs)};
        double df[]{3.0 - 2.0, static_cast<double>(range - 3)};
        double pValue{common::CTools::oneMinusPowOneMinusX(
            common::CStatisticalTests::rightTailFTest(std::max(v[0] - v[1], 0.0),
                                                      v[1], df[0], df[1]),
            static_cast<double>(range - 10))};
        LOG_TRACE(<< "  p-value = " << pValue);

        if (pValue < pValueToSegment) {
            fitTopDownPiecewiseTimeShifted(begin, split, depth + 1, offset,
                                           predictions, maxDepth, pValueToSegment,
                                           segmentation, depthAndPValue);
            fitTopDownPiecewiseTimeShifted(split, end, depth + 1, splitOffset,
                                           predictions, maxDepth, pValueToSegment,
                                           segmentation, depthAndPValue);
            segmentation.push_back(splitOffset);
            depthAndPValue.emplace_back(static_cast<double>(depth), pValue);
        }
    }
}

template<typename ITR>
CTimeSeriesSegmentation::TMeanVarAccumulator
CTimeSeriesSegmentation::centredResidualMoments(ITR begin, ITR end, double startTime) {
    if (begin == end) {
        return common::CBasicStatistics::momentsAccumulator(0.0, 0.0, 0.0);
    }
    if (std::distance(begin, end) == 1) {
        return common::CBasicStatistics::momentsAccumulator(
            common::CBasicStatistics::count(*begin), 0.0, 0.0);
    }
    TRegression model{fitLinearModel(begin, end, startTime)};
    TMeanVarAccumulator moments{residualMoments(begin, end, startTime, model)};
    common::CBasicStatistics::moment<0>(moments) = 0.0;
    return moments;
}

template<typename ITR>
CTimeSeriesSegmentation::TMeanVarAccumulator
CTimeSeriesSegmentation::centredResidualMoments(ITR begin,
                                                ITR end,
                                                std::size_t offset,
                                                const TSeasonality& model) {
    if (begin == end) {
        return common::CBasicStatistics::momentsAccumulator(0.0, 0.0, 0.0);
    }
    if (std::distance(begin, end) == 1) {
        return common::CBasicStatistics::momentsAccumulator(
            common::CBasicStatistics::count(*begin), 0.0, 0.0);
    }

    TMeanAccumulator projection;
    TMeanAccumulator Z;
    for (ITR i = begin; i != end; ++i) {
        double x{common::CBasicStatistics::mean(*i)};
        double w{common::CBasicStatistics::count(*i)};
        double p{model(offset + std::distance(begin, i))};
        if (w > 0.0) {
            projection.add(x * p, w);
            Z.add(p * p, w);
        }
    }
    double scale{common::CBasicStatistics::mean(Z) == 0.0
                     ? 1.0
                     : std::max(common::CBasicStatistics::mean(projection) /
                                    common::CBasicStatistics::mean(Z),
                                0.0)};
    LOG_TRACE(<< "  scale = " << scale);

    TMeanVarAccumulator moments;
    for (ITR i = begin; i != end; ++i) {
        double x{common::CBasicStatistics::mean(*i)};
        double w{common::CBasicStatistics::count(*i)};
        double p{scale * model(offset + std::distance(begin, i))};
        if (w > 0.0) {
            moments.add(x - p, w);
        }
    }
    common::CBasicStatistics::moment<0>(moments) = 0.0;
    LOG_TRACE(<< "  moments = " << moments);

    return moments;
}

template<typename ITR>
CTimeSeriesSegmentation::TMeanVarAccumulator
CTimeSeriesSegmentation::residualMoments(ITR begin, ITR end, double startTime, const TRegression& model) {
    TMeanVarAccumulator moments;
    TRegressionArray params;
    model.parameters(params);
    for (double time = startTime; begin != end; ++begin, time += 1.0) {
        double w{common::CBasicStatistics::count(*begin)};
        if (w > 0.0) {
            double x{common::CBasicStatistics::mean(*begin) -
                     TRegression::predict(params, time)};
            moments.add(x, w);
        }
    }
    return moments;
}

template<typename ITR>
CTimeSeriesSegmentation::TMeanVarAccumulatorSizePr
CTimeSeriesSegmentation::residualMoments(ITR begin,
                                         ITR end,
                                         std::size_t offset,
                                         const TDoubleVecVec& predictions) {

    std::size_t range{static_cast<std::size_t>(std::distance(begin, end))};

    TMeanVarAccumulator bestMoments;
    std::size_t bestShift{predictions.size()};
    double bestVariance{std::numeric_limits<double>::max()};

    for (std::size_t i = 0; i < predictions.size(); ++i) {
        TMeanVarAccumulator shiftMoments;
        for (std::size_t j = 0; j < range; ++j) {
            const auto& value = *(begin + j);
            shiftMoments.add(common::CBasicStatistics::mean(value) -
                                 predictions[i][offset + j],
                             common::CBasicStatistics::count(value));
        }
        if (common::CBasicStatistics::maximumLikelihoodVariance(shiftMoments) < bestVariance) {
            bestMoments = shiftMoments;
            bestShift = i;
            bestVariance = common::CBasicStatistics::maximumLikelihoodVariance(shiftMoments);
        }
    }

    return std::make_pair(bestMoments, bestShift);
}

template<typename ITR>
CTimeSeriesSegmentation::TRegression
CTimeSeriesSegmentation::fitLinearModel(ITR begin, ITR end, double startTime) {
    TRegression result;
    for (double time = startTime; begin != end; ++begin) {
        double x{common::CBasicStatistics::mean(*begin)};
        double w{common::CBasicStatistics::count(*begin)};
        if (w > 0.0) {
            result.add(time, x, w);
        }
        time += 1.0;
    }
    return result;
}

template<typename ITR>
CTimeSeriesSegmentation::TMeanAccumulatorVec
CTimeSeriesSegmentation::fitSeasonalModel(ITR begin,
                                          ITR end,
                                          const TSeasonalComponent& period,
                                          const TPredictor& predictor,
                                          const TScale& scale) {
    TMeanAccumulatorVec model(period.period());
    for (std::size_t i = 0; begin != end; ++i, ++begin) {
        if (period.contains(i)) {
            double w{common::CBasicStatistics::count(*begin) * scale(i)};
            if (w > 0.0) {
                double x{common::CBasicStatistics::mean(*begin) / scale(i)};
                model[period.offset(i)].add(x - predictor(static_cast<double>(i)), w);
            }
        }
    }
    return model;
}
}
}
}
