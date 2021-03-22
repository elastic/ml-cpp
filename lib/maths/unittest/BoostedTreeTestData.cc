/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "BoostedTreeTestData.h"

#include <test/CRandomNumbers.h>

#include <vector>

using namespace ml;
using TBoolVec = std::vector<bool>;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TRowItr = core::CDataFrame::TRowItr;
using TRowRef = core::CDataFrame::TRowRef;

std::unique_ptr<core::CDataFrame> setupRegressionProblem(test::CRandomNumbers& rng,
                                                         const TTargetFunc& target,
                                                         double noiseVariance,
                                                         std::size_t rows,
                                                         std::size_t cols) {
    TDoubleVecVec x(cols - 1);
    for (std::size_t i = 0; i < cols - 1; ++i) {
        rng.generateUniformSamples(0.0, 10.0, rows, x[i]);
    }

    TDoubleVec noise;
    rng.generateNormalSamples(0.0, noiseVariance, rows, noise);

    auto frame = core::makeMainStorageDataFrame(cols, rows).first;

    frame->categoricalColumns(TBoolVec(cols, false));
    for (std::size_t i = 0; i < rows; ++i) {
        frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t&) {
            for (std::size_t j = 0; j < cols - 1; ++j, ++column) {
                *column = x[j][i];
            }
        });
    }
    frame->finishWritingRows();
    frame->writeColumns(1, [&](TRowItr beginRows, TRowItr endRows) {
        for (auto row = beginRows; row != endRows; ++row) {
            double targetValue{target(*row) + noise[row->index()]};
            row->writeColumn(cols - 1, targetValue);
        }
    });

    return frame;
}

TTargetFunc linearRegression(test::CRandomNumbers& rng, std::size_t cols) {
    TDoubleVec m;
    TDoubleVec s;
    rng.generateUniformSamples(0.0, 10.0, cols - 1, m);
    rng.generateUniformSamples(-10.0, 10.0, cols - 1, s);
    return [=](const TRowRef& row) {
        double result{0.0};
        for (std::size_t i = 0; i < cols - 1; ++i) {
            result += m[i] + s[i] * row[i];
        }
        return result;
    };
}

std::unique_ptr<core::CDataFrame> setupLinearRegressionProblem(std::size_t rows,
                                                               std::size_t cols) {
    test::CRandomNumbers rng;
    double noiseVariance{100.0};
    auto target = linearRegression(rng, cols);
    return setupRegressionProblem(rng, target, noiseVariance, rows, cols);
}
