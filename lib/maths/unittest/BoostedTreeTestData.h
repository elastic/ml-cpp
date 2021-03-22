/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_BoostedTreeTestData_h
#define INCLUDED_BoostedTreeTestData_h

#include <core/CDataFrame.h>

#include <test/CRandomNumbers.h>

#include <cstddef>
#include <functional>
#include <memory>

using TTargetFunc = std::function<double(const ml::core::CDataFrame::TRowRef&)>;

std::unique_ptr<ml::core::CDataFrame> setupRegressionProblem(ml::test::CRandomNumbers& rng,
                                                             const TTargetFunc& target,
                                                             double noiseVariance,
                                                             std::size_t rows,
                                                             std::size_t cols);
TTargetFunc linearRegression(ml::test::CRandomNumbers& rng, std::size_t cols);
std::unique_ptr<ml::core::CDataFrame>
setupLinearRegressionProblem(std::size_t rows, std::size_t cols);

#endif // INCLUDED_BoostedTreeTestData_h
