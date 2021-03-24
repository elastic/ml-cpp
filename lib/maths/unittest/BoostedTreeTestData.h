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
#include <vector>

using TTargetFunc = std::function<double(const ml::core::CDataFrame::TRowRef&)>;

//! Add regression data which comprises feature vectors \p x and the predictions
//! of \p target on \p x to \p frame.
//!
//! \param rng The random number generator.
//! \param target Defines the regression target as a function of \p cols - 1 features.
//! \param x The feature values indexed by feature/column then row.
//! \param noiseVariance The variance of the noise added to the output of \p target.
//! \param frame The data frame to which to add the examples \p x.
void addRegressionData(ml::test::CRandomNumbers& rng,
                       const TTargetFunc& target,
                       const std::vector<std::vector<double>>& x,
                       double noiseVariance,
                       ml::core::CDataFrame& frame);

//! Create and return a data frame containing the regression problem defined by
//! \p target.
//!
//! The regression target is written into the last column of \p frame.
//!
//! \param rng The random number generator.
//! \param target Defines the regression target as a function of \p cols - 1 features.
//! \param noiseVariance The variance of the noise added to the output of \p target.
//! \param rows The number of examples of the regression relationship to generate.
//! \param cols The number of features + one for the regression target.
std::unique_ptr<ml::core::CDataFrame> setupRegressionProblem(ml::test::CRandomNumbers& rng,
                                                             const TTargetFunc& target,
                                                             double noiseVariance,
                                                             std::size_t rows,
                                                             std::size_t cols);

//! Set up a random linear regression with \p cols - 1 features.
TTargetFunc linearRegression(ml::test::CRandomNumbers& rng, std::size_t cols);

//! Create and return a data frame containing a noisy linear regression problem.
//!
//! \param rows The number of examples of the regression relationship to generate.
//! \param cols The number of features + one for the regression target.
std::unique_ptr<ml::core::CDataFrame>
setupLinearRegressionProblem(std::size_t rows, std::size_t cols);

#endif // INCLUDED_BoostedTreeTestData_h
