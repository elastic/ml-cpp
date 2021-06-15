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

//! Add data which comprises feature vectors \p x and the output of \p target
//! on each \p x to \p frame.
//!
//! \param rng The random number generator.
//! \param target Defines the target variable as a function of \p cols - 1 features.
//! \param x The feature values indexed by feature/column then row.
//! \param noiseVariance The variance of the noise added to the output of \p target.
//! \param frame The data frame to which to add the examples \p x.
void addData(ml::test::CRandomNumbers& rng,
             const TTargetFunc& target,
             const std::vector<std::vector<double>>& x,
             double noiseVariance,
             ml::core::CDataFrame& frame);

//! Create and return a data frame containing the output of \p target on \p rows
//! of \p cols - 1 features. The target variable is written into the last column
//! of \p frame.
//!
//! \param rng The random number generator.
//! \param target Defines the target variable as a function of \p cols - 1 features.
//! \param noiseVariance The variance of the noise added to the output of \p target.
//! \param rows The number of examples of the regression relationship to generate.
//! \param cols The number of features + one for the regression target.
std::unique_ptr<ml::core::CDataFrame> setupRegressionProblem(ml::test::CRandomNumbers& rng,
                                                             const TTargetFunc& target,
                                                             double noiseVariance,
                                                             std::size_t rows,
                                                             std::size_t cols);

//! A random linear function of \p cols - 1 features.
TTargetFunc linearRegression(ml::test::CRandomNumbers& rng, std::size_t cols);

//! Create and return a data frame containing a noisy linear relationship between
//! \p cols - 1 features and a target variable. This is written into the last column
//! of \p frame.
//!
//! \param rows The number of examples of the regression relationship to generate.
//! \param cols The number of features + one for the regression target.
std::unique_ptr<ml::core::CDataFrame>
setupLinearRegressionProblem(std::size_t rows, std::size_t cols);

//! Create and return a data frame containing the output of \p target on \p rows
//! of \p cols - 1 features. This is written into the last column of \p frame.
//!
//! \param rng The random number generator.
//! \param target Defines the target variable as a function of \p cols - 1 features.
//! The range of this is expected to be categorical.
//! \param rows The number of examples of the regression relationship to generate.
//! \param cols The number of features + one for the regression target.
std::unique_ptr<ml::core::CDataFrame>
setupClassificationProblem(ml::test::CRandomNumbers& rng,
                           const TTargetFunc& target,
                           std::size_t rows,
                           std::size_t cols);

//! Create and return a data frame containing a noisy linear relationship between
//! \p cols - 1 features and the log-odds of a categorical target variable. This
//! is written into the last column of \p frame.
//!
//! \param rows The number of examples of the regression relationship to generate.
//! \param cols The number of features + one for the regression target.
std::unique_ptr<ml::core::CDataFrame>
setupLinearBinaryClassificationProblem(std::size_t rows, std::size_t cols);

#endif // INCLUDED_BoostedTreeTestData_h
