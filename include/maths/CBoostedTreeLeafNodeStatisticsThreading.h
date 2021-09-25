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

#ifndef INCLUDED_ml_maths_CBoostedTreeLeafNodeStatisticsThreading_h
#define INCLUDED_ml_maths_CBoostedTreeLeafNodeStatisticsThreading_h

#include <maths/CLinearAlgebraFwd.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <functional>

namespace ml {
namespace maths {

//! \brief Utilities for threading CBoostedTreeLeafNodeStatistics code.
//!
//! DESCRIPTION:\n
//! This encapsulates the logic to select the number of threads to use for
//! different operations computing leaf node statistics. Depending on the
//! amount of work to perform using fewer threads (which incur an overhead)
//! can improve throughput. Here, where possible, we estimate the work to
//! perform as a function of the thread overhead and minimize throughput.
//!
//! IMPLEMENTATION:\n
//! The constants in the cost model for each operation were extracted using
//! ordinary least squares regression for a couple of typical environments.
//! This provides a route to calibrate at runtime if we choose. The danger
//! of doing this is that timing statistics could be skewed by temporary
//! environmental factors. We can revisit this if we decide it is important.
class MATHS_EXPORT CBoostedTreeLeafNodeStatisticsThreading {
public:
    using TMinimumLoss =
        std::function<double(const Eigen::VectorXd&, const Eigen::MatrixXd&)>;

public:
    //! Get the number of threads to use to aggregate loss derivatives.
    static std::size_t numberThreadsForAggregateLossDerivatives(std::size_t numberThreads,
                                                                std::size_t features,
                                                                std::size_t rows);

    //! Get the number of threads to use to add split derivatives.
    static std::size_t
    numberThreadsForAddSplitsDerivatives(std::size_t numberThreads,
                                         std::size_t features,
                                         std::size_t numberLossParameters,
                                         std::size_t numberDerivatives);

    //! Get the number of threads to use to subtract split derivatives.
    static std::size_t
    numberThreadsForSubtractSplitsDerivatives(std::size_t numberThreads,
                                              std::size_t features,
                                              std::size_t numberLossParameters,
                                              std::size_t numberDerivatives);

    //! Get the number of threads to use to remap curvatures of split derivatives.
    static std::size_t
    numberThreadsForRemapSplitsDerivatives(std::size_t numberThreads,
                                           std::size_t features,
                                           std::size_t numberLossParameters,
                                           std::size_t numberDerivatives);

    //! Get the number of threads to use to compute the best split statistics.
    static std::size_t
    numberThreadsForComputeBestSplitStats(std::size_t numberThreads,
                                          std::size_t features,
                                          std::size_t numberLossParameters,
                                          std::size_t numberDerivatives);

    //! Compute the number of threads which optimises throughput given \p totalWork.
    static std::size_t maximumThroughputNumberThreads(double totalWork);

    //! Get the overhead of using \p numberThreads.
    static double threadCost(double numberThreads);

    //! Make the loss function to use for computing leaf node split gain.
    static TMinimumLoss makeThreadLocalMinimumLossFunction(int numberLossParameters,
                                                           double lambda);

private:
    static double addSplitsDerivativesTotalWork(std::size_t numberLossParameters,
                                                std::size_t numberDerivatives);
    static double subtractSplitsDerivativesTotalWork(std::size_t numberLossParameters,
                                                     std::size_t numberDerivatives);
    static double remapSplitsDerivativesTotalWork(std::size_t numberLossParameters,
                                                  std::size_t numberDerivatives);
    static double computeBestSplitStatsTotalWork(std::size_t numberLossParameters,
                                                 std::size_t numberDerivatives);
    template<int d>
    static TMinimumLoss makeThreadLocalMinimumLossFunction(double lambda);
};
}
}

#endif // INCLUDED_ml_maths_CBoostedTreeLeafNodeStatisticsThreading_h
