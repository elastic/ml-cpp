/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CPca.h>

#include <maths/CBasicStatistics.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraShims.h>
#include <maths/COrderings.h>
#include <maths/CPRNG.h>
#include <maths/CSampling.h>

#include <numeric>

namespace ml {
namespace maths {
using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TDenseMatrix = maths::CDenseMatrix<double>;
using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
using TDenseVectorMeanAccumulator = CBasicStatistics::SSampleMean<CPca::TDenseVector>::TAccumulator;

void CPca::projectOntoPrincipleComponents(std::size_t numberComponents_,
                                          const TSizeVec& support,
                                          TDenseVectorVec& data) {
    if (support.empty() || numberComponents_ >= las::dimension(data[0])) {
        return;
    }

    // Centre the data.
    TDenseVectorMeanAccumulator mean_(las::zero(data[0]));
    for (auto i : support) {
        mean_.add(data[i]);
    }
    const TDenseVector& mean{CBasicStatistics::mean(mean_)};
    for (auto&& vector : data) {
        vector -= mean;
    }

    // Setup the matrix.
    std::ptrdiff_t rows{static_cast<std::ptrdiff_t>(support.size())};
    std::ptrdiff_t cols{static_cast<std::ptrdiff_t>(las::dimension(data[0]))};
    TDenseMatrix matrix(rows, cols);
    for (std::size_t i = 0u; i < support.size(); ++i) {
        matrix.row(i) = data[support[i]];
    }

    // Project.
    std::ptrdiff_t numberComponents{static_cast<std::ptrdiff_t>(numberComponents_)};
    auto svd = matrix.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    LOG_TRACE(<< "singularValues = " << svd.singularValues().transpose());
    for (auto&& vector : data) {
        vector = svd.matrixV().leftCols(numberComponents).transpose() * (mean + vector);
    }
}

void CPca::projectOntoPrincipleComponentsRandom(std::size_t numberComponents,
                                                std::ptrdiff_t dimension,
                                                const TSizeVec& support,
                                                const TIntDoublePrVecVec& data,
                                                TDenseVectorVec& projected) {
    // Rather than trying to estimate the rank of the matrix and hence the
    // number of samples to ensure a given bound on the approximation error
    // in the SVD, since the error is monotonic decreasing with increasing
    // sample size we simply compute the number of samples to match the
    // memory used by the dense samples and the sparse representation.

    std::size_t size{std::accumulate(data.begin(), data.end(), std::size_t(0),
                                     [](std::size_t size_, const TIntDoublePrVec& vector) {
                                         return size_ + vector.size();
                                     })};
    projectOntoPrincipleComponentsRandom(numberComponents, 3 * size, dimension,
                                         support, data, projected);
}

void CPca::projectOntoPrincipleComponentsRandom(std::size_t numberComponents,
                                                std::size_t numberSamples,
                                                std::ptrdiff_t dimension,
                                                const TSizeVec& support,
                                                const TIntDoublePrVecVec& data,
                                                TDenseVectorVec& projected) {
    projected.clear();

    if (numberSamples == 0) {
        return;
    }

    LOG_TRACE(<< "numberSamples = " << numberSamples << " / " << support.size() * dimension);

    projected.reserve(data.size());

    if (numberSamples >= dimension * support.size()) {
        TSizeVec columns(dimension);
        std::iota(columns.begin(), columns.end(), 0);
        for (const auto& vector : data) {
            projected.push_back(toDense(columns, vector));
        }
        projectOntoPrincipleComponents(numberComponents, support, projected);
        return;
    }

    // TODO we can boost this result by running with different random
    // samples and choosing the projection with maximum product for
    // its singular values at the expense of increased runtime.

    // Compute the centroid of the data.
    TMeanAccumulatorVec mean_(dimension);
    for (auto i : support) {
        for (const auto& component : data[i]) {
            mean_[component.first].add(component.second);
        }
    }
    for (auto&& mi : mean_) {
        mi.add(0.0, static_cast<double>(support.size()) - CBasicStatistics::count(mi));
    }
    TDenseVector mean(static_cast<std::size_t>(dimension));
    for (std::size_t i = 0u; i < mean_.size(); ++i) {
        mean(i) = CBasicStatistics::mean(mean_[i]);
    }
    LOG_TRACE(<< "mean = " << mean.transpose());

    // Compute the # columns to sample and the probability with which
    // to sample them. Because we sample columns without replacement
    // to compute the column weights, so we get an unbiased estimate
    // of the SVD, we need to know the probability that each column is
    // sampled in the set as a whole. This is estimated by Monte Carlo:
    // the analytic formula involves summing O("dimension" ^ "# columns")
    // terms. (We could also investigate Sampford's method for sampling
    // without replacement, for which a column weight would be min(1,n*p)
    // with n the number of samples and p the column probability.)

    TSizeVec columns;
    TDoubleVec columnProbabilities(dimension);
    TDoubleVec columnWeights(dimension, 1.0);
    double Z{0.0};
    for (std::ptrdiff_t i = 0u; i < dimension; ++i) {
        columnProbabilities[i] = centredColumnNorm2(i, data, mean);
        Z += columnProbabilities[i];
    }
    std::for_each(columnProbabilities.begin(), columnProbabilities.end(),
                  [Z](double& probability) { probability /= Z; });
    CPRNG::CXorShift1024Mult rng;
    std::ptrdiff_t numberColumns{std::max(
        std::count_if(columnProbabilities.begin(), columnProbabilities.end(),
                      [dimension](double p) {
                          return p > 0.05 / static_cast<double>(dimension);
                      }),
        static_cast<std::ptrdiff_t>(numberComponents + 2))};
    LOG_TRACE(<< "# columns = " << numberColumns);
    LOG_TRACE(<< "probabilities = " << core::CContainerPrinter::print(columnProbabilities));

    if (numberColumns == dimension) {
        columns.resize(dimension);
        std::iota(columns.begin(), columns.end(), 0);
    } else {
        TDoubleVec probabilities;
        for (std::size_t trial = 0u; trial < 50; ++trial) {
            probabilities = columnProbabilities;
            CSampling::categoricalSampleWithoutReplacement(rng, probabilities,
                                                           numberColumns, columns);
            std::for_each(columns.begin(), columns.end(), [&columnWeights](std::size_t i) {
                columnWeights[i] += 1.0;
            });
        }
        std::sort(columns.begin(), columns.end());
        std::for_each(columnWeights.begin(), columnWeights.end(), [](double& weight) {
            weight = std::sqrt((weight - 1.0) / 50.0);
        });
    }
    LOG_TRACE(<< "columns = " << core::CContainerPrinter::print(columns));
    LOG_TRACE(<< "weights = " << core::CContainerPrinter::print(columnWeights));

    // Sample rows.

    TSizeVec rows;
    TDoubleVec rowProbabilities(support.size());
    std::size_t numberRows{std::max(numberSamples / numberColumns, std::size_t(1))};
    LOG_TRACE(<< "# rows = " << std::min(numberRows, data.size()));

    if (numberRows >= support.size()) {
        rows.resize(support.size());
        std::iota(rows.begin(), rows.end(), 0);
        std::fill_n(rowProbabilities.begin(), support.size(), 1.0);
    } else {
        Z = 0.0;
        double meanNorm2{0.0};
        for (auto i : columns) {
            meanNorm2 += mean(i) * mean(i);
        }
        for (auto i : support) {
            rowProbabilities[i] = centredRowNorm2(i, columns, data, meanNorm2, mean);
            Z += rowProbabilities[i];
        }
        std::for_each(rowProbabilities.begin(), rowProbabilities.end(),
                      [Z](double& probability) { probability /= Z; });
        TDoubleVec probabilities(rowProbabilities);
        CSampling::categoricalSampleWithReplacement(rng, probabilities, numberRows, rows);
    }
    LOG_TRACE(<< "rows = " << core::CContainerPrinter::print(rows));
    LOG_TRACE(<< "probabilities = " << core::CContainerPrinter::print(rowProbabilities));

    TDenseVector sampledWeights(columns.size());
    TDenseVector sampledMean(columns.size());
    for (std::size_t i = 0u; i < columns.size(); ++i) {
        sampledWeights(i) = columnWeights[columns[i]];
        sampledMean(i) = mean(columns[i]);
    }
    LOG_TRACE(<< "sampledWeights = " << sampledWeights.transpose());
    LOG_TRACE(<< "sampledMean = " << sampledMean.transpose());

    TDenseMatrix matrix(rows.size(), columns.size());
    for (std::size_t i = 0u; i < rows.size(); ++i) {
        const TIntDoublePrVec& vector{data[support[rows[i]]]};
        matrix.row(i) = (toDense(columns, vector) - sampledMean).cwiseQuotient(sampledWeights) /
                        std::sqrt(rowProbabilities[rows[i]]);
    }

    // Project.
    auto svd = matrix.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    LOG_TRACE(<< "singularValues = " << svd.singularValues().transpose());
    for (const auto& vector : data) {
        projected.push_back(svd.matrixV().leftCols(numberComponents).transpose() *
                            toDense(columns, vector));
    }
}

double CPca::numericRank(std::ptrdiff_t dimension, TIntDoublePrVecVec& data) {
    CPRNG::CXorShift1024Mult rng;

    TDoubleVec b;
    CSampling::uniformSample(rng, -1.0, 1.0, static_cast<std::size_t>(dimension), b);

    for (std::size_t i = 0u; i < 9; ++i) {
        double n{std::sqrt(norm2(b))};
        for (auto&& component : b) {
            component /= n;
        }
        dot(data, b);
    }

    return frobenius(data) / norm2(b);
}

double CPca::norm2(const TDoubleVec& x) {
    double result{0.0};
    for (const auto& component : x) {
        result += component * component;
    }
    return result;
}

double CPca::centredColumnNorm2(std::size_t i, const TIntDoublePrVecVec& x, const TDenseVector& mean) {
    double mi{mean(i)};
    double result{static_cast<double>(x.size()) * mi * mi};
    for (const auto& row : x) {
        double xij{element(i, row)};
        result += (xij - mi) * (xij - mi) - mi * mi;
    }
    return result;
}

double CPca::centredRowNorm2(std::size_t i,
                             const TSizeVec& columns,
                             const TIntDoublePrVecVec& x,
                             double meanNorm2,
                             const TDenseVector& mean) {
    double result{meanNorm2};
    for (auto j : columns) {
        double xij{element(j, x[i])};
        double mj{mean(j)};
        result += (xij - mj) * (xij - mj) - mj * mj;
    }
    return result;
}

void CPca::dot(const TIntDoublePrVecVec& m, TDoubleVec& x) {
    TDoubleVec result;
    result.reserve(x.size());
    for (std::size_t i = 0u; i < m.size(); ++i) {
        double xi{0.0};
        for (const auto& element : m[i]) {
            xi += element.second * x[element.first];
        }
        result.push_back(xi);
    }
    x.swap(result);
}

double CPca::frobenius(const TIntDoublePrVecVec& m) {
    double result{0.0};
    for (const auto& row : m) {
        for (const auto& element : row) {
            result += element.second * element.second;
        }
    }
    return result;
}

CPca::TDenseVector CPca::toDense(const TSizeVec& columns, const TIntDoublePrVec& x) {
    TDenseVector result(SConstant<TDenseVector>::get(columns.size(), 0.0));
    for (std::size_t i = 0u; i < columns.size(); ++i) {
        result(i) = element(columns[i], x);
    }
    return result;
}

double CPca::element(std::ptrdiff_t i, const TIntDoublePrVec& row) {
    auto xi = std::lower_bound(row.begin(), row.end(), i, COrderings::SFirstLess());
    return xi != row.end() && xi->first == i ? xi->second : 0.0;
}
}
}
