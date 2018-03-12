/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#include <maths/CAssignment.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <boost/numeric/conversion/bounds.hpp>

#include <algorithm>

namespace ml {
namespace maths {

namespace {

typedef std::vector<bool> TBoolVec;
typedef std::vector<double> TDoubleVec;
typedef std::vector<TDoubleVec> TDoubleVecVec;
typedef std::pair<double, double> TDoubleDoublePr;
typedef std::vector<std::size_t> TSizeVec;
typedef std::pair<std::size_t, std::size_t> TSizeSizePr;
typedef std::vector<TSizeSizePr> TSizeSizePrVec;

const std::size_t UNMATCHED = boost::numeric::bounds<std::size_t>::highest();
const double MAXIMUM_COST = boost::numeric::bounds<double>::highest();

//! Extract the cost of the i'th row and j'th column. Indices
//! out of bounds are treated as zero.
//!
//! \param[in] costs The matrix of costs.
//! \param[in] i The row index.
//! \param[in] j The column index.
inline double cost(const TDoubleVecVec &costs,
                   std::size_t i,
                   std::size_t j) {
    return (i < costs.size() ? (j < costs[i].size() ? costs[i][j] : 0.0) : 0.0);
}

//! Extract the cost of the i'th row and j'th column and adjust
//! by the row and column potentials, i.e.
//! <pre class="fragment">
//!   \f$c'_{i,j} = c_{i,j} - P_i^r - P_j^r\f$
//! </pre>
//!
//! Indices out of bounds are treated as zero.
//!
//! \param[in] costs The matrix of costs.
//! \param[in] rowPotential The row potential function.
//! \param[in] columnPotential The column potential function.
//! \param[in] i The row index.
//! \param[in] j The column index.
inline double adjustedCost(const TDoubleVecVec &costs,
                           const TDoubleVec &rowPotential,
                           const TDoubleVec &columnPotential,
                           std::size_t i,
                           std::size_t j) {
    // The bracketing is important in this expression since
    // it ensures we find the correct initial feasible solution.
    return (cost(costs, i, j) - columnPotential[j]) - rowPotential[i];
}

//! Update the matching to include (\p i, \p j).
//!
//! \param[in] i The row index to match.
//! \param[in] j The column index to match.
//! \param[out] matchColumnByRow The columns matching each row.
//! \param[out] matchRowByColumn The rows matching each column.
inline void match(std::size_t i,
                  std::size_t j,
                  TSizeVec &matchColumnByRow,
                  TSizeVec &matchRowByColumn) {
    matchColumnByRow[i] = j;
    matchRowByColumn[j] = i;
}

//! Update the augmenting path to include the pivot row.
//!
//! \param[in] costs The matrix of costs.
//! \param[in] rowPotential The row potential function.
//! \param[in] columnPotential The column potential function.
//! \param[in] parentRowByCommittedColumn The parent of each
//! committed column.
//! \param[in] pivot The row to include in the augmenting path.
//! \param[out] committedRows The rows which have been committed
//! to the augmenting path.
//! \param[out] minSlackRowByColumn The minimum slack row by
//! committed column.
//! \param[out] minSlackValueByColumn The slack of the minimum
//! slack row by committed column.
//! \param[out] minSlackRow The row of the minimum slack edge.
//! \param[out] minSlackColumn The column of the minimum slack
//! edge.
//! \param[out] minSlackValue The minimum slack.
void grow(const TDoubleVecVec &costs,
          const TDoubleVec &rowPotential,
          const TDoubleVec &columnPotential,
          const TSizeVec &parentRowByCommittedColumn,
          std::size_t pivot,
          TBoolVec &committedRows,
          TSizeVec &minSlackRowByColumn,
          TDoubleVec &minSlackValueByColumn,
          std::size_t &minSlackRow,
          std::size_t &minSlackColumn,
          double &minSlackValue) {
    minSlackRow = UNMATCHED;
    minSlackColumn = UNMATCHED;
    minSlackValue = MAXIMUM_COST;
    committedRows[pivot] = true;
    for (std::size_t j = 0u; j < parentRowByCommittedColumn.size(); ++j) {
        if (parentRowByCommittedColumn[j] == UNMATCHED) {
            double slack = adjustedCost(costs, rowPotential, columnPotential, pivot, j);
            if (minSlackValueByColumn[j] > slack) {
                minSlackValueByColumn[j] = slack;
                minSlackRowByColumn[j] = pivot;
            }
            if (minSlackValueByColumn[j] < minSlackValue) {
                minSlackValue = minSlackValueByColumn[j];
                minSlackRow = minSlackRowByColumn[j];
                minSlackColumn = j;
            }
        }
    }
}

}

bool CAssignment::kuhnMunkres(const TDoubleVecVec &costs,
                              TSizeSizePrVec &matching) {
    matching.clear();

    if (costs.empty()) {
        return true;
    }

    // Sanity check cost matrix.
    std::size_t n = costs.size();
    std::size_t m = costs[0].size();
    for (std::size_t i = 1u; i < costs.size(); ++i) {
        if (costs[i].size() != m) {
            LOG_ERROR("Irregular cost matrix");
            return false;
        }
    }
    std::size_t N = std::max(n, m);
    LOG_TRACE("N = " << N);

    // Initialize the potential function with the minimum cost
    // edge incident on each column and row.
    TDoubleVec columnPotential(N, 0.0);
    TDoubleVec rowPotential(N, 0.0);
    for (std::size_t j = 0u; j < m; ++j) {
        double min = costs[0][j];
        for (std::size_t i = 1u; i < N; ++i) {
            min = std::min(min, cost(costs, i, j));
        }
        columnPotential[j] = min;
    }
    for (std::size_t i = 0u; i < n; ++i) {
        double min = costs[i][0] - columnPotential[0];
        for (std::size_t j = 1u; j < N; ++j) {
            min = std::min(min, cost(costs, i, j) - columnPotential[j]);
        }
        rowPotential[i] = min;
    }
    LOG_TRACE("row potential = " << core::CContainerPrinter::print(rowPotential));
    LOG_TRACE("column potential = " << core::CContainerPrinter::print(columnPotential));

    // An initial matching candidate matching (note that this
    // may not be complete).
    TSizeVec matchColumnByRow(N, UNMATCHED);
    TSizeVec matchRowByColumn(N, UNMATCHED);
    std::size_t unmatched = N;
    for (std::size_t i = 0u; i < N; ++i) {
        for (std::size_t j = 0u; j < N; ++j) {
            if (matchColumnByRow[i] == UNMATCHED &&
                matchRowByColumn[j] == UNMATCHED &&
                adjustedCost(costs, rowPotential, columnPotential, i, j) == 0.0) {
                match(i, j, matchColumnByRow, matchRowByColumn);
                --unmatched;
            }
        }
    }

    TBoolVec committedRows(N, false);
    TSizeVec parentRowByCommittedColumn(N, UNMATCHED);
    TSizeVec minSlackRowByColumn(N, UNMATCHED);
    TDoubleVec minSlackValueByColumn(N, MAXIMUM_COST);

    while (unmatched > 0) {
        LOG_TRACE("matchColumnByRow = " << core::CContainerPrinter::print(matchColumnByRow));
        LOG_TRACE("matchRowByColumn = " << core::CContainerPrinter::print(matchRowByColumn));
        LOG_TRACE("unmatched = " << unmatched);

        LOG_TRACE("*** Initialize augmenting path ***");

        // Find an unmatched row. We look for the augmenting
        // path which matches this row in the loop below.
        std::size_t pivot = N;
        for (std::size_t i = 0u; i < N; ++i) {
            if (matchColumnByRow[i] == UNMATCHED) {
                pivot = i;
                break;
            }
        }
        LOG_TRACE("pivot = " << pivot);
        if (pivot == N) {
            LOG_ERROR("Bad pivot: costs = " << core::CContainerPrinter::print(costs));
            return false;
        }

        // Update the path to include the unmatched row.
        std::size_t minSlackRow;
        std::size_t minSlackColumn;
        double minSlackValue;
        grow(costs, rowPotential, columnPotential,
             parentRowByCommittedColumn,
             pivot,
             committedRows,
             minSlackRowByColumn,
             minSlackValueByColumn,
             minSlackRow,
             minSlackColumn,
             minSlackValue);
        LOG_TRACE("committedRows = " << core::CContainerPrinter::print(committedRows));
        LOG_TRACE("minSlackRowByColumn = "
                  << core::CContainerPrinter::print(minSlackRowByColumn));
        LOG_TRACE("minSlackValueByColumn = "
                  << core::CContainerPrinter::print(minSlackValueByColumn));

        // Search for an augmenting path following zero slack
        // edges. In each iteration the minimum potential is
        // checked and adjusted if the minimum slack to any
        // unvisited row is greater than zero. The adjusted
        // edge weights in the visited subgraph are unmodified
        // by this operation. If an unmatched column is found
        // the matching is updated to include the augmenting
        // path and the search reinitialized.

        LOG_TRACE("*** Search for augmenting path ***");

        std::size_t check = 0u;
        for (/**/; check < N; ++check) {
            LOG_TRACE(" minSlackValue = " << minSlackValue
                      << ", minSlackRow = " << minSlackRow
                      << ", minSlackColumn = " << minSlackColumn);

            // Checking greater than zero here is important since
            // due to non-associativity of floating point arithmetic
            // it may be that after adjusting potentials some slacks
            // are slightly negative.
            if (minSlackValue > 0.0) {
                double adjustment = minSlackValue;
                for (std::size_t i = 0; i < N; ++i) {
                    if (committedRows[i]) {
                        rowPotential[i] += adjustment;
                    }
                }
                for (std::size_t j = 0u; j < N; ++j) {
                    if (parentRowByCommittedColumn[j] == UNMATCHED) {
                        minSlackValueByColumn[j] -= adjustment;
                    } else {
                        columnPotential[j] -= adjustment;
                    }
                }
                LOG_TRACE(" rowPotential = " << core::CContainerPrinter::print(rowPotential));
                LOG_TRACE(" columnPotential = " << core::CContainerPrinter::print(columnPotential));
            }

            parentRowByCommittedColumn[minSlackColumn] = minSlackRow;
            pivot = matchRowByColumn[minSlackColumn];
            if (pivot == UNMATCHED) {
                // Update the matching by backtracking.
                std::size_t committedColumn = minSlackColumn;
                check = 0u;
                for (/**/; check < N; ++check) {
                    std::size_t parentRow = parentRowByCommittedColumn[committedColumn];
                    std::size_t tmp = matchColumnByRow[parentRow];
                    match(parentRow,
                          committedColumn,
                          matchColumnByRow,
                          matchRowByColumn);
                    committedColumn = tmp;
                    if (committedColumn == UNMATCHED) {
                        break;
                    }
                }
                if (check == N) {
                    LOG_ERROR("Bad augmenting path: costs = "
                              << core::CContainerPrinter::print(costs));
                    return false;
                }
                --unmatched;

                break;
            } else {
                LOG_TRACE(" pivot = " << pivot);
                LOG_TRACE(" parentRowByCommittedColumn = "
                          << core::CContainerPrinter::print(parentRowByCommittedColumn));

                // Grow the path to include the pivot row.
                grow(costs, rowPotential, columnPotential,
                     parentRowByCommittedColumn,
                     pivot,
                     committedRows,
                     minSlackRowByColumn,
                     minSlackValueByColumn,
                     minSlackRow,
                     minSlackColumn,
                     minSlackValue);
                LOG_TRACE(" committedRows = " << core::CContainerPrinter::print(committedRows));
                LOG_TRACE(" minSlackRowByColumn = "
                          << core::CContainerPrinter::print(minSlackRowByColumn));
                LOG_TRACE(" minSlackValueByColumn = "
                          << core::CContainerPrinter::print(minSlackValueByColumn));
            }
        }
        if (check == N) {
            LOG_ERROR("Failed to find path: costs "
                      << core::CContainerPrinter::print(costs));
            return false;
        }

        // Reset the augmenting path data.
        std::fill_n(committedRows.begin(), N, false);
        std::fill_n(parentRowByCommittedColumn.begin(), N, UNMATCHED);
        std::fill_n(minSlackValueByColumn.begin(), N, MAXIMUM_COST);
    }

    // Extract the matching.
    matching.reserve(std::min(m, n));
    for (std::size_t i = 0u; i < n; ++i) {
        if (matchColumnByRow[i] < m) {
            matching.emplace_back(i, matchColumnByRow[i]);
        }
    }

    return true;
}

}
}
