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

#ifndef INCLUDED_ml_maths_CAssignment_h
#define INCLUDED_ml_maths_CAssignment_h

#include <maths/ImportExport.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace ml
{
namespace maths
{

//! \brief Implementation of algorithms for assignment problems.
//!
//! DESCRIPTION:\n
//! Implements the Kuhn-Munkres algorithm to find the minimum
//! cost perfect matching given an n x m cost matrix.
class MATHS_EXPORT CAssignment
{
    public:
        using TDoubleVec = std::vector<double>;
        using TDoubleVecVec = std::vector<TDoubleVec>;
        using TSizeSizePr = std::pair<std::size_t, std::size_t>;
        using TSizeSizePrVec = std::vector<TSizeSizePr>;

    public:
        //! \brief The Kuhn-Munkres algorithm for solving the
        //! assignment problem.
        //!
        //! The assignment problem consists of a number of rows
        //! along with a number of columns and a cost matrix which
        //! gives the cost of assigning the i'th row to the j'th
        //! column at position (i, j). The goal is to find a one-to-one
        //! assignment of rows to columns whilst minimizing the total
        //! cost of the assignment.
        //!
        //! \param[in] costs The cost matrix.
        //! \param[out] matching Filled in with the optimal matching.
        //! \warning The numbers of columns should be the same for
        //! every row.
        //! \note This implementation is O(m * n^2) where m is the
        //! minimum of the # rows and columns and n is the maximum.
        static bool kuhnMunkres(const TDoubleVecVec &costs,
                                TSizeSizePrVec &matching);
};

}
}
#endif // INCLUDED_ml_maths_CAssignment_h
