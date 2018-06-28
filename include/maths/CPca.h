/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CPca_h
#define INCLUDED_ml_maths_CPca_h

#include <maths/CLinearAlgebraEigen.h>
#include <maths/ImportExport.h>

namespace ml {
namespace maths {

//! \brief Methods for principle component analysis.
class CPca {
public:
    using TIntDoublePr = std::pair<std::ptrdiff_t, double>;
    using TSizeVec = std::vector<std::size_t>;
    using TIntDoublePrVec = std::vector<TIntDoublePr>;
    using TIntDoublePrVecVec = std::vector<TIntDoublePrVec>;
    using TDenseVector = maths::CDenseVector<double>;
    using TDenseVectorVec = std::vector<TDenseVector>;

public:
    //! Project \p data onto the first \p numberComponents principle components.
    //!
    //! \param[in] numberComponents The number of components on which to project.
    //! \param[in] support The indices of the vectors to use to compute the
    //! projection.
    //! \param[in,out] data The vectors to project.
    static void projectOntoPrincipleComponents(std::size_t numberComponents,
                                               const TSizeVec& support,
                                               TDenseVectorVec& data);

    //! Project \p data onto the first \p numberComponents principle components.
    //!
    //! This computes a number of samples to use based on the data sparsity.
    //!
    //! \param[in] numberComponents The number of components on which to project.
    //! \param[in] dimension The data dimension.
    //! \param[in] support The indices of the vectors to use to compute the
    //! projection.
    //! \param[in] data The vectors to project.
    //! \param[out] projected Filled in with the projection of \p data.
    static void projectOntoPrincipleComponentsRandom(std::size_t numberComponents,
                                                     std::ptrdiff_t dimension,
                                                     const TSizeVec& support,
                                                     const TIntDoublePrVecVec& data,
                                                     TDenseVectorVec& projected);

    //! Project \p data onto the first \p numberComponents principle components.
    //!
    //! This assumes the data have a low rank approximation and uses a random
    //! sample of the data rows and columns, with rows x columns = \p numberSamples,
    //! to compute an approximate SVD. See for example https://www.math.cmu.edu/~af1p/Texfiles/SVD.pdf
    //! for background.
    //!
    //! We make a number of modifications to the basic approach. In particular,
    //! we compute a desired number of columns based on retaining columns with
    //! significant probability of being sampled. Furthermore, we sample columns
    //! *without* replacement. This should be strictly better and testing suggests
    //! it is a big win if the number of columns to be sampled is a significant
    //! fraction of all columns. However, getting an unbiased estimator of the
    //! SVD requires estimates of the sampling probability in the sample set *as
    //! a whole*. An exact calculation of these is prohibitively slow so we use
    //! a Monte-Carlo estimate.
    //!
    //! \param[in] numberComponents The number of components on which to project.
    //! \param[in] numberSamples The number of samples of \p support to use to
    //! compute the projection.
    //! \param[in] dimension The data dimension.
    //! \param[in] support The indices of the vectors to use to compute the
    //! projection.
    //! \param[in] data The vectors to project.
    //! \param[out] projected Filled in with the projection of \p data.
    static void projectOntoPrincipleComponentsRandom(std::size_t numberComponents,
                                                     std::size_t numberSamples,
                                                     std::ptrdiff_t dimension,
                                                     const TSizeVec& support,
                                                     const TIntDoublePrVecVec& data,
                                                     TDenseVectorVec& projected);

    //! Get the numeric rank of the \p data matrix.
    //!
    //! This is Frobenius norm divided by the square spectral norm.
    static double numericRank(std::ptrdiff_t dimension, TIntDoublePrVecVec& data);

private:
    using TDoubleVec = std::vector<double>;

private:
    //! Get the square of the Euclidean norm of \p x.
    static double norm2(const TDoubleVec& x);

    //! Get the square of the Euclidean norm of \p i'th column
    //! of \p x - \p mean.
    static double centredColumnNorm2(std::size_t i,
                                     const TIntDoublePrVecVec& x,
                                     const TDenseVector& mean);

    //! Get the square of the Euclidean norm of \p i'th row
    //! of \p x - \p mean.
    static double centredRowNorm2(std::size_t i,
                                  const TSizeVec& columns,
                                  const TIntDoublePrVecVec& x,
                                  double meanNorm2,
                                  const TDenseVector& mean);

    //! Get the inner product of \p m and \p x.
    static void dot(const TIntDoublePrVecVec& m, TDoubleVec& x);

    //! Get the Frobenius norm of the matrix \p m.
    static double frobenius(const TIntDoublePrVecVec& m);

    //! Convert \p x to a dense vector representation.
    static TDenseVector toDense(const TSizeVec& columns, const TIntDoublePrVec& x);

    //! Get the \p i'th element of \p row.
    static double element(std::ptrdiff_t i, const TIntDoublePrVec& row);
};
}
}

#endif // INCLUDED_ml_maths_CPca_h
