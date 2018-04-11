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

#ifndef INCLUDED_ml_maths_CLassoLogisticRegression_h
#define INCLUDED_ml_maths_CLassoLogisticRegression_h

#include <maths/COrderings.h>
#include <maths/ImportExport.h>

#include <boost/unordered_set.hpp>

#include <cstddef>
#include <vector>

namespace ml {
namespace maths {

namespace lasso_logistic_regression_detail {

using TDoubleVec = std::vector<double>;
using TSizeSizePr = std::pair<std::size_t, std::size_t>;
using TSizeSizePrDoublePr = std::pair<TSizeSizePr, double>;
using TSizeSizePrDoublePrVec = std::vector<TSizeSizePrDoublePr>;

//! Very simple dynamically sized dense matrix.
//!
//! DESCRIPTION:\n
//! Used to represent dense feature vectors. This only implements
//! the interface needed by the CLG algorithm.
class MATHS_EXPORT CDenseMatrix {
public:
    using iterator = TDoubleVec::const_iterator;
    using TDoubleVecVec = std::vector<TDoubleVec>;

public:
    CDenseMatrix();
    CDenseMatrix(TDoubleVecVec& elements);

    //! Efficiently swap the contents of two matrices.
    void swap(CDenseMatrix& other);

    //! Get the number of rows.
    std::size_t rows() const {
        return m_Elements.empty() ? 0 : m_Elements[0].size();
    }
    //! Get the number of columns.
    std::size_t columns() const { return m_Elements.size(); }
    //! Get the beginning of the rows present for the j'th column.
    iterator beginRows(std::size_t j) const { return m_Elements[j].begin(); }
    //! Get the end of the rows present for the j'th column.
    iterator endRows(std::size_t j) const { return m_Elements[j].end(); }
    //! Get the row represented by the j'th column row iterator.
    std::size_t row(iterator itr, std::size_t j) const {
        return itr - m_Elements[j].begin();
    }
    //! Get the element represented by the iterator.
    double element(iterator itr) const { return *itr; }

private:
    //! The actual matrix.
    TDoubleVecVec m_Elements;
};

//! Very simple dynamically sized sparse matrix.
//!
//! DESCRIPTION:\n
//! This only implements the interface needed by the CLG algorithm.
class MATHS_EXPORT CSparseMatrix {
public:
    using iterator = TSizeSizePrDoublePrVec::const_iterator;

public:
    CSparseMatrix();
    CSparseMatrix(std::size_t rows, std::size_t columns, TSizeSizePrDoublePrVec& elements);

    //! Efficiently swap the contents of two matrices.
    void swap(CSparseMatrix& other);

    //! Get the number of rows.
    std::size_t rows() const { return m_Rows; }
    //! Get the number of columns.
    std::size_t columns() const { return m_Columns; }
    //! Get the beginning of the rows present for the j'th column.
    iterator beginRows(std::size_t j) const {
        return std::lower_bound(m_Elements.begin(), m_Elements.end(),
                                TSizeSizePr(j, size_t(0)), COrderings::SFirstLess());
    }
    //! Get the end of the rows present for the j'th column.
    iterator endRows(std::size_t j) const {
        return std::upper_bound(m_Elements.begin(), m_Elements.end(),
                                TSizeSizePr(j, m_Rows), COrderings::SFirstLess());
    }
    //! Get the row represented by the j'th column row iterator.
    std::size_t row(iterator itr, std::size_t /*j*/) const {
        return itr->first.second;
    }
    //! Get the element represented by the iterator.
    double element(iterator itr) const { return itr->second; }

private:
    //! The number of rows.
    std::size_t m_Rows;
    //! The number of columns.
    std::size_t m_Columns;
    //! Representation of the non-zero elements.
    TSizeSizePrDoublePrVec m_Elements;
};

//! \brief Implements Zhang and Oles cyclic coordinate descent scheme,
//! CLG, adapted for Bayesian Logistic Regression with Laplace prior.
//!
//! DESCRIPTION:\n
//! This implements a Gauss-Seidel optimization scheme with trust
//! region used to ensure convergence. Effectively the algorithm
//! is:
//! <pre>
//! \f$\Delta\beta_j \leftarrow 0\f$ for \f$j = 1,..,d\f$ and \f$r_i \leftarrow 0\f$ for \f$i = 1,...,n\f$
//!  for \f$k = 1,2,...\f$
//!    for \f$j = 1,...,d\f$
//!      find \f$\Delta\beta_j\f$ by approximately minimizing \f$\frac{1}{n}\sum_i{f(r_i + \Delta\beta_j x_{ij} y_i) - \lambda_j |\beta_j|}\f$
//!      set \f$r_i \leftarrow r_i + \Delta\beta_j x_{ij} y_i\f$
//!      set \f$\beta_j \leftarrow \beta_j + \Delta\beta_j\f$
//!    end
//!  end
//! </pre>
//! The objective function is guaranteed to decrease monotonically
//! by carefully choosing the step size \f$\Delta\beta_j\f$ in the
//! inner loop (based on an upper bound of the curvature in the trust
//! region).
//!
//! Convergence is tested in the relative and absolute change in the
//! objective function and the maximum number of iterations of the
//! outer loop can also be limited.
//!
//! \see http://www.stat.columbia.edu/~madigan/PAPERS/techno.pdf for
//! more details.
class MATHS_EXPORT CCyclicCoordinateDescent {
public:
    CCyclicCoordinateDescent(std::size_t maxIterations, double eps);

    //! Compute the regression parameters for dense feature vectors.
    //!
    //! \param[in] x The feature vectors in the training data.
    //! \param[in] y The class labels of the feature vectors.
    //! \param[in] lambda The precision of the Laplace prior.
    //! \param[out] beta The MAP parameters of the LASSO logistic
    //! regression.
    //! \param[out] numberIterations The number of iterations of
    //! the main optimization loop used.
    bool run(const CDenseMatrix& x,
             const TDoubleVec& y,
             const TDoubleVec& lambda,
             TDoubleVec& beta,
             std::size_t& numberIterations);

    //! Compute the regression parameters for sparse feature vectors.
    //!
    //! \param[in] x The feature vectors in the training data.
    //! \param[in] y The class labels of the feature vectors.
    //! \param[in] lambda The precision of the Laplace prior.
    //! \param[out] beta The MAP parameters of the LASSO logistic
    //! regression.
    //! \param[out] numberIterations The number of iterations of
    //! the main optimization loop used.
    bool run(const CSparseMatrix& x,
             const TDoubleVec& y,
             const TDoubleVec& lambda,
             TDoubleVec& beta,
             std::size_t& numberIterations);

    //! Compute the regression parameters for dense feature vectors
    //! using the input value of beta to initialize the optimization
    //! loop.
    //!
    //! \param[in] x The feature vectors in the training data.
    //! \param[in] y The class labels of the feature vectors.
    //! \param[in] lambda The precision of the Laplace prior.
    //! \param[in,out] beta The MAP parameters of the LASSO logistic
    //! regression.
    //! \param[out] numberIterations The number of iterations of
    //! the main optimization loop used.
    bool runIncremental(const CDenseMatrix& x,
                        const TDoubleVec& y,
                        const TDoubleVec& lambda,
                        TDoubleVec& beta,
                        std::size_t& numberIterations);

    //! Compute the regression parameters for sparse feature vectors
    //! using the input value of beta to initialize the optimization
    //! loop.
    //!
    //! \param[in] x The feature vectors in the training data.
    //! \param[in] y The class labels of the feature vectors.
    //! \param[in] lambda The precision of the Laplace prior.
    //! \param[in,out] beta The MAP parameters of the LASSO logistic
    //! regression.
    //! \param[out] numberIterations The number of iterations of
    //! the main optimization loop used.
    bool runIncremental(const CSparseMatrix& x,
                        const TDoubleVec& y,
                        const TDoubleVec& lambda,
                        TDoubleVec& beta,
                        std::size_t& numberIterations);

private:
    //! Check the validity of the training data and the prior parameters.
    template<typename MATRIX>
    static bool
    checkInputs(const MATRIX& x, const TDoubleVec& y, const TDoubleVec& lambda);

private:
    //! The maximum number of iterations of the main loop.
    std::size_t m_MaxIterations;
    //! The relative convergence threshold.
    double m_Eps;
};

//! The possible styles for learning hyperparameter \f$\lambda\f$.
//!   -# Lambda norm based, \f$\lambda_n\f$, uses the \f$L_{2,2}\f$
//!      of the training data matrix.
//!   -# Lambda cross validated, \f$\lambda_{cv}\f$, uses two-fold
//!      cross validation and searches the interval
//!      \f$[\frac{\lambda_n}{10}, 10 \lambda_n]\f$ for the value
//!      maximizing the test data likelihood.
enum EHyperparametersStyle { E_LambdaNormBased, E_LambdaCrossValidated };

} // lasso_logistic_regression_detail::

//! \brief A logistic regression model.
//!
//! DESCRIPTION:\n
//! A logistic regression model takes the form:
//! <pre class="fragment">
//!   \f$\displaystyle p(y | x) = \frac{e^{\beta^t x + \beta_0}}{1 + e^{\beta^t x + \beta_0}}\f$
//! </pre>
//!
//! The parameters \f$(\beta, \beta_0)\f$ are chosen to minimize
//! the log likelihood of a collection training data. For more
//! information on fitting this see CLassoLogisticRegression.
class MATHS_EXPORT CLogisticRegressionModel {
public:
    using TDoubleVec = std::vector<double>;
    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TSizeDoublePrVec = std::vector<TSizeDoublePr>;

public:
    CLogisticRegressionModel();
    CLogisticRegressionModel(double beta0, TSizeDoublePrVec& beta);

    //! Efficiently swap the contents of two models.
    void swap(CLogisticRegressionModel& other);

    //! Get the probability of the dense feature vector \p x.
    bool operator()(const TDoubleVec& x, double& probability) const;

    //! Get the probability of the sparse feature vector \p x.
    double operator()(const TSizeDoublePrVec& x) const;

private:
    //! The intercept.
    double m_Beta0;

    //! The non-zero beta parameters.
    TSizeDoublePrVec m_Beta;
};

//! \brief Implements shared functionality for the different Lasso
//! regression implementations.
//!
//! DESCRIPTION:\n
//! This fits the logistic model:
//! <pre class="fragment">
//!   \f$\displaystyle p(y | x) = \frac{e^{\beta^t x}}{1 + e^{\beta^t x}}\f$
//! </pre>
//!
//! to a collection of training data \f$(x, y)\f$, where \f$x\f$
//! is the feature vector and \f$y\in \{-1, 1\}\f$ are the labels.
//!
//! This uses a Bayesian approach, in particular calculating the
//! MAP estimate of the regression parameters \f$\beta\f$. The prior
//! is chosen to be Laplace whose precision can be based either on
//! the mean norm of the feature vectors or learned using 2-fold
//! cross-validation. This means that the solution is sparse, i.e.
//! the least correlated component of the regression parameter
//! vector are driven to zero for high precision.
//!
//! \see http://www.stat.columbia.edu/~madigan/PAPERS/techno.pdf for
//! more details.
//!
//! IMPLEMENTATION DESCISIONS:\n
//! It is useful to have both dense and sparse logistic regression
//! models for different types of training data. This implements
//! the functionality to train the hyperparameters, which can be
//! shared between the two implementations.
template<typename STORAGE>
class MATHS_EXPORT CLassoLogisticRegression {
public:
    using TDoubleVec = std::vector<double>;
    using EHyperparametersStyle = lasso_logistic_regression_detail::EHyperparametersStyle;

protected:
    CLassoLogisticRegression();

    //! Learn the value of precision of the Laplace prior.
    template<typename MATRIX>
    void doLearnHyperparameter(EHyperparametersStyle style);

    //! Learn the parameters of the logistic model based on the
    //! training data added so far.
    template<typename MATRIX>
    bool doLearn(CLogisticRegressionModel& result);

    //! Check whether it is possible to learn a model.
    //!
    //! It is only possible to learn a model if the training
    //! data contains a mixture of both positive and negative
    //! examples.
    bool sanityChecks() const;

    //! Get the training feature vectors.
    inline const STORAGE& x() const { return m_X; }
    //! Get the training feature vectors.
    inline STORAGE& x() { return m_X; }

    //! Get the feature vector dimension.
    inline std::size_t d() const { return m_D; }
    //! Get the feature vector dimension.
    inline std::size_t& d() { return m_D; }

    //! Get the training feature vectors.
    inline const TDoubleVec& y() const { return m_Y; }
    //! Get the training feature vectors.
    inline TDoubleVec& y() { return m_Y; }

private:
    //! The feature vectors.
    STORAGE m_X;
    //! The dimension of the feature vectors.
    std::size_t m_D;
    //! The feature vector labels.
    TDoubleVec m_Y;
    //! The precision of the Laplace prior.
    double m_Lambda;
    //! The (last) learned regression parameters.
    TDoubleVec m_Beta;
};

using TDenseStorage = std::vector<std::vector<double>>;
using TSparseStorage = std::vector<std::vector<std::pair<std::size_t, double>>>;

//! \brief Lasso logistic regression using dense encoding of the
//! feature vectors.
//!
//! DESCRIPTION:\n
//! \see CLassoLogisticRegression for details.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This uses a dense encoding of the feature vector for the case that
//! they are small and mostly non-zero.
class MATHS_EXPORT CLassoLogisticRegressionDense
    : public CLassoLogisticRegression<TDenseStorage> {
public:
    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TSizeDoublePrVec = std::vector<TSizeDoublePr>;

public:
    //! Add a labeled feature vector \p x. The label is either
    //! interesting or boring.
    //!
    //! \param[in] x The feature vector.
    //! \param[in] interesting The label of \p x.
    void addTrainingData(const TDoubleVec& x, bool interesting);

    //! Learn the value of precision of the Laplace prior.
    //!
    //! \param[in] style The style of training to use.
    //! \see EHyperparametersStyle for more details on the options.
    void learnHyperparameter(EHyperparametersStyle style);

    //! Learn the parameters of the logistic model based on the
    //! training data added so far.
    //!
    //! \param[out] result The trained logistic model.
    bool learn(CLogisticRegressionModel& result);
};

//! \brief Lasso logistic regression using sparse encoding of the
//! feature vectors.
//!
//! DESCRIPTION:\n
//! \see CLassoLogisticRegression for details.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This uses a sparse encoding of the feature vector for the case
//! that they are high dimensional, but most components are zero.
class MATHS_EXPORT CLassoLogisticRegressionSparse : CLassoLogisticRegression<TSparseStorage> {
public:
    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TSizeDoublePrVec = std::vector<TSizeDoublePr>;
    using EHyperparametersStyle = lasso_logistic_regression_detail::EHyperparametersStyle;

public:
    //! Add a labeled feature vector \p x. The label is either
    //! interesting or boring.
    //!
    //! \param[in] x The feature vector.
    //! \param[in] interesting The label of \p x.
    void addTrainingData(const TSizeDoublePrVec& x, bool interesting);

    //! Learn the value of precision of the Laplace prior.
    //!
    //! \param[in] style The style of training to use.
    //! \see EHyperparametersStyle for more details on the options.
    void learnHyperparameter(EHyperparametersStyle style);

    //! Learn the parameters of the logistic model based on the
    //! training data added so far.
    //!
    //! \param[out] result The trained logistic model.
    bool learn(CLogisticRegressionModel& result);
};
}
}

#endif // INCLUDED_ml_maths_CLassoLogisticRegression_h
