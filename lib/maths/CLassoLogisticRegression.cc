/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CLassoLogisticRegression.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CSampling.h>
#include <maths/CSolvers.h>
#include <maths/CTools.h>

#include <boost/bind.hpp>
#include <boost/range.hpp>

#include <math.h>

namespace ml
{
namespace maths
{

namespace
{
typedef std::vector<double> TDoubleVec;

//! An upper bound on the second derivative of minus the log of the
//! logistic link function in a ball, i.e.
//! <pre class="fragment">
//!   \f$F(r, \delta) \ge sup_{|\Delta r| \le \delta}{ f''(r+\Delta r) }\f$
//! </pre>
//! where \f$f(r)=log(1 + e^-r)\f$. This is used to implement trust
//! region concept for guaranteeing convergence of Gauss-Seidel
//! iterative solver.
double F(double r, double delta)
{
    r = ::fabs(r);
    if (r <= delta)
    {
        return 0.25;
    }
    double s = ::exp(r - delta);
    return 1.0 / (2.0 + s + 1.0 / s);
}

//! Computes the step for the CLG solver for the LASSO logistic
//! regression objective.
double lassoStep(double beta, double lambda, double n, double d)
{
    double dv = 0.0;
    if (beta == 0.0)
    {
        // Second derivative is undefined check that moving away
        // from zero one gets consistent sign for the desired step.
        dv = (n - lambda) / d;
        if (dv <= 0.0)
        {
            dv = (n + lambda) / d;
            if (dv >= 0)
            {
                return 0.0;
            }
        }
    }
    else
    {
        double sign = CTools::sign(beta);
        dv = (n - sign * lambda) / d;
        if (sign * (beta + dv) < 0.0)
        {
            // Don't allow the sign of the beta to change in one
            // step, since the bound on second derivative doesn't
            // hold on a sign change.
            dv = -beta;
        }
    }
    return dv;
}

//! The log-likelihood (minus some constants).
//!
//! The log-likelihood is proportional to
//! <pre class="fragment">
//!   \f$-\sum_{i=1}^{n}\log(1+\exp(-\boldsymbol{\beta}'\mathbf{x_i} y_i)) - \sum_{j=1}^d\left(\lambda_j |\beta_j|\right)\f$
//! </pre>
//!
//! \note That this should decrease monotonically in each iteration
//! of the inner solver loop.
template<typename MATRIX>
double logLikelihood(const MATRIX &x,
                     const TDoubleVec &y,
                     const TDoubleVec &lambda,
                     const TDoubleVec &beta)
{
    typedef typename MATRIX::iterator iterator;

    double result = 0.0;
    TDoubleVec f(y.size(), 0.0);
    for (std::size_t j = 0u; j < beta.size(); ++j)
    {
        for (iterator itr = x.beginRows(j); itr != x.endRows(j); ++itr)
        {
            std::size_t i = x.row(itr, j);
            double xij = x.element(itr);
            f[i] += beta[j] * xij;
        }
    }

    for (std::size_t i = 0u; i < f.size(); ++i)
    {
        result -= ::log(1.0 + ::exp(-f[i] * y[i]));
    }
    for (std::size_t j = 0u; j < beta.size(); ++j)
    {
        result -= lambda[j] * ::fabs(beta[j]);
    }

    return result;
}

//! Zhang and Oles cyclic coordinate descent scheme, CLG, adapted
//! for Bayesian Logistic Regression with Laplace prior.
template<typename MATRIX>
void CLG(std::size_t maxIterations,
         double eps,
         const MATRIX &x,
         const TDoubleVec &y,
         const TDoubleVec &lambda,
         TDoubleVec &beta,
         TDoubleVec &r,
         std::size_t &numberIterations)
{
    typedef typename MATRIX::iterator iterator;

    std::size_t d = x.columns();
    LOG_DEBUG("d = " << d << ", n = " << y.size());

    // Initialize the regression parameters.
    beta.resize(d, 0.0);

    // The trust region.
    TDoubleVec delta(d, 1.0);

    // The reachable values.
    TDoubleVec rlast;

    // Auxiliary variables used to compute solver step.
    TDoubleVec num(d, 0.0);
    TDoubleVec den(d, 0.0);
    for (std::size_t j = 0u; j < d; ++j)
    {
        double Dj = delta[j];
        for (iterator itr = x.beginRows(j); itr != x.endRows(j); ++itr)
        {
            std::size_t i = x.row(itr, j);
            double xij = x.element(itr);
            double xy  = xij * y[i];
            double xx  = xij * xij;
            double ri  = r[i];
            num[j] += xy / (1.0 + ::exp(ri));
            den[j] += xx * F(ri, Dj * ::fabs(xij));
        }
    }

    for (std::size_t k = 0u; k < maxIterations; ++k)
    {
        rlast = r;
        LOG_TRACE("numerator   = " << core::CContainerPrinter::print(num));
        LOG_TRACE("denominator = " << core::CContainerPrinter::print(den));

        for (std::size_t j = 0u; j < d; ++j)
        {
            double bj = beta[j];
            double Dj = delta[j];
            double dvj = lassoStep(bj, lambda[j], num[j], den[j]);
            double dbj = CTools::truncate(dvj, -Dj, +Dj);

            beta[j] += dbj;
            delta[j] = std::max(2.0 * ::fabs(dbj), Dj / 2.0);
            if (dbj != 0.0 || j+1 == d)
            {
                for (iterator itr = x.beginRows(j); itr != x.endRows(j); ++itr)
                {
                    std::size_t i = x.row(itr, j);
                    r[i] += dbj * x.element(itr) * y[i];
                }

                std::size_t jPlus1 = (j + 1) % d;
                double &numjPlus1 = num[jPlus1];
                double &denjPlus1 = den[jPlus1];
                double DjPlus1    = delta[jPlus1];
                numjPlus1 = denjPlus1 = 0.0;
                for (iterator itr = x.beginRows(jPlus1); itr != x.endRows(jPlus1); ++itr)
                {
                    std::size_t i = x.row(itr, jPlus1);
                    double xij = x.element(itr);
                    double xy  = xij * y[i];
                    double xx  = xij * xij;
                    double ri  = r[i];
                    numjPlus1 += xy / (1.0 + ::exp(ri));
                    denjPlus1 += xx * F(ri, DjPlus1 * ::fabs(xij));
                }
            }
        }
        LOG_TRACE("r = " << core::CContainerPrinter::print(r));
        LOG_TRACE("beta = " << core::CContainerPrinter::print(beta));
        LOG_TRACE("delta = " << core::CContainerPrinter::print(delta));
        LOG_TRACE("-log(likelihood) = " << logLikelihood(x, y, lambda, beta));

        ++numberIterations;

        // Check for convergence.
        double dsum = 0.0;
        double sum  = 0.0;
        for (std::size_t i = 0u; i < r.size(); ++i)
        {
            dsum += ::fabs(r[i] - rlast[i]);
            sum  += ::fabs(r[i]);
        }
        LOG_TRACE("sum |dr| = " << dsum << ", sum |r| = " << sum);
        if (dsum < eps * (1.0 + sum))
        {
            break;
        }
    }
}

} // unnamed::

namespace lasso_logistic_regression_detail
{

////// CDenseMatrix //////

CDenseMatrix::CDenseMatrix(void)
{
}

CDenseMatrix::CDenseMatrix(TDoubleVecVec &elements)
{
    m_Elements.swap(elements);
}

void CDenseMatrix::swap(CDenseMatrix &other)
{
    m_Elements.swap(other.m_Elements);
}


////// CSparseMatrix //////

CSparseMatrix::CSparseMatrix(void) :
        m_Rows(0),
        m_Columns(0)
{
}

CSparseMatrix::CSparseMatrix(std::size_t rows,
                             std::size_t columns,
                             TSizeSizePrDoublePrVec &elements) :
        m_Rows(rows),
        m_Columns(columns)
{
    m_Elements.swap(elements);
    std::sort(m_Elements.begin(), m_Elements.end(), COrderings::SFirstLess());
}

void CSparseMatrix::swap(CSparseMatrix &other)
{
    std::swap(m_Rows, other.m_Rows);
    std::swap(m_Columns, other.m_Columns);
    m_Elements.swap(other.m_Elements);
}


////// CCyclicCoordinateDescent //////

CCyclicCoordinateDescent::CCyclicCoordinateDescent(std::size_t maxIterations,
                                                   double eps) :
        m_MaxIterations(maxIterations),
        m_Eps(eps)
{
}

template<typename MATRIX>
bool CCyclicCoordinateDescent::checkInputs(const MATRIX &x,
                                           const TDoubleVec &y,
                                           const TDoubleVec &lambda)
{
    if (x.rows() == 0)
    {
        LOG_ERROR("No training data");
        return false;
    }
    if (x.rows() != y.size())
    {
        LOG_ERROR("Inconsistent training data |x| = "
                  << x.rows() << ", |y| = " << y.size());
        return false;
    }
    if (lambda.size() != x.columns())
    {
        LOG_ERROR("Inconsistent prior |lambda| = "
                  << lambda.size() << ", D = " << x.columns());
        return false;
    }
    return true;
}

bool CCyclicCoordinateDescent::run(const CDenseMatrix &x,
                                   const TDoubleVec &y,
                                   const TDoubleVec &lambda,
                                   TDoubleVec &beta,
                                   std::size_t &numberIterations)
{
    beta.clear();
    numberIterations = 0;
    if (!checkInputs(x, y, lambda))
    {
        return false;
    }
    TDoubleVec r(x.rows(), 0.0);
    CLG(m_MaxIterations, m_Eps, x, y, lambda, beta, r, numberIterations);
    return true;
}

bool CCyclicCoordinateDescent::run(const CSparseMatrix &x,
                                   const TDoubleVec &y,
                                   const TDoubleVec &lambda,
                                   TDoubleVec &beta,
                                   std::size_t &numberIterations)
{
    beta.clear();
    numberIterations = 0;
    if (!checkInputs(x, y, lambda))
    {
        return false;
    }
    TDoubleVec r(x.rows(), 0.0);
    CLG(m_MaxIterations, m_Eps, x, y, lambda, beta, r, numberIterations);
    return true;
}

bool CCyclicCoordinateDescent::runIncremental(const CDenseMatrix &x,
                                              const TDoubleVec &y,
                                              const TDoubleVec &lambda,
                                              TDoubleVec &beta,
                                              std::size_t &numberIterations)
{
    numberIterations = 0;
    if (!checkInputs(x, y, lambda))
    {
        return false;
    }
    if (beta.size() != lambda.size())
    {
        LOG_ERROR("Inconsistent seed parameter vector |beta| = "
                 << beta.size() << ", D = " << lambda.size());
        return false;
    }

    // Initialize the reachable values.
    TDoubleVec r(x.rows(), 0.0);
    for (std::size_t j = 0u; j < x.columns(); ++j)
    {
        double bj = beta[j];
        for (CDenseMatrix::iterator itr = x.beginRows(j); itr != x.endRows(j); ++itr)
        {
            std::size_t i = x.row(itr, j);
            r[i] = bj * x.element(itr) * y[i];
        }
    }

    CLG(m_MaxIterations, m_Eps, x, y, lambda, beta, r, numberIterations);
    return true;
}

bool CCyclicCoordinateDescent::runIncremental(const CSparseMatrix &x,
                                              const TDoubleVec &y,
                                              const TDoubleVec &lambda,
                                              TDoubleVec &beta,
                                              std::size_t &numberIterations)
{
    numberIterations = 0;
    if (!checkInputs(x, y, lambda))
    {
        return false;
    }
    if (beta.size() != lambda.size())
    {
        LOG_ERROR("Inconsistent seed parameter vector |beta| = "
                 << beta.size() << ", D = " << lambda.size());
        return false;
    }

    // Initialize the reachable values.
    TDoubleVec r(x.rows(), 0.0);
    for (std::size_t j = 0u; j < x.columns(); ++j)
    {
        double bj = beta[j];
        for (CSparseMatrix::iterator itr = x.beginRows(j); itr != x.endRows(j); ++itr)
        {
            std::size_t i = x.row(itr, j);
            r[i] = bj * x.element(itr) * y[i];
        }
    }

    CLG(m_MaxIterations, m_Eps, x, y, lambda, beta, r, numberIterations);
    return true;
}

} // lasso_logistic_regression_detail::

////// CLogisticRegressionModel //////

CLogisticRegressionModel::CLogisticRegressionModel(void) :
        m_Beta0(0.0),
        m_Beta()
{
}

CLogisticRegressionModel::CLogisticRegressionModel(double beta0,
                                                   TSizeDoublePrVec &beta) :
        m_Beta0(beta0),
        m_Beta()
{
    m_Beta.swap(beta);
}

void CLogisticRegressionModel::swap(CLogisticRegressionModel &other)
{
    std::swap(m_Beta0, other.m_Beta0);
    m_Beta.swap(other.m_Beta);
}

bool CLogisticRegressionModel::operator()(const TDoubleVec &x,
                                          double &probability) const
{
    probability = 0.5;
    if (m_Beta.empty())
    {
        return true;
    }

    std::size_t n = m_Beta.size();
    if (x.size() <= m_Beta[n - 1].first)
    {
        LOG_ERROR("Invalid feature vector |x| = " << x.size()
                  << ", D = " << m_Beta[n - 1].first + 1)
    }
    double r = -m_Beta0;
    for (std::size_t i = 0u; i < m_Beta.size(); ++i)
    {
        r -= m_Beta[i].second * x[m_Beta[i].first];
    }
    probability = 1.0 / (1.0 + ::exp(-r));
    return true;
}

double CLogisticRegressionModel::operator()(const TSizeDoublePrVec &x) const
{
    if (m_Beta.empty())
    {
        return 0.5;
    }

    double r = -m_Beta0;
    for (std::size_t i = 0u, j = 0u; i < m_Beta.size() && j < x.size(); /**/)
    {
        if (m_Beta[i].first < x[j].first)
        {
            ++i;
        }
        else if (x[j].first < m_Beta[i].first)
        {
            ++j;
        }
        else
        {
            r -= m_Beta[i].second * x[j].second;
            ++i;
            ++j;
        }
    }
    return 1.0 / (1.0 + ::exp(-r));
}

namespace
{

using namespace lasso_logistic_regression_detail;
typedef std::vector<TDoubleVec> TDoubleVecVec;
typedef std::pair<std::size_t, double> TSizeDoublePr;
typedef std::vector<TSizeDoublePr> TSizeDoublePrVec;
typedef std::vector<TSizeDoublePrVec> TSizeDoublePrVecVec;
typedef boost::unordered_set<std::size_t> TSizeUSet;

//! Set up the masked training data.
//!
//! \param[in] x The set of all feature vectors.
//! \param[in] y The set of all feature vectors labels.
//! \param[in] mask The indices of the feature vectors to remove.
//! \param[out] xMasked The training matrix corresponding to \p x.
//! \param[out] yMasked The training labels corresponding to \p y.
void setupTrainingData(const TDoubleVecVec &x,
                       const TDoubleVec &y,
                       const TSizeUSet &mask,
                       CDenseMatrix &xMasked,
                       TDoubleVec &yMasked)
{
    xMasked = CDenseMatrix();
    yMasked.clear();

    if (x.empty())
    {
        return;
    }

    std::size_t n = x.size();
    std::size_t m = n - mask.size();
    std::size_t d = x[0].size();
    LOG_TRACE("n = " << n << ", d = " << d);

    TDoubleVecVec xTranspose(d + 1, TDoubleVec(m, 1.0));
    yMasked.reserve(m);
    for (std::size_t i = 0u, i_ = 0u; i < n; ++i)
    {
        if (mask.count(i) == 0)
        {
            for (std::size_t j = 0u; j < d; ++j)
            {
                xTranspose[j][i_] = x[i][j];
            }
            yMasked.push_back(y[i]);
            ++i_;
        }
    }
    CDenseMatrix tmp(xTranspose);
    xMasked.swap(tmp);
}

//! Overload for sparse feature vectors.
void setupTrainingData(const TSizeDoublePrVecVec &x,
                       const TDoubleVec &y,
                       const TSizeUSet &mask,
                       CSparseMatrix &xMasked,
                       TDoubleVec &yMasked)
{
    xMasked = CSparseMatrix();
    yMasked.clear();

    if (x.empty())
    {
        return;
    }

    std::size_t n = x.size();
    std::size_t m = n - mask.size();
    LOG_TRACE("n = " << n);

    TSizeSizePrDoublePrVec xTranspose;
    yMasked.reserve(m);
    std::size_t rows = m;
    std::size_t columns = 0u;
    for (std::size_t i = 0u, i_ = 0u; i < n; ++i)
    {
        if (mask.count(i) == 0)
        {
            for (std::size_t j = 0u, d = x[i].size(); j < d; ++j)
            {
                std::size_t j_ = x[i][j].first;
                xTranspose.push_back(TSizeSizePrDoublePr(TSizeSizePr(j_, i_),
                                                         x[i][j].second));
                columns = std::max(columns, j_ + 1);
            }
            yMasked.push_back(y[i]);
            ++i_;
        }
    }
    for (std::size_t i = 0u; i < rows; ++i)
    {
        xTranspose.emplace_back(TSizeSizePr(columns, i), 1.0);
    }
    CSparseMatrix tmp(rows, columns + 1, xTranspose);
    xMasked.swap(tmp);
}

//! Learn the regression parameters for the training data (\p x, \p y).
//!
//! This chooses between full and incremental CLG implementations
//! based on whether \p beta is pre-populated.
//!
//! \param[in] x The feature vectors.
//! \param[in] y The feature vector labels.
//! \param[in] lambda The precision of the Laplace prior.
//! \param[out] beta Filled in with the learned regression parameters.
template<typename MATRIX>
bool learn(const MATRIX &x,
           const TDoubleVec &y,
           const TDoubleVec &lambda,
           TDoubleVec &beta)
{
    using namespace lasso_logistic_regression_detail;

    // TODO parameters.
    CCyclicCoordinateDescent clg(100, 0.001);

    // Compute the regression parameters.
    std::size_t numberIterations;
    if (beta.empty())
    {
        if (!clg.run(x, y, lambda, beta, numberIterations))
        {
            LOG_ERROR("Failed to solve for parameters");
            return false;
        }
    }
    else
    {
        if (!clg.runIncremental(x, y, lambda, beta, numberIterations))
        {
            LOG_ERROR("Failed to solve for parameters");
            return false;
        }
    }
    LOG_TRACE("Solved for parameters using "
              << numberIterations << " iterations");

    return true;
}

//! Extract a matrix element from dense storage.
double element(double xij)
{
    return xij;
}

//! Extract a matrix element from sparse storage.
double element(const TSizeDoublePr &xij)
{
    return xij.second;
}

//! Not strictly speaking the square \f$L_{2,2}\f$ norm, but closely
//! related. In particular, this is defined as:
//! <pre class="fragment">
//!   \f$\displaystyle \frac{1}{nd}\sum_{i,j}{ x_{i,j}^2 }\f$
//! </pre>
//!
//! Here, \f$n\f$ is the number of rows and \f$d\f$ is the number
//! of columns.
template<typename STORAGE>
double l22Norm(const STORAGE &x)
{
    typedef CBasicStatistics::SSampleMean<double>::TAccumulator TMeanAccumulator;
    TMeanAccumulator result;
    for (std::size_t i = 0u; i < x.size(); ++i)
    {
        for (std::size_t j = 0u; j < x[i].size(); ++j)
        {
            double xij = element(x[i][j]);
            result.add(xij * xij);
        }
    }
    return CBasicStatistics::mean(result);
}

//! \brief Implements the 2-fold cross validation optimization
//! objective.
//!
//! DESCRIPTION:\n
//! This computes the log-likelihood of a collection of two splits
//! of the data for specified values of the lambda parameter. (Any
//! number of random 2-splits of the data may be added, via the
//! addSplit member function.) In particular, labeling one subset
//! train and one test, this first trains a logistic regression
//! model on train and evaluates the likelihood of the test set
//! then relabels the subsets and repeats.
//!
//! It is up to the caller to ensure that the models can be trained,
//! for example by ensuring that both subsets contain both positive
//! and negative examples. This can be achieved by randomly splitting
//! these sets independently.
template<typename MATRIX>
class C2FoldCrossValidatedLogLikelihood
{
    public:
        typedef double result_type;

    public:
        C2FoldCrossValidatedLogLikelihood(std::size_t d) :
            m_D(d + 1),
            m_Splits(0)
        {}

        //! Add a 2-split of the training data.
        void addSplit(MATRIX &xTrain,
                      TDoubleVec &yTrain,
                      MATRIX &xTest,
                      TDoubleVec &yTest)
        {
            if (xTrain.rows() != m_D || xTest.rows() != m_D)
            {
                LOG_ERROR("Bad training data: |train| = " << xTrain.rows()
                          << ", |test| = " << xTest.rows()
                          << ", D = " << m_D);
                return;
            }
            ++m_Splits;
            m_X[0].push_back(MATRIX());
            m_X[0].back().swap(xTrain);
            m_Y[0].push_back(TDoubleVec());
            m_Y[0].back().swap(yTrain);
            m_X[1].push_back(MATRIX());
            m_X[1].back().swap(xTest);
            m_Y[1].push_back(TDoubleVec());
            m_Y[1].back().swap(yTest);
        }

        //! Calculate the 2-fold cross-validation objective for the
        //! prior precision \p lambda.
        double operator()(double lambda) const
        {
            m_Beta.clear();
            m_Lambda.assign(m_D, lambda);
            double result = 0.0;
            for (std::size_t j = 0u; j < m_Splits; ++j)
            {
                for (std::size_t i = 0u; i < 2; ++i)
                {
                    learn(m_X[i][j], m_Y[i][j], m_Lambda, m_Beta);
                    result += logLikelihood(m_X[(i+1) % 2][j],
                                            m_Y[(i+1) % 2][j],
                                            m_Lambda,
                                            m_Beta);
                }
            }
            return result;
        }

    private:
        typedef std::vector<MATRIX> TMatrixVec;

    private:
        //! The feature vector dimension.
        std::size_t m_D;
        //! The number of 2-splits.
        std::size_t m_Splits;
        //! The feature vectors of the 2-splits.
        TMatrixVec m_X[2];
        //! The feature vector labels of the 2-splits.
        TDoubleVecVec m_Y[2];
        //! A placeholder for lambda so that it doesn't need to be
        //! re-initialized on each call to operator().
        mutable TDoubleVec m_Lambda;
        //! A placeholder for beta so that it doesn't need to be
        //! re-initialized on each call to operator().
        mutable TDoubleVec m_Beta;
};

} // unnamed::

////// CLassoLogisticRegression //////

template<typename STORAGE>
CLassoLogisticRegression<STORAGE>::CLassoLogisticRegression(void) :
        m_X(),
        m_D(0),
        m_Y(),
        m_Lambda(1.0),
        m_Beta()
{
}

template<typename STORAGE>
template<typename MATRIX>
void CLassoLogisticRegression<STORAGE>::doLearnHyperparameter(EHyperparametersStyle style)
{
    if (m_X.empty())
    {
        return;
    }

    typedef std::vector<std::size_t> TSizeVec;

    std::size_t n = m_X.size();
    LOG_DEBUG("d = " << m_D << ", n = " << n);

    double lambda = ::sqrt(l22Norm(m_X) / 2.0);
    m_Lambda = lambda;
    if (n <= 1)
    {
        return;
    }
    switch (style)
    {
    case E_LambdaNormBased:      return;
    case E_LambdaCrossValidated: break;
    }

    // Set up the cross-validation optimization objective.
    // (Note, we need to ensure we have positive and negative
    // examples in both train and test sets so we treat these
    // indices separately.)
    C2FoldCrossValidatedLogLikelihood<MATRIX> objective(m_D);
    TSizeVec positive;
    TSizeVec negative;
    positive.reserve(n);
    negative.reserve(n);
    for (std::size_t i = 0u; i < n; ++i)
    {
        (m_Y[i] > 0.0 ? positive : negative).push_back(i);
    }
    if (positive.size() <= 1 || negative.size() <= 1)
    {
        LOG_WARN("Can't cross-validate: insufficient "
                 << (positive.size() <= 1 ? "" : "un")
                 << "interesting examples provided");
        return;
    }
    for (std::size_t i = 0u, np = positive.size(), nn = negative.size(); i < 2; ++i)
    {
        CSampling::random_shuffle(positive.begin(), positive.end());
        CSampling::random_shuffle(negative.begin(), negative.end());

        TSizeUSet maskTrain;
        maskTrain.insert(positive.begin(), positive.begin() + np / 2);
        maskTrain.insert(negative.begin(), negative.begin() + nn / 2);
        MATRIX xTrain;
        TDoubleVec yTrain;
        setupTrainingData(m_X, m_Y, maskTrain, xTrain, yTrain);

        TSizeUSet maskTest;
        maskTest.insert(positive.begin() + np / 2, positive.end());
        maskTest.insert(negative.begin() + nn / 2, negative.end());
        MATRIX xTest;
        TDoubleVec yTest;
        setupTrainingData(m_X, m_Y, maskTest, xTest, yTest);

        objective.addSplit(xTrain, yTrain, xTest, yTest);
    }

    double scales[]         = { 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0 };
    double logLikelihoods[] = { 0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,   0.0 };

    double min = lambda / 10.0;
    for (std::size_t i = 0u; i < boost::size(scales); ++i)
    {
        logLikelihoods[i] = objective(scales[i] * min);
    }
    LOG_TRACE("log(L) = " << core::CContainerPrinter::print(logLikelihoods));

    double *max = std::max_element(boost::begin(logLikelihoods),
                                   boost::end(logLikelihoods));
    ptrdiff_t a = std::max(max - logLikelihoods - 1, ptrdiff_t(0));
    ptrdiff_t b = std::min(max - logLikelihoods + 1, ptrdiff_t(7));
    LOG_TRACE("a = " << a << ", b = " << b);

    std::size_t maxIterations = boost::size(scales) / 2;
    double logLikelihood;
    CSolvers::maximize(scales[a] * lambda, scales[b] * lambda,
                       logLikelihoods[a], logLikelihoods[b],
                       objective, 0.0, maxIterations,
                       lambda, logLikelihood);
    LOG_TRACE("lambda = " << lambda << " log(L(lambda)) = " << logLikelihood);

    m_Lambda = logLikelihood > *max ?
               lambda : scales[max - logLikelihoods] * min;
}

template<typename STORAGE>
template<typename MATRIX>
bool CLassoLogisticRegression<STORAGE>::doLearn(CLogisticRegressionModel &result)
{
    result = CLogisticRegressionModel();

    if (!this->sanityChecks())
    {
        return false;
    }

    MATRIX x;
    TDoubleVec y;
    TSizeUSet excludeNone;
    setupTrainingData(m_X, m_Y, excludeNone, x, y);
    TDoubleVec lambda(m_D, m_Lambda);
    TDoubleVec beta(m_Beta);
    if (!learn(x, y, lambda, beta))
    {
        return false;
    }
    m_Beta.swap(beta);

    // Create the model.
    TSizeDoublePrVec sparse;
    sparse.reserve(std::count_if(m_Beta.begin(), m_Beta.end(),
                                 boost::bind(std::greater<double>(), _1, 0.0)));
    for (std::size_t j = 0u; j < m_D; ++j)
    {
        if (m_Beta[j] > 0.0)
        {
            sparse.emplace_back(j, m_Beta[j]);
        }
    }
    CLogisticRegressionModel model(m_Beta[m_D], sparse);
    result.swap(model);

    return true;
}

template<typename STORAGE>
bool CLassoLogisticRegression<STORAGE>::sanityChecks(void) const
{
    if (m_Y.empty())
    {
        LOG_WARN("No training data");
        return false;
    }

    bool positive = false;
    bool negative = false;
    for (std::size_t i = 0u;
         (!positive || !negative) && i < m_Y.size();
         ++i)
    {
        (m_Y[i] < 0.0 ? negative : positive) = true;
    }
    if (!negative || !positive)
    {
        LOG_WARN("Only " << (negative ? "un" : "")
                  << "interesting examples provided: problem is ill posed");
        return false;
    }
    return true;
}


////// CLassoLogisticRegressionDense //////

void CLassoLogisticRegressionDense::addTrainingData(const TDoubleVec &x,
                                                    bool interesting)
{
    if (this->x().empty())
    {
        this->d() = x.size();
    }
    if (x.size() != this->d())
    {
        LOG_ERROR("Ignoring inconsistent training data |x| = "
                  << x.size() << ", D = " << this->x()[0].size());
        return;
    }
    this->x().push_back(x);
    this->y().push_back(interesting ? +1.0 : -1.0);
}

void CLassoLogisticRegressionDense::learnHyperparameter(EHyperparametersStyle style)
{
    this->doLearnHyperparameter<CDenseMatrix>(style);
}

bool CLassoLogisticRegressionDense::learn(CLogisticRegressionModel &result)
{
    return this->doLearn<CDenseMatrix>(result);
}


////// CLassoLogisticRegressionSparse //////

void CLassoLogisticRegressionSparse::addTrainingData(const TSizeDoublePrVec &x,
                                                     bool interesting)
{
    for (std::size_t i = 0u; i < x.size(); ++i)
    {
        this->d() = std::max(this->d(), x[i].first);
    }
    this->x().push_back(x);
    this->y().push_back(interesting ? +1.0 : -1.0);
}

void CLassoLogisticRegressionSparse::learnHyperparameter(EHyperparametersStyle style)
{
    this->doLearnHyperparameter<CSparseMatrix>(style);
}

bool CLassoLogisticRegressionSparse::learn(CLogisticRegressionModel &result)
{
    return this->doLearn<CSparseMatrix>(result);
}

}
}
