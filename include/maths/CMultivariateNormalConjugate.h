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

#ifndef INCLUDED_ml_maths_CMultivariateNormalConjugate_h
#define INCLUDED_ml_maths_CMultivariateNormalConjugate_h

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/Constants.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStringUtils.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CChecksum.h>
#include <maths/CRestoreParams.h>
#include <maths/CEqualWithTolerance.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraEigen.h>
#include <maths/CLinearAlgebraPersist.h>
#include <maths/CLinearAlgebraTools.h>
#include <maths/CMathsFuncs.h>
#include <maths/CMathsFuncsForMatrixAndVectorTypes.h>
#include <maths/CMultivariatePrior.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/CSampling.h>
#include <maths/ProbabilityAggregators.h>

#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/optional.hpp>

#include <string>
#include <utility>
#include <vector>

namespace ml
{
namespace maths
{

//! \brief A conjugate prior distribution for a stationary multivariate Normal
//! process.
//!
//! DESCRIPTION:\n
//! Our normal process is a stationary process described by Y, where Y is a
//! multivariate normal R.V. with unknown mean vector and precision matrix.
//!
//! The parameters of Y are modeled as a joint Wishart-normal distribution (which
//! is the conjugate prior for a multivariate normal with unknown mean and precision).
//!
//! All prior distributions implement a process whereby they relax back to the
//! non-informative over some period without update (see propagateForwardsByTime).
//! The rate at which they relax is controlled by the decay factor supplied to the
//! constructor.
//!
//! IMPORTANT: Other than for testing, this class should not be constructed
//! directly. Creation of objects is managed by CMultivariateNormalConjugateFactory.
//!
//! IMPLEMENTATION DECISIONS:\n
//! All derived priors are templated on their size. In practice, it is very hard
//! to estimate densities in more than a small number of dimensions. We achieve
//! this by partitioning the matrix into block diagonal form with one prior used
//! to model each block. Therefore, we never need priors for more than a small
//! number of dimensions. This means that we template them on their size so we
//! can use stack (mathematical) vectors and matrices.
//!
//! All priors are derived from CMultivariatePrior which defines the contract that
//! is used by composite priors. This allows us to select the most appropriate model
//! for the data when using one-of-n composition (see CMultivariateOneOfNPrior).
//! From a design point of view this is the composite pattern.
template<std::size_t N>
class CMultivariateNormalConjugate : public CMultivariatePrior
{
    public:
        //! See core::CMemory.
        static bool dynamicSizeAlwaysZero(void) { return true; }

        typedef std::vector<double> TDoubleVec;
        typedef CVectorNx1<double, N> TPoint;
        typedef std::vector<TPoint> TPointVec;
        typedef core::CSmallVector<TPoint, 4> TPoint4Vec;
        typedef CSymmetricMatrixNxN<double, N> TMatrix;
        typedef std::vector<TMatrix> TMatrixVec;
        typedef typename CBasicStatistics::SSampleCovariances<double, N> TCovariance;

        // Lift all overloads of into scope.
        //{
        using CMultivariatePrior::addSamples;
        using CMultivariatePrior::print;
        //}

    private:
        typedef typename SDenseVector<TPoint>::Type TDenseVector;
        typedef typename SDenseMatrix<TPoint>::Type TDenseMatrix;

    public:
        //! \name Life-cycle
        //@{
        //!  \param[in] dataType The type of data being modeled (see maths_t::EDataType
        //! for details).
        //! \param[in] gaussianMean The mean of the normal component of the prior.
        //! \param[in] gaussianPrecision The precision of the normal component of
        //! the prior.
        //! \param[in] wishartDegreesFreedom The degrees freedom of Wishart component
        //! of the prior.
        //! \param[in] wishartScaleMatrix The scale matrix of Wishart component
        //! of the prior.
        //! \param[in] decayRate The rate at which to revert to non-informative.
        CMultivariateNormalConjugate(maths_t::EDataType dataType,
                                     const TPoint &gaussianMean,
                                     const TPoint &gaussianPrecision,
                                     double wishartDegreesFreedom,
                                     const TMatrix &wishartScaleMatrix,
                                     double decayRate = 0.0) :
                CMultivariatePrior(dataType, decayRate),
                m_GaussianMean(gaussianMean),
                m_GaussianPrecision(gaussianPrecision),
                m_WishartDegreesFreedom(wishartDegreesFreedom),
                m_WishartScaleMatrix(wishartScaleMatrix)
        {}

        //! Construct from sample central moments.
        CMultivariateNormalConjugate(maths_t::EDataType dataType,
                                     const TCovariance &covariance,
                                     double decayRate = 0.0) :
                CMultivariatePrior(dataType, decayRate),
                m_GaussianMean(CBasicStatistics::mean(covariance)),
                m_GaussianPrecision(covariance.s_Count),
                m_WishartDegreesFreedom(this->smallest(covariance.s_Count.template toVector<TDouble10Vec>())),
                m_WishartScaleMatrix(covariance.s_Count * covariance.s_Covariances)
        {
            this->numberSamples(CBasicStatistics::count(covariance));
        }

        //! Construct from part of a state document.
        CMultivariateNormalConjugate(const SDistributionRestoreParams &params,
                                     core::CStateRestoreTraverser &traverser) :
                CMultivariatePrior(params.s_DataType, params.s_DecayRate)
        {
            traverser.traverseSubLevel(boost::bind(&CMultivariateNormalConjugate::acceptRestoreTraverser, this, _1));
        }

        virtual ~CMultivariateNormalConjugate(void) {}

        // Default copy constructor and assignment operator work.

        //! Create an instance of a non-informative prior.
        //!
        //! \param[in] dataType The type of data being modeled (see maths_t::EDataType
        //! for details).
        //! \param[in] decayRate The rate at which to revert to the non-informative prior.
        //! \return A non-informative prior.
        static CMultivariateNormalConjugate nonInformativePrior(maths_t::EDataType dataType,
                                                                double decayRate = 0.0)
        {
            return CMultivariateNormalConjugate(dataType,
                                                NON_INFORMATIVE_MEAN,
                                                TPoint(NON_INFORMATIVE_PRECISION),
                                                NON_INFORMATIVE_DEGREES_FREEDOM,
                                                NON_INFORMATIVE_SCALE,
                                                decayRate);
        }
        //@}

        //! \name Prior Contract
        //@{
        //! Create a copy of the prior.
        //!
        //! \warning Caller owns returned object.
        virtual CMultivariateNormalConjugate *clone(void) const
        {
            return new CMultivariateNormalConjugate(*this);
        }

        //! Get the dimension of the prior.
        std::size_t dimension(void) const
        {
            return N;
        }

        //! Reset the prior to non-informative.
        virtual void setToNonInformative(double /*offset = 0.0*/, double decayRate = 0.0)
        {
            *this = nonInformativePrior(this->dataType(), decayRate);
        }

        //! No-op.
        virtual void adjustOffset(const TWeightStyleVec &/*weightStyles*/,
                                  const TDouble10Vec1Vec &/*samples*/,
                                  const TDouble10Vec4Vec1Vec &/*weights*/)
        {
        }

        //! Update the prior with a collection of independent samples from the
        //! process.
        //!
        //! \param[in] weightStyles Controls the interpretation of the weight(s)
        //! that are associated with each sample. See maths_t::ESampleWeightStyle
        //! for more details.
        //! \param[in] samples A collection of samples of the process.
        //! \param[in] weights The weights of each sample in \p samples.
        virtual void addSamples(const TWeightStyleVec &weightStyles,
                                const TDouble10Vec1Vec &samples,
                                const TDouble10Vec4Vec1Vec &weights)
        {
            if (samples.empty())
            {
                return;
            }
            if (!this->check(samples, weights))
            {
                return;
            }

            this->CMultivariatePrior::addSamples(weightStyles, samples, weights);

            // Note that if either count weight or Winsorisation weights are supplied
            // the weight of the sample x(i) is interpreted as its count, so for example
            // updating with {(x, 2)} is equivalent to updating with {x, x}.
            //
            // If the data are discrete then we approximate the discrete distribution
            // by saying it is uniform on the intervals [n,n+1] for each integral n.
            // This is like saying that the data are samples from:
            //   X' = X + Z
            //
            // where,
            //   {[Z]_i} are IID uniform in the interval [0,1].
            //
            // We care about the limiting behaviour of the filter, i.e. as the number
            // of samples n -> inf. In this case, the law of large numbers give that:
            //   mean(x(i) + z(i))
            //     -> 1/n * Sum_i( x(i) ) + E[Z]
            //
            // and
            //   cov(x(i) + z(i))
            //      = Sum_i( (x(i) + z(i) - 1/n * Sum_i( x(i) + z(i) ))'
            //               (x(i) + z(i) - 1/n * Sum_i( x(i) + z(i) )) )
            //     -> Sum_i( (x(i) - 1/n * Sum_j( x(j) ) + z(i) - E[Z])'
            //               (x(i) - 1/n * Sum_j( x(j) ) + z(i) - E[Z]) )
            //     -> cov(x(i)) + n * E[(Z - E[Z])'(Z - E[Z])]
            //
            // Since Z is uniform on the interval [0,1]
            //   E[Z] = 1/2 1
            //   E[(Z - E[Z])^2] = 1/12 I

            TPoint numberSamples(0.0);
            TCovariance covariancePost;
            try
            {
                for (std::size_t i = 0u; i < samples.size(); ++i)
                {
                    TPoint x(samples[i]);
                    TPoint n(maths_t::countForUpdate(N, weightStyles, weights[i]));
                    TPoint varianceScale =  TPoint(maths_t::seasonalVarianceScale(N, weightStyles, weights[i]))
                                          * TPoint(maths_t::countVarianceScale(N, weightStyles, weights[i]));
                    numberSamples += n;
                    covariancePost.add(x, n / varianceScale);
                }
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Failed to update likelihood: " << e.what());
                return;
            }
            TPoint scaledNumberSamples = covariancePost.s_Count;

            if (this->isInteger())
            {
                covariancePost.s_Mean += TPoint(0.5);
                covariancePost.s_Covariances += TPoint(1.0 / 12.0).diagonal();
            }

            if (m_WishartDegreesFreedom > 0.0)
            {
                TPoint scale = TPoint(1.0) / m_GaussianPrecision;
                TMatrix covariances = m_WishartScaleMatrix;
                scaleCovariances(scale, covariances);
                TCovariance covariancePrior = CBasicStatistics::accumulator(m_GaussianPrecision,
                                                                            m_GaussianMean,
                                                                            covariances);
                covariancePost += covariancePrior;
            }
            m_GaussianMean = CBasicStatistics::mean(covariancePost);
            m_GaussianPrecision += scaledNumberSamples;
            m_WishartDegreesFreedom += this->smallest(numberSamples.template toVector<TDouble10Vec>());
            m_WishartScaleMatrix = covariancePost.s_Covariances;
            scaleCovariances(covariancePost.s_Count, m_WishartScaleMatrix);

            // If the coefficient of variation of the data is too small we run
            // in to numerical problems. We truncate the variation by modeling
            // the impact of an actual variation (standard deviation divided by
            // mean) in the data of size MINIMUM_COEFFICIENT_OF_VARATION on the
            // prior parameters.

            if (!this->isNonInformative())
            {
                double truncatedMean = max(m_GaussianMean, TPoint(1e-8)).euclidean();
                double minimumDeviation =  truncatedMean
                                         * MINIMUM_COEFFICIENT_OF_VARIATION;
                double minimumDiagonal =  m_WishartDegreesFreedom
                                        * minimumDeviation * minimumDeviation;
                for (std::size_t i = 0u; i < N; ++i)
                {
                    m_WishartScaleMatrix(i, i) = std::max(m_WishartScaleMatrix(i, i), minimumDiagonal);
                }
            }

            LOG_TRACE("numberSamples = " << numberSamples
                      << ", scaledNumberSamples = " << scaledNumberSamples
                      << ", m_WishartDegreesFreedom = " << m_WishartDegreesFreedom
                      << ", m_WishartScaleMatrix = " << m_WishartScaleMatrix
                      << ", m_GaussianMean = " << m_GaussianMean
                      << ", m_GaussianPrecision = " << m_GaussianPrecision);

            if (this->isBad())
            {
                LOG_ERROR("Update failed (" << this->debug() << ")"
                          << ", samples = " << core::CContainerPrinter::print(samples)
                          << ", weights = " << core::CContainerPrinter::print(weights));
                this->setToNonInformative(this->offsetMargin(), this->decayRate());
            }
        }

        //! Update the prior for the specified elapsed time.
        virtual void propagateForwardsByTime(double time)
        {
            if (!CMathsFuncs::isFinite(time) || time < 0.0)
            {
                LOG_ERROR("Bad propagation time " << time);
                return;
            }

            if (this->isNonInformative())
            {
                // Nothing to be done.
                return;
            }

            double alpha = ::exp(-this->scaledDecayRate() * time);

            m_GaussianPrecision =         alpha  * m_GaussianPrecision
                                 + (1.0 - alpha) * TPoint(NON_INFORMATIVE_PRECISION);

            // The mean of the Wishart distribution is n V and the variance
            // is [V]_ij = n ( V_ij^2 + V_ii * V_jj), note V is the inverse
            // of teh scale matrix. We want to increase the variance while
            // holding its mean constant s.t. in the limit t -> inf var -> inf.
            // Choosing a factor f in the range [0, 1] and the update is as
            // follows:
            //   n -> f * n
            //   V -> f * V
            //
            // Thus the mean is unchanged and variance is increased by 1 / f.

            double factor = std::min(         (alpha  * m_WishartDegreesFreedom
                                     + (1.0 - alpha) * NON_INFORMATIVE_DEGREES_FREEDOM)
                                     / m_WishartDegreesFreedom, 1.0);

            m_WishartDegreesFreedom *= factor;
            m_WishartScaleMatrix *= factor;

            this->numberSamples(this->numberSamples() * alpha);

            LOG_TRACE("time = " << time
                      << ", alpha = " << alpha
                      << ", m_WishartDegreesFreedom = " << m_WishartDegreesFreedom
                      << ", m_WishartScaleMatrix = " << m_WishartScaleMatrix
                      << ", m_GaussianMean = " << m_GaussianMean
                      << ", m_GaussianPrecision = " << m_GaussianPrecision
                      << ", numberSamples = " << this->numberSamples());
        }

        //! Compute the univariate prior marginalizing over the variables
        //! \p marginalize and conditioning on the variables \p condition.
        //!
        //! \param[in] marginalize The variables to marginalize out.
        //! \param[in] condition The variables to condition on.
        //! \warning The caller owns the result.
        //! \note The variables are passed by the index of their dimension
        //! which must therefore be in range.
        //! \note The caller must specify dimension - 1 variables between
        //! \p marginalize and \p condition so the resulting distribution
        //! is univariate.
        virtual TUnivariatePriorPtrDoublePr univariate(const TSize10Vec &marginalize,
                                                       const TSizeDoublePr10Vec &condition) const
        {
            if (!this->check(marginalize, condition))
            {
                return TUnivariatePriorPtrDoublePr();
            }

            TSize10Vec i1;
            this->remainingVariables(marginalize, condition, i1);
            if (i1.size() != 1)
            {
                LOG_ERROR("Invalid variables for computing univariate distribution: "
                          << "marginalize '" << core::CContainerPrinter::print(marginalize) << "'"
                          << ", condition '" << core::CContainerPrinter::print(condition) << "'");
                return TUnivariatePriorPtrDoublePr();
            }

            maths_t::EDataType dataType = this->dataType();
            double decayRate = this->decayRate();
            if (this->isNonInformative())
            {
                return {TUnivariatePriorPtr(CNormalMeanPrecConjugate::nonInformativePrior(dataType, decayRate).clone()), 0.0};
            }

            double p = m_GaussianPrecision(i1[0]);
            double s = m_WishartDegreesFreedom / 2.0;
            double v = m_WishartDegreesFreedom - static_cast<double>(N) - 1.0;
            const TPoint &m = this->mean();
            TMatrix c = m_WishartScaleMatrix / v;

            double m1  = m(i1[0]);
            double c11 = c(i1[0], i1[0]);
            if (condition.empty())
            {
                return {TUnivariatePriorPtr(new CNormalMeanPrecConjugate(dataType, m1, p, s, c11 * v / 2.0, decayRate)), 0.0};
            }

            TSize10Vec condition_;
            condition_.reserve(condition.size() + 1);
            CDenseVector<double> xc(condition.size());
            this->unpack(condition, condition_, xc);

            try
            {
                std::size_t n = condition_.size();
                CDenseVector<double> m2  = projectedVector(condition_, m);
                condition_.push_back(i1[0]);
                CDenseMatrix<double> cp  = projectedMatrix(condition_, c);
                CDenseVector<double> c12 = cp.topRightCorner(n, 1);
                Eigen::JacobiSVD<CDenseMatrix<double>> c22(cp.topLeftCorner(n, n),
                                                           Eigen::ComputeThinU | Eigen::ComputeThinV);
                LOG_TRACE("c22 = " << cp.topLeftCorner(n, n)
                          << ", c12 = " << c12
                          << ", a = " << xc
                          << ", m2 = " << m2);

                CDenseVector<double> c22SolvexcMinusm2 = c22.solve(xc - m2);

                double mean = m1 + c12.transpose() * c22SolvexcMinusm2;
                double variance = std::max(c11 - c12.transpose() * c22.solve(c12),
                                           MINIMUM_COEFFICIENT_OF_VARIATION * ::fabs(mean));
                double weight = 0.5 * (::log(variance) - (xc - m2).transpose() * c22SolvexcMinusm2);
                LOG_TRACE("mean = " << mean
                          << ", variance = " << variance
                          << ", weight = " << weight);

                return {TUnivariatePriorPtr(
                            new CNormalMeanPrecConjugate(dataType, mean, p, s, variance * v / 2.0, decayRate)), weight};
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Failed to get univariate prior: " << e.what());
            }

            return TUnivariatePriorPtrDoublePr();
        }

        //! Compute the bivariate prior marginalizing over the variables
        //! \p marginalize and conditioning on the variables \p condition.
        //!
        //! \param[in] marginalize The variables to marginalize out.
        //! \param[in] condition The variables to condition on.
        //! \warning The caller owns the result.
        //! \note The variables are passed by the index of their dimension
        //! which must therefore be in range.
        //! \note It is assumed that the variables are in sorted order.
        //! \note The caller must specify dimension - 2 variables between
        //! \p marginalize and \p condition so the resulting distribution
        //! is univariate.
        virtual TPriorPtrDoublePr bivariate(const TSize10Vec &marginalize,
                                            const TSizeDoublePr10Vec &condition) const
        {
            if (N == 2)
            {
                return TPriorPtrDoublePr(boost::shared_ptr<CMultivariatePrior>(this->clone()), 0.0);
            }

            if (!this->check(marginalize, condition))
            {
                return TPriorPtrDoublePr();
            }

            TSize10Vec i1;
            this->remainingVariables(marginalize, condition, i1);
            if (i1.size() != 2)
            {
                return TPriorPtrDoublePr();
            }

            maths_t::EDataType dataType = this->dataType();
            double decayRate = this->decayRate();
            if (this->isNonInformative())
            {
                return {TPriorPtr(CMultivariateNormalConjugate<2>::nonInformativePrior(dataType, decayRate).clone()), 0.0};
            }

            typedef CVectorNx1<double, 2> TPoint2;
            typedef CSymmetricMatrixNxN<double, 2> TMatrix2;

            TPoint2 p;
            p(0) = m_GaussianPrecision(i1[0]);
            p(1) = m_GaussianPrecision(i1[1]);
            double f = m_WishartDegreesFreedom;
            const TPoint &m  = this->mean();
            const TMatrix &c = m_WishartScaleMatrix;

            TPoint2 m1;
            TMatrix2 c11;
            for (std::size_t i = 0u; i < 2; ++i)
            {
                m1(i) = m(i1[i]);
                for (std::size_t j = 0u; j < 2; ++j)
                {
                    c11(i,j) = c(i1[i],i1[j]);
                }
            }
            if (condition.empty())
            {
                return {TPriorPtr(new CMultivariateNormalConjugate<2>(dataType, m1, p, f, c11, decayRate)), 0.0};
            }

            TSize10Vec condition_;
            condition_.reserve(condition.size() + 1);
            CDenseVector<double> xc(condition.size());
            this->unpack(condition, condition_, xc);

            try
            {
                std::size_t n = condition_.size();
                CDenseVector<double> m2  = projectedVector(condition_, m);
                condition_.push_back(i1[0]);
                condition_.push_back(i1[1]);
                CDenseMatrix<double> cp  = projectedMatrix(condition_, c);
                CDenseVector<double> c12 = cp.topRightCorner(n, 1);
                Eigen::JacobiSVD<CDenseMatrix<double>> c22(cp.topLeftCorner(n, n),
                                                           Eigen::ComputeThinU | Eigen::ComputeThinV);
                LOG_TRACE("c22 = " << cp.topLeftCorner(n, n)
                          << ", c12 = " << c12
                          << ", a = " << xc
                          << ", m2 = " << m2);

                CDenseVector<double> c22SolvexcMinusm2 = c22.solve(xc - m2);

                TPoint2 mean(fromDenseVector(toDynamicDenseVector(m1) + c12.transpose() * c22SolvexcMinusm2));
                TMatrix2 covariance(fromDenseMatrix(toDynamicDenseMatrix(c11) - c12.transpose() * c22.solve(c12)));
                double weight;
                logDeterminant(covariance, weight, false);
                weight -= 0.5 * (xc - m2).transpose() * c22SolvexcMinusm2;
                LOG_TRACE("mean = " << mean << ", covariance = " << covariance);

                return {TPriorPtr(new CMultivariateNormalConjugate<2>(dataType, mean, p, f, covariance, decayRate)), weight};
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Failed to get univariate prior: " << e.what());
            }

            return TPriorPtrDoublePr();
        }

        //! Get the support for the marginal likelihood function.
        virtual TDouble10VecDouble10VecPr marginalLikelihoodSupport(void) const
        {
            return {TPoint::smallest().template toVector<TDouble10Vec>(),
                    TPoint::largest().template toVector<TDouble10Vec>()};
        }

        //! Get the mean of the marginal likelihood function.
        virtual TDouble10Vec marginalLikelihoodMean(void) const
        {
            return this->mean().template toVector<TDouble10Vec>();
        }

        //! Get the mode of the marginal likelihood function.
        virtual TDouble10Vec marginalLikelihoodMode(const TWeightStyleVec &/*weightStyles*/,
                                                    const TDouble10Vec4Vec &/*weights*/) const
        {
            return this->marginalLikelihoodMean();
        }

        //! Get the covariance matrix for the marginal likelihood.
        virtual TDouble10Vec10Vec marginalLikelihoodCovariance(void) const
        {
            return this->covarianceMatrix().template toVectors<TDouble10Vec10Vec>();
        }

        //! Get the diagonal of the covariance matrix for the marginal likelihood.
        virtual TDouble10Vec marginalLikelihoodVariances(void) const
        {
            return this->covarianceMatrix().template diagonal<TDouble10Vec>();
        }

        //! Calculate the log marginal likelihood function, integrating over the
        //! prior density function.
        //!
        //! \param[in] weightStyles Controls the interpretation of the weight(s)
        //! that are associated with each sample. See maths_t::ESampleWeightStyle
        //! for more details.
        //! \param[in] samples A collection of samples of the process.
        //! \param[in] weights The weights of each sample in \p samples.
        //! \param[out] result Filled in with the joint likelihood of \p samples.
        virtual maths_t::EFloatingPointErrorStatus
            jointLogMarginalLikelihood(const TWeightStyleVec &weightStyles,
                                       const TDouble10Vec1Vec &samples,
                                       const TDouble10Vec4Vec1Vec &weights,
                                       double &result) const
        {
            result = 0.0;

            if (samples.empty())
            {
                LOG_ERROR("Can't compute likelihood for empty sample set");
                return maths_t::E_FpFailed;
            }
            if (!this->check(samples, weights))
            {
                return maths_t::E_FpFailed;
            }

            result = boost::numeric::bounds<double>::lowest();

            if (this->isNonInformative())
            {
                // The non-informative likelihood is improper and effectively
                // zero everywhere. We use minus max double because
                // log(0) = HUGE_VALUE, which causes problems for Windows.
                // Calling code is notified when the calculation overflows
                // and should avoid taking the exponential since this will
                // underflow and pollute the floating point environment. This
                // may cause issues for some library function implementations
                // (see fe*exceptflag for more details).
                return maths_t::E_FpOverflowed;
            }

            // We evaluate the integral over the latent variable by Monte-Carlo
            // approximation.

            maths_t::EFloatingPointErrorStatus status;

            if (this->isInteger())
            {
                double logLikelihood;
                status = this->jointLogMarginalLikelihood(weightStyles, samples, TPoint(0.5), weights, logLikelihood);
                if (status != maths_t::E_FpNoErrors)
                {
                    return status;
                }

                double sum = 0.0;
                double n = 0.0;
                double maxLogLikelihood = logLikelihood;

                TDoubleVec z;
                CSampling::uniformSample(0.0, 1.0, 3 * N, z);
                for (std::size_t i = 0u; i < z.size(); i += N)
                {
                    status = this->jointLogMarginalLikelihood(weightStyles,
                                                              samples,
                                                              TPoint(&z[i], &z[i + N]),
                                                              weights,
                                                              logLikelihood);
                    if (status & maths_t::E_FpFailed)
                    {
                        return maths_t::E_FpFailed;
                    }
                    if (status & maths_t::E_FpOverflowed)
                    {
                        continue;
                    }

                    if (logLikelihood > maxLogLikelihood)
                    {
                        sum *= ::exp(maxLogLikelihood - logLikelihood);
                        maxLogLikelihood = logLikelihood;
                    }
                    sum += ::exp(logLikelihood - maxLogLikelihood);
                    n   += 1.0;
                }

                result = maxLogLikelihood + ::log(sum / n);
            }
            else
            {
                status = this->jointLogMarginalLikelihood(weightStyles, samples, TPoint(0.0), weights, result);
            }

            if (status & maths_t::E_FpFailed)
            {
                LOG_ERROR("Failed to compute log likelihood (" << this->debug() << ")");
                LOG_ERROR("samples = " << core::CContainerPrinter::print(samples));
                LOG_ERROR("weights = " << core::CContainerPrinter::print(weights));
            }
            else if (status & maths_t::E_FpOverflowed)
            {
                LOG_TRACE("Log likelihood overflowed for (" << this->debug() << ")");
                LOG_TRACE("samples = " << core::CContainerPrinter::print(samples));
                LOG_TRACE("weights = " << core::CContainerPrinter::print(weights));
            }
            return status;
        }

        //! Sample the marginal likelihood function.
        //!
        //! The marginal likelihood functions are sampled in quantile intervals
        //! of the generalized cumulative density function, specifically intervals
        //! between contours of constant probability density.
        //!
        //! The idea is to capture a set of samples that accurately and efficiently
        //! represent the information in the prior. Random sampling (although it
        //! has nice asymptotic properties) doesn't fulfill the second requirement:
        //! typically requiring many more samples than sampling in quantile intervals
        //! to capture the same amount of information.
        //!
        //! This is to allow us to transform one prior distribution into another
        //! completely generically and relatively efficiently, by updating the target
        //! prior with these samples. As such the prior needs to maintain a count of
        //! the number of samples to date so that it isn't over sampled.
        //!
        //! \param[in] numberSamples The number of samples required.
        //! \param[out] samples Filled in with samples from the prior.
        //! \note \p numberSamples is truncated to the number of samples received.
        virtual void sampleMarginalLikelihood(std::size_t numberSamples,
                                              TDouble10Vec1Vec &samples) const
        {
            samples.clear();

            if (numberSamples == 0 || this->numberSamples() == 0.0)
            {
                return;
            }

            if (this->isNonInformative())
            {
                // We can't sample the marginal likelihood directly. This should
                // only happen if we've had one sample so just return that sample.
                samples.push_back(m_GaussianMean.template toVector<TDouble10Vec>());
                return;
            }

            // We sample the moment matched Gaussian (to the marginal likelihood).
            // Clearly, E_{m*,p*}[ X ] = m. To calculate the covariance matrix we
            // note that:
            //   E_{m*,p*}[ (X - m)'(X - m) ]
            //       = E_{m*,p*}[ (X - m*)'(X - m*) - (m* - m)'(m* - m) ]
            //       = E_{p*}[ (1 + 1/t) (p* ^ -1) ]
            //       = (1 + 1/t) V ^ (-1) / (v - d - 1)
            //
            // In the last line we have used the fact that if X ~ W_d(V, n) and
            // Y = X^(-1), then Y ~ W_d^(-1)(V^(-1), n), i.e. the inverse Wishart
            // with the same degrees of freedom, but inverse scale matrix.
            //
            // See sampleGaussian for details on the sampling strategy.

            double d = static_cast<double>(N);
            double v = m_WishartDegreesFreedom - d - 1.0;
            TPoint mean(m_GaussianMean);
            TMatrix covariance(m_WishartScaleMatrix);
            for (std::size_t i = 0u; i < N; ++i)
            {
                if (m_GaussianPrecision(i) > 0.0 && v > 0.0)
                {
                    scaleCovariances(i, (1.0 - 1.0 / m_GaussianPrecision(i)) / v, covariance);
                }
            }
            TPointVec samples_;
            sampleGaussian(numberSamples, mean, covariance, samples_);
            samples.reserve(samples_.size());
            for (const auto &sample : samples_)
            {
                samples.push_back(sample.template toVector<TDouble10Vec>());
            }
        }

        //! Check if this is a non-informative prior.
        virtual bool isNonInformative(void) const
        {
            return m_WishartDegreesFreedom <= static_cast<double>(N + 1);
        }

        //! Get a human readable description of the prior.
        //!
        // \param[in] separator String used to separate priors.
        // \param[in,out] result Filled in with the description.
        virtual void print(const std::string &separator, std::string &result) const
        {
            result += "\n" + separator + " multivariate normal";
            if (this->isNonInformative())
            {
                result += " non-informative";
            }
            else
            {
                std::ostringstream mean;
                mean << this->mean();
                std::ostringstream covariance;
                covariance << this->covarianceMatrix();
                result += ":\n" + mean.str() + covariance.str();
                result += "\n" + separator;
            }
        }

        //! Get a checksum for this object.
        virtual uint64_t checksum(uint64_t seed = 0) const
        {
            seed = this->CMultivariatePrior::checksum(seed);
            seed = CChecksum::calculate(seed, m_GaussianMean);
            seed = CChecksum::calculate(seed, m_GaussianPrecision);
            seed = CChecksum::calculate(seed, m_WishartDegreesFreedom);
            return CChecksum::calculate(seed, m_WishartScaleMatrix);
        }

        //! Get the memory used by this component
        virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
        {
            mem->setName("CMultivariateNormalConjugate");
        }

        //! Get the memory used by this component
        virtual std::size_t memoryUsage(void) const
        {
            return 0;
        }

        //! Get the static size of this object - used for virtual hierarchies
        virtual std::size_t staticSize(void) const
        {
            return sizeof(*this);
        }

        //! Get the tag name for this prior.
        virtual std::string persistenceTag(void) const
        {
            return NORMAL_TAG + core::CStringUtils::typeToString(N);
        }

        //! Read parameters from \p traverser.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser)
        {
            do
            {
                const std::string &name = traverser.name();
                RESTORE_SETUP_TEARDOWN(DECAY_RATE_TAG,
                                       double decayRate,
                                       core::CStringUtils::stringToType(traverser.value(), decayRate),
                                       this->decayRate(decayRate))
                RESTORE_SETUP_TEARDOWN(NUMBER_SAMPLES_TAG,
                                       double numberSamples,
                                       core::CStringUtils::stringToType(traverser.value(), numberSamples),
                                       this->numberSamples(numberSamples))
                RESTORE(GAUSSIAN_MEAN_TAG, m_GaussianMean.fromDelimited(traverser.value()))
                RESTORE(GAUSSIAN_PRECISION_TAG, m_GaussianPrecision.fromDelimited(traverser.value()))
                RESTORE(WISHART_DEGREES_FREEDOM_TAG,
                        core::CStringUtils::stringToType(traverser.value(), m_WishartDegreesFreedom))
                RESTORE(WISHART_SCALE_MATRIX_TAG, m_WishartScaleMatrix.fromDelimited(traverser.value()))
            }
            while (traverser.next());

            return true;
        }

        //! Persist state by passing information to the supplied inserter
        virtual void acceptPersistInserter(core::CStatePersistInserter &inserter) const
        {
            inserter.insertValue(DECAY_RATE_TAG, this->decayRate(), core::CIEEE754::E_SinglePrecision);
            inserter.insertValue(NUMBER_SAMPLES_TAG,
                                 this->numberSamples(),
                                 core::CIEEE754::E_SinglePrecision);
            inserter.insertValue(GAUSSIAN_MEAN_TAG, m_GaussianMean.toDelimited());
            inserter.insertValue(GAUSSIAN_PRECISION_TAG, m_GaussianPrecision.toDelimited());
            inserter.insertValue(WISHART_DEGREES_FREEDOM_TAG,
                                 m_WishartDegreesFreedom,
                                 core::CIEEE754::E_DoublePrecision);
            inserter.insertValue(WISHART_SCALE_MATRIX_TAG, m_WishartScaleMatrix.toDelimited());
        }
        //@}

        //! \name Sampling
        //@{
        //! Randomly sample the covariance matrix prior.
        void randomSamplePrecisionMatrixPrior(std::size_t n, TMatrixVec &result)
        {
            // The prior on the precision matrix is Wishart with matrix V equal
            // to the inverse of the scale matrix and degrees freedom equal to
            // degrees freedom minus the data dimension. To sample from the Wishart
            // we use the Bartlett transformation. In particular, if X ~ W(V, n),
            // V = LL^t (i.e. the Cholesky factorization), then X = L A A^t L^t
            // with
            //       | c1   0  .  .  0  |
            //       | n21 c2        .  |
            //   A = |  .      .     .  |
            //       |  .         .  0  |
            //       | nd1  .  .  .  cd |
            //
            // Here, ci^2 ~ chi2(n - i + 1) and nij ~ N(0, 1).

            result.clear();

            if (this->isNonInformative())
            {
                return;
            }

            double d = static_cast<double>(N);
            double f = m_WishartDegreesFreedom - d - 1.0;
            LOG_TRACE("f = " << f);

            Eigen::JacobiSVD<TDenseMatrix> precision(toDenseMatrix(m_WishartScaleMatrix),
                                                     Eigen::ComputeFullU | Eigen::ComputeFullV);

            // Note we can extract the (non-zero vectors of the Cholesky
            // factorization by noting that U = V^t and multiplying each
            // column of U by the square root of the corresponding singular
            // value).

            std::size_t rank = static_cast<std::size_t>(precision.rank());

            TDenseVector diag = precision.singularValues();
            for (std::size_t i = 0u; i < rank; ++i)
            {
                diag(i) = 1.0 / ::sqrt(diag(i));
            }
            for (std::size_t i = rank; i < N; ++i)
            {
                diag(i) = 0.0;
            }
            TDenseMatrix L = TDenseMatrix::Zero(N, N);
            L.leftCols(rank) = precision.matrixU().leftCols(rank);
            L = L * diag.asDiagonal();
            LOG_TRACE("L = " << L);

            TDoubleVec chi;
            TDoubleVec chii;
            for (std::size_t i = 0u; i < rank; ++i)
            {
                chii.clear();
                CSampling::chiSquaredSample(f - static_cast<double>(i), n, chii);
                chi.insert(chi.end(), chii.begin(), chii.end());
            }
            TDoubleVec normal;
            CSampling::normalSample(0.0, 1.0, n * rank * (rank - 1) / 2, normal);

            TDenseMatrix A(N,N);
            for (std::size_t s = 0u; s < n; ++s)
            {
                A.setZero();
                for (std::size_t i = 0, k = 0u; i < rank; ++i)
                {
                    A(i,i) = ::sqrt(chi[i * n + s]);
                    for (std::size_t j = 0u; j < i; ++j, ++k)
                    {
                        A(i,j) = normal[(s * rank * (rank - 1)) / 2 + k];
                    }
                }
                result.emplace_back(fromDenseMatrix(L * A * A.transpose() * L.transpose()));
            }
        }

        //! Randomly sample from the marginal over the mean.
        void randomSampleMeanPrior(std::size_t n, TPointVec &result)
        {
            result.clear();

            if (this->isNonInformative())
            {
                return;
            }

            // The marginal distribution of the prior is multivariate t with
            // mean equal to the Gaussian prior mean and covariance matrix
            // equal to 1 / (f - d + 1) * S where S and f are the scale matrix
            // and degrees freedom of the Wishart prior and d is the data
            // dimension. We use the fact that a multivariate t variable
            // T(m, V, v) = m + (W)^(1/2) N(0, V), where m is its mean and
            // W ~ v / chi^2(v).

            double d = static_cast<double>(N);
            double f = m_WishartDegreesFreedom - d + 1.0;
            TPoint mean(m_GaussianMean);
            TMatrix covariance = m_WishartScaleMatrix;
            for (std::size_t i = 0u; i < N; ++i)
            {
                if (m_GaussianPrecision(i) > 0.0 && f > 0.0)
                {
                    scaleCovariances(i, 1.0 / (m_GaussianPrecision(i) * f), covariance);
                }
            }
            LOG_TRACE("mean = " << mean);
            LOG_TRACE("covariance = " << covariance);
            LOG_TRACE("fmin = " << f);

            TDoubleVec chi;
            CSampling::chiSquaredSample(f, n, chi);
            TPoint zero(0.0);
            CSampling::multivariateNormalSample(zero, covariance, n, result);

            for (std::size_t i = 0u; i < n; ++i)
            {
                result[i] = mean + (f / chi[i]) * result[i];
            }
        }

        //! Randomly sample from the predictive distribution.
        void randomSamplePredictive(std::size_t n, TPointVec &result)
        {
            result.clear();

            if (this->isNonInformative())
            {
                return;
            }

            // The predictive distribution is multivariate t with mean, m,
            // equal to the Gaussian prior mean, scale matrix, V, equal to
            // the Wishart prior scale matrix, and degrees freedom, f, equal
            // to its degrees of freedom. See randomSampleMeanPrior for how
            // to sample from multivariate t.

            double d = static_cast<double>(N);
            double f = m_WishartDegreesFreedom - d;
            TPoint mean(m_GaussianMean);

            TDoubleVec chi;
            CSampling::chiSquaredSample(f, n, chi);
            TPoint zero(0.0);
            CSampling::multivariateNormalSample(zero, m_WishartScaleMatrix, n, result);

            for (std::size_t i = 0u; i < n; ++i)
            {
                result[i] = mean + (f / chi[i]) * result[i];
            }
        }
        //@}

        //! Get the expected mean of the marginal likelihood.
        TPoint mean(void) const
        {
            return this->isInteger() ? m_GaussianMean - TPoint(0.5) : m_GaussianMean;
        }

        //! Get the covariance matrix for the marginal likelihood.
        TMatrix covarianceMatrix(void) const
        {
            // This can be found by change of variables from the prior on the
            // precision matrix. In particular, if X ~ W_d(V, n) and Y = X^(-1),
            // then Y ~ W_d^(-1)(V^(-1), n), i.e. the inverse Wishart with the
            // same degrees of freedom, but inverse scale matrix.

            double d = static_cast<double>(N);
            double v = m_WishartDegreesFreedom - d - 1.0;
            TMatrix covariance(m_WishartScaleMatrix);
            for (std::size_t i = 0u; i < N; ++i)
            {
                if (m_GaussianPrecision(i) > 0.0 && v > 0.0)
                {
                    scaleCovariances(i, (1.0 - 1.0 / m_GaussianPrecision(i)) / v, covariance);
                }
            }
            return covariance;
        }

        //! \name Test Functions
        //@{
        //! Get the expected precision matrix of the marginal likelhood.
        TMatrix precision(void) const
        {
            if (this->isNonInformative())
            {
                return TMatrix(0.0);
            }

            TMatrix result(m_WishartScaleMatrix / m_WishartDegreesFreedom);
            return TMatrix(fromDenseMatrix(toDenseMatrix(result).inverse()));
        }

        //! Check if two priors are equal to the specified tolerance.
        bool equalTolerance(const CMultivariateNormalConjugate &rhs,
                            unsigned int toleranceType,
                            double epsilon) const
        {
            LOG_DEBUG(m_GaussianMean << " " << rhs.m_GaussianMean);
            LOG_DEBUG(m_GaussianPrecision << " " << rhs.m_GaussianPrecision);
            LOG_DEBUG(m_WishartDegreesFreedom << " " << rhs.m_WishartDegreesFreedom);
            LOG_DEBUG(m_WishartScaleMatrix << " " << rhs.m_WishartScaleMatrix);

            CEqualWithTolerance<double> equalScalar(toleranceType, epsilon);
            CEqualWithTolerance<TPoint> equalVector(toleranceType, TPoint(epsilon));
            CEqualWithTolerance<TMatrix> equalMatrix(toleranceType, TMatrix(epsilon));

            return    equalVector(m_GaussianMean, rhs.m_GaussianMean)
                   && equalVector(m_GaussianPrecision, rhs.m_GaussianPrecision)
                   && equalScalar(m_WishartDegreesFreedom, rhs.m_WishartDegreesFreedom)
                   && equalMatrix(m_WishartScaleMatrix, rhs.m_WishartScaleMatrix);
        }
        //@}

    private:
        //! The mean parameter of a non-informative prior.
        static const TPoint NON_INFORMATIVE_MEAN;

        //! The precision parameter of a non-informative prior.
        static const double NON_INFORMATIVE_PRECISION;

        //! The degrees freedom of a non-informative prior.
        static const double NON_INFORMATIVE_DEGREES_FREEDOM;

        //! The scale matrix of a non-informative prior.
        static const TMatrix NON_INFORMATIVE_SCALE;

        //! The minimum degrees freedom for the Wishart distribution for
        //! which we'll treat the predictive distribution as Gaussian.
        static const double MINIMUM_GAUSSIAN_DEGREES_FREEDOM;

        //! \name State tags for model persistence.
        //@{
        static const std::string NUMBER_SAMPLES_TAG;
        static const std::string GAUSSIAN_MEAN_TAG;
        static const std::string GAUSSIAN_PRECISION_TAG;
        static const std::string WISHART_DEGREES_FREEDOM_TAG;
        static const std::string WISHART_SCALE_MATRIX_TAG;
        static const std::string DECAY_RATE_TAG;
        //@}

    private:
        //! Unpack the variable values on which to condition.
        void unpack(const TSizeDoublePr10Vec &condition,
                    TSize10Vec &condition_,
                    CDenseVector<double> &x) const
        {
            condition_.reserve(condition.size());
            for (std::size_t i = 0u; i < condition.size(); ++i)
            {
                condition_.push_back(condition[i].first);
                x(i) = condition[i].second;
            }
        }

        //! Compute the marginal likelihood for \p samples at the offset
        //! \p offset.
        maths_t::EFloatingPointErrorStatus
            jointLogMarginalLikelihood(const TWeightStyleVec &weightStyles,
                                       const TDouble10Vec1Vec &samples,
                                       const TPoint &offset,
                                       const TDouble10Vec4Vec1Vec &weights,
                                       double &result) const
        {
            // As usual, one can find the marginal likelihood by noting that
            // it is proportional to the ratio of the normalization factors
            // of the conjugate distribution before and after update with the
            // samples.

            double d = static_cast<double>(N);

            double numberSamples = 0.0;
            TCovariance covariancePost;
            double logCountVarianceScales = 0.0;
            try
            {
                TPoint m(this->marginalLikelihoodMean());
                for (std::size_t i = 0u; i < samples.size(); ++i)
                {
                    TPoint x(samples[i]);
                    TPoint n(maths_t::countForUpdate(N, weightStyles, weights[i]));
                    TPoint seasonalScale = sqrt(TPoint(maths_t::seasonalVarianceScale(N, weightStyles, weights[i])));
                    TPoint countVarianceScale(maths_t::countVarianceScale(N, weightStyles, weights[i]));
                    x = m + (x + offset - m) / seasonalScale;
                    numberSamples += this->smallest(n.template toVector<TDouble10Vec>());
                    covariancePost.add(x, n / countVarianceScale);
                    for (std::size_t j = 0u; j < N; ++j)
                    {
                        logCountVarianceScales -= 0.5 * ::log(countVarianceScale(j));
                    }
                }
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Failed to update likelihood: " << e.what());
                return maths_t::E_FpFailed;
            }
            TPoint scaledNumberSamples = covariancePost.s_Count;
            TCovariance covariancePrior =
                    CBasicStatistics::accumulator(m_WishartDegreesFreedom,
                                                  m_GaussianMean,
                                                  m_WishartScaleMatrix / m_WishartDegreesFreedom);
            covariancePost += covariancePrior;

            double logGaussianPrecisionPrior = 0.0;
            double logGaussianPrecisionPost  = 0.0;
            for (std::size_t i = 0u; i < N; ++i)
            {
                logGaussianPrecisionPrior += ::log(m_GaussianPrecision(i));
                logGaussianPrecisionPost  += ::log(m_GaussianPrecision(i) + scaledNumberSamples(i));
            }
            double wishartDegreesFreedomPrior = m_WishartDegreesFreedom;
            double wishartDegreesFreedomPost  = m_WishartDegreesFreedom + numberSamples;
            TMatrix wishartScaleMatrixPost = covariancePost.s_Covariances;
            scaleCovariances(covariancePost.s_Count, wishartScaleMatrixPost);
            double logDeterminantPrior;
            if (logDeterminant(m_WishartScaleMatrix, logDeterminantPrior, false) & maths_t::E_FpFailed)
            {
                LOG_ERROR("Failed to calculate log det " << m_WishartScaleMatrix);
                return maths_t::E_FpFailed;
            }
            double logDeterminantPost;
            if (logDeterminant(wishartScaleMatrixPost, logDeterminantPost) & maths_t::E_FpFailed)
            {
                LOG_ERROR("Failed to calculate log det " << wishartScaleMatrixPost);
                return maths_t::E_FpFailed;
            }

            try
            {
                double logGammaPostMinusPrior = 0.0;
                for (std::size_t i = 0u; i < N; ++i)
                {
                    logGammaPostMinusPrior +=
                            boost::math::lgamma(0.5 * (wishartDegreesFreedomPost - static_cast<double>(i)))
                          - boost::math::lgamma(0.5 * (wishartDegreesFreedomPrior - static_cast<double>(i)));
                }
                LOG_TRACE("numberSamples = " << numberSamples);
                LOG_TRACE("logGaussianPrecisionPrior = " << logGaussianPrecisionPrior
                          << ", logGaussianPrecisionPost  = " << logGaussianPrecisionPost);
                LOG_TRACE("wishartDegreesFreedomPrior = " << wishartDegreesFreedomPrior
                          << ", wishartDegreesFreedomPost  = " << wishartDegreesFreedomPost);
                LOG_TRACE("wishartScaleMatrixPrior = " << m_WishartScaleMatrix);
                LOG_TRACE("wishartScaleMatrixPost  = " << wishartScaleMatrixPost);
                LOG_TRACE("logDeterminantPrior = " << logDeterminantPrior
                          << ", logDeterminantPost = " << logDeterminantPost);
                LOG_TRACE("logGammaPostMinusPrior = " << logGammaPostMinusPrior);
                LOG_TRACE("logCountVarianceScales = " << logCountVarianceScales);

                result = 0.5 * (  wishartDegreesFreedomPrior * logDeterminantPrior
                                - wishartDegreesFreedomPost * logDeterminantPost
                                - d * (logGaussianPrecisionPost - logGaussianPrecisionPrior)
                                + (wishartDegreesFreedomPost - wishartDegreesFreedomPrior) * d * core::constants::LOG_TWO
                                + 2.0 * logGammaPostMinusPrior
                                - numberSamples * d * core::constants::LOG_TWO_PI
                                - logCountVarianceScales);
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Failed to calculate marginal likelihood: " << e.what());
                return maths_t::E_FpFailed;
            }
            return static_cast<maths_t::EFloatingPointErrorStatus>(CMathsFuncs::fpStatus(result));
        }

        //! Check that the state is valid.
        bool isBad(void) const
        {
            return    !CMathsFuncs::isFinite(m_GaussianMean)
                   || !CMathsFuncs::isFinite(m_GaussianPrecision)
                   || !CMathsFuncs::isFinite(m_WishartDegreesFreedom)
                   || !CMathsFuncs::isFinite(m_WishartScaleMatrix);
        }

        //! Full debug dump of the state of this prior.
        std::string debug(void) const
        {
            std::ostringstream result;
            result << std::scientific << std::setprecision(15)
                   << m_GaussianMean << " "
                   << m_GaussianPrecision << " "
                   << m_WishartDegreesFreedom << " "
                   << m_WishartScaleMatrix;
            return result.str();
        }

    private:
        //! The mean of the multivariate Gaussian prior.
        TPoint m_GaussianMean;

        //! The precision scale of the multivariate Gaussian prior.
        TPoint m_GaussianPrecision;

        //! The degrees freedom of the Wishart prior.
        double m_WishartDegreesFreedom;

        //! The scale matrix of the Wishart prior.
        TMatrix m_WishartScaleMatrix;
};

template<std::size_t N>
const std::string CMultivariateNormalConjugate<N>::NUMBER_SAMPLES_TAG("a");
template<std::size_t N>
const std::string CMultivariateNormalConjugate<N>::GAUSSIAN_MEAN_TAG("b");
template<std::size_t N>
const std::string CMultivariateNormalConjugate<N>::GAUSSIAN_PRECISION_TAG("c");
template<std::size_t N>
const std::string CMultivariateNormalConjugate<N>::WISHART_DEGREES_FREEDOM_TAG("d");
template<std::size_t N>
const std::string CMultivariateNormalConjugate<N>::WISHART_SCALE_MATRIX_TAG("e");
template<std::size_t N>
const std::string CMultivariateNormalConjugate<N>::DECAY_RATE_TAG("f");
template<std::size_t N>
const typename CMultivariateNormalConjugate<N>::TPoint CMultivariateNormalConjugate<N>::NON_INFORMATIVE_MEAN = TPoint(0);
template<std::size_t N>
const double CMultivariateNormalConjugate<N>::NON_INFORMATIVE_PRECISION(0.0);
template<std::size_t N>
const double CMultivariateNormalConjugate<N>::NON_INFORMATIVE_DEGREES_FREEDOM(0.0);
template<std::size_t N>
const typename CMultivariateNormalConjugate<N>::TMatrix CMultivariateNormalConjugate<N>::NON_INFORMATIVE_SCALE = TMatrix(0);
template<std::size_t N>
const double CMultivariateNormalConjugate<N>::MINIMUM_GAUSSIAN_DEGREES_FREEDOM(100.0);

}
}

#endif // INCLUDED_ml_maths_CMultivariateNormalConjugate_h
