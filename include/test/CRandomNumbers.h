/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_test_CRandomNumbers_h
#define INCLUDED_ml_test_CRandomNumbers_h

#include <test/ImportExport.h>

#include <maths/CLinearAlgebraFwd.h>
#include <maths/CPRNG.h>

#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include <boost/shared_ptr.hpp>

#include <vector>

namespace ml
{
namespace test
{

//! \brief Creates random numbers from a variety of distributions.
class TEST_EXPORT CRandomNumbers
{
    public:
        using TDoubleVec = std::vector<double>;
        using TDoubleVecVec = std::vector<TDoubleVec>;
        using TUIntVec = std::vector<unsigned int>;
        using TSizeVec = std::vector<std::size_t>;
        using TStrVec = std::vector<std::string>;
        using TGenerator = maths::CPRNG::CXorShift1024Mult;
        using TGeneratorPtr = boost::shared_ptr<TGenerator>;

    public:
        //! A uniform generator on the interval [a,b].
        class TEST_EXPORT CUniform0nGenerator
        {
            public:
                CUniform0nGenerator(const TGenerator &generator);

                std::size_t operator()(std::size_t n) const;

            private:
                TGeneratorPtr m_Generator;
        };

    public:
        //! \brief Generate random samples from the specified distribution
        //! using a custom random number generator.
        template<typename RNG,
                 typename Distribution,
                 typename Container>
        static void generateSamples(RNG &randomNumberGenerator,
                                    const Distribution &distribution,
                                    std::size_t numberSamples,
                                    Container &samples);

        //! Shuffle the elements of a sequence using a random number generator.
        //!
        //! Reorders the elements in the range \p [first,last) using the
        //! internal random number generator to provide a random distribution.
        //!
        //! \note We provide our own implementation of std::random_shuffle
        //! based on the libc++ implementation because this is different from
        //! the libstdc++ implementation which can cause platform specific test
        //! failures.
        template<typename ITR>
        void random_shuffle(ITR first, ITR last);

        //! Generate normal random samples with the specified mean and
        //! variance using the default random number generator.
        void generateNormalSamples(double mean,
                                   double variance,
                                   std::size_t numberSamples,
                                   TDoubleVec &samples);

        //! Generate multivariate normal random samples with the specified
        //! mean and covariance matrix the default random number generator.
        void generateMultivariateNormalSamples(const TDoubleVec &mean,
                                               const TDoubleVecVec &covariances,
                                               std::size_t numberSamples,
                                               TDoubleVecVec &samples);

        //! Generate Poisson random samples with the specified rate using
        //! the default random number generator.
        void generatePoissonSamples(double rate,
                                    std::size_t numberSamples,
                                    TUIntVec &samples);

        //! Generate Student's t random samples with the specified degrees
        //! freedom using the default random number generator.
        void generateStudentsSamples(double degreesFreedom,
                                     std::size_t numberSamples,
                                     TDoubleVec &samples);

        //! Generate log-normal random samples with the specified location
        //! and scale using the default random number generator.
        void generateLogNormalSamples(double location,
                                      double squareScale,
                                      std::size_t numberSamples,
                                      TDoubleVec &samples);

        //! Generate uniform random samples in the interval [a,b) using
        //! the default random number generator.
        void generateUniformSamples(double a,
                                    double b,
                                    std::size_t numberSamples,
                                    TDoubleVec &samples);

        //! Generate uniform integer samples from the the set [a, a+1, ..., b)
        //! using the default random number generator.
        void generateUniformSamples(std::size_t a,
                                    std::size_t b,
                                    std::size_t numberSamples,
                                    TSizeVec &samples);

        //! Generate gamma random samples with the specified shape and rate
        //! using the default random number generator.
        void generateGammaSamples(double shape,
                                  double scale,
                                  std::size_t numberSamples,
                                  TDoubleVec &samples);

        //! Generate multinomial random samples on the specified categories
        //! using the default random number generator.
        void generateMultinomialSamples(const TDoubleVec &categories,
                                        const TDoubleVec &probabilities,
                                        std::size_t numberSamples,
                                        TDoubleVec &samples);

        //! Generate random samples from a Diriclet distribution with
        //! concentration parameters \p concentrations.
        void generateDirichletSamples(const TDoubleVec &concentrations,
                                      std::size_t numberSamples,
                                      TDoubleVecVec &samples);

        //! Generate a collection of random words of specified length using
        //! the default random number generator.
        void generateWords(std::size_t length,
                           std::size_t numberSamples,
                           TStrVec &samples);

        //! Generate a collection of |\p sizes| random mean vectors and
        //! covariance matrices and a collection of samples from those
        //! distributions.
        //!
        //! \param[in] sizes The number of points to generate from each
        //! cluster.
        //! \param[out] means Filled in with the distribution mean for
        //! each cluster.
        //! \param[out] covariances Filled in with the distribution covariance
        //! matrix for each cluster.
        //! \param[out] points Filled in with the samples from each cluster.
        template<typename T, std::size_t N>
        void generateRandomMultivariateNormals(const TSizeVec &sizes,
                                               std::vector<maths::CVectorNx1<T, N> > &means,
                                               std::vector<maths::CSymmetricMatrixNxN<T, N> > &covariances,
                                               std::vector<std::vector<maths::CVectorNx1<T, N> > > &points);

        //! Get a uniform generator in the range [0, n). This can be used
        //! in conjunction with std::random_shuffle if you want a seeded
        //! platform independent implementation.
        CUniform0nGenerator uniformGenerator();

        //! Throw away \p n random numbers.
        void discard(std::size_t n);

    private:
        //! The random number generator.
        TGenerator m_Generator;
};


}
}

#endif // INCLUDED_ml_test_CRandomNumbers_h

