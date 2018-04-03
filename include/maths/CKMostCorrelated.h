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

#ifndef INCLUDED_ml_maths_CKMostCorrelated_h
#define INCLUDED_ml_maths_CKMostCorrelated_h

#include <maths/CBasicStatistics.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CPackedBitVector.h>
#include <maths/CPRNG.h>
#include <maths/ImportExport.h>

#include <boost/unordered_map.hpp>

#include <utility>
#include <vector>

#include <stdint.h>


namespace ml
{
namespace maths
{

//! \brief Randomized linear complexity search for the most correlated
//! pairs of variables.
//!
//! DESCRIPTION:\n
//! This uses random projections to find a specified number \f$M\f$ of
//! the most correlated pairs of \f$N\f$ random variables with time and
//! space complexity \f$O(N + M)\f$.
//!
//! In particular, we project the residuals \f$\{ \frac{x_i - m_x}{\sigma_x} \}\f$
//! using sequences of \f$n\f$ samples of the random variable
//! \f$P : \Omega \rightarrow \{-1, 1}\f$. It is easy to show that the
//! expectation of
//! <pre class="fragment">
//!   \f$\left( \sum_{i=1}{n}{ P_i \frac{x_i - m_x}{\sigma_x} - \sum_{i=1}{n}{ P_i \frac{y_i - m_y}{\sigma_y} \right)\f$
//! </pre>
//! is equal to
//! <pre class="fragment">
//!   \f$2 (1 - cov(X,Y))\f$
//! </pre>
//! where \f$cov(X,Y)\f$ is the covariance between \f$X\f$ and \f$Y\f$.
//! We can average this over different copies of the random sequences
//! to "boost" the result. Since the average is proportional to the
//! Euclidean norm of the difference between the two points, whose
//! components are the projected normalised residuals, finding the
//! most correlated variables amounts to a collection neighbourhood
//! searches around each point.
class MATHS_EXPORT CKMostCorrelated
{
    public:
        //! The number of projections of the data to maintain
        //! simultaneously.
        static const std::size_t NUMBER_PROJECTIONS = 10u;

    public:
        using TDoubleVec = std::vector<double>;
        using TSizeVec = std::vector<std::size_t>;
        using TSizeSizePr = std::pair<std::size_t, std::size_t>;
        using TSizeSizePrVec = std::vector<TSizeSizePr>;
        using TVector = CVectorNx1<maths::CFloatStorage, NUMBER_PROJECTIONS>;
        using TVectorVec = std::vector<TVector>;
        using TSizeVectorUMap = boost::unordered_map<std::size_t, TVector>;
        using TVectorPackedBitVectorPr = std::pair<TVector, CPackedBitVector>;
        using TSizeVectorPackedBitVectorPrUMap = boost::unordered_map<std::size_t, TVectorPackedBitVectorPr>;

    public:
        CKMostCorrelated(std::size_t k, double decayRate, bool initialize = true);

        //! Create from part of a state document.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

        //! Persist state by passing to the supplied inserter.
        void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! Get the most correlated variables.
        void mostCorrelated(TSizeSizePrVec &result) const;

        //! Get the \p n most correlated variables.
        void mostCorrelated(std::size_t n,
                            TSizeSizePrVec &correlates,
                            TDoubleVec *pearson = 0) const;

        //! Get the most correlated variables correlations.
        void correlations(TDoubleVec &result) const;

        //! Get the \p n most correlated variables correlations.
        void correlations(std::size_t n, TDoubleVec &result) const;

        //! Resize the relevant statistics to accommodate up to \p n variables.
        void addVariables(std::size_t n);

        //! Remove the variables \p remove.
        void removeVariables(const TSizeVec &remove);

        //! Check if the correlations may have just changed.
        bool changed(void) const;

        //! Add the value \p x for the variable \p X.
        void add(std::size_t X, double x);

        //! Capture the projections of all variables added.
        void capture(void);

        //! Get the checksum of this object.
        uint64_t checksum(uint64_t seed = 0) const;

        //! Debug the memory used by this object.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this object.
        std::size_t memoryUsage(void) const;

    protected:
        //! The length of the projected sequence to capture.
        static const std::size_t PROJECTION_DIMENSION;
        //! The minimum sparseness, in terms of proportion of missing values,
        //! for a variable we'll consider trying to correlate.
        static const double MINIMUM_SPARSENESS;
        //! The proportion of values to replace for each projection.
        static const double REPLACE_FRACTION;

    protected:
        using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
        using TMeanVarAccumulatorVec = std::vector<TMeanVarAccumulator>;
        using TSizeVectorUMapCItr = TSizeVectorUMap::const_iterator;
        using TSizeVectorPackedBitVectorPrUMapItr = TSizeVectorPackedBitVectorPrUMap::iterator;
        using TSizeVectorPackedBitVectorPrUMapCItr = TSizeVectorPackedBitVectorPrUMap::const_iterator;

        //! \brief A pair of variables and their correlation.
        //!
        //! DESCRIPTION:\n
        //! This manages the estimation of the sample correlation,
        //! i.e. \f$\frac{1}{n}\sum_{i=1}{n}{\frac{(x-m_x)(y-m_y)}{\sigma_x\sigma_y}}\f$,
        //! from the projected data.
        struct MATHS_EXPORT SCorrelation
        {
            //! See core::CMemory.
            static bool dynamicSizeAlwaysZero(void) { return true; }

            SCorrelation(void);
            SCorrelation(std::size_t X,
                         const TVector &px,
                         const CPackedBitVector &ix,
                         std::size_t Y,
                         const TVector &py,
                         const CPackedBitVector &iy);

            //! Create from part of a state document.
            bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

            //! Persist state by passing to the supplied inserter.
            void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

            //! Complete ordering of correlations by _increasing_
            //! absolute correlation.
            bool operator<(const SCorrelation &rhs) const;

            //! Update the correlation with a new projection.
            void update(const TSizeVectorPackedBitVectorPrUMap &projected);

            //! Get the Euclidean distance between points corresponding
            //! to this correlation.
            double distance(double amax) const;

            //! Get (a lower bound) on the estimated absolute correlation.
            double absCorrelation(void) const;

            //! Estimate the correlation based on the projections
            //! \p px and \p py.
            static double correlation(const TVector &px,
                                      const CPackedBitVector &ix,
                                      const TVector &py,
                                      const CPackedBitVector &iy);

            //! Get the checksum of this object.
            uint64_t checksum(uint64_t seed) const;

            //! Print for debug.
            std::string print(void) const;

            //! The correlation.
            TMeanVarAccumulator s_Correlation;
            //! The first variable.
            std::size_t s_X;
            //! The second variable.
            std::size_t s_Y;
        };

        //! \brief Checks if a correlation includes a specified variable.
        class MATHS_EXPORT CMatches
        {
            public:
                CMatches(std::size_t x);

                bool operator()(const SCorrelation &correlation) const;

            private:
                std::size_t m_X;
        };

        using TCorrelationVec = std::vector<SCorrelation>;

    protected:
        //! Get the most correlated variables based on the current
        //! projections.
        void mostCorrelated(TCorrelationVec &result) const;

        //! Generate the next projection and reinitialize related state.
        void nextProjection(void);

        //! Get the projections.
        const TVectorVec &projections(void) const;

        //! Get the projected residuals.
        const TSizeVectorPackedBitVectorPrUMap &projected(void) const;

        //! Get the current correlation collection.
        const TCorrelationVec &correlations(void) const;

        //! Get the variable moments.
        const TMeanVarAccumulatorVec &moments(void) const;

    private:
        //! The number of correlations to find.
        std::size_t m_K;

        //! The rate at which to forget about historical correlations.
        double m_DecayRate;

        //! The random number generator.
        mutable CPRNG::CXorShift1024Mult m_Rng;

        //! The random projections.
        TVectorVec m_Projections;

        //! The values to add in the next capture.
        TSizeVectorUMap m_CurrentProjected;

        //! The projected variables' "normalised" residuals.
        TSizeVectorPackedBitVectorPrUMap m_Projected;

        //! The maximum possible metric measurement count.
        double m_MaximumCount;

        //! The variables' means and variances.
        TMeanVarAccumulatorVec m_Moments;

        //! The 2 * m_Size most correlated variables.
        TCorrelationVec m_MostCorrelated;
};

}
}

#endif // INCLUDED_ml_maths_CKMostCorrelated_h
