/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2018 Elasticsearch BV. All Rights Reserved.
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

#ifndef INCLUDED_ml_maths_CNaiveBayes_h
#define INCLUDED_ml_maths_CNaiveBayes_h

#include <maths/ImportExport.h>

#include <maths/CPrior.h>

#include <boost/unordered_map.hpp>

#include <cstddef>
#include <vector>

namespace ml
{
namespace core
{
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths
{
struct SDistributionRestoreParams;

//! \brief The interface expected by CNaiveBayes for implementations
//! of the class conditional density functions.
class MATHS_EXPORT CNaiveBayesFeatureDensity
{
    public:
        using TDouble1Vec = core::CSmallVector<double, 1>;

    public:
        virtual ~CNaiveBayesFeatureDensity() = default;

        //! Create and return a clone.
        //!
        //! \note The caller owns this.
        virtual CNaiveBayesFeatureDensity *clone() const = 0;

        //! Initialize by reading state from \p traverser.
        virtual bool acceptRestoreTraverser(const SDistributionRestoreParams &params,
                                            core::CStateRestoreTraverser &traverser) = 0;

        //! Persist state by passing information to \p inserter.
        virtual void acceptPersistInserter(core::CStatePersistInserter &inserter) const = 0;

        //! Add the value \p x.
        virtual void add(const TDouble1Vec &x) = 0;

        //! Compute the log value of the density function at \p x.
        virtual double logValue(const TDouble1Vec &x) const = 0;

        //! Age out old values density to account for \p time passing.
        virtual void propagateForwardsByTime(double time) = 0;

        //! Debug the memory used by this object.
        virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const = 0;

        //! Get the static size of this object.
        virtual std::size_t staticSize() const = 0;

        //! Get the memory used by this object.
        virtual std::size_t memoryUsage() const = 0;

        //! Get a checksum for this object.
        virtual uint64_t checksum(uint64_t seed) const = 0;
};

//! \brief An implementation of the class conditional density function
//! based on the CPrior hierarchy.
class MATHS_EXPORT CNaiveBayesFeatureDensityFromPrior final : public CNaiveBayesFeatureDensity
{
    public:
        CNaiveBayesFeatureDensityFromPrior() = default;
        CNaiveBayesFeatureDensityFromPrior(CPrior &prior);

        //! Create and return a clone.
        //!
        //! \note The caller owns this.
        virtual CNaiveBayesFeatureDensityFromPrior *clone() const;

        //! Initialize by reading state from \p traverser.
        virtual bool acceptRestoreTraverser(const SDistributionRestoreParams &params,
                                            core::CStateRestoreTraverser &traverser);

        //! Persist state by passing information to \p inserter.
        virtual void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! Add the value \p x.
        virtual void add(const TDouble1Vec &x);

        //! Compute the log value of the density function at \p x.
        virtual double logValue(const TDouble1Vec &x) const;

        //! Age out old values density to account for \p time passing.
        virtual void propagateForwardsByTime(double time);

        //! Debug the memory used by this object.
        virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the static size of this object.
        virtual std::size_t staticSize() const;

        //! Get the memory used by this object.
        virtual std::size_t memoryUsage() const;

        //! Get a checksum for this object.
        virtual uint64_t checksum(uint64_t seed) const;

    private:
        using TPriorPtr = boost::shared_ptr<CPrior>;

    private:
        //! The density model.
        TPriorPtr m_Prior;
};

//! \brief Implements a Naive Bayes classifier.
class MATHS_EXPORT CNaiveBayes
{
    public:
        using TDoubleSizePr = std::pair<double, std::size_t>;
        using TDoubleSizePrVec = std::vector<TDoubleSizePr>;
        using TDouble1Vec = core::CSmallVector<double, 1>;
        using TDouble1VecVec = std::vector<TDouble1Vec>;

    public:
        explicit CNaiveBayes(const CNaiveBayesFeatureDensity &exemplar,
                             double decayRate = 0.0);
        CNaiveBayes(const SDistributionRestoreParams &params,
                    core::CStateRestoreTraverser &traverser);

        //! Persist state by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! This can be used to optionally seed the class counts
        //! with \p counts. These are added on to data class counts
        //! to compute the class posterior probabilities.
        void initialClassCounts(const TDoubleSizePrVec &counts);

        //! Add a training data point comprising the pair \f$(x,l)\f$
        //! for feature vector \f$x\f$ and class label \f$l\f$.
        //!
        //! \param[in] label The class label for \p x.
        //! \param[in] x The feature values.
        //! \note \p x size should be equal to the number of features.
        //! A feature is missing is indicated by passing an empty vector
        //! for that feature.
        void addTrainingDataPoint(std::size_t label, const TDouble1VecVec &x);

        //! Age out old values from the class conditional densities
        //! to account for \p time passing.
        void propagateForwardsByTime(double time);

        //! Get the top \p n class probabilities for \p features.
        //!
        //! \param[in] n The number of class probabilities to estimate.
        //! \param[in] x The feature values.
        //! \note \p x size should be equal to the number of features.
        //! A feature is missing is indicated by passing an empty vector
        //! for that feature.
        TDoubleSizePrVec highestClassProbabilities(std::size_t n,
                                                   const TDouble1VecVec &x) const;

        //! Debug the memory used by this object.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this object.
        std::size_t memoryUsage() const;

        //! Get a checksum for this object.
        uint64_t checksum(uint64_t seed = 0) const;

    private:
        using TFeatureDensityPtr = boost::shared_ptr<CNaiveBayesFeatureDensity>;
        using TFeatureDensityPtrVec = std::vector<TFeatureDensityPtr>;

        //! \brief The data associated with a class.
        struct SClass
        {
            //! Initialize by reading state from \p traverser.
            bool acceptRestoreTraverser(const SDistributionRestoreParams &params,
                                        core::CStateRestoreTraverser &traverser);
            //! Persist state by passing information to \p inserter.
            void acceptPersistInserter(core::CStatePersistInserter &inserter) const;
            //! Debug the memory used by this object.
            void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;
            //! Get the memory used by this object.
            std::size_t memoryUsage() const;
            //! Get a checksum for this object.
            uint64_t checksum(uint64_t seed = 0) const;

            //! The number of examples in this class.
            double s_Count = 0.0;
            //! The feature conditional densities for this class.
            TFeatureDensityPtrVec s_ConditionalDensities;
        };

        using TSizeClassUMap = boost::unordered_map<std::size_t, SClass>;

    private:
        //! Initialize by reading state from \p traverser.
        bool acceptRestoreTraverser(const SDistributionRestoreParams &params,
                                    core::CStateRestoreTraverser &traverser);

        //! Validate \p x.
        bool validate(const TDouble1VecVec &x) const;

    private:
        //! Controls the rate at which data are aged out.
        double m_DecayRate;

        //! An exemplar for creating conditional densities.
        TFeatureDensityPtr m_Exemplar;

        //! The class conditional density estimates and weights.
        TSizeClassUMap m_ClassConditionalDensities;
};

}
}

#endif // INCLUDED_ml_maths_CNaiveBayes_h
