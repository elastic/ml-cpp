/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CNaturalBreaksClassifier_h
#define INCLUDED_ml_maths_CNaturalBreaksClassifier_h

#include <core/CMemory.h>

#include <maths/CBasicStatistics.h>
#include <maths/Constants.h>

#include <maths/ImportExport.h>

#include <cstddef>
#include <utility>
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

//! \brief This does online segmentation with fixed space by approximate
//! minimization of the within class total deviation (defined below).
//!
//! DESCRIPTION:\n
//! Segmentation or data classification is a 1-D analogue of the
//! general clustering problem. This problem is generally treated
//! by trying to minimize the within class total variation by looking
//! for a partition of the ordered samples, i.e. the set satisfying
//! <pre class="fragment">
//!   \f$\displaystyle I=argmin{\sum_{(i,j)\in J}\sum_{k=i}^{k<j}{(x_k-\mu_{i,j})^2}}\f$
//! </pre>
//! Here, \f$\mu_{i,j}=1/(j-i)\sum_{k\in[i,j)}{x_k}\f$. See, for
//! example, <a href="http://en.wikipedia.org/wiki/Jenks_natural_breaks_optimization">here</a>
//! for more details on this area. Note that this is equivalent to
//! the k-means objective in 1-D because the points would always be
//! assigned to their nearest cluster so we only need to consider
//! ordered partitions. We instead optimize the total within class
//! deviation, which is obtained by replacing the summand above with
//! <pre class="fragment">
//!   \f$\displaystyle {(\sum_{k=i}^{k<j}{(x_k-\mu_{i,j})^2})^{1/p}}\f$
//! </pre>
//! Taking the limit \f$p\rightarrow\infty\f$ this means that rather
//! than assigning points to the class with the nearest by mean, points
//! are assigned to the class based on the distance to the class
//! mean in standard deviations. Using large values for \f$p\f$
//! causes numerical problems, i.e. results in large cancellation
//! errors. We use \f$p = 2\f$ by default, which gives reasonable results.
//!
//! For a given data set this problem can be solved exactly by
//! a dynamic programming approach described in this
//! <a href="http://journal.r-project.org/archive/2011-2/RJournal_2011-2_Wang+Song.pdf">paper</a>
//!
//! The algorithm implemented by this class is inspired by
//! <a href="http://en.wikipedia.org/wiki/BIRCH_(data_clustering)">BIRCH</a>.
//! The idea is to fix the space required and then store a set of
//! tuples (count, mean, variance) which correspond to possible classes
//! that are good candidates for minimizing the within class total
//! deviation. Each time a point is received it is added to a buffer
//! until the buffer is full. At this point the buffered points are
//! converted to new clusters and the number of clusters reduced, by
//! greedily minimizing the within class total deviation, until the
//! space threshold is satisfied. As such the, the following types of
//! operation are considered:
//!   -# merging two tuples
//!   -# adding a point to a tuple
//!   -# creating a new tuple from a point
//!
//! An optimal algorithm is implemented to return \f$m \leq n\f$
//! classes, given the tuples which have been identified, that uses
//! the more efficient of either a branch and bound search of all
//! possible partitions or straightforward adaptation of the dynamic
//! program. Here, \f$n\f$ is the maximum number of categories which
//! can be described subject to the space constraint. The intention
//! is for the caller to use a space requirement somewhat larger
//! than the eventual number of partitions of the data wanted.
//!
//! A mechanism to age out the clusters and prune dead clusters has
//! been implemented by scaling the cluster counts and removing low
//! count clusters.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This class uses float storage for the tuples. This is because
//! it is intended for use in cases where space is at a premium.
//! *DO NOT* use floats unless doing so gives a significant overall
//! space improvement to the *program* footprint. Note also that the
//! interface to this class is double precision. If floats are used
//! they should be used for storage only and transparent to the rest
//! of the code base.
class MATHS_EXPORT CNaturalBreaksClassifier
{
    public:
        using TSizeVec = std::vector<std::size_t>;
        using TDoubleVec = std::vector<double>;
        using TDoubleDoublePr = std::pair<double, double>;
        using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
        using TDoubleTuple = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
        using TDoubleTupleVec = std::vector<TDoubleTuple>;
        using TTuple = CBasicStatistics::SSampleMeanVar<CFloatStorage>::TAccumulator;
        using TTupleVec = std::vector<TTuple>;
        using TClassifierVec = std::vector<CNaturalBreaksClassifier>;

    public:
        //! The type of optimization object which it is possible
        //! to target. In particular,
        //!   -# Deviation is the square root of the total sample
        //!      variation.
        //!   -# Variation is the total sample variation, i.e. the
        //!      sum of the square differences from the sample mean.
        enum EObjective
        {
            E_TargetDeviation,
            E_TargetVariation
        };

    public:
        //! Create a new classifier with the specified space limit.
        //!
        //! \param[in] space The maximum space in numbers of tuples.
        //! A tuple comprises three floats.
        //! \param[in] decayRate The rate at which we data ages out
        //! of the classifier.
        //! \param[in] minimumCategoryCount The minimum permitted count
        //! for a category.
        //! \note This will store as much information about the points
        //! subject to this constraint so will generally hold approximately
        //! \p space tuples.
        CNaturalBreaksClassifier(std::size_t space,
                                 double decayRate = 0.0,
                                 double minimumCategoryCount = MINIMUM_CATEGORY_COUNT);

        //! Create from part of a state document.
        bool acceptRestoreTraverser(const SDistributionRestoreParams &params,
                                    core::CStateRestoreTraverser &traverser);

        //! Persist state by passing information to the supplied inserter.
        void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! Get the count \p p percentile position.
        double percentile(double p) const;

        //! Get the total number of categories in the classifier.
        std::size_t size() const;

        //! Split this classifier into the n-categories identified by
        //! the categories function.
        //!
        //! \param[in] n The desired size of the split.
        //! \param[in] p The minimum category size.
        //! \param[out] result Filled in with the classifiers representing
        //! the split.
        //! \sa categories for details on the split.
        bool split(std::size_t n, std::size_t p, TClassifierVec &result);

        //! Split this classifier into the n-categories corresponding to
        //! \p split.
        //!
        //! \param[in] split The desired partition.
        //! \param[out] result Filled in with the classifiers representing
        //! \p split if it is a valid partition and cleared otherwise.
        //! \note \p split should be ordered and the maximum value should
        //! be equal to the number of points in the classifier.
        bool split(const TSizeVec &split, TClassifierVec &result);

        //! Get the minimum within class total deviation partition
        //! of size at most \p n.
        //!
        //! \param[in] n The number of partitions.
        //! \param[in] p The minimum category size.
        //! \param[out] result Filled in with the indices at which to break.
        bool naturalBreaks(std::size_t n, std::size_t p, TSizeVec &result);

        //! Get as many tuples as possible, but not more than \p n,
        //! describing our best estimate of the categories in the data.
        //!
        //! \param[in] n The desired size for the partition.
        //! \param[in] p The minimum category size.
        //! \param[out] result Filled in with the minimum within class
        //! total deviation partition of size at most n.
        //! \param[in] append If true the categories are appended to
        //! \p result.
        //! \note This finds the optimum partition using a dynamic
        //! programming approach in complexity \f$O(N^2n)\f$ where
        //! \f$N\f$ the number of tuples and \f$n\f$ is the desired
        //! size for the partition.
        bool categories(std::size_t n,
                        std::size_t p,
                        TTupleVec &result,
                        bool append = false);

        //! Get the categories corresponding to \p split.
        //!
        //! \param[in] split The desired partition.
        //! \param[out] result Filled in with the categories corresponding
        //! to \p split if it is a valid partition and cleared otherwise.
        //! \note \p split should be ordered and the maximum value should
        //! be equal to the number of points in the classifier.
        bool categories(const TSizeVec &split, TTupleVec &result);

        //! Add \p x to the classifier.
        //!
        //! \param[in] x A point to add to the classifier.
        //! \param[in] count The count weight of this point.
        void add(double x, double count = 1.0);

        //! Merge \p other with this classifier.
        //!
        //! \param[in] other Another classifier to merge with this one.
        void merge(const CNaturalBreaksClassifier &other);

        //! Set the rate at which information is aged out.
        void decayRate(double decayRate);

        //! Propagate the clusters forwards by \p time.
        void propagateForwardsByTime(double time);

        //! Check if we are currently buffering points.
        bool buffering() const;

        //! Get \p n samples of the distribution corresponding to the
        //! categories we are maintaining.
        //!
        //! \param[in] numberSamples The desired number of samples.
        //! \param[in] smallest The smallest permitted sample.
        //! \param[in] largest The largest permitted sample.
        //! \param[out] result Filled in with the samples of the distribution.
        void sample(std::size_t numberSamples,
                    double smallest,
                    double largest,
                    TDoubleVec &result) const;

        //! Print this classifier for debug.
        std::string print() const;

        //! Get a checksum for this object.
        uint64_t checksum(uint64_t seed = 0) const;

        //! Get the memory used by this component
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this component
        std::size_t memoryUsage() const;

        //! Get the minimum within class total deviation partition
        //! of the categories \p categories with size at most \p n
        //! subject to the constraint that no category contains fewer
        //! than \p p values.
        //!
        //! \param[in] categories The categories to partition.
        //! \param[in] n The number of partitions.
        //! \param[in] p The minimum category size.
        //! \param[in] target The optimization objective to target.
        //! \param[out] result Filled in with the indices at which to
        //! break.
        //! \note This finds the optimum partition using a dynamic
        //! programming approach in complexity \f$O(N^2n)\f$ where
        //! \f$N\f$ the number of tuples and \f$n\f$ is the desired
        //! size for the partition.
        static bool naturalBreaks(const TTupleVec &categories,
                                  std::size_t n,
                                  std::size_t p,
                                  EObjective target,
                                  TSizeVec &result);

        //! Double tuple version.
        //!
        //! \see naturalBreaks for more details.
        static bool naturalBreaks(const TDoubleTupleVec &categories,
                                  std::size_t n,
                                  std::size_t p,
                                  EObjective target,
                                  TSizeVec &result);

    private:
        using TSizeSizePr = std::pair<std::size_t, std::size_t>;

    private:
        //! Implementation called by naturalBreaks with explicit
        //! tuple types.
        template<typename TUPLE>
        static bool naturalBreaksImpl(const std::vector<TUPLE> &categories,
                                      std::size_t n,
                                      std::size_t p,
                                      EObjective target,
                                      TSizeVec &result);

    private:
        //! The minimum permitted size for the classifier.
        static const std::size_t MINIMUM_SPACE;

        //! The maximum allowed size of the points buffer.
        static const std::size_t MAXIMUM_BUFFER_SIZE;

    private:
        //! Construct a new classifier with the specified space limit
        //! \p space and categories \p categories.
        CNaturalBreaksClassifier(std::size_t space,
                                 double decayRate,
                                 double minimumCategoryCount,
                                 TTupleVec &categories);

        //! Reduce the number of tuples until we satisfy the space constraint.
        void reduce();

        //! Get the indices of the closest categories.
        TSizeSizePr closestPair() const;

        //! Get the total deviation of the specified class.
        static double deviation(const TTuple &category);

        //! Get the total variation of the specified class.
        static double variation(const TTuple &category);

        //! Wrapper to evaluate the specified object function.
        static inline double objective(EObjective objective, const TTuple &category)
        {
            switch (objective)
            {
            case E_TargetDeviation: return deviation(category);
            case E_TargetVariation: return variation(category);
            }
            return deviation(category);
        }

    private:
        //! The maximum space in doubles.
        std::size_t m_Space;

        //! The rate at which the categories lose information.
        double m_DecayRate;

        //! The minimum permitted count for a category.
        double m_MinimumCategoryCount;

        //! The categories we are maintaining.
        TTupleVec m_Categories;

        //! A buffer of the points added while the space constraint is satisfied.
        TDoubleDoublePrVec m_PointsBuffer;
};

}
}

#endif // INCLUDED_ml_maths_CNaturalBreaksClassifier_h
