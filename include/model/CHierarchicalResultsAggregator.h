/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#ifndef INCLUDED_ml_model_CHierarchicalResultsAggregator_h
#define INCLUDED_ml_model_CHierarchicalResultsAggregator_h

#include <maths/CQuantileSketch.h>

#include <model/CDetectorEqualizer.h>
#include <model/CHierarchicalResultsLevelSet.h>
#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <cstddef>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace model {
class CAnomalyDetectorModelConfig;
class CLimits;

//! \brief Aggregates the probabilities up a collection hierarchical
//! results.
//!
//! DESCRIPTION:\n
//! This can be used to compute the aggregate probabilities for the
//! internal nodes of a hierarchical results object in a bottom up
//! breadth first pass.
//!
//! This uses different aggregation styles for aggregating partitions,
//! people in a population and collections of individual results in
//! system change analysis. Special logic is used for named people,
//! i.e. aggregations of multiple compatible simple searches.
class MODEL_EXPORT CHierarchicalResultsAggregator
    : public CHierarchicalResultsLevelSet<CDetectorEqualizer> {
public:
    //! Enumeration of the possible jobs that the aggregator can
    //! perform when invoked.
    enum EJob { E_UpdateAndCorrect, E_Correct, E_NoOp };

public:
    CHierarchicalResultsAggregator(const CAnomalyDetectorModelConfig& modelConfig);

    //! Add a job for the subsequent invocations of the normalizer.
    void setJob(EJob job);

    //! Update the parameters to reflect changes to the model configuration.
    void refresh(const CAnomalyDetectorModelConfig& modelConfig);

    //! Clear all state such that all equalizers restart from scratch.
    void clear();

    //! Compute the aggregate probability for \p node.
    virtual void visit(const CHierarchicalResults& results, const TNode& node, bool pivot);

    //! Age the quantile sketches.
    void propagateForwardByTime(double time);

    //! \name Persistence
    //@{
    //! Persist state by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Restore reading state from \p traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    // Clone for persistence?
    //@}

    //! Get a checksum of this object.
    uint64_t checksum() const;

private:
    using TBase = CHierarchicalResultsLevelSet<CDetectorEqualizer>;
    using TDetectorEqualizer = TBase::Type;
    using TDetectorEqualizerPtrVec = TBase::TTypePtrVec;
    using TIntSizePr = std::pair<int, std::size_t>;
    using TDouble1Vec = core::CSmallVector<double, 1>;
    using TIntSizePrDouble1VecUMap = boost::unordered_map<TIntSizePr, TDouble1Vec>;

private:
    static const std::size_t N = model_t::E_AggregateAttributes + 1;

private:
    //! Aggregate at a leaf node.
    void aggregateLeaf(const TNode& node);

    //! Aggregate at internal node.
    void aggregateNode(const TNode& node, bool pivot);

    //! Partition the child probabilities into groups to aggregate together.
    bool partitionChildProbabilities(const TNode& node,
                                     bool pivot,
                                     std::size_t& numberDetectors,
                                     TIntSizePrDouble1VecUMap (&partition)[N]);

    //! Compute the probability for each of the detectors.
    void detectorProbabilities(const TNode& node,
                               bool pivot,
                               std::size_t numberDetectors,
                               const TIntSizePrDouble1VecUMap (&partition)[N],
                               int& detector,
                               int& aggregation,
                               TDouble1Vec& probabilities);

    //! Compute a hash of \p node for gathering up related results.
    std::size_t hash(const TNode& node) const;

    //! Correct the probability for \p node to equalize probabilities
    //! across detectors.
    double correctProbability(const TNode& node, bool pivot, int detector, double probability);

private:
    //! The jobs that the aggregator will perform when invoked can be:
    //! update or update + correct.
    EJob m_Job;

    //! The rate information is lost from the quantile sketches.
    double m_DecayRate;

    //! The weights to use for the different aggregation styles.
    double m_Parameters[model_t::NUMBER_AGGREGATION_STYLES][model_t::NUMBER_AGGREGATION_PARAMS];

    //! The maximum anomalous probability.
    double m_MaximumAnomalousProbability;
};
}
}

#endif // INCLUDED_ml_model_CHierarchicalResultsAggregator_h
