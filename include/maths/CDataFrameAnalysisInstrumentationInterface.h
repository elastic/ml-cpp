/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CDataFrameAnalysisInstrumentationInterface_h
#define INCLUDED_ml_maths_CDataFrameAnalysisInstrumentationInterface_h

#include <maths/CBoostedTree.h>
#include <maths/COutliers.h>
#include <maths/ImportExport.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ml {
namespace maths {

//! \brief Interface class for collecting data frame analysis job statistics owned
//! by the maths module.
class MATHS_EXPORT CDataFrameAnalysisInstrumentationInterface {
public:
    using TProgressCallback = std::function<void(double)>;
    using TMemoryUsageCallback = std::function<void(std::int64_t)>;
    using TFlushCallback = std::function<void(const std::string&)>;

public:
    virtual ~CDataFrameAnalysisInstrumentationInterface() = default;

    //! Adds \p delta to the memory usage statistics.
    virtual void updateMemoryUsage(std::int64_t delta) = 0;

    //! Start progress monitoring of \p task.
    //!
    //! \note This resets the current progress to zero.
    virtual void startNewProgressMonitoredTask(const std::string& task) = 0;

    //! This adds \p fractionalProgress to the current progress.
    //!
    //! \note The caller should try to ensure that the sum of the values added
    //! at the end of the analysis is equal to one.
    //! \note This is converted to an integer - so we can atomically add - by
    //! scaling by 1024. Therefore, this shouldn't be called with values less
    //! than 0.001. In fact, it is unlikely that such high resolution is needed
    //! and typically this would be called significantly less frequently.
    virtual void updateProgress(double fractionalProgress) = 0;

    //! Flush then reinitialize the instrumentation data. This will trigger
    //! writing them to the results pipe.
    virtual void flush(const std::string& tag = "") = 0;

    //! Factory for the updateProgress() callback function object.
    TProgressCallback progressCallback() {
        return [this](double fractionalProgress) {
            this->updateProgress(fractionalProgress);
        };
    }

    //! Factory for the updateMemoryUsage() callback function object.
    TMemoryUsageCallback memoryUsageCallback() {
        return [this](std::int64_t delta) { this->updateMemoryUsage(delta); };
    }

    //! Factory for the flush() callback function object.
    TFlushCallback flushCallback() {
        return [this](const std::string& tag) { this->flush(tag); };
    }
};

//! \brief Instrumentation interface for Outlier Detection jobs.
//!
//! DESCRIPTION:\n
//! This interface extends CDataFrameAnalysisInstrumentationInterface with a setters
//! for analysis parameters and elapsed time.
class MATHS_EXPORT CDataFrameOutliersInstrumentationInterface
    : virtual public CDataFrameAnalysisInstrumentationInterface {
public:
    virtual void parameters(const maths::COutliers::SComputeParameters& parameters) = 0;
    virtual void elapsedTime(std::uint64_t time) = 0;
    virtual void featureInfluenceThreshold(double featureInfluenceThreshold) = 0;
};

//! \brief Instrumentation interface for Supervised Learning jobs.
//!
//! DESCRIPTION:\n
//! This interface extends CDataFrameAnalysisInstrumentationInterface with a setters
//! for hyperparameters, validation loss results, and job timing.
class MATHS_EXPORT CDataFrameTrainBoostedTreeInstrumentationInterface
    : virtual public CDataFrameAnalysisInstrumentationInterface {
public:
    enum EStatsType { E_Regression, E_Classification };
    struct SHyperparameters {
        double s_Eta{-1.0};
        double s_RetrainedTreeEta{-1.0};
        CBoostedTree::EClassAssignmentObjective s_ClassAssignmentObjective{
            CBoostedTree::E_MinimumRecall};
        double s_DepthPenaltyMultiplier{-1.0};
        double s_SoftTreeDepthLimit{-1.0};
        double s_SoftTreeDepthTolerance{-1.0};
        double s_TreeSizePenaltyMultiplier{-1.0};
        double s_LeafWeightPenaltyMultiplier{-1.0};
        double s_TreeTopologyChangePenalty{-1.0};
        double s_DownsampleFactor{-1.0};
        std::size_t s_NumFolds{0};
        std::size_t s_MaxTrees{0};
        double s_FeatureBagFraction{-1.0};
        double s_EtaGrowthRatePerTree{-1.0};
        double s_PredictionChangeCost{-1.0};
        std::size_t s_MaxAttemptsToAddTree{0};
        std::size_t s_NumSplitsPerFeature{0};
        std::size_t s_MaxOptimizationRoundsPerHyperparameter{0};
    };
    using TDoubleVec = std::vector<double>;

public:
    //! Set the supervised learning job \p type, can be E_Regression or E_Classification.
    virtual void type(EStatsType type) = 0;
    //! Set the current \p iteration number.
    virtual void iteration(std::size_t iteration) = 0;
    //! Set the run time of the current iteration.
    virtual void iterationTime(std::uint64_t delta) = 0;
    //! Set the type of the validation loss result, e.g. "mse".
    virtual void lossType(const std::string& lossType) = 0;
    //! Set the validation loss values for \p fold for each forest size to \p lossValues.
    virtual void lossValues(std::size_t fold, TDoubleVec&& lossValues) = 0;
    //! \return A writable object containing the training hyperparameters.
    virtual SHyperparameters& hyperparameters() = 0;
};

//! \brief Dummies out all instrumentation for outlier detection.
class MATHS_EXPORT CDataFrameOutliersInstrumentationStub
    : public CDataFrameOutliersInstrumentationInterface {
public:
    void updateMemoryUsage(std::int64_t) override {}
    void startNewProgressMonitoredTask(const std::string& /* task */) override {}
    void updateProgress(double) override {}
    void flush(const std::string& /* tag */) override {}
    void parameters(const maths::COutliers::SComputeParameters& /* parameters */) override {}
    void elapsedTime(std::uint64_t /* time */) override {}
    void featureInfluenceThreshold(double /* featureInfluenceThreshold */) override {}
};

//! \brief Dummies out all instrumentation for supervised learning.
class MATHS_EXPORT CDataFrameTrainBoostedTreeInstrumentationStub
    : public CDataFrameTrainBoostedTreeInstrumentationInterface {
public:
    void updateMemoryUsage(std::int64_t) override {}
    void startNewProgressMonitoredTask(const std::string& /* task */) override {}
    void updateProgress(double) override {}
    void flush(const std::string& /* tag */) override {}
    void type(EStatsType /* type */) override {}
    void iteration(std::size_t /* iteration */) override {}
    void iterationTime(std::uint64_t /* delta */) override {}
    void lossType(const std::string& /* lossType */) override {}
    void lossValues(std::size_t /* fold */, TDoubleVec&& /* lossValues */) override {}
    SHyperparameters& hyperparameters() override { return m_Hyperparameters; }

private:
    SHyperparameters m_Hyperparameters;
};
}
}

#endif // INCLUDED_ml_maths_CDataFrameAnalysisInstrumentationInterface_h
