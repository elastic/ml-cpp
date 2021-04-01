/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_api_CDataFrameTrainBoostedTreeRunner_h
#define INCLUDED_ml_api_CDataFrameTrainBoostedTreeRunner_h

#include <maths/CBasicStatistics.h>

#include <api/CDataFrameAnalysisInstrumentation.h>
#include <api/CDataFrameAnalysisRunner.h>
#include <api/CDataFrameAnalysisSpecification.h>
#include <api/ImportExport.h>

#include <rapidjson/document.h>

#include <memory>

namespace ml {
namespace maths {
namespace boosted_tree {
class CLoss;
}
class CBoostedTree;
class CBoostedTreeFactory;
}
namespace api {
class CDataFrameAnalysisConfigReader;
class CDataFrameAnalysisParameters;
class CBoostedTreeInferenceModelBuilder;

//! \brief Runs boosted tree regression on a core::CDataFrame.
class API_EXPORT CDataFrameTrainBoostedTreeRunner : public CDataFrameAnalysisRunner {
public:
    enum ETask { E_Train, E_Update };
    
    static const std::string DEPENDENT_VARIABLE_NAME;
    static const std::string PREDICTION_FIELD_NAME;
    static const std::string DOWNSAMPLE_ROWS_PER_FEATURE;
    static const std::string DOWNSAMPLE_FACTOR;
    static const std::string ALPHA;
    static const std::string LAMBDA;
    static const std::string GAMMA;
    static const std::string ETA;
    static const std::string ETA_GROWTH_RATE_PER_TREE;
    static const std::string SOFT_TREE_DEPTH_LIMIT;
    static const std::string SOFT_TREE_DEPTH_TOLERANCE;
    static const std::string MAX_TREES;
    static const std::string FEATURE_BAG_FRACTION;
    static const std::string NUM_FOLDS;
    static const std::string STOP_CROSS_VALIDATION_EARLY;
    static const std::string MAX_OPTIMIZATION_ROUNDS_PER_HYPERPARAMETER;
    static const std::string BAYESIAN_OPTIMISATION_RESTARTS;
    static const std::string NUM_TOP_FEATURE_IMPORTANCE_VALUES;
    static const std::string TRAINING_PERCENT_FIELD_NAME;
    static const std::string FEATURE_PROCESSORS;
    static const std::string EARLY_STOPPING_ENABLED;
    static const std::string TASK;
    static const std::string TASK_TRAIN;
    static const std::string TASK_UPDATE;

    // Output
    static const std::string IS_TRAINING_FIELD_NAME;
    static const std::string FEATURE_NAME_FIELD_NAME;
    static const std::string IMPORTANCE_FIELD_NAME;
    static const std::string FEATURE_IMPORTANCE_FIELD_NAME;

public:
    ~CDataFrameTrainBoostedTreeRunner() override;

    //! \return The number of columns this adds to the data frame.
    std::size_t numberExtraColumns() const override;

    //! \return The capacity of the data frame slice to use.
    std::size_t dataFrameSliceCapacity() const override;

    //! The boosted tree.
    const maths::CBoostedTree& boostedTree() const;

    //! The boosted tree factory.
    const maths::CBoostedTreeFactory& boostedTreeFactory() const;

    //! \return Reference to the analysis state.
    const CDataFrameAnalysisInstrumentation& instrumentation() const override;
    //! \return Reference to the analysis state.
    CDataFrameAnalysisInstrumentation& instrumentation() override;

    //! \return A serialisable data summarization for the trained model.
    TDataSummarizationUPtr dataSummarization(const core::CDataFrame& dataFrame) const override;

protected:
    using TLossFunctionUPtr = std::unique_ptr<maths::boosted_tree::CLoss>;

protected:
    CDataFrameTrainBoostedTreeRunner(const CDataFrameAnalysisSpecification& spec,
                                     const CDataFrameAnalysisParameters& parameters,
                                     TLossFunctionUPtr loss);

    //! Parameter reader handling parameters that are shared by subclasses.
    static const CDataFrameAnalysisConfigReader& parameterReader();
    //! Name of dependent variable field.
    const std::string& dependentVariableFieldName() const;
    //! Name of prediction field.
    const std::string& predictionFieldName() const;
    //! The boosted tree factory.
    maths::CBoostedTreeFactory& boostedTreeFactory();

    //! Validate if \p frame is suitable for running the analysis on.
    bool validate(const core::CDataFrame& frame) const override;

    //! Write the boosted tree and custom processors to \p builder.
    void accept(CBoostedTreeInferenceModelBuilder& builder) const;

private:
    using TBoostedTreeFactoryUPtr = std::unique_ptr<maths::CBoostedTreeFactory>;
    using TBoostedTreeUPtr = std::unique_ptr<maths::CBoostedTree>;
    using TDataSearcherUPtr = CDataFrameAnalysisSpecification::TDataSearcherUPtr;

private:
    void runImpl(core::CDataFrame& frame) override;
    bool restoreBoostedTree(core::CDataFrame& frame,
                            std::size_t dependentVariableColumn,
                            TDataSearcherUPtr& restoreSearcher);
    std::size_t estimateBookkeepingMemoryUsage(std::size_t numberPartitions,
                                               std::size_t totalNumberRows,
                                               std::size_t partitionNumberRows,
                                               std::size_t numberColumns) const override;

    virtual void validate(const core::CDataFrame& frame,
                          std::size_t dependentVariableColumn) const = 0;

private:
    // Note custom config is written directly to the factory object.

    rapidjson::Document m_CustomProcessors;
    std::string m_DependentVariableFieldName;
    std::string m_PredictionFieldName;
    double m_TrainingPercent;
    TBoostedTreeFactoryUPtr m_BoostedTreeFactory;
    TBoostedTreeUPtr m_BoostedTree;
    CDataFrameTrainBoostedTreeInstrumentation m_Instrumentation;
    ETask m_Task;
};
}
}

#endif // INCLUDED_ml_api_CDataFrameTrainBoostedTreeRunner_h
