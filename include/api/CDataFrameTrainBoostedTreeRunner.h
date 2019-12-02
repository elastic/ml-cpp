/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_api_CDataFrameTrainBoostedTreeRunner_h
#define INCLUDED_ml_api_CDataFrameTrainBoostedTreeRunner_h

#include <api/CDataFrameAnalysisRunner.h>
#include <api/CDataFrameAnalysisSpecification.h>
#include <api/ImportExport.h>

#include <rapidjson/fwd.h>

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

//! \brief Runs boosted tree regression on a core::CDataFrame.
class API_EXPORT CDataFrameTrainBoostedTreeRunner : public CDataFrameAnalysisRunner {
public:
    static const std::string DEPENDENT_VARIABLE_NAME;
    static const std::string PREDICTION_FIELD_NAME;
    static const std::string DOWNSAMPLE_ROWS_PER_FEATURE;
    static const std::string ALPHA;
    static const std::string LAMBDA;
    static const std::string GAMMA;
    static const std::string ETA;
    static const std::string SOFT_TREE_DEPTH_LIMIT;
    static const std::string SOFT_TREE_DEPTH_TOLERANCE;
    static const std::string MAXIMUM_NUMBER_TREES;
    static const std::string FEATURE_BAG_FRACTION;
    static const std::string NUMBER_FOLDS;
    static const std::string NUMBER_ROUNDS_PER_HYPERPARAMETER;
    static const std::string BAYESIAN_OPTIMISATION_RESTARTS;
    static const std::string TOP_SHAP_VALUES;

public:
    ~CDataFrameTrainBoostedTreeRunner() override;

    //! \return The number of columns this adds to the data frame.
    std::size_t numberExtraColumns() const override;

    //! The boosted tree.
    const maths::CBoostedTree& boostedTree() const;

    //! The boosted tree factory.
    const maths::CBoostedTreeFactory& boostedTreeFactory() const;

    //! Specified number of SHAP values to return
    size_t topShapValues() const;

protected:
    using TBoostedTreeUPtr = std::unique_ptr<maths::CBoostedTree>;
    using TLossFunctionUPtr = std::unique_ptr<maths::boosted_tree::CLoss>;

protected:
    CDataFrameTrainBoostedTreeRunner(const CDataFrameAnalysisSpecification& spec,
                                     const CDataFrameAnalysisParameters& parameters);
    CDataFrameTrainBoostedTreeRunner(const CDataFrameAnalysisSpecification& spec);

    //! Parameter reader handling parameters that are shared by subclasses.
    static const CDataFrameAnalysisConfigReader& parameterReader();
    //! Name of dependent variable field.
    const std::string& dependentVariableFieldName() const;
    //! Name of prediction field.
    const std::string& predictionFieldName() const;

    //! The boosted tree factory.
    maths::CBoostedTreeFactory& boostedTreeFactory();

private:
    using TBoostedTreeFactoryUPtr = std::unique_ptr<maths::CBoostedTreeFactory>;
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

    virtual TLossFunctionUPtr chooseLossFunction(const core::CDataFrame& frame,
                                                 std::size_t dependentVariableColumn) const = 0;

private:
    // Note custom config is written directly to the factory object.

    std::string m_DependentVariableFieldName;
    std::string m_PredictionFieldName;
    TBoostedTreeFactoryUPtr m_BoostedTreeFactory;
    TBoostedTreeUPtr m_BoostedTree;
    std::size_t m_TopShapValues = 0;
};
}
}

#endif // INCLUDED_ml_api_CDataFrameTrainBoostedTreeRunner_h
