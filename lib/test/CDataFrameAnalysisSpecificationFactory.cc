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

#include <test/CDataFrameAnalysisSpecificationFactory.h>

#include <core/CDataFrame.h>
#include <core/CStringUtils.h>

#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameAnalysisSpecificationJsonWriter.h>
#include <api/CDataFrameOutliersRunner.h>
#include <api/CDataFrameTrainBoostedTreeClassifierRunner.h>
#include <api/CDataFrameTrainBoostedTreeRegressionRunner.h>
#include <api/CDataFrameTrainBoostedTreeRunner.h>

#include <test/CTestTmpDir.h>

#include <memory>

namespace ml {
namespace test {
using TJsonLineWriter = core::CStreamWriter;

CDataFrameAnalysisSpecificationFactory::CDataFrameAnalysisSpecificationFactory()
    : m_MissingString{core::CDataFrame::DEFAULT_MISSING_STRING} {
}

const std::string& CDataFrameAnalysisSpecificationFactory::classification() {
    return api::CDataFrameTrainBoostedTreeClassifierRunnerFactory::NAME;
}

const std::string& CDataFrameAnalysisSpecificationFactory::regression() {
    return api::CDataFrameTrainBoostedTreeRegressionRunnerFactory::NAME;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::rows(std::size_t rows) {
    m_Rows = rows;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::columns(std::size_t columns) {
    m_Columns = columns;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::memoryLimit(std::size_t memoryLimit) {
    m_MemoryLimit = memoryLimit;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::missingString(const std::string& missing) {
    m_MissingString = missing;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::diskUsageAllowed(bool disk) {
    m_DiskUsageAllowed = disk;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::outlierMethod(std::string method) {
    m_Method = method;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::outlierNumberNeighbours(std::size_t number) {
    m_NumberNeighbours = number;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::outlierComputeInfluence(bool compute) {
    m_ComputeFeatureInfluence = compute;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::predicitionNumberRoundsPerHyperparameter(std::size_t rounds) {
    m_NumberRoundsPerHyperparameter = rounds;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::predictionBayesianOptimisationRestarts(std::size_t restarts) {
    m_BayesianOptimisationRestarts = restarts;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::predictionFieldName(const std::string& name) {
    m_PredictionFieldName = name;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::predictionCategoricalFieldNames(const TStrVec& categorical) {
    m_CategoricalFieldNames = categorical;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::predictionAlpha(double alpha) {
    m_Alpha = alpha;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::predictionLambda(double lambda) {
    m_Lambda = lambda;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::predictionGamma(double gamma) {
    m_Gamma = gamma;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::predictionSoftTreeDepthLimit(double limit) {
    m_SoftTreeDepthLimit = limit;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::predictionSoftTreeDepthTolerance(double tolerance) {
    m_SoftTreeDepthTolerance = tolerance;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::predictionEta(double eta) {
    m_Eta = eta;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::predictionEtaGrowthRatePerTree(double etaGrowthRatePerTree) {
    m_EtaGrowthRatePerTree = etaGrowthRatePerTree;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::predictionMaximumNumberTrees(std::size_t number) {
    m_MaximumNumberTrees = number;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::predictionDownsampleFactor(double downsampleFactor) {
    m_DownsampleFactor = downsampleFactor;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::predictionFeatureBagFraction(double fraction) {
    m_FeatureBagFraction = fraction;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::predictionNumberTopShapValues(std::size_t number) {
    m_NumberTopShapValues = number;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::predictionPersisterSupplier(TPersisterSupplier* persisterSupplier) {
    m_PersisterSupplier = persisterSupplier;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::predictionRestoreSearcherSupplier(
    TRestoreSearcherSupplier* restoreSearcherSupplier) {
    m_RestoreSearcherSupplier = restoreSearcherSupplier;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::earlyStoppingEnabled(bool earlyStoppingEnabled) {
    m_EarlyStoppingEnabled = earlyStoppingEnabled;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::task(TTask task) {
    m_Task = task;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::dataSummarizationFraction(double fraction) {
    m_DataSummarizationFraction = fraction;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::previousTrainLossGap(double lossGap) {
    m_PreviousTrainLossGap = lossGap;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::previousTrainNumberRows(std::size_t number) {
    m_PreviousTrainNumberRows = number;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::numberClasses(std::size_t number) {
    m_NumberClasses = number;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::numberTopClasses(std::size_t number) {
    m_NumberTopClasses = number;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::predictionFieldType(const std::string& type) {
    m_PredictionFieldType = type;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::classificationWeights(const TStrDoublePrVec& weights) {
    m_ClassificationWeights = weights;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::regressionLossFunction(TLossFunctionType lossFunction) {
    m_RegressionLossFunction = lossFunction;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::predictionCustomProcessor(const json::value& value) {
    m_CustomProcessors = value;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::regressionLossFunctionParameter(double lossFunctionParameter) {
    m_RegressionLossFunctionParameter = lossFunctionParameter;
    return *this;
}

std::string CDataFrameAnalysisSpecificationFactory::outlierParams() const {

    std::ostringstream os;
    TJsonLineWriter writer(os);

    writer.onObjectBegin();
    if (m_Method != "") {
        writer.onKey(api::CDataFrameOutliersRunner::METHOD);
        writer.onString(m_Method);
    }
    if (m_NumberNeighbours > 0) {
        writer.onKey(api::CDataFrameOutliersRunner::N_NEIGHBORS);
        writer.onUint64(m_NumberNeighbours);
    }
    if (m_ComputeFeatureInfluence == false) {
        writer.onKey(api::CDataFrameOutliersRunner::COMPUTE_FEATURE_INFLUENCE);
        writer.onBool(m_ComputeFeatureInfluence);
    } else {
        writer.onKey(api::CDataFrameOutliersRunner::FEATURE_INFLUENCE_THRESHOLD);
        writer.onDouble(0.0);
    }
    writer.onObjectEnd();
    writer.flush();

    return os.str();
}

CDataFrameAnalysisSpecificationFactory::TSpecificationUPtr
CDataFrameAnalysisSpecificationFactory::outlierSpec(TDataFrameUPtrTemporaryDirectoryPtrPr* frameAndDirectory) const {

    std::size_t rows{m_Rows ? *m_Rows : 110};
    std::size_t columns{m_Columns ? *m_Columns : 5};
    std::size_t memoryLimit{m_MemoryLimit ? *m_MemoryLimit : 100000};

    std::string spec{api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
        "testJob", rows, columns, memoryLimit, 1, m_MissingString, {},
        m_DiskUsageAllowed, CTestTmpDir::tmpDir(), "ml",
        api::CDataFrameOutliersRunnerFactory::NAME, this->outlierParams())};

    LOG_TRACE(<< "spec =\n" << spec);

    return std::make_unique<api::CDataFrameAnalysisSpecification>(spec, frameAndDirectory);
}

std::string
CDataFrameAnalysisSpecificationFactory::predictionParams(const std::string& analysis,
                                                         const std::string& dependentVariable) const {

    using TRunner = api::CDataFrameTrainBoostedTreeRunner;
    using TClassificationRunner = api::CDataFrameTrainBoostedTreeClassifierRunner;
    using TRegressionRunner = api::CDataFrameTrainBoostedTreeRegressionRunner;

    std::ostringstream os;
    TJsonLineWriter writer(os);

    writer.onObjectBegin();
    writer.onKey(TRunner::DEPENDENT_VARIABLE_NAME);
    writer.onString(dependentVariable);
    if (m_Alpha >= 0.0) {
        writer.onKey(TRunner::ALPHA);
        writer.onArrayBegin();
        writer.onDouble(m_Alpha);
        writer.onArrayEnd();
    }
    if (m_Lambda >= 0.0) {
        writer.onKey(TRunner::LAMBDA);
        writer.onArrayBegin();
        writer.onDouble(m_Lambda);
        writer.onArrayEnd();
    }
    if (m_Gamma >= 0.0) {
        writer.onKey(TRunner::GAMMA);
        writer.onArrayBegin();
        writer.onDouble(m_Gamma);
        writer.onArrayEnd();
    }
    if (m_SoftTreeDepthLimit >= 0.0) {
        writer.onKey(TRunner::SOFT_TREE_DEPTH_LIMIT);
        writer.onArrayBegin();
        writer.onDouble(m_SoftTreeDepthLimit);
        writer.onArrayEnd();
    }
    if (m_SoftTreeDepthTolerance >= 0.0) {
        writer.onKey(TRunner::SOFT_TREE_DEPTH_TOLERANCE);
        writer.onArrayBegin();
        writer.onDouble(m_SoftTreeDepthTolerance);
        writer.onArrayEnd();
    }
    if (m_Eta > 0.0) {
        writer.onKey(TRunner::ETA);
        writer.onArrayBegin();
        writer.onDouble(m_Eta);
        writer.onArrayEnd();
    }
    if (m_EtaGrowthRatePerTree > 0.0) {
        writer.onKey(TRunner::ETA_GROWTH_RATE_PER_TREE);
        writer.onArrayBegin();
        writer.onDouble(m_EtaGrowthRatePerTree);
        writer.onArrayEnd();
    }
    if (m_DownsampleFactor > 0.0) {
        writer.onKey(TRunner::DOWNSAMPLE_FACTOR);
        writer.onArrayBegin();
        writer.onDouble(m_DownsampleFactor);
        writer.onArrayEnd();
    }
    if (m_MaximumNumberTrees > 0) {
        writer.onKey(TRunner::MAX_TREES);
        writer.onUint64(m_MaximumNumberTrees);
    }
    if (m_FeatureBagFraction > 0.0) {
        writer.onKey(TRunner::FEATURE_BAG_FRACTION);
        writer.onArrayBegin();
        writer.onDouble(m_FeatureBagFraction);
        writer.onArrayEnd();
    }
    if (m_NumberRoundsPerHyperparameter > 0) {
        writer.onKey(TRunner::MAX_OPTIMIZATION_ROUNDS_PER_HYPERPARAMETER);
        writer.onUint64(m_NumberRoundsPerHyperparameter);
    }
    if (m_BayesianOptimisationRestarts > 0) {
        writer.onKey(TRunner::BAYESIAN_OPTIMISATION_RESTARTS);
        writer.onUint64(m_BayesianOptimisationRestarts);
    }
    if (m_NumberTopShapValues > 0) {
        writer.onKey(TRunner::NUM_TOP_FEATURE_IMPORTANCE_VALUES);
        writer.onUint64(m_NumberTopShapValues);
    }
    if (m_PredictionFieldName.empty() == false) {
        writer.onKey(TRunner::PREDICTION_FIELD_NAME);
        writer.onString(m_PredictionFieldName);
    }
    if (m_CustomProcessors.is_null() == false) {
        writer.onKey(TRunner::FEATURE_PROCESSORS);
        writer.write(m_CustomProcessors);
    }
    if (m_EarlyStoppingEnabled == false) {
        writer.onKey(TRunner::EARLY_STOPPING_ENABLED);
        writer.onBool(m_EarlyStoppingEnabled);
    }
    if (m_DataSummarizationFraction > 0.0) {
        writer.onKey(TRunner::DATA_SUMMARIZATION_FRACTION);
        writer.onDouble(m_DataSummarizationFraction);
    }
    if (m_PreviousTrainLossGap > 0.0) {
        writer.onKey(TRunner::PREVIOUS_TRAIN_LOSS_GAP);
        writer.onDouble(m_PreviousTrainLossGap);
    }
    if (m_PreviousTrainNumberRows > 0) {
        writer.onKey(TRunner::PREVIOUS_TRAIN_NUM_ROWS);
        writer.onUint64(m_PreviousTrainNumberRows);
    }

    writer.onKey(TRunner::TASK);
    switch (m_Task) {
    case TTask::E_Encode:
        writer.onString(TRunner::TASK_ENCODE);
        break;
    case TTask::E_Train:
        writer.onString(TRunner::TASK_TRAIN);
        break;
    case TTask::E_Update:
        writer.onString(TRunner::TASK_UPDATE);
        break;
    case TTask::E_Predict:
        writer.onString(TRunner::TASK_PREDICT);
        break;
    }

    if (analysis == classification()) {
        writer.onKey(TClassificationRunner::NUM_CLASSES);
        writer.onUint64(m_NumberClasses);
        writer.onKey(TClassificationRunner::NUM_TOP_CLASSES);
        writer.onUint64(m_NumberTopClasses);
        if (m_PredictionFieldType.empty() == false) {
            writer.onKey(TClassificationRunner::PREDICTION_FIELD_TYPE);
            writer.onString(m_PredictionFieldType);
        }
        if (m_ClassificationWeights.empty() == false) {
            writer.onKey(TClassificationRunner::CLASS_ASSIGNMENT_OBJECTIVE);
            writer.onString(
                TClassificationRunner::CLASS_ASSIGNMENT_OBJECTIVE_VALUES[maths::analytics::CDataFramePredictiveModel::E_Custom]);
            writer.onKey(TClassificationRunner::CLASSIFICATION_WEIGHTS);
            writer.onArrayBegin();
            for (const auto& weight : m_ClassificationWeights) {
                writer.onObjectBegin();
                writer.onKey(TClassificationRunner::CLASSIFICATION_WEIGHTS_CLASS);
                writer.onString(weight.first);
                writer.onKey(TClassificationRunner::CLASSIFICATION_WEIGHTS_WEIGHT);
                writer.onDouble(weight.second);
                writer.onObjectEnd();
            }
            writer.onArrayEnd();
        }
    }
    if (analysis == regression()) {
        if (m_RegressionLossFunction) {
            writer.onKey(TRegressionRunner::LOSS_FUNCTION);
            switch (*m_RegressionLossFunction) {
            case TLossFunctionType::E_MsleRegression:
                writer.onString(TRegressionRunner::MSLE);
                break;
            case TLossFunctionType::E_MseRegression:
                writer.onString(TRegressionRunner::MSE);
                break;
            case TLossFunctionType::E_HuberRegression:
                writer.onString(TRegressionRunner::PSEUDO_HUBER);
                break;
            case TLossFunctionType::E_BinaryClassification:
            case TLossFunctionType::E_MulticlassClassification:
                LOG_ERROR(<< "Input error: regression loss type is expected but classification type is provided.");
                break;
            }
        }
        if (m_RegressionLossFunctionParameter != std::nullopt) {
            writer.onKey(TRegressionRunner::LOSS_FUNCTION_PARAMETER);
            writer.onDouble(*m_RegressionLossFunctionParameter);
        }
    }

    writer.onObjectEnd();

    return os.str();
}

CDataFrameAnalysisSpecificationFactory::TSpecificationUPtr
CDataFrameAnalysisSpecificationFactory::predictionSpec(
    const std::string& analysis,
    const std::string& dependentVariable,
    TDataFrameUPtrTemporaryDirectoryPtrPr* frameAndDirectory) const {

    std::size_t rows{m_Rows ? *m_Rows : 100};
    std::size_t columns{m_Columns ? *m_Columns : 5};
    std::size_t memoryLimit{m_MemoryLimit ? *m_MemoryLimit : 7000000};

    std::string spec{api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
        "testJob", rows, columns, memoryLimit, 1, m_MissingString,
        m_CategoricalFieldNames, true, CTestTmpDir::tmpDir(), "ml", analysis,
        this->predictionParams(analysis, dependentVariable))};

    LOG_DEBUG(<< "spec =\n" << spec);

    if (m_RestoreSearcherSupplier != nullptr && m_PersisterSupplier != nullptr) {
        return std::make_unique<api::CDataFrameAnalysisSpecification>(
            spec, frameAndDirectory, *m_PersisterSupplier, *m_RestoreSearcherSupplier);
    }
    if (m_RestoreSearcherSupplier == nullptr && m_PersisterSupplier != nullptr) {
        return std::make_unique<api::CDataFrameAnalysisSpecification>(
            spec, frameAndDirectory, *m_PersisterSupplier);
    }
    if (m_RestoreSearcherSupplier != nullptr && m_PersisterSupplier == nullptr) {
        return std::make_unique<api::CDataFrameAnalysisSpecification>(
            spec, frameAndDirectory, api::CDataFrameAnalysisSpecification::noopPersisterSupplier,
            *m_RestoreSearcherSupplier);
    }
    return std::make_unique<api::CDataFrameAnalysisSpecification>(spec, frameAndDirectory);
}
}
}
