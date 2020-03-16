/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
using TRapidJsonLineWriter = core::CRapidJsonLineWriter<rapidjson::StringBuffer>;

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
CDataFrameAnalysisSpecificationFactory::predictionMaximumNumberTrees(std::size_t number) {
    m_MaximumNumberTrees = number;
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
CDataFrameAnalysisSpecificationFactory::numberClasses(std::size_t number) {
    m_NumberClasses = number;
    return *this;
}

CDataFrameAnalysisSpecificationFactory&
CDataFrameAnalysisSpecificationFactory::predictionFieldType(const std::string& type) {
    m_PredictionFieldType = type;
    return *this;
}

std::string CDataFrameAnalysisSpecificationFactory::outlierParams() const {

    rapidjson::StringBuffer parameters;
    TRapidJsonLineWriter writer;
    writer.Reset(parameters);

    writer.StartObject();
    if (m_Method != "") {
        writer.Key(api::CDataFrameOutliersRunner::METHOD);
        writer.String(m_Method);
    }
    if (m_NumberNeighbours > 0) {
        writer.Key(api::CDataFrameOutliersRunner::N_NEIGHBORS);
        writer.Uint64(m_NumberNeighbours);
    }
    if (m_ComputeFeatureInfluence == false) {
        writer.Key(api::CDataFrameOutliersRunner::COMPUTE_FEATURE_INFLUENCE);
        writer.Bool(m_ComputeFeatureInfluence);
    } else {
        writer.Key(api::CDataFrameOutliersRunner::FEATURE_INFLUENCE_THRESHOLD);
        writer.Double(0.0);
    }
    writer.EndObject();
    writer.Flush();

    return parameters.GetString();
}

CDataFrameAnalysisSpecificationFactory::TSpecificationUPtr
CDataFrameAnalysisSpecificationFactory::outlierSpec() const {

    std::size_t rows{m_Rows ? *m_Rows : 110};
    std::size_t columns{m_Columns ? *m_Columns : 5};
    std::size_t memoryLimit{m_MemoryLimit ? *m_MemoryLimit : 100000};

    std::string spec{api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
        "testJob", rows, columns, memoryLimit, 1, m_MissingString, {},
        m_DiskUsageAllowed, CTestTmpDir::tmpDir(), "ml",
        api::CDataFrameOutliersRunnerFactory::NAME, this->outlierParams())};

    LOG_TRACE(<< "spec =\n" << spec);

    return std::make_unique<api::CDataFrameAnalysisSpecification>(spec);
}

std::string
CDataFrameAnalysisSpecificationFactory::predictionParams(const std::string& analysis,
                                                         const std::string& dependentVariable) const {

    rapidjson::StringBuffer parameters;
    TRapidJsonLineWriter writer;
    writer.Reset(parameters);

    writer.StartObject();
    writer.Key(api::CDataFrameTrainBoostedTreeRunner::DEPENDENT_VARIABLE_NAME);
    writer.String(dependentVariable);
    if (m_Alpha >= 0.0) {
        writer.Key(api::CDataFrameTrainBoostedTreeRunner::ALPHA);
        writer.Double(m_Alpha);
    }
    if (m_Lambda >= 0.0) {
        writer.Key(api::CDataFrameTrainBoostedTreeRunner::LAMBDA);
        writer.Double(m_Lambda);
    }
    if (m_Gamma >= 0.0) {
        writer.Key(api::CDataFrameTrainBoostedTreeRunner::GAMMA);
        writer.Double(m_Gamma);
    }
    if (m_SoftTreeDepthLimit >= 0.0) {
        writer.Key(api::CDataFrameTrainBoostedTreeRunner::SOFT_TREE_DEPTH_LIMIT);
        writer.Double(m_SoftTreeDepthLimit);
    }
    if (m_SoftTreeDepthTolerance >= 0.0) {
        writer.Key(api::CDataFrameTrainBoostedTreeRunner::SOFT_TREE_DEPTH_TOLERANCE);
        writer.Double(m_SoftTreeDepthTolerance);
    }
    if (m_Eta > 0.0) {
        writer.Key(api::CDataFrameTrainBoostedTreeRunner::ETA);
        writer.Double(m_Eta);
    }
    if (m_MaximumNumberTrees > 0) {
        writer.Key(api::CDataFrameTrainBoostedTreeRunner::MAX_TREES);
        writer.Uint64(m_MaximumNumberTrees);
    }
    if (m_FeatureBagFraction > 0.0) {
        writer.Key(api::CDataFrameTrainBoostedTreeRunner::FEATURE_BAG_FRACTION);
        writer.Double(m_FeatureBagFraction);
    }
    if (m_NumberRoundsPerHyperparameter > 0) {
        writer.Key(api::CDataFrameTrainBoostedTreeRunner::NUMBER_ROUNDS_PER_HYPERPARAMETER);
        writer.Uint64(m_NumberRoundsPerHyperparameter);
    }
    if (m_BayesianOptimisationRestarts > 0) {
        writer.Key(api::CDataFrameTrainBoostedTreeRunner::BAYESIAN_OPTIMISATION_RESTARTS);
        writer.Uint64(m_BayesianOptimisationRestarts);
    }
    if (m_NumberTopShapValues > 0) {
        writer.Key(api::CDataFrameTrainBoostedTreeRunner::NUM_TOP_FEATURE_IMPORTANCE_VALUES);
        writer.Uint64(m_NumberTopShapValues);
    }
    if (m_PredictionFieldName.empty() == false) {
        writer.Key(api::CDataFrameTrainBoostedTreeRunner::PREDICTION_FIELD_NAME);
        writer.String(m_PredictionFieldName);
    }
    if (m_PredictionFieldType.empty() == false) {
        writer.Key(api::CDataFrameTrainBoostedTreeClassifierRunner::PREDICTION_FIELD_TYPE);
        writer.String(m_PredictionFieldType);
    }
    if (analysis == classification()) {
        writer.Key(api::CDataFrameTrainBoostedTreeClassifierRunner::NUM_CLASSES);
        writer.Uint64(m_NumberClasses);
        writer.Key(api::CDataFrameTrainBoostedTreeClassifierRunner::NUM_TOP_CLASSES);
        writer.Uint64(1);
    }
    writer.EndObject();

    return parameters.GetString();
}

CDataFrameAnalysisSpecificationFactory::TSpecificationUPtr
CDataFrameAnalysisSpecificationFactory::predictionSpec(const std::string& analysis,
                                                       const std::string& dependentVariable) const {

    std::size_t rows{m_Rows ? *m_Rows : 100};
    std::size_t columns{m_Columns ? *m_Columns : 5};
    std::size_t memoryLimit{m_MemoryLimit ? *m_MemoryLimit : 7000000};

    std::string spec{api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
        "testJob", rows, columns, memoryLimit, 1, m_MissingString,
        m_CategoricalFieldNames, true, CTestTmpDir::tmpDir(), "ml", analysis,
        this->predictionParams(analysis, dependentVariable))};

    LOG_TRACE(<< "spec =\n" << spec);

    if (m_RestoreSearcherSupplier != nullptr && m_PersisterSupplier != nullptr) {
        return std::make_unique<api::CDataFrameAnalysisSpecification>(
            spec, *m_PersisterSupplier, *m_RestoreSearcherSupplier);
    } else if (m_RestoreSearcherSupplier == nullptr && m_PersisterSupplier != nullptr) {
        return std::make_unique<api::CDataFrameAnalysisSpecification>(spec, *m_PersisterSupplier);
    } else {
        return std::make_unique<api::CDataFrameAnalysisSpecification>(spec);
    }
}
}
}
