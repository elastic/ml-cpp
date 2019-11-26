/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <test/CDataFrameAnalysisSpecificationFactory.h>

#include <core/CStringUtils.h>

#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameAnalysisSpecificationJsonWriter.h>
#include <api/CDataFrameOutliersRunner.h>
#include <api/CDataFrameTrainBoostedTreeRunner.h>

#include <test/CTestTmpDir.h>

#include <memory>

namespace ml {
namespace test {
using TRapidJsonLineWriter = core::CRapidJsonLineWriter<rapidjson::StringBuffer>;

CDataFrameAnalysisSpecificationFactory::TSpecificationUPtr
CDataFrameAnalysisSpecificationFactory::outlierSpec(std::size_t rows,
                                                    std::size_t cols,
                                                    std::size_t memoryLimit,
                                                    const std::string& method,
                                                    std::size_t numberNeighbours,
                                                    bool computeFeatureInfluence,
                                                    bool diskUsageAllowed) {

    rapidjson::StringBuffer parameters;
    TRapidJsonLineWriter writer;
    writer.Reset(parameters);

    writer.StartObject();
    if (method != "") {
        writer.Key(api::CDataFrameOutliersRunner::METHOD);
        writer.String(method);
    }
    if (numberNeighbours > 0) {
        writer.Key(api::CDataFrameOutliersRunner::N_NEIGHBORS);
        writer.Uint64(numberNeighbours);
    }
    if (computeFeatureInfluence == false) {
        writer.Key(api::CDataFrameOutliersRunner::COMPUTE_FEATURE_INFLUENCE);
        writer.Bool(computeFeatureInfluence);
    } else {
        writer.Key(api::CDataFrameOutliersRunner::FEATURE_INFLUENCE_THRESHOLD);
        writer.Double(0.0);
    }
    writer.EndObject();
    writer.Flush();

    std::string spec{api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
        "testJob", rows, cols, memoryLimit, 1, {}, diskUsageAllowed, CTestTmpDir::tmpDir(),
        "ml", api::CDataFrameOutliersRunnerFactory::NAME, parameters.GetString())};

    LOG_TRACE(<< "spec =\n" << spec);

    return std::make_unique<api::CDataFrameAnalysisSpecification>(spec);
}

CDataFrameAnalysisSpecificationFactory::TSpecificationUPtr CDataFrameAnalysisSpecificationFactory::predictionSpec(
        const std::string& analysis,
        const std::string& dependentVariable,
        std::size_t rows,
        std::size_t cols,
        std::size_t memoryLimit,
        std::size_t numberRoundsPerHyperparameter,
        std::size_t bayesianOptimisationRestarts,
        const TStrVec& categoricalFieldNames,
        double alpha,
        double lambda,
        double gamma,
        double softTreeDepthLimit,
        double softTreeDepthTolerance,
        double eta,
        std::size_t maximumNumberTrees,
        double featureBagFraction,
        size_t topShapValues,
        TPersisterSupplier* persisterSupplier,
        TRestoreSearcherSupplier* restoreSearcherSupplier) {

    rapidjson::StringBuffer parameters;
    TRapidJsonLineWriter writer;
    writer.Reset(parameters);

    writer.StartObject();
    writer.Key(api::CDataFrameTrainBoostedTreeRunner::DEPENDENT_VARIABLE_NAME);
    writer.String(dependentVariable);
    if (alpha >= 0.0) {
        writer.Key(api::CDataFrameTrainBoostedTreeRunner::ALPHA);
        writer.Double(alpha);
    }
    if (lambda >= 0.0) {
        writer.Key(api::CDataFrameTrainBoostedTreeRunner::LAMBDA);
        writer.Double(lambda);
    }
    if (gamma >= 0.0) {
        writer.Key(api::CDataFrameTrainBoostedTreeRunner::GAMMA);
        writer.Double(gamma);
    }
    if (softTreeDepthLimit >= 0.0) {
        writer.Key(api::CDataFrameTrainBoostedTreeRunner::SOFT_TREE_DEPTH_LIMIT);
        writer.Double(softTreeDepthLimit);
    }
    if (softTreeDepthTolerance >= 0.0) {
        writer.Key(api::CDataFrameTrainBoostedTreeRunner::SOFT_TREE_DEPTH_TOLERANCE);
        writer.Double(softTreeDepthTolerance);
    }
    if (eta > 0.0) {
        writer.Key(api::CDataFrameTrainBoostedTreeRunner::ETA);
        writer.Double(eta);
    }
    if (maximumNumberTrees > 0) {
        writer.Key(api::CDataFrameTrainBoostedTreeRunner::MAXIMUM_NUMBER_TREES);
        writer.Uint64(maximumNumberTrees);
    }
    if (featureBagFraction > 0.0) {
        writer.Key(api::CDataFrameTrainBoostedTreeRunner::FEATURE_BAG_FRACTION);
        writer.Double(featureBagFraction);
    }
    if (numberRoundsPerHyperparameter > 0) {
        writer.Key(api::CDataFrameTrainBoostedTreeRunner::NUMBER_ROUNDS_PER_HYPERPARAMETER);
        writer.Uint64(numberRoundsPerHyperparameter);
    }
    if (bayesianOptimisationRestarts > 0) {
        writer.Key(api::CDataFrameTrainBoostedTreeRunner::BAYESIAN_OPTIMISATION_RESTARTS);
        writer.Uint64(bayesianOptimisationRestarts);
    }
    if (topShapValues > 0) {
        writer.Key(api::CDataFrameTrainBoostedTreeRunner::TOP_SHAP_VALUES);
        writer.Uint64(topShapValues);
    }
    writer.EndObject();

    std::string spec{api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
        "testJob", rows, cols, memoryLimit, 1, categoricalFieldNames, true,
        CTestTmpDir::tmpDir(), "ml", analysis, parameters.GetString())};

    LOG_TRACE(<< "spec =\n" << spec);

    if (restoreSearcherSupplier != nullptr && persisterSupplier != nullptr) {
        return std::make_unique<api::CDataFrameAnalysisSpecification>(
            spec, *persisterSupplier, *restoreSearcherSupplier);
    } else if (restoreSearcherSupplier == nullptr && persisterSupplier != nullptr) {
        return std::make_unique<api::CDataFrameAnalysisSpecification>(spec, *persisterSupplier);
    } else {
        return std::make_unique<api::CDataFrameAnalysisSpecification>(spec);
    }
}
}
}
