/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <test/CDataFrameAnalysisSpecificationFactory.h>

#include <core/CStringUtils.h>

#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameAnalysisSpecificationJsonWriter.h>

#include <test/CTestTmpDir.h>

#include <memory>

using namespace ml;

test::CDataFrameAnalysisSpecificationFactory::TSpecificationUPtr
test::CDataFrameAnalysisSpecificationFactory::outlierSpec(std::size_t rows,
                                                          std::size_t cols,
                                                          std::size_t memoryLimit,
                                                          const std::string& method,
                                                          std::size_t numberNeighbours,
                                                          bool computeFeatureInfluence) {
    std::string parameters = "{\n";
    bool hasTrailingParameter{false};
    if (method != "") {
        parameters += "\"method\": \"" + method + "\"";
        hasTrailingParameter = true;
    }
    if (numberNeighbours > 0) {
        parameters += (hasTrailingParameter ? ",\n" : "");
        parameters += "\"n_neighbors\": " + core::CStringUtils::typeToString(numberNeighbours);
        hasTrailingParameter = true;
    }
    if (computeFeatureInfluence == false) {
        parameters += (hasTrailingParameter ? ",\n" : "");
        parameters += "\"compute_feature_influence\": false";
        hasTrailingParameter = true;
    } else {
        parameters += (hasTrailingParameter ? ",\n" : "");
        parameters += "\"feature_influence_threshold\": 0.0";
        hasTrailingParameter = true;
    }
    parameters += (hasTrailingParameter ? "\n" : "");
    parameters += "}\n";

    std::string spec{api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
        "testJob", rows, cols, memoryLimit, 1, {}, true,
        test::CTestTmpDir::tmpDir(), "ml", "outlier_detection", parameters)};

    LOG_DEBUG(<< "spec =\n" << spec);

    return std::make_unique<api::CDataFrameAnalysisSpecification>(spec);
}

test::CDataFrameAnalysisSpecificationFactory::TSpecificationUPtr
test::CDataFrameAnalysisSpecificationFactory::predictionSpec(
    const std::string& analysis,
    const std::string& dependentVariable,
    std::size_t rows,
    std::size_t cols,
    std::size_t memoryLimit,
    std::size_t numberRoundsPerHyperparameter,
    std::size_t bayesianOptimisationRestarts,
    const ml::test::CDataFrameAnalysisSpecificationFactory::TStrVec& categoricalFieldNames,
    double alpha,
    double lambda,
    double gamma,
    double softTreeDepthLimit,
    double softTreeDepthTolerance,
    double eta,
    std::size_t maximumNumberTrees,
    double featureBagFraction,
    ml::test::CDataFrameAnalysisSpecificationFactory::TPersisterSupplier* persisterSupplier,
    ml::test::CDataFrameAnalysisSpecificationFactory::TRestoreSearcherSupplier* restoreSearcherSupplier) {
    std::string parameters = "{\n\"dependent_variable\": \"" + dependentVariable + "\"";
    if (alpha >= 0.0) {
        parameters += ",\n\"alpha\": " + core::CStringUtils::typeToString(alpha);
    }
    if (lambda >= 0.0) {
        parameters += ",\n\"lambda\": " + core::CStringUtils::typeToString(lambda);
    }
    if (gamma >= 0.0) {
        parameters += ",\n\"gamma\": " + core::CStringUtils::typeToString(gamma);
    }
    if (softTreeDepthLimit >= 0.0) {
        parameters += ",\n\"soft_tree_depth_limit\": " +
                      core::CStringUtils::typeToString(softTreeDepthLimit);
    }
    if (softTreeDepthTolerance >= 0.0) {
        parameters += ",\n\"soft_tree_depth_tolerance\": " +
                      core::CStringUtils::typeToString(softTreeDepthTolerance);
    }
    if (eta > 0.0) {
        parameters += ",\n\"eta\": " + core::CStringUtils::typeToString(eta);
    }
    if (maximumNumberTrees > 0) {
        parameters += ",\n\"maximum_number_trees\": " +
                      core::CStringUtils::typeToString(maximumNumberTrees);
    }
    if (featureBagFraction > 0.0) {
        parameters += ",\n\"feature_bag_fraction\": " +
                      core::CStringUtils::typeToString(featureBagFraction);
    }
    if (numberRoundsPerHyperparameter > 0) {
        parameters += ",\n\"number_rounds_per_hyperparameter\": " +
                      core::CStringUtils::typeToString(numberRoundsPerHyperparameter);
    }
    if (bayesianOptimisationRestarts > 0) {
        parameters += ",\n\"bayesian_optimisation_restarts\": " +
                      core::CStringUtils::typeToString(bayesianOptimisationRestarts);
    }
    parameters += "\n}";

    std::string spec{api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
        "testJob", rows, cols, memoryLimit, 1, categoricalFieldNames, true,
        test::CTestTmpDir::tmpDir(), "ml", analysis, parameters)};

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

test::CDataFrameAnalysisSpecificationFactory::TSpecificationUPtr
test::CDataFrameAnalysisSpecificationFactory::diskUsageTestSpec(std::size_t rows,
                                                                std::size_t cols,
                                                                bool diskUsageAllowed) {
    std::string spec{api::CDataFrameAnalysisSpecificationJsonWriter::jsonString(
        "testJob", rows, cols, 500000, 1, {}, diskUsageAllowed,
        test::CTestTmpDir::tmpDir(), "", "outlier_detection", "")};
    return std::make_unique<api::CDataFrameAnalysisSpecification>(spec);
}
