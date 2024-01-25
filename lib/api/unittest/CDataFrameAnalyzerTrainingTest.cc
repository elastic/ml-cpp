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

#include <core/CContainerPrinter.h>
#include <core/CDataFrame.h>
#include <core/CDataSearcher.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CProgramCounters.h>
#include <core/CStateDecompressor.h>
#include <core/CStopWatch.h>

#include <maths/analytics/CBoostedTree.h>
#include <maths/analytics/CBoostedTreeFactory.h>
#include <maths/analytics/CBoostedTreeHyperparameters.h>
#include <maths/analytics/CBoostedTreeLoss.h>
#include <maths/analytics/CDataFrameUtils.h>

#include <maths/common/CBasicStatistics.h>

#include <api/CDataFrameAnalyzer.h>
#include <api/CDataFrameTrainBoostedTreeRegressionRunner.h>
#include <api/CSingleStreamDataAdder.h>
#include <api/CSingleStreamSearcher.h>
#include <api/ElasticsearchStateIndex.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CDataFrameAnalysisSpecificationFactory.h>
#include <test/CDataFrameAnalyzerTrainingFactory.h>
#include <test/CRandomNumbers.h>

#include <boost/make_shared.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/unordered_map.hpp>

#include <algorithm>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

using TDoubleVec = std::vector<double>;
using TStrVec = std::vector<std::string>;
BOOST_TEST_DONT_PRINT_LOG_VALUE(TDoubleVec::iterator)
BOOST_TEST_DONT_PRINT_LOG_VALUE(TStrVec::iterator)

BOOST_AUTO_TEST_SUITE(CDataFrameAnalyzerTrainingTest)

using namespace ml;

namespace {
using TBoolVec = std::vector<bool>;
using TSizeVec = std::vector<std::size_t>;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TRowItr = core::CDataFrame::TRowItr;
using TRowRef = core::CDataFrame::TRowRef;
using TDataFrameUPtr = std::unique_ptr<core::CDataFrame>;
using TDataAdderUPtr = std::unique_ptr<core::CDataAdder>;
using TPersisterSupplier = std::function<TDataAdderUPtr()>;
using TDataSearcherUPtr = std::unique_ptr<core::CDataSearcher>;
using TRestoreSearcherSupplier = std::function<TDataSearcherUPtr()>;
using TLossFunctionType = maths::analytics::boosted_tree::ELossType;
using TDataFrameUPtrTemporaryDirectoryPtrPr =
    test::CDataFrameAnalysisSpecificationFactory::TDataFrameUPtrTemporaryDirectoryPtrPr;

class CTestDataSearcher : public core::CDataSearcher {
public:
    explicit CTestDataSearcher(const std::string& data)
        : m_Stream(new std::istringstream(data)) {}

    TIStreamP search(std::size_t /*currentDocNum*/, std::size_t /*limit*/) override {
        std::istringstream* intermediateStateStream{
            static_cast<std::istringstream*>(m_Stream.get())};
        // Discard first line, which contains the state id.
        intermediateStateStream->ignore(256, '\n');
        std::string intermediateState;
        std::getline(*intermediateStateStream, intermediateState);
        return std::make_shared<std::istringstream>(intermediateState);
    }

private:
    TIStreamP m_Stream;
};

TStrVec splitOnNull(std::stringstream&& tokenStream) {
    TStrVec results;
    std::string token;
    while (std::getline(tokenStream, token, '\0')) {
        results.push_back(token);
    }
    return results;
}

json::object treeToJsonDocument(const maths::analytics::CBoostedTree& tree) {
    std::stringstream persistStream;
    {
        core::CJsonStatePersistInserter inserter(persistStream);
        tree.acceptPersistInserter(inserter);
        persistStream.flush();
    }
    json::error_code ec;
    json::value results = json::parse(persistStream.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    BOOST_TEST_REQUIRE(results.is_object());
    return results.as_object();
}

auto restoreTree(std::string persistedState, TDataFrameUPtr& frame, std::size_t dependentVariable) {
    CTestDataSearcher dataSearcher(persistedState);
    auto decompressor = std::make_unique<core::CStateDecompressor>(dataSearcher);
    auto stream = decompressor->search(1, 1);
    return maths::analytics::CBoostedTreeFactory::constructFromString(*stream).restoreFor(
        *frame, dependentVariable);
}

auto generateCategoricalData(test::CRandomNumbers& rng,
                             std::size_t rows,
                             const TDoubleVec& expectedFrequencies) {

    TDoubleVecVec frequencies;
    rng.generateDirichletSamples(expectedFrequencies, 1, frequencies);

    TDoubleVec values(1);
    for (std::size_t j = 0; j < frequencies[0].size(); ++j) {
        std::size_t target{static_cast<std::size_t>(
            static_cast<double>(rows) * frequencies[0][j] + 0.5)};
        values.resize(values.size() + target, static_cast<double>(j));
    }
    values.resize(rows, values.back());
    rng.random_shuffle(values.begin(), values.end());
    rng.discard(1000000); // Make sure the categories are not correlated

    return values;
}

template<typename F>
void testOneRunOfBoostedTreeTrainingWithStateRecovery(
    F makeSpec,
    std::size_t iterationToRestartFrom,
    TLossFunctionType lossFunction = TLossFunctionType::E_MseRegression) {

    std::stringstream outputStream;
    auto outputWriterFactory = [&outputStream]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(outputStream);
    };

    std::size_t numberExamples{100};
    TStrVec fieldNames{"c1", "c2", "c3", "c4", "target", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    TDoubleVec weights{0.1, 2.0, 0.4, -0.5};
    TDoubleVec regressors;
    test::CRandomNumbers rng;
    rng.generateUniformSamples(-10.0, 10.0, weights.size() * numberExamples, regressors);

    auto persistenceStream = std::make_shared<std::ostringstream>();
    TPersisterSupplier persisterSupplier{[&persistenceStream]() {
        return std::make_unique<api::CSingleStreamDataAdder>(persistenceStream);
    }};

    // Compute expected tree.

    TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
    auto spec = makeSpec("target", numberExamples, frameAndDirectory,
                         &persisterSupplier, nullptr);
    api::CDataFrameAnalyzer analyzer{std::move(spec), std::move(frameAndDirectory),
                                     outputWriterFactory};
    std::size_t dependentVariable(std::find(fieldNames.begin(), fieldNames.end(), "target") -
                                  fieldNames.begin());

    TStrVec targets;

    // Avoid negative targets for MSLE.
    auto targetTransformer = [&lossFunction](double x) {
        return (lossFunction == TLossFunctionType::E_MsleRegression) ? x * x : x;
    };
    auto frame = test::CDataFrameAnalyzerTrainingFactory::setupLinearRegressionData(
        fieldNames, fieldValues, analyzer, weights, regressors, targets, targetTransformer);
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    TStrVec persistedStates{splitOnNull(std::stringstream{persistenceStream->str()})};
    auto expectedTree = restoreTree(std::move(persistedStates.back()), frame, dependentVariable);

    persistenceStream->str("");
    persistenceStream->clear();

    std::string intermediateStateStream{persistedStates[iterationToRestartFrom]};
    TRestoreSearcherSupplier restorerSupplier{[&intermediateStateStream]() {
        return std::make_unique<CTestDataSearcher>(intermediateStateStream);
    }};

    spec = makeSpec("target", numberExamples, frameAndDirectory,
                    &persisterSupplier, &restorerSupplier);
    api::CDataFrameAnalyzer restoredAnalyzer{
        std::move(spec), std::move(frameAndDirectory), outputWriterFactory};

    targets.clear();
    test::CDataFrameAnalyzerTrainingFactory::setupLinearRegressionData(
        fieldNames, fieldValues, restoredAnalyzer, weights, regressors, targets,
        targetTransformer);
    restoredAnalyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    persistedStates = splitOnNull(std::stringstream{persistenceStream->str()});
    auto actualTree = restoreTree(std::move(persistedStates.back()), frame, dependentVariable);

    // Compare hyperparameters.

    json::object expectedResults{treeToJsonDocument(*expectedTree)};
    const json::object& expectedHyperparameters =
        expectedResults.at("hyperparameters").as_object();

    json::object actualResults{treeToJsonDocument(*actualTree)};
    const auto& actualHyperparameters = actualResults["hyperparameters"];

    for (const auto& key : maths::analytics::CBoostedTreeHyperparameters::names()) {
        if (expectedHyperparameters.contains(key)) {
            double expected{std::stod(expectedHyperparameters.at(key)
                                          .as_object()
                                          .at("value")
                                          .as_string()
                                          .c_str())};
            double actual{std::stod(actualHyperparameters.at(key)
                                        .as_object()
                                        .at("value")
                                        .as_string()
                                        .c_str())};
            BOOST_REQUIRE_CLOSE(expected, actual, 1e-3);
        } else {
            BOOST_FAIL("Missing " + key);
        }
    }
}

void testRegressionTrainingWithParams(TLossFunctionType lossFunction) {

    // Test the regression hyperparameter settings are correctly propagated to the
    // analysis runner.

    std::size_t numberSamples{100};
    double alpha{2.0};
    double lambda{1.0};
    double gamma{10.0};
    double softTreeDepthLimit{3.0};
    double softTreeDepthTolerance{0.1};
    double downsampleFactor{0.3};
    double eta{0.9};
    double etaGrowthRatePerTree{1.2};
    std::size_t maximumNumberTrees{2};
    double featureBagFraction{0.3};

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
    auto spec = test::CDataFrameAnalysisSpecificationFactory{}
                    .predictionAlpha(alpha)
                    .predictionLambda(lambda)
                    .predictionGamma(gamma)
                    .predictionSoftTreeDepthLimit(softTreeDepthLimit)
                    .predictionSoftTreeDepthTolerance(softTreeDepthTolerance)
                    .predictionDownsampleFactor(downsampleFactor)
                    .predictionEta(eta)
                    .predictionEtaGrowthRatePerTree(etaGrowthRatePerTree)
                    .predictionMaximumNumberTrees(maximumNumberTrees)
                    .predictionFeatureBagFraction(featureBagFraction)
                    .regressionLossFunction(lossFunction)
                    .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::regression(),
                                    "target", &frameAndDirectory);
    api::CDataFrameAnalyzer analyzer{std::move(spec), std::move(frameAndDirectory),
                                     std::move(outputWriterFactory)};

    TDoubleVec expectedPredictions;

    TStrVec fieldNames{"f1", "f2", "f3", "f4", "target", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
        lossFunction, fieldNames, fieldValues, analyzer, expectedPredictions,
        numberSamples, alpha, lambda, gamma, softTreeDepthLimit, softTreeDepthTolerance,
        eta, maximumNumberTrees, downsampleFactor, featureBagFraction);
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    // Check the hyperparameter values match the overrides.
    const auto* runner{dynamic_cast<const api::CDataFrameTrainBoostedTreeRegressionRunner*>(
        analyzer.runner())};
    const auto& boostedTree{runner->boostedTree()};
    const auto& hyperparameters{boostedTree.hyperparameters()};
    BOOST_TEST_REQUIRE(hyperparameters.eta().value() == eta);
    BOOST_TEST_REQUIRE(hyperparameters.featureBagFraction().value() == featureBagFraction);
    BOOST_TEST_REQUIRE(hyperparameters.downsampleFactor().value() == downsampleFactor);
    BOOST_TEST_REQUIRE(hyperparameters.etaGrowthRatePerTree().value() == etaGrowthRatePerTree);
    BOOST_TEST_REQUIRE(hyperparameters.depthPenaltyMultiplier().value() == alpha);
    BOOST_TEST_REQUIRE(hyperparameters.leafWeightPenaltyMultiplier().value() == lambda);
    BOOST_TEST_REQUIRE(hyperparameters.treeSizePenaltyMultiplier().value() == gamma);
    BOOST_TEST_REQUIRE(hyperparameters.softTreeDepthLimit().value() == softTreeDepthLimit);
    BOOST_TEST_REQUIRE(hyperparameters.softTreeDepthTolerance().value() == softTreeDepthTolerance);

    json::error_code ec;
    json::value results = json::parse(output.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);

    auto expectedPrediction = expectedPredictions.begin();
    bool progressCompleted{false};
    for (const auto& result_ : results.as_array()) {
        BOOST_TEST_REQUIRE(result_.is_object());
        const json::object& result = result_.as_object();
        if (result.contains("row_results")) {
            BOOST_TEST_REQUIRE(expectedPrediction != expectedPredictions.end());
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                *expectedPrediction,
                result_.at_pointer("/row_results/results/ml/target_prediction").as_double(),
                1e-4 * std::fabs(*expectedPrediction));
            ++expectedPrediction;
            BOOST_TEST_REQUIRE(result.contains("progress_percent") == false);
        } else if (result.contains("phase_progress")) {
            BOOST_TEST_REQUIRE(
                result_.at_pointer("/phase_progress/progress_percent").as_int64() >= 0);
            BOOST_TEST_REQUIRE(
                result_.at_pointer("/phase_progress/progress_percent").as_int64() <= 100);
            BOOST_TEST_REQUIRE(result.contains("row_results") == false);
            progressCompleted =
                result_.at_pointer("/phase_progress/progress_percent").as_int64() == 100;
        }
    }
    BOOST_TEST_REQUIRE(expectedPrediction == expectedPredictions.end());
    BOOST_TEST_REQUIRE(progressCompleted);
}

void readIncrementalTrainingState(const std::string& resultsJson,
                                  double& alpha,
                                  double& lambda,
                                  double& gamma,
                                  double& softTreeDepthLimit,
                                  double& softTreeDepthTolerance,
                                  double& eta,
                                  double& etaGrowthRatePerTree,
                                  double& downsampleFactor,
                                  double& featureBagFraction,
                                  double& lossGap,
                                  std::ostream& incrementalTrainingState) {

    json::error_code ec;
    json::value results = json::parse(resultsJson, ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    BOOST_TEST_REQUIRE(results.is_array());

    std::stringstream inferenceModelStream;
    core::CStreamWriter inferenceModelWriter{inferenceModelStream};

    core::CStreamWriter dataSummarizationWriter{incrementalTrainingState};

    LOG_DEBUG(<< "Results size: " << results.as_array().size() << ", " << results);
    // Read the state used to initialize incremental training.
    int i{0};
    for (const auto& result_ : results.as_array()) {
        LOG_DEBUG(<< "result_[" << i << "]: " << result_);
        BOOST_TEST_REQUIRE(result_.is_object());
        const json::object& result = result_.as_object();
        if (result.contains("compressed_inference_model")) {
            inferenceModelWriter.write(result);
            LOG_DEBUG(<< "Inference Model definition found");
        } else if (result.contains("compressed_data_summarization")) {
            dataSummarizationWriter.write(result);
            LOG_DEBUG(<< "Data summarization found");
        } else if (result.contains("model_metadata")) {
            LOG_DEBUG(<< "Metadata found");
            json::error_code ec;
            for (const auto& item :
                 result_.at_pointer("/model_metadata/hyperparameters").as_array()) {
                LOG_DEBUG(<< "/model_metadata/hyperparameters: " << item);
                if (item.at("name").as_string() == "alpha") {
                    alpha = item.at("value").to_number<double>(ec);
                    BOOST_REQUIRE(ec.failed() == false);
                } else if (item.at("name").as_string() == "lambda") {
                    lambda = item.at("value").to_number<double>(ec);
                    BOOST_REQUIRE(ec.failed() == false);
                } else if (item.at("name").as_string() == "gamma") {
                    gamma = item.at("value").to_number<double>(ec);
                    BOOST_REQUIRE(ec.failed() == false);
                } else if (item.at("name").as_string() == "soft_tree_depth_limit") {
                    softTreeDepthLimit = item.at("value").to_number<double>(ec);
                    BOOST_REQUIRE(ec.failed() == false);
                } else if (item.at("name").as_string() == "soft_tree_depth_tolerance") {
                    softTreeDepthTolerance = item.at("value").to_number<double>(ec);
                    BOOST_REQUIRE(ec.failed() == false);
                } else if (item.at("name").as_string() == "eta") {
                    eta = item.at("value").to_number<double>(ec);
                    BOOST_REQUIRE(ec.failed() == false);
                } else if (item.at("name").as_string() == "eta_growth_rate_per_tree") {
                    etaGrowthRatePerTree = item.at("value").to_number<double>(ec);
                    BOOST_REQUIRE(ec.failed() == false);
                } else if (item.at("name").as_string() == "downsample_factor") {
                    downsampleFactor = item.at("value").to_number<double>(ec);
                    BOOST_REQUIRE(ec.failed() == false);
                } else if (item.at("name").as_string() == "feature_bag_fraction") {
                    featureBagFraction = item.at("value").to_number<double>(ec);
                    BOOST_REQUIRE(ec.failed() == false);
                }
            }
            if (result.at("model_metadata").as_object().contains("train_properties") &&
                result_.at_pointer("/model_metadata/train_properties")
                    .as_object()
                    .contains("loss_gap")) {
                lossGap = result_
                              .at_pointer("/model_metadata/train_properties/loss_gap")
                              .to_number<double>(ec);
                BOOST_REQUIRE(ec.failed() == false);
            }
        }
        ++i;
    }
    incrementalTrainingState << '\0' << inferenceModelStream.str() << '\0';
}

void readIncrementalTrainingState(const std::string& resultsJson,
                                  std::ostream& incrementalTrainingState) {

    json::error_code ec;
    //    std::string str = "[{\"analytics_memory_usage\":{\"job_id\":\"testJob\",\"timestamp\":1704805891956,\"peak_usage_bytes\":4032,\"status\":\"ok\"}}\n,{\"analytics_memory_usage\":{\"job_id\":\"testJob\",\"timestamp\":1704805893487,\"peak_usage_bytes\":4272,\"status\":\"ok\"}}\n,{\"phase_progress\":{\"phase\":\"feature_selection\",\"progress_percent\":0}}\n,{\"phase_progress\":{\"phase\":\"feature_selection\",\"progress_percent\":1}}\n,{\"analytics_memory_usage\":{\"job_id\":\"testJob\",\"timestamp\":1704805893525,\"peak_usage_bytes\":4752,\"status\":\"ok\"}}\n,{\"analytics_memory_usage\":{\"job_id\":\"testJob\",\"timestamp\":1704805893526,\"peak_usage_bytes\":9092,\"status\":\"ok\"}}\n,{\"phase_progress\":{\"phase\":\"feature_selection\",\"progress_percent\":100}}\n,{\"phase_progress\":{\"phase\":\"coarse_parameter_search\",\"progress_percent\":0}}\n,{\"phase_progress\":{\"phase\":\"coarse_parameter_search\",\"progress_percent\":1}}\n,{\"phase_progress\":{\"phase\":\"coarse_parameter_search\",\"progress_percent\":4}}\n,{\"phase_progress\":{\"phase\":\"coarse_parameter_search\",\"progress_percent\":5}}\n,{\"phase_progress\":{\"phase\":\"coarse_parameter_search\",\"progress_percent\":8}}\n,{\"phase_progress\":{\"phase\":\"coarse_parameter_search\",\"progress_percent\":100}}\n,{\"phase_progress\":{\"phase\":\"fine_tuning_parameters\",\"progress_percent\":100}}\n,{\"analytics_memory_usage\":{\"job_id\":\"testJob\",\"timestamp\":1704805923077,\"peak_usage_bytes\":235303,\"status\":\"ok\"},\"regression_stats\":{\"job_id\":\"testJob\",\"timestamp\":1704805923077,\"iteration\":0,\"hyperparameters\":{\"eta\":0.3805847,\"alpha\":7.102547,\"soft_tree_depth_limit\":9.228819,\"soft_tree_depth_tolerance\":0.15,\"gamma\":0.02713613,\"lambda\":0.7970777,\"downsample_factor\":0.3423561,\"num_folds\":0,\"max_trees\":10,\"feature_bag_fraction\":0.4,\"eta_growth_rate_per_tree\":1.190292,\"max_attempts_to_add_tree\":0,\"num_splits_per_feature\":0,\"max_optimization_rounds_per_hyperparameter\":2},\"validation_loss\":{\"loss_type\":\"mse\",\"fold_values\":[]},\"timing_stats\":{\"elapsed_time\":0,\"iteration_time\":0}}}\n,{\"phase_progress\":{\"phase\":\"final_training\",\"progress_percent\":100}}\n,{\"phase_progress\":{\"phase\":\"final_training\",\"progress_percent\":100}}\n,{\"model_size_info\":{\"preprocessors\":[],\"trained_model_size\":{\"ensemble_model_size\":{\"feature_name_lengths\":[2,2,2,2],\"tree_sizes\":[{\"num_nodes\":0,\"num_leaves\":1},{\"num_nodes\":8,\"num_leaves\":9},{\"num_nodes\":10,\"num_leaves\":11},{\"num_nodes\":10,\"num_leaves\":11},{\"num_nodes\":6,\"num_leaves\":7},{\"num_nodes\":6,\"num_leaves\":7},{\"num_nodes\":8,\"num_leaves\":9},{\"num_nodes\":7,\"num_leaves\":8},{\"num_nodes\":9,\"num_leaves\":10},{\"num_nodes\":4,\"num_leaves\":5},{\"num_nodes\":7,\"num_leaves\":8}],\"num_output_processor_weights\":11,\"num_operations\":46}}}}\n,{\"compressed_inference_model\":{\"doc_num\":0,\"definition\":\"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\",\"eos\":true}}\n,{\"compressed_inference_model\":{\"doc_num\":0,\"definition\":\"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\",\"eos\":true}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-4.978199,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-0.1856499,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-0.1454616,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-4.460028,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":1.643608,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":11.66935,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":6.449406,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-1.651949,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-1.431112,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":5.957064,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":6.531394,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":1.916028,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-3.202722,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":4.401132,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":3.176036,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-3.262762,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":9.79813,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-0.4056594,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-2.103299,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":4.711451,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":6.600036,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-10.10891,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":5.268023,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":6.372826,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-2.329821,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-6.211882,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":3.840094,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":2.392636,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-4.116221,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-0.06490951,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-7.632752,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-6.97257,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-0.283581,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":1.867715,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-1.77209,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-11.01916,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-0.6186131,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":3.61408,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":4.495405,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-4.042377,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":2.327649,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":1.776945,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-1.152582,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":4.170098,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-0.701755,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":5.661319,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":0.8796449,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":5.454208,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-5.780083,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-0.8118424,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":8.506525,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-0.9182748,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":0.7343156,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-6.68997,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":11.22411,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":6.840355,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":8.387185,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-6.993784,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":5.391653,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-3.177985,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-7.313002,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-1.248203,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-4.542601,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":0.9235312,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-6.91104,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-0.902207,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-3.477778,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":-1.264715,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":6.052814,\"is_training\":true}}}}\n,{\"row_results\":{\"checksum\":0,\"results\":{\"ml\":{\"target_prediction\":3.770232,\"is_training\":true}}}}\n,{\"model_metadata\":{\"total_feature_importance\":[],\"hyperparameters\":[{\"name\":\"soft_tree_depth_tolerance\",\"value\":0.15,\"absolute_importance\":0.06414481,\"relative_importance\":0.6031168,\"supplied\":false},{\"name\":\"lambda\",\"value\":0.9963471,\"absolute_importance\":0.03006686,\"relative_importance\":0.2827014,\"supplied\":false},{\"name\":\"eta\",\"value\":0.3805847,\"absolute_importance\":0.01167569,\"relative_importance\":0.1097798,\"supplied\":false},{\"name\":\"gamma\",\"value\":0.03392016,\"absolute_importance\":0,\"relative_importance\":0,\"supplied\":false},{\"name\":\"soft_tree_depth_limit\",\"value\":9.228819,\"absolute_importance\":0,\"relative_importance\":0,\"supplied\":false},{\"name\":\"downsample_factor\",\"value\":0.3423561,\"absolute_importance\":0,\"relative_importance\":0,\"supplied\":false},{\"name\":\"alpha\",\"value\":8.878184,\"absolute_importance\":0,\"relative_importance\":0,\"supplied\":false},{\"name\":\"eta_growth_rate_per_tree\",\"value\":1.190292,\"absolute_importance\":0,\"relative_importance\":0,\"supplied\":false},{\"name\":\"max_trees\",\"value\":10,\"supplied\":true},{\"name\":\"feature_bag_fraction\",\"value\":0.4,\"absolute_importance\":0,\"relative_importance\":0,\"supplied\":false}],\"train_properties\":{\"num_train_rows\":70,\"loss_gap\":3.44645,\"trained_model_memory_usage\":0},\"data_summarization\":{\"num_rows\":7}}}\n,{\"compressed_data_summarization\":{\"doc_num\":0,\"data_summarization\":\"H4sIAAAAAAAA/3VS7arbMAz9v8fw78ZYH7asvkopIaTZJdCmo00vG6XvPtm57TqWQWIi+eicIyl3N91ObX8+3k7T1W3jxi3f7dSdBkvsXA/OklgOKgfbMXeXj2F2+xd6vLZ9Nw8f58vYd0cr+94dr8PmP6fVDVN/PozTx7Udp8PYV6n7K1vV3XbRfiUNOfx02/DYrCBxBQmrSFpB4iqSV5C0hvyax79ofrz36rZ3Jz5awTv0c+jn86XcjYdhmsf5V/u8K8k3yh+3uX0OfOF3f1k8jb2lgns8HraZP/t4Fn12x1ud9G6/eXvM4qGbu5J32aeQNRJnI27IM+QsAMki9hQCiZAsAZHGJGXuTfBJBBmVov0UO7sNknJiLPYa8EIRQsBS2BQa4wyiFiUvSRkBFhzGGMV4K0nwmlQwAceKxExKDEUQfVTmAJmrFclmJGCxHD0qZ5WQK0ejxUkiZqgcLEGRinJj0hqyKUAlBCIoUAvIC4tiMveVI3sOAvaWP0fMPkZKmha1oGryud4gW9MaCj2gDxADcFo4yOfAIrGqZQ8SM2WWKq0siJpKY+Ctwyi5jj/5iDY0zUsv4hWIKXKqYlGRc/gyDGYfNeuyDFIw4VyNlNXY2gjFnOwf334DxRlEGfEDAAA=\",\"eos\":true}}\n]";

    json::value results = json::parse(resultsJson, ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);

    std::stringstream inferenceModelStream;
    core::CStreamWriter inferenceModelWriter{inferenceModelStream};

    core::CStreamWriter dataSummarizationWriter{incrementalTrainingState};

    // Read the state used to initialize incremental training.
    for (const auto& result_ : results.as_array()) {
        const json::object& result = result_.as_object();
        if (result.contains("compressed_inference_model")) {
            inferenceModelWriter.write(result);
            LOG_DEBUG(<< "Inference Model definition found");
        } else if (result.contains("compressed_data_summarization")) {
            dataSummarizationWriter.write(result);
            LOG_DEBUG(<< "Data summarization found");
        }
    }
    incrementalTrainingState << '\0' << inferenceModelStream.str() << '\0';
}

void readPredictions(const std::string& resultsJson,
                     const std::string& targetPredictionName,
                     TDoubleVec& predictions) {

    json::error_code ec;
    json::value results = json::parse(resultsJson, ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    BOOST_TEST_REQUIRE(results.is_array());

    for (const auto& result_ : results.as_array()) {
        const json::object& result = result_.as_object();
        if (result.contains("row_results")) {
            predictions.emplace_back(result_
                                         .at_pointer("/row_results/results/ml/" + targetPredictionName)
                                         .as_double());
        }
    }
}
}

BOOST_AUTO_TEST_CASE(testMissingString) {

    // Test that the special missing value string is respected.

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TStrVec fieldNames{"f1", "f2", "f3", "f4", "target", ".", "."};
    TStrVec fieldValues{"a", "2.0", "3.0", "4.0", "5.0", "0", ""};

    // Test default value.
    {
        std::string a{"a"};
        std::string b{"b"};
        std::string missing{core::CDataFrame::DEFAULT_MISSING_STRING};

        TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
        auto spec = test::CDataFrameAnalysisSpecificationFactory{}
                        .rows(5)
                        .predictionCategoricalFieldNames({"f1"})
                        .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::regression(),
                                        "target", &frameAndDirectory);
        api::CDataFrameAnalyzer analyzer{
            std::move(spec), std::move(frameAndDirectory), outputWriterFactory};

        TBoolVec isMissing;
        for (const auto& category : {a, missing, b, a, missing}) {
            fieldValues[0] = category;
            analyzer.handleRecord(fieldNames, fieldValues);
            isMissing.push_back(category == missing);
        }
        analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

        analyzer.dataFrame().readRows(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
            std::size_t i{0};
            for (auto row = beginRows; row != endRows; ++row, ++i) {
                BOOST_REQUIRE_EQUAL(
                    isMissing[row->index()],
                    maths::analytics::CDataFrameUtils::isMissing((*row)[0]));
            }
        });
    }

    // Test custom value.
    {
        std::string a{"a"};
        std::string b{"b"};
        std::string missing{"foo"};

        TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
        auto spec = test::CDataFrameAnalysisSpecificationFactory{}
                        .rows(5)
                        .predictionCategoricalFieldNames({"f1"})
                        .missingString("foo")
                        .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::regression(),
                                        "target", &frameAndDirectory);
        api::CDataFrameAnalyzer analyzer{
            std::move(spec), std::move(frameAndDirectory), outputWriterFactory};

        TBoolVec isMissing;
        for (const auto& category : {a, missing, b, a, missing}) {
            fieldValues[0] = category;
            analyzer.handleRecord(fieldNames, fieldValues);
            isMissing.push_back(category == missing);
        }
        analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

        analyzer.dataFrame().readRows(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
            std::size_t i{0};
            for (auto row = beginRows; row != endRows; ++row, ++i) {
                BOOST_REQUIRE_EQUAL(
                    isMissing[row->index()],
                    maths::analytics::CDataFrameUtils::isMissing((*row)[0]));
            }
        });
    }
}

BOOST_AUTO_TEST_CASE(testMemoryLimitHandling) {
    TStrVec errors;
    auto errorHandler = [&errors](std::string error) { errors.push_back(error); };
    core::CLogger::CScopeSetFatalErrorHandler scope{errorHandler};

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TDoubleVec expectedPredictions;
    std::size_t numberSamples{50};

    TStrVec fieldNames{"f1", "f2", "f3", "f4", "target", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};

    TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
    auto spec = test::CDataFrameAnalysisSpecificationFactory{}
                    .rows(numberSamples)
                    .predictionMaximumNumberTrees(2)
                    .memoryLimit(6000)
                    .predicitionNumberRoundsPerHyperparameter(1)
                    .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::regression(),
                                    "target", &frameAndDirectory);
    api::CDataFrameAnalyzer analyzer{std::move(spec), std::move(frameAndDirectory),
                                     std::move(outputWriterFactory)};
    test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
        TLossFunctionType::E_MseRegression, fieldNames, fieldValues, analyzer,
        expectedPredictions, numberSamples);

    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    BOOST_TEST_REQUIRE(errors.size() > 0);
    bool memoryLimitExceed{false};
    for (const auto& error : errors) {
        if (error.find("Input error: memory limit") != std::string::npos) {
            memoryLimitExceed = true;
            break;
        }
    }
    BOOST_TEST_REQUIRE(memoryLimitExceed);

    // Verify memory status change. Initially we should be ok, but hit the hard
    // limit during training.
    json::error_code ec;
    json::value results = json::parse(output.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    BOOST_TEST_REQUIRE(results.is_array());

    bool memoryStatusOk{false};
    bool memoryStatusHardLimit{false};
    bool memoryReestimateAvailable{false};
    for (const auto& result_ : results.as_array()) {
        const json::object& result = result_.as_object();
        if (result.contains("analytics_memory_usage")) {
            std::string status{
                result_.at_pointer("/analytics_memory_usage/status").as_string()};
            if (status == "ok") {
                memoryStatusOk = true;
            } else if (status == "hard_limit") {
                memoryStatusHardLimit = true;
                if (result_.at_pointer("/analytics_memory_usage").as_object().contains("memory_reestimate_bytes") &&
                    result_.at_pointer("/analytics_memory_usage/memory_reestimate_bytes")
                            .as_int64() > 0) {
                    memoryReestimateAvailable = true;
                }
            }
        }
    }
    BOOST_TEST_REQUIRE(memoryStatusOk);
    BOOST_TEST_REQUIRE(memoryStatusHardLimit);
    BOOST_TEST_REQUIRE(memoryReestimateAvailable);
}

BOOST_AUTO_TEST_CASE(testRegressionTraining) {

    // Test the results the analyzer produces match running the regression directly.

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TDoubleVec expectedPredictions;

    TStrVec fieldNames{"f1", "f2", "f3", "f4", "target", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
    auto spec = test::CDataFrameAnalysisSpecificationFactory{}.predictionSpec(
        test::CDataFrameAnalysisSpecificationFactory::regression(), "target", &frameAndDirectory);
    api::CDataFrameAnalyzer analyzer{std::move(spec), std::move(frameAndDirectory),
                                     std::move(outputWriterFactory)};
    test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
        TLossFunctionType::E_MseRegression, fieldNames, fieldValues, analyzer,
        expectedPredictions);

    core::CStopWatch watch{true};
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});
    std::uint64_t duration{watch.stop()};

    json::error_code ec;
    json::value results = json::parse(output.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    BOOST_TEST_REQUIRE(results.is_array());

    auto expectedPrediction = expectedPredictions.begin();
    bool progressCompleted{false};
    for (const auto& result_ : results.as_array()) {
        const json::object& result = result_.as_object();
        if (result.contains("row_results")) {
            BOOST_TEST_REQUIRE(expectedPrediction != expectedPredictions.end());
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                *expectedPrediction,
                result_.at_pointer("/row_results/results/ml/target_prediction").as_double(),
                1e-4 * std::fabs(*expectedPrediction));
            ++expectedPrediction;
            BOOST_TEST_REQUIRE(result.contains("phase_progress") == false);
        } else if (result.contains("phase_progress")) {
            BOOST_TEST_REQUIRE(
                result_.at_pointer("/phase_progress/progress_percent").as_int64() >= 0);
            BOOST_TEST_REQUIRE(
                result_.at_pointer("/phase_progress/progress_percent").as_int64() <= 100);
            BOOST_TEST_REQUIRE(result.contains("row_results") == false);
            progressCompleted =
                result_.at_pointer("/phase_progress/progress_percent").as_int64() == 100;
        }
    }
    BOOST_TEST_REQUIRE(expectedPrediction == expectedPredictions.end());
    BOOST_TEST_REQUIRE(progressCompleted);

    LOG_DEBUG(<< "estimated memory usage = "
              << core::CProgramCounters::counter(counter_t::E_DFTPMEstimatedPeakMemoryUsage));
    LOG_DEBUG(<< "peak memory = "
              << core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage));
    LOG_DEBUG(<< "time to train = " << core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain)
              << "ms");

    BOOST_TEST_REQUIRE(core::CProgramCounters::counter(
                           counter_t::E_DFTPMEstimatedPeakMemoryUsage) < 7000000);
    BOOST_TEST_REQUIRE(core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage) < 2100000);
    BOOST_TEST_REQUIRE(core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain) > 0);
    BOOST_TEST_REQUIRE(core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain) <= duration);
}

BOOST_AUTO_TEST_CASE(testRegressionTrainingWithMse) {

    // Test the regression hyperparameter settings are correctly propagated to the
    // analysis runner.
    testRegressionTrainingWithParams(TLossFunctionType::E_MseRegression);
}

BOOST_AUTO_TEST_CASE(testRegressionTrainingWithParamsMsle) {
    // Test the regression hyperparameter settings are correctly propagated to the
    // analysis runner.
    testRegressionTrainingWithParams(TLossFunctionType::E_MsleRegression);
}

BOOST_AUTO_TEST_CASE(testRegressionTrainingWithParamsPseudoHuber) {
    // Test the regression hyperparameter settings are correctly propagated to the
    // analysis runner.
    testRegressionTrainingWithParams(TLossFunctionType::E_HuberRegression);
}

BOOST_AUTO_TEST_CASE(testRegressionTrainingWithRowsMissingTargetValue) {

    // Test we are able to predict value rows for which the dependent variable
    // is missing.

    test::CRandomNumbers rng;

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    auto target = [](double feature) { return 10.0 * feature; };
    TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
    auto spec = test::CDataFrameAnalysisSpecificationFactory{}
                    .rows(50)
                    .columns(2)
                    .memoryLimit(4000000)
                    .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::regression(),
                                    "target", &frameAndDirectory);
    api::CDataFrameAnalyzer analyzer{std::move(spec), std::move(frameAndDirectory),
                                     std::move(outputWriterFactory)};

    TDoubleVec feature;
    rng.generateUniformSamples(1.0, 3.0, 50, feature);

    TStrVec fieldNames{"feature", "target", ".", "."};
    TStrVec fieldValues(4);

    for (std::size_t i = 0; i < 40; ++i) {
        fieldValues[0] = std::to_string(feature[i]);
        fieldValues[1] = std::to_string(target(feature[i]));
        fieldValues[2] = std::to_string(i);
        analyzer.handleRecord(fieldNames, fieldValues);
    }
    for (std::size_t i = 40; i < 50; ++i) {
        fieldValues[0] = std::to_string(feature[i]);
        fieldValues[1] = core::CDataFrame::DEFAULT_MISSING_STRING;
        fieldValues[2] = std::to_string(i);
        analyzer.handleRecord(fieldNames, fieldValues);
    }
    analyzer.handleRecord(fieldNames, {"", "", "", "$"});

    json::error_code ec;
    json::value results = json::parse(output.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    BOOST_TEST_REQUIRE(results.is_array());

    std::size_t numberResults{0};
    for (const auto& result_ : results.as_array()) {
        const json::object& result = result_.as_object();
        if (result.contains("row_results")) {
            json::error_code ec;
            std::size_t index(
                result_.at_pointer("/row_results/checksum").to_number<std::size_t>(ec));
            BOOST_REQUIRE(ec.failed() == false);
            double expected{target(feature[index])};
            BOOST_REQUIRE_CLOSE_ABSOLUTE(
                expected,
                result_.at_pointer("/row_results/results/ml/target_prediction").as_double(),
                0.2 * expected);
            BOOST_REQUIRE_EQUAL(
                index < 40,
                result_.at_pointer("/row_results/results/ml/is_training").as_bool());
            ++numberResults;
        }
    }
    BOOST_REQUIRE_EQUAL(std::size_t{50}, numberResults);
}

// TODO
//BOOST_AUTO_TEST_CASE(testRegressionTrainingWithStateRecovery) {
//
//    // Test that restoring state and resuming training from different checkpoints
//    // always produces the same result.
//
//    test::CRandomNumbers rng;
//
//    for (const auto& lossFunction :
//         {TLossFunctionType::E_MseRegression, TLossFunctionType::E_MsleRegression,
//          TLossFunctionType::E_HuberRegression}) {
//
//        LOG_DEBUG(<< "Loss function type " << lossFunction);
//
//        for (std::size_t restart = 6; restart < 16; restart += 2) {
//
//            auto makeSpec = [&](const std::string& dependentVariable, std::size_t numberExamples,
//                                TDataFrameUPtrTemporaryDirectoryPtrPr& frameAndDirectory,
//                                TPersisterSupplier* persisterSupplier,
//                                TRestoreSearcherSupplier* restorerSupplier) {
//                return test::CDataFrameAnalysisSpecificationFactory{}
//                    .rows(numberExamples)
//                    .memoryLimit(15000000)
//                    .predictionMaximumNumberTrees(10)
//                    .predictionPersisterSupplier(persisterSupplier)
//                    .predictionRestoreSearcherSupplier(restorerSupplier)
//                    .regressionLossFunction(lossFunction)
//                    .earlyStoppingEnabled(false)
//                    .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::regression(),
//                                    dependentVariable, &frameAndDirectory);
//            };
//
//            LOG_DEBUG(<< "restart from " << restart);
//            testOneRunOfBoostedTreeTrainingWithStateRecovery(makeSpec, restart, lossFunction);
//        }
//    }
//}

//TODO
//BOOST_AUTO_TEST_CASE(testRegressionPredictionNumericalOnly, *utf::tolerance(0.000001)) {
//
//    // This tests that a prediction only task produces the same predictions as the
//    // result of training if rerunning with numeric features only.
//    using TTask = test::CDataFrameAnalysisSpecificationFactory::TTask;
//    double dataSummarizationFraction{0.1};
//    std::size_t trainExamples{70};
//    std::size_t predictExamples{50};
//
//    test::CRandomNumbers rng;
//    TSizeVec seed{1};
//    rng.generateUniformSamples(0, 1000, 1, seed);
//
//    std::stringstream outputStream;
//    auto outputWriterFactory = [&outputStream]() {
//        return std::make_unique<core::CJsonOutputStreamWrapper>(outputStream);
//    };
//    auto makeSpec = [&](const std::string& dependentVariable, std::size_t numberExamples,
//                        TDataFrameUPtrTemporaryDirectoryPtrPr& frameAndDirectory,
//                        TRestoreSearcherSupplier* restorerSupplier, TTask task) {
//        return test::CDataFrameAnalysisSpecificationFactory{}
//            .rows(numberExamples)
//            .memoryLimit(15000000)
//            .predictionMaximumNumberTrees(10)
//            .predictionRestoreSearcherSupplier(restorerSupplier)
//            .regressionLossFunction(TLossFunctionType::E_MseRegression)
//            .task(task)
//            .dataSummarizationFraction(dataSummarizationFraction)
//            .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::regression(),
//                            dependentVariable, &frameAndDirectory);
//    };
//    auto runAnalyzer = [&](std::size_t totalExamples, std::size_t newExamples, TTask task,
//                           TRestoreSearcherSupplier* restorerSupplier = nullptr) {
//        TStrVec fieldNames{"c1", "c2", "c3", "c4", "target", ".", "."};
//        TStrVec fieldValues{"", "", "", "", "", "0", ""};
//        outputStream.str("");
//        TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
//        auto spec = makeSpec("target", totalExamples, frameAndDirectory,
//                             restorerSupplier, task);
//        api::CDataFrameAnalyzer analyzer{
//            std::move(spec), std::move(frameAndDirectory), outputWriterFactory};
//
//        test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
//            TLossFunctionType::E_MseRegression, fieldNames, fieldValues,
//            analyzer, newExamples, seed[0]);
//        analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});
//    };
//
//    // Run initial training.
//    TDoubleVec expectedPredictions;
//    expectedPredictions.reserve(trainExamples);
//    std::stringstream incrementalTrainingState;
//    {
//        runAnalyzer(trainExamples, trainExamples, TTask::E_Train);
//        readIncrementalTrainingState(outputStream.str(), incrementalTrainingState);
//        readPredictions(outputStream.str(), "target_prediction", expectedPredictions);
//    }
//    BOOST_REQUIRE_EQUAL(expectedPredictions.size(), trainExamples);
//
//    // Run prediction.
//    TDoubleVec actualPredictions;
//    actualPredictions.reserve(predictExamples);
//    {
//        // Pass incremental training state into the restore stream.
//        auto restoreStreamPtr =
//            std::make_shared<std::stringstream>(std::move(incrementalTrainingState));
//        TRestoreSearcherSupplier restorerSupplier{[&restoreStreamPtr]() {
//            return std::make_unique<api::CSingleStreamSearcher>(restoreStreamPtr);
//        }};
//
//        std::size_t numberExamples{
//            static_cast<std::size_t>(trainExamples * dataSummarizationFraction) + predictExamples};
//        runAnalyzer(numberExamples, predictExamples, TTask::E_Predict, &restorerSupplier);
//        readPredictions(outputStream.str(), "target_prediction", actualPredictions);
//    }
//    BOOST_REQUIRE_EQUAL(actualPredictions.size(), predictExamples);
//    for (std::size_t i = 0; i < predictExamples; ++i) {
//        BOOST_TEST_REQUIRE(actualPredictions[i] == expectedPredictions[i]);
//    }
//}

BOOST_AUTO_TEST_CASE(testRegressionPredictionNumericalCategoricalMix,
                     *utf::tolerance(0.000001)) {

    // This tests that a prediction only task produces the same predictions as the
    // result of training if rerunning with a mixture of categorical and numeric
    // features.
    using TTask = test::CDataFrameAnalysisSpecificationFactory::TTask;
    double dataSummarizationFraction{0.1};
    std::size_t trainExamples{70};
    std::size_t predictExamples{50};
    std::size_t cols = 3;

    test::CRandomNumbers rng;
    TSizeVec seed{1};
    rng.generateUniformSamples(0, 1000, 1, seed);

    // Generate the training values.
    TDoubleVec weights{10.0, 50.0};
    TDoubleVecVec values(cols);
    rng.generateUniformSamples(-10.0, 10.0, trainExamples, values[0]);
    values[1] = generateCategoricalData(rng, trainExamples, {5.0, 5.0, 5.0});
    for (std::size_t i = 0; i < trainExamples; ++i) {
        values[2].push_back(values[0][i] * weights[0] + values[1][i] * weights[1]);
    }

    std::stringstream outputStream;
    auto outputWriterFactory = [&outputStream]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(outputStream);
    };
    auto makeSpec = [&](const std::string& dependentVariable, std::size_t numberExamples,
                        TDataFrameUPtrTemporaryDirectoryPtrPr& frameAndDirectory,
                        TRestoreSearcherSupplier* restorerSupplier, TTask task) {
        return test::CDataFrameAnalysisSpecificationFactory{}
            .rows(numberExamples)
            .columns(cols)
            .memoryLimit(30000000)
            .predictionCategoricalFieldNames({"categorical_col"})
            .predictionRestoreSearcherSupplier(restorerSupplier)
            .dataSummarizationFraction(dataSummarizationFraction)
            .task(task)
            .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::regression(),
                            dependentVariable, &frameAndDirectory);
    };
    auto runAnalyzer = [&](std::size_t totalExamples, std::size_t newExamples, TTask task,
                           TRestoreSearcherSupplier* restorerSupplier = nullptr) {
        TStrVec fieldNames{"numeric_col", "categorical_col", "target", ".", "."};
        TStrVec fieldValues{"", "", "0", "", ""};
        outputStream.str("");
        TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
        auto spec = makeSpec("target", totalExamples, frameAndDirectory,
                             restorerSupplier, task);
        api::CDataFrameAnalyzer analyzer{
            std::move(spec), std::move(frameAndDirectory), outputWriterFactory};
        for (std::size_t i = 0; i < newExamples; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                    values[j][i], core::CIEEE754::E_DoublePrecision);
            }
            analyzer.handleRecord(fieldNames, fieldValues);
        }
        analyzer.handleRecord(fieldNames, {"", "", "", "", "$"});
    };

    // Run inital training.
    TDoubleVec expectedPredictions;
    expectedPredictions.reserve(trainExamples);
    std::stringstream incrementalTrainingState;
    {
        runAnalyzer(trainExamples, trainExamples, TTask::E_Train);
        readIncrementalTrainingState(outputStream.str(), incrementalTrainingState);
        readPredictions(outputStream.str(), "target_prediction", expectedPredictions);
    }
    BOOST_REQUIRE_EQUAL(expectedPredictions.size(), trainExamples);

    // Run prediction.
    TDoubleVec actualPredictions;
    actualPredictions.reserve(predictExamples);
    {
        // Pass incremental training state into the restore stream.
        auto restoreStreamPtr =
            std::make_shared<std::stringstream>(std::move(incrementalTrainingState));
        TRestoreSearcherSupplier restorerSupplier{[&restoreStreamPtr]() {
            return std::make_unique<api::CSingleStreamSearcher>(restoreStreamPtr);
        }};

        std::size_t numberExamples{
            static_cast<std::size_t>(trainExamples * dataSummarizationFraction) + predictExamples};
        runAnalyzer(numberExamples, predictExamples, TTask::E_Predict, &restorerSupplier);
        readPredictions(outputStream.str(), "target_prediction", actualPredictions);
    }
    BOOST_REQUIRE_EQUAL(actualPredictions.size(), predictExamples);
    for (std::size_t i = 0; i < predictExamples; ++i) {
        BOOST_TEST_REQUIRE(actualPredictions[i] == expectedPredictions[i]);
    }
}

//TODO
//BOOST_AUTO_TEST_CASE(testRegressionIncrementalTraining) {
//
//    // Test running incremental training from the analyzer matches running directly.
//
//    std::size_t maximumNumberTrees{30};
//    std::size_t numberExamples{100};
//
//    auto makeTrainSpec = [&](const std::string& dependentVariable,
//                             TDataFrameUPtrTemporaryDirectoryPtrPr& frameAndDirectory,
//                             TPersisterSupplier* persisterSupplier,
//                             TRestoreSearcherSupplier* restorerSupplier) {
//        test::CDataFrameAnalysisSpecificationFactory specFactory;
//        return specFactory.rows(numberExamples)
//            .memoryLimit(15000000)
//            .predictionMaximumNumberTrees(maximumNumberTrees)
//            .predictionPersisterSupplier(persisterSupplier)
//            .predictionRestoreSearcherSupplier(restorerSupplier)
//            .regressionLossFunction(TLossFunctionType::E_MseRegression)
//            .task(test::CDataFrameAnalysisSpecificationFactory::TTask::E_Train)
//            .dataSummarizationFraction(0.1)
//            .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::regression(),
//                            dependentVariable, &frameAndDirectory);
//    };
//
//    double alpha;
//    double lambda;
//    double gamma;
//    double softTreeDepthLimit;
//    double softTreeDepthTolerance;
//    double eta;
//    double etaGrowthRatePerTree;
//    double downsampleFactor;
//    double featureBagFraction;
//    double lossGap;
//
//    auto makeUpdateSpec = [&](const std::string& dependentVariable,
//                              TDataFrameUPtrTemporaryDirectoryPtrPr& frameAndDirectory,
//                              TPersisterSupplier* persisterSupplier,
//                              TRestoreSearcherSupplier* restorerSupplier) {
//        test::CDataFrameAnalysisSpecificationFactory specFactory;
//        return specFactory.rows(2 * numberExamples)
//            .memoryLimit(15000000)
//            .predictionAlpha(alpha)
//            .predictionLambda(lambda)
//            .predictionGamma(gamma)
//            .predictionSoftTreeDepthLimit(softTreeDepthLimit)
//            .predictionSoftTreeDepthTolerance(softTreeDepthTolerance)
//            .predictionEta(eta)
//            .predictionEtaGrowthRatePerTree(etaGrowthRatePerTree)
//            .predictionMaximumNumberTrees(maximumNumberTrees)
//            .predictionDownsampleFactor(downsampleFactor)
//            .predictionFeatureBagFraction(featureBagFraction)
//            .previousTrainLossGap(lossGap)
//            .previousTrainNumberRows(numberExamples)
//            .predictionPersisterSupplier(persisterSupplier)
//            .predictionRestoreSearcherSupplier(restorerSupplier)
//            .regressionLossFunction(TLossFunctionType::E_MseRegression)
//            .task(test::CDataFrameAnalysisSpecificationFactory::TTask::E_Update)
//            .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::regression(),
//                            dependentVariable, &frameAndDirectory);
//    };
//
//    std::stringstream outputStream;
//    auto outputWriterFactory = [&outputStream]() {
//        return std::make_unique<core::CJsonOutputStreamWrapper>(outputStream);
//    };
//
//    // Run once.
//    TStrVec fieldNames{"c1", "c2", "c3", "c4", "target", ".", "."};
//    TStrVec fieldValues{"", "", "", "", "", "0", ""};
//    TDoubleVec weights{0.1, 2.0, 0.4, -0.5};
//    TDoubleVec regressors;
//    test::CRandomNumbers rng;
//    rng.generateUniformSamples(-10.0, 10.0, weights.size() * numberExamples, regressors);
//    TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
//    auto spec = makeTrainSpec("target", frameAndDirectory, nullptr, nullptr);
//    api::CDataFrameAnalyzer analyzer{std::move(spec), std::move(frameAndDirectory),
//                                     outputWriterFactory};
//    TStrVec targets;
//
//    auto frame = test::CDataFrameAnalyzerTrainingFactory::setupLinearRegressionData(
//        fieldNames, fieldValues, analyzer, weights, regressors, targets);
//    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});
//
//    // Train a model for comparison.
//    auto regression = maths::analytics::CBoostedTreeFactory::constructFromParameters(
//                          1, std::make_unique<maths::analytics::boosted_tree::CMse>())
//                          .maximumNumberTrees(maximumNumberTrees)
//                          .dataSummarizationFraction(0.1)
//                          .buildForTrain(*frame, weights.size());
//    regression->train();
//    regression->predict();
//
//    // Retrieve documents from the result stream that will be used to restore the model.
//
//    std::stringstream incrementalTrainingState;
//    readIncrementalTrainingState(outputStream.str(), alpha, lambda, gamma,
//                                 softTreeDepthLimit, softTreeDepthTolerance,
//                                 eta, etaGrowthRatePerTree, downsampleFactor,
//                                 featureBagFraction, lossGap, incrementalTrainingState);
//
//    // Pass model definition and data summarization into the restore stream.
//    auto restoreStreamPtr =
//        std::make_shared<std::stringstream>(std::move(incrementalTrainingState));
//    TRestoreSearcherSupplier restorerSupplier{[&restoreStreamPtr]() {
//        return std::make_unique<api::CSingleStreamSearcher>(restoreStreamPtr);
//    }};
//
//    outputStream.clear();
//    outputStream.str("");
//
//    // Create a new spec for incremental training.
//    spec = makeUpdateSpec("target", frameAndDirectory, nullptr, &restorerSupplier);
//
//    // Create new analyzer and run incremental training.
//    api::CDataFrameAnalyzer analyzerIncremental{
//        std::move(spec), std::move(frameAndDirectory), outputWriterFactory};
//    auto newTrainingDataFrame = test::CDataFrameAnalyzerTrainingFactory::setupLinearRegressionData(
//        fieldNames, fieldValues, analyzerIncremental, weights, regressors, targets);
//    analyzerIncremental.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});
//
//    json::error_code ec;
//    json::value results = json::parse(outputStream.str(), ec);
//    BOOST_TEST_REQUIRE(ec.failed() == false);
//    BOOST_TEST_REQUIRE(results.is_array());
//
//    // Read the predictions.
//    TDoubleVec predictions;
//    for (const auto& result_ : results.as_array()) {
//        const json::object& result = result_.as_object();
//        if (result.contains("row_results")) {
//            predictions.emplace_back(
//                result_.at_pointer("/row_results/results/ml/target_prediction").as_double());
//        }
//    }
//    BOOST_REQUIRE_EQUAL(numberExamples, predictions.size());
//
//    frame->resizeColumns(1, weights.size() + 1);
//
//    auto summarisation = regression->dataSummarization();
//
//    TDoubleVecVec newTrainingData;
//    newTrainingData.reserve(numberExamples +
//                            static_cast<std::size_t>(summarisation.manhattan()));
//    frame->readRows(1, 0, frame->numberRows(),
//                    [&](const TRowItr& beginRows, const TRowItr& endRows) {
//                        for (auto row = beginRows; row != endRows; ++row) {
//                            newTrainingData.push_back(TDoubleVec(row->numberColumns()));
//                            row->copyTo(newTrainingData.back().begin());
//                        }
//                    },
//                    &summarisation);
//    newTrainingDataFrame->readRows(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
//        for (auto row = beginRows; row != endRows; ++row) {
//            newTrainingData.push_back(TDoubleVec(row->numberColumns()));
//            row->copyTo(newTrainingData.back().begin());
//        }
//    });
//    frame->resizeRows(0);
//    for (std::size_t i = 0; i < newTrainingData.size(); ++i) {
//        frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t& id) {
//            for (std::size_t j = 0; j < newTrainingData[i].size(); ++j, ++column) {
//                *column = newTrainingData[i][j];
//                id = static_cast<std::int32_t>(i);
//            }
//        });
//    }
//    frame->finishWritingRows();
//
//    core::CPackedBitVector newTrainingRowMask(
//        static_cast<std::size_t>(summarisation.manhattan()), false);
//    newTrainingRowMask.extend(true, numberExamples);
//
//    regression = maths::analytics::CBoostedTreeFactory::constructFromModel(std::move(regression))
//                     .newTrainingRowMask(newTrainingRowMask)
//                     .buildForTrainIncremental(*frame, weights.size());
//
//    regression->trainIncremental();
//    regression->predict();
//
//    auto prediction = predictions.begin();
//    frame->readRows(1, 0, frame->numberRows(),
//                    [&](const TRowItr& beginRows, const TRowItr& endRows) {
//                        for (auto row = beginRows; row != endRows; ++row) {
//                            BOOST_REQUIRE_CLOSE_ABSOLUTE(
//                                (*prediction++), regression->prediction(*row)[0], 1e-6);
//                        }
//                    },
//                    &newTrainingRowMask);
//}

// TODO
//BOOST_AUTO_TEST_CASE(testClassificationTraining) {
//
//    // Test the results the analyzer produces match running classification directly.
//
//    std::stringstream output;
//    auto outputWriterFactory = [&output]() {
//        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
//    };
//
//    TStrVec expectedPredictions;
//
//    TStrVec fieldNames{"f1", "f2", "f3", "f4", "target", ".", "."};
//    TStrVec fieldValues{"", "", "", "", "", "0", ""};
//    TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
//    auto spec = test::CDataFrameAnalysisSpecificationFactory{}
//                    .memoryLimit(6000000)
//                    .predictionCategoricalFieldNames({"target"})
//                    .numberTopClasses(1)
//                    .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::classification(),
//                                    "target", &frameAndDirectory);
//    api::CDataFrameAnalyzer analyzer{std::move(spec), std::move(frameAndDirectory),
//                                     std::move(outputWriterFactory)};
//    test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
//        TLossFunctionType::E_BinaryClassification, fieldNames, fieldValues,
//        analyzer, expectedPredictions);
//
//    core::CStopWatch watch{true};
//    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});
//    std::uint64_t duration{watch.stop()};
//
//    json::error_code ec;
//    json::value results = json::parse(output.str(), ec);
//    BOOST_TEST_REQUIRE(ec.failed() == false);
//    BOOST_TEST_REQUIRE(results.is_array());
//
//    auto expectedPrediction = expectedPredictions.begin();
//    bool progressCompleted{false};
//    for (const auto& result_ : results.as_array()) {
//        const json::object& result = result_.as_object();
//        if (result.contains("row_results")) {
//            BOOST_TEST_REQUIRE(expectedPrediction != expectedPredictions.end());
//            std::string actualPrediction{
//                result_.at_pointer("/row_results/results/ml/target_prediction").as_string()};
//            BOOST_REQUIRE_EQUAL(*expectedPrediction, actualPrediction);
//            // Check the prediction values match the first entry in the top-classes.
//            BOOST_REQUIRE_EQUAL(
//                result_.at_pointer("/row_results/results/ml/target_prediction").as_string(),
//                result_.at_pointer("/row_results/results/ml/top_classes")
//                    .as_array()[0]
//                    .as_object().at("class_name")
//                    .as_string());
//            BOOST_REQUIRE_EQUAL(
//                result_.at_pointer("/row_results/results/ml/prediction_probability").as_double(),
//                result_.at_pointer("/row_results/results/ml/top_classes")
//                    .as_array()[0]
//                    .as_object()
//                    .at("class_probability")
//                    .as_double());
//            BOOST_REQUIRE_EQUAL(
//                result_.at_pointer("/row_results/results/ml/prediction_score").as_double(),
//                result_.at_pointer("/row_results/results/ml/top_classes")
//                    .as_array()[0]
//                    .as_object()
//                    .at("class_score")
//                    .as_double());
//            ++expectedPrediction;
//            BOOST_TEST_REQUIRE(result.contains("phase_progress") == false);
//        } else if (result.contains("phase_progress")) {
//            BOOST_TEST_REQUIRE(result_.at_pointer("/phase_progress/progress_percent").as_int64() >= 0);
//            BOOST_TEST_REQUIRE(result_.at_pointer("/phase_progress/progress_percent").as_int64() <= 100);
//            BOOST_TEST_REQUIRE(result.contains("row_results") == false);
//            progressCompleted = result_.at_pointer("/phase_progress/progress_percent").as_int64() == 100;
//        }
//    }
//    BOOST_TEST_REQUIRE(expectedPrediction == expectedPredictions.end());
//    BOOST_TEST_REQUIRE(progressCompleted);
//
//    LOG_DEBUG(<< "estimated memory usage = "
//              << core::CProgramCounters::counter(counter_t::E_DFTPMEstimatedPeakMemoryUsage));
//    LOG_DEBUG(<< "peak memory = "
//              << core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage));
//    LOG_DEBUG(<< "time to train = " << core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain)
//              << "ms");
//
//    BOOST_TEST_REQUIRE(core::CProgramCounters::counter(
//                           counter_t::E_DFTPMEstimatedPeakMemoryUsage) < 7000000);
//    BOOST_TEST_REQUIRE(core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage) < 2100000);
//    BOOST_TEST_REQUIRE(core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain) > 0);
//    BOOST_TEST_REQUIRE(core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain) <= duration);
//}

// TODO
//BOOST_AUTO_TEST_CASE(testClassificationImbalancedClasses) {
//
//    // Test for the default configuration we get high average recall for each class
//    // when the training data are imbalanced.
//
//    using TStrSizeUMap = boost::unordered_map<std::string, std::size_t>;
//
//    std::stringstream output;
//    auto outputWriterFactory = [&output]() {
//        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
//    };
//
//    std::size_t numberExamples{500};
//    TStrVec fieldNames{"f1", "f2", "f3", "target", ".", "."};
//    TStrVec fieldValues{"", "", "", "", "0", ""};
//    test::CRandomNumbers rng;
//    TDoubleVec weights{1.0, 1.0, 1.0};
//    TDoubleVec regressors;
//    rng.generateUniformSamples(-5.0, 10.0, numberExamples * weights.size(), regressors);
//
//    TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
//    auto spec = test::CDataFrameAnalysisSpecificationFactory{}
//                    .rows(numberExamples)
//                    .columns(4)
//                    .memoryLimit(18000000)
//                    .predictionCategoricalFieldNames({"target"})
//                    .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::classification(),
//                                    "target", &frameAndDirectory);
//    api::CDataFrameAnalyzer analyzer{std::move(spec), std::move(frameAndDirectory),
//                                     std::move(outputWriterFactory)};
//
//    TStrVec actuals;
//    test::CDataFrameAnalyzerTrainingFactory::setupBinaryClassificationData(
//        fieldNames, fieldValues, analyzer, weights, regressors, actuals);
//    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "$"});
//
//    json::error_code ec;
//    json::value results = json::parse(output.str(), ec);
//    BOOST_TEST_REQUIRE(ec.failed() == false);
//    BOOST_TEST_REQUIRE(results.is_array());
//
//    TStrSizeUMap correct;
//    TStrSizeUMap counts;
//
//    auto actual = actuals.begin();
//    for (const auto& result_ : results.as_array()) {
//        const json::object& result = result_.as_object();
//        if (result.contains("row_results")) {
//            BOOST_TEST_REQUIRE(actual != actuals.end());
//            std::string prediction{
//                result_.at_pointer("/row_results/results/ml/target_prediction").as_string()};
//
//            if (*actual == prediction) {
//                ++correct[*actual];
//            }
//            ++counts[*actual];
//            ++actual;
//        }
//    }
//
//    for (const auto& label : {"foo", "bar"}) {
//        double recall{static_cast<double>(correct[label]) /
//                      static_cast<double>(counts[label])};
//        BOOST_TEST_REQUIRE(recall > 0.84);
//    }
//}

// TODO
//BOOST_AUTO_TEST_CASE(testClassificationWithUserClassWeights) {
//
//    // Test when the user supplies class weights to control class assignments the
//    // correct weights are propagated through to training.
//
//    std::stringstream output;
//    auto outputWriterFactory = [&output]() {
//        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
//    };
//
//    std::size_t numberExamples{500};
//    TStrVec fieldNames{"f1", "f2", "f3", "target", ".", "."};
//    TStrVec fieldValues{"", "", "", "", "0", ""};
//    test::CRandomNumbers rng;
//    TDoubleVec weights{1.0, 1.0, 1.0};
//    TDoubleVec regressors;
//    rng.generateUniformSamples(-5.0, 10.0, numberExamples * weights.size(), regressors);
//
//    TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
//    auto spec = test::CDataFrameAnalysisSpecificationFactory{}
//                    .rows(numberExamples)
//                    .columns(4)
//                    .memoryLimit(18000000)
//                    .predictionCategoricalFieldNames({"target"})
//                    .classificationWeights({{"foo", 0.8}, {"bar", 0.2}})
//                    .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::classification(),
//                                    "target", &frameAndDirectory);
//    api::CDataFrameAnalyzer analyzer{std::move(spec), std::move(frameAndDirectory),
//                                     std::move(outputWriterFactory)};
//
//    TStrVec actuals;
//    auto frame = test::CDataFrameAnalyzerTrainingFactory::setupBinaryClassificationData(
//        fieldNames, fieldValues, analyzer, weights, regressors, actuals);
//    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "$"});
//
//    auto classifier =
//        maths::analytics::CBoostedTreeFactory::constructFromParameters(
//            1, std::make_unique<maths::analytics::boosted_tree::CBinomialLogisticLoss>())
//            .classAssignmentObjective(maths::analytics::CBoostedTree::E_Custom)
//            .classificationWeights({{"foo", 0.8}, {"bar", 0.2}})
//            .buildForTrain(*frame, 3);
//
//    classifier->train();
//    classifier->predict();
//
//    TStrVec expectedPredictions(frame->numberRows());
//    frame->readRows(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
//        for (auto row = beginRows; row != endRows; ++row) {
//            auto scores = classifier->adjustedPrediction(*row);
//            std::size_t prediction(scores[1] < scores[0] ? 0 : 1);
//            expectedPredictions[row->index()] = frame->categoricalColumnValues()[3][prediction];
//        }
//    });
//
//    json::error_code ec;
//    json::value results = json::parse(output.str(), ec);
//    BOOST_TEST_REQUIRE(ec.failed() == false);
//    BOOST_TEST_REQUIRE(results.is_array());
//
//    auto expectedPrediction = expectedPredictions.begin();
//    for (const auto& result_ : results.as_array()) {
//        BOOST_TEST_REQUIRE(result_.is_object());
//        const json::object& result = result_.as_object();
//        if (result.contains("row_results")) {
//            BOOST_TEST_REQUIRE(expectedPrediction != expectedPredictions.end());
//            std::string prediction{
//                result_.at_pointer("/row_results/results/ml/target_prediction").as_string()};
//            BOOST_TEST_REQUIRE(*expectedPrediction == prediction);
//            ++expectedPrediction;
//        }
//    }
//}

// TODO
//BOOST_AUTO_TEST_CASE(testClassificationIncrementalTraining) {
//
//    // Test running incremental training from the analyzer matches running directly.
//
//    std::size_t maximumNumberTrees{30};
//    std::size_t numberExamples{100};
//
//    auto makeTrainSpec = [&](const std::string& dependentVariable,
//                             TDataFrameUPtrTemporaryDirectoryPtrPr& frameAndDirectory,
//                             TPersisterSupplier* persisterSupplier,
//                             TRestoreSearcherSupplier* restorerSupplier) {
//        test::CDataFrameAnalysisSpecificationFactory specFactory;
//        return specFactory.rows(numberExamples)
//            .memoryLimit(15000000)
//            .predictionCategoricalFieldNames({dependentVariable})
//            .predictionMaximumNumberTrees(maximumNumberTrees)
//            .predictionPersisterSupplier(persisterSupplier)
//            .predictionRestoreSearcherSupplier(restorerSupplier)
//            .regressionLossFunction(TLossFunctionType::E_BinaryClassification)
//            .dataSummarizationFraction(0.1)
//            .task(test::CDataFrameAnalysisSpecificationFactory::TTask::E_Train)
//            .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::classification(),
//                            dependentVariable, &frameAndDirectory);
//    };
//
//    double alpha;
//    double lambda;
//    double gamma;
//    double softTreeDepthLimit;
//    double softTreeDepthTolerance;
//    double eta;
//    double etaGrowthRatePerTree;
//    double downsampleFactor;
//    double featureBagFraction;
//    double lossGap;
//
//    auto makeUpdateSpec = [&](const std::string& dependentVariable,
//                              TDataFrameUPtrTemporaryDirectoryPtrPr& frameAndDirectory,
//                              TPersisterSupplier* persisterSupplier,
//                              TRestoreSearcherSupplier* restorerSupplier) {
//        test::CDataFrameAnalysisSpecificationFactory specFactory;
//        return specFactory.rows(2 * numberExamples)
//            .memoryLimit(15000000)
//            .predictionCategoricalFieldNames({dependentVariable})
//            .predictionAlpha(alpha)
//            .predictionLambda(lambda)
//            .predictionGamma(gamma)
//            .predictionSoftTreeDepthLimit(softTreeDepthLimit)
//            .predictionSoftTreeDepthTolerance(softTreeDepthTolerance)
//            .predictionEta(eta)
//            .predictionEtaGrowthRatePerTree(etaGrowthRatePerTree)
//            .predictionMaximumNumberTrees(maximumNumberTrees)
//            .predictionDownsampleFactor(downsampleFactor)
//            .predictionFeatureBagFraction(featureBagFraction)
//            .previousTrainLossGap(lossGap)
//            .previousTrainNumberRows(numberExamples)
//            .predictionPersisterSupplier(persisterSupplier)
//            .predictionRestoreSearcherSupplier(restorerSupplier)
//            .regressionLossFunction(TLossFunctionType::E_BinaryClassification)
//            .task(test::CDataFrameAnalysisSpecificationFactory::TTask::E_Update)
//            .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::classification(),
//                            dependentVariable, &frameAndDirectory);
//    };
//
//    std::stringstream outputStream;
//    auto outputWriterFactory = [&outputStream]() {
//        return std::make_unique<core::CJsonOutputStreamWrapper>(outputStream);
//    };
//
//    // Run once.
//    TStrVec fieldNames{"f1", "f2", "f3", "f4", "target", ".", "."};
//    TStrVec fieldValues{"", "", "", "", "", "0", ""};
//    TDoubleVec weights{0.1, 2.0, 0.4, -0.5};
//    TDoubleVec regressors;
//    test::CRandomNumbers rng;
//    rng.generateUniformSamples(-10.0, 10.0, weights.size() * numberExamples, regressors);
//    TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
//    auto spec = makeTrainSpec("target", frameAndDirectory, nullptr, nullptr);
//    api::CDataFrameAnalyzer analyzer{std::move(spec), std::move(frameAndDirectory),
//                                     outputWriterFactory};
//    TStrVec targets;
//    auto frame = test::CDataFrameAnalyzerTrainingFactory::setupBinaryClassificationData(
//        fieldNames, fieldValues, analyzer, weights, regressors, targets);
//    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});
//
//    // Train a model for comparison.
//    auto classification =
//        maths::analytics::CBoostedTreeFactory::constructFromParameters(
//            1, std::make_unique<maths::analytics::boosted_tree::CBinomialLogisticLoss>())
//            .maximumNumberTrees(maximumNumberTrees)
//            .dataSummarizationFraction(0.1)
//            .buildForTrain(*frame, weights.size());
//    classification->train();
//    classification->predict();
//
//    // Retrieve documents from the result stream that will be used to restore the model.
//
//    std::stringstream incrementalTrainingState;
//    readIncrementalTrainingState(outputStream.str(), alpha, lambda, gamma,
//                                 softTreeDepthLimit, softTreeDepthTolerance,
//                                 eta, etaGrowthRatePerTree, downsampleFactor,
//                                 featureBagFraction, lossGap, incrementalTrainingState);
//
//    // Pass model definition and data summarization into the restore stream.
//    auto restoreStreamPtr =
//        std::make_shared<std::stringstream>(std::move(incrementalTrainingState));
//    TRestoreSearcherSupplier restorerSupplier{[&restoreStreamPtr]() {
//        return std::make_unique<api::CSingleStreamSearcher>(restoreStreamPtr);
//    }};
//
//    outputStream.clear();
//    outputStream.str("");
//
//    // Create a new spec for incremental training.
//    spec = makeUpdateSpec("target", frameAndDirectory, nullptr, &restorerSupplier);
//
//    // Create new analyzer and run incremental training.
//    api::CDataFrameAnalyzer analyzerIncremental{
//        std::move(spec), std::move(frameAndDirectory), outputWriterFactory};
//    auto newTrainingDataFrame = test::CDataFrameAnalyzerTrainingFactory::setupBinaryClassificationData(
//        fieldNames, fieldValues, analyzerIncremental, weights, regressors, targets);
//    analyzerIncremental.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});
//
//    json::error_code ec;
//    json::value results = json::parse(outputStream.str(), ec);
//    BOOST_TEST_REQUIRE(ec.failed() == false);
//    BOOST_TEST_REQUIRE(results.is_array());
//
//    // Read the predictions.
//    TDoubleVec predictions;
//    for (const auto& result_ : results.as_array()) {
//        const json::object& result = result_.as_object();
//        if (result.contains("row_results")) {
//            predictions.emplace_back(
//                result_.at_pointer("/row_results/results/ml/prediction_probability").as_double());
//        }
//    }
//    BOOST_REQUIRE_EQUAL(numberExamples, predictions.size());
//
//    frame->resizeColumns(1, weights.size() + 1);
//
//    auto summarisation = classification->dataSummarization();
//
//    TDoubleVecVec newTrainingData;
//    newTrainingData.reserve(numberExamples +
//                            static_cast<std::size_t>(summarisation.manhattan()));
//    frame->readRows(1, 0, frame->numberRows(),
//                    [&](const TRowItr& beginRows, const TRowItr& endRows) {
//                        for (auto row = beginRows; row != endRows; ++row) {
//                            newTrainingData.push_back(TDoubleVec(row->numberColumns()));
//                            row->copyTo(newTrainingData.back().begin());
//                        }
//                    },
//                    &summarisation);
//    newTrainingDataFrame->readRows(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
//        for (auto row = beginRows; row != endRows; ++row) {
//            newTrainingData.push_back(TDoubleVec(row->numberColumns()));
//            row->copyTo(newTrainingData.back().begin());
//        }
//    });
//
//    frame->resizeRows(0);
//    for (std::size_t i = 0; i < newTrainingData.size(); ++i) {
//        frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t& id) {
//            for (std::size_t j = 0; j < newTrainingData[i].size(); ++j, ++column) {
//                *column = newTrainingData[i][j];
//                id = static_cast<std::int32_t>(i);
//            }
//        });
//    }
//    frame->finishWritingRows();
//
//    core::CPackedBitVector newTrainingRowMask(
//        static_cast<std::size_t>(summarisation.manhattan()), false);
//    newTrainingRowMask.extend(true, numberExamples);
//
//    classification = maths::analytics::CBoostedTreeFactory::constructFromModel(
//                         std::move(classification))
//                         .newTrainingRowMask(newTrainingRowMask)
//                         .buildForTrainIncremental(*frame, weights.size());
//
//    classification->trainIncremental();
//    classification->predict();
//
//    auto prediction = predictions.begin();
//    frame->readRows(
//        1, 0, frame->numberRows(),
//        [&](const TRowItr& beginRows, const TRowItr& endRows) {
//            for (auto row = beginRows; row != endRows; ++row) {
//                double expectedPrediction{classification->prediction(*row)[0]};
//                // The prediction_probability result contains the highest scoring
//                // class probability which is usually, but not always, the highest
//                // class probability. The probability of the prediction result is
//                // therefore not a consistent class while readPrediction always
//                // returns the probability of class 1. Here, we simply check that
//                // the prediction_probability matches the probability of one of the
//                // classes, since it is very unlikely to match the wrong class by
//                // chance.
//                BOOST_REQUIRE((std::fabs(*prediction - expectedPrediction) < 1e-6) ||
//                              (std::fabs(*prediction + expectedPrediction - 1.0) < 1e-6));
//                ++prediction;
//            }
//        },
//        &newTrainingRowMask);
//}

// TODO
//BOOST_AUTO_TEST_CASE(testIncrementalTrainingFieldMismatch) {
//
//    // Test running incremental with reordered fields and extra fields.
//
//    std::size_t maximumNumberTrees{30};
//    std::size_t numberExamples{100};
//
//    auto makeTrainSpec = [&](const std::string& dependentVariable,
//                             TDataFrameUPtrTemporaryDirectoryPtrPr& frameAndDirectory,
//                             TPersisterSupplier* persisterSupplier,
//                             TRestoreSearcherSupplier* restorerSupplier) {
//        test::CDataFrameAnalysisSpecificationFactory specFactory;
//        return specFactory.rows(numberExamples)
//            .memoryLimit(15000000)
//            .predictionCategoricalFieldNames({dependentVariable})
//            .predictionMaximumNumberTrees(maximumNumberTrees)
//            .predictionPersisterSupplier(persisterSupplier)
//            .predictionRestoreSearcherSupplier(restorerSupplier)
//            .regressionLossFunction(TLossFunctionType::E_BinaryClassification)
//            .dataSummarizationFraction(0.1)
//            .task(test::CDataFrameAnalysisSpecificationFactory::TTask::E_Train)
//            .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::classification(),
//                            dependentVariable, &frameAndDirectory);
//    };
//
//    double alpha;
//    double lambda;
//    double gamma;
//    double softTreeDepthLimit;
//    double softTreeDepthTolerance;
//    double eta;
//    double etaGrowthRatePerTree;
//    double downsampleFactor;
//    double featureBagFraction;
//    double lossGap;
//
//    auto makeUpdateSpec = [&](const std::string& dependentVariable, TStrVec categoricalFields,
//                              TDataFrameUPtrTemporaryDirectoryPtrPr& frameAndDirectory,
//                              TPersisterSupplier* persisterSupplier,
//                              TRestoreSearcherSupplier* restorerSupplier) {
//        test::CDataFrameAnalysisSpecificationFactory specFactory;
//        categoricalFields.push_back(dependentVariable);
//        return specFactory.rows(2 * numberExamples)
//            .memoryLimit(15000000)
//            .predictionCategoricalFieldNames(categoricalFields)
//            .predictionAlpha(alpha)
//            .predictionLambda(lambda)
//            .predictionGamma(gamma)
//            .predictionSoftTreeDepthLimit(softTreeDepthLimit)
//            .predictionSoftTreeDepthTolerance(softTreeDepthTolerance)
//            .predictionEta(eta)
//            .predictionEtaGrowthRatePerTree(etaGrowthRatePerTree)
//            .predictionMaximumNumberTrees(maximumNumberTrees)
//            .predictionDownsampleFactor(downsampleFactor)
//            .predictionFeatureBagFraction(featureBagFraction)
//            .previousTrainLossGap(lossGap)
//            .previousTrainNumberRows(numberExamples)
//            .predictionPersisterSupplier(persisterSupplier)
//            .predictionRestoreSearcherSupplier(restorerSupplier)
//            .regressionLossFunction(TLossFunctionType::E_BinaryClassification)
//            .task(test::CDataFrameAnalysisSpecificationFactory::TTask::E_Update)
//            .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::classification(),
//                            dependentVariable, &frameAndDirectory);
//    };
//
//    std::stringstream outputStream;
//    auto outputWriterFactory = [&outputStream]() {
//        return std::make_unique<core::CJsonOutputStreamWrapper>(outputStream);
//    };
//
//    // Run once.
//    TStrVec fieldNames{"f1", "f2", "f3", "f4", "target", ".", "."};
//    TStrVec fieldValues{"", "", "", "", "", "0", ""};
//    TDoubleVec weights{0.1, 2.0, 0.4, -0.5};
//    TDoubleVec regressors;
//    test::CRandomNumbers rng;
//    rng.generateUniformSamples(-10.0, 10.0, weights.size() * numberExamples, regressors);
//    TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
//    auto spec = makeTrainSpec("target", frameAndDirectory, nullptr, nullptr);
//    api::CDataFrameAnalyzer analyzer{std::move(spec), std::move(frameAndDirectory),
//                                     outputWriterFactory};
//    TStrVec targets;
//    auto frame = test::CDataFrameAnalyzerTrainingFactory::setupBinaryClassificationData(
//        fieldNames, fieldValues, analyzer, weights, regressors, targets);
//    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});
//
//    // Train a model for comparison.
//    auto classification =
//        maths::analytics::CBoostedTreeFactory::constructFromParameters(
//            1, std::make_unique<maths::analytics::boosted_tree::CBinomialLogisticLoss>())
//            .maximumNumberTrees(maximumNumberTrees)
//            .dataSummarizationFraction(0.1)
//            .buildForTrain(*frame, weights.size());
//    classification->train();
//    classification->predict();
//
//    // Retrieve documents from the result stream that will be used to restore the model.
//    std::stringstream incrementalTrainingState;
//    readIncrementalTrainingState(outputStream.str(), alpha, lambda, gamma,
//                                 softTreeDepthLimit, softTreeDepthTolerance,
//                                 eta, etaGrowthRatePerTree, downsampleFactor,
//                                 featureBagFraction, lossGap, incrementalTrainingState);
//
//    outputStream.clear();
//    outputStream.str("");
//
//    LOG_DEBUG(<< "Test extra field");
//
//    {
//        std::stringstream incrementalTrainingStateCopy;
//        incrementalTrainingStateCopy << incrementalTrainingState.str();
//
//        // Pass model definition and data summarization into the restore stream.
//        auto restoreStreamPtr = std::make_shared<std::stringstream>(
//            std::move(incrementalTrainingStateCopy));
//        TRestoreSearcherSupplier restorerSupplier{[&restoreStreamPtr]() {
//            return std::make_unique<api::CSingleStreamSearcher>(restoreStreamPtr);
//        }};
//
//        // Create a new spec for incremental training.
//        spec = makeUpdateSpec("target", {}, frameAndDirectory, nullptr, &restorerSupplier);
//
//        api::CDataFrameAnalyzer analyzerIncremental{
//            std::move(spec), std::move(frameAndDirectory), outputWriterFactory};
//
//        auto errorHandler = [](std::string error) {
//            throw std::runtime_error(error);
//        };
//        core::CLogger::CScopeSetFatalErrorHandler scope{errorHandler};
//
//        fieldNames.assign({"f1", "f2", "f3", "f5", "target", ".", "."});
//
//        // Adding an unseen field should be a fatal error.
//        bool throws{false};
//        try {
//            test::CDataFrameAnalyzerTrainingFactory::setupBinaryClassificationData(
//                fieldNames, fieldValues, analyzerIncremental, weights, regressors, targets);
//        } catch (const std::exception& e) {
//            LOG_DEBUG(<< "Caught '" << e.what() << "'");
//            throws = true;
//        }
//        BOOST_TEST_REQUIRE(throws);
//    }
//
//    LOG_DEBUG(<< "Test mismatching categorical fields");
//
//    {
//        std::stringstream incrementalTrainingStateCopy;
//        incrementalTrainingStateCopy << incrementalTrainingState.str();
//
//        // Pass model definition and data summarization into the restore stream.
//        auto restoreStreamPtr = std::make_shared<std::stringstream>(
//            std::move(incrementalTrainingStateCopy));
//        TRestoreSearcherSupplier restorerSupplier{[&restoreStreamPtr]() {
//            return std::make_unique<api::CSingleStreamSearcher>(restoreStreamPtr);
//        }};
//
//        // Create a new spec for incremental training.
//        spec = makeUpdateSpec("target", {"f4"}, frameAndDirectory, nullptr, &restorerSupplier);
//
//        api::CDataFrameAnalyzer analyzerIncremental{
//            std::move(spec), std::move(frameAndDirectory), outputWriterFactory};
//
//        auto errorHandler = [](std::string error) {
//            throw std::runtime_error(error);
//        };
//        core::CLogger::CScopeSetFatalErrorHandler scope{errorHandler};
//
//        fieldNames.assign({"f1", "f2", "f3", "f4", "target", ".", "."});
//
//        // Adding an unseen field should be a fatal error.
//        bool throws{false};
//        try {
//            test::CDataFrameAnalyzerTrainingFactory::setupBinaryClassificationData(
//                fieldNames, fieldValues, analyzerIncremental, weights, regressors, targets);
//        } catch (const std::exception& e) {
//            LOG_DEBUG(<< "Caught '" << e.what() << "'");
//            throws = true;
//        }
//        BOOST_TEST_REQUIRE(throws);
//    }
//
//    LOG_DEBUG(<< "Test permute fields");
//
//    auto restoreStreamPtr =
//        std::make_shared<std::stringstream>(std::move(incrementalTrainingState));
//    TRestoreSearcherSupplier restorerSupplier{[&restoreStreamPtr]() {
//        return std::make_unique<api::CSingleStreamSearcher>(restoreStreamPtr);
//    }};
//
//    // Create a new spec for incremental training.
//    spec = makeUpdateSpec("target", {}, frameAndDirectory, nullptr, &restorerSupplier);
//
//    api::CDataFrameAnalyzer analyzerIncremental{
//        std::move(spec), std::move(frameAndDirectory), outputWriterFactory};
//
//    // Permuting the field order should have no effect.
//    fieldNames.assign({"f2", "f3", "f4", "f1", "target", ".", "."});
//
//    // Create new analyzer and run incremental training.
//    auto newTrainingDataFrame = test::CDataFrameAnalyzerTrainingFactory::setupBinaryClassificationData(
//        fieldNames, fieldValues, analyzerIncremental, weights, regressors, targets);
//    analyzerIncremental.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});
//
//    json::error_code ec;
//    json::value results = json::parse(outputStream.str(), ec);
//    BOOST_TEST_REQUIRE(ec.failed() == false);
//    BOOST_TEST_REQUIRE(results.is_array());
//
//    // Read the predictions.
//    TDoubleVec predictions;
//    for (const auto& result : results.as_array()) {
//        if (result.as_object().contains("row_results")) {
//            predictions.emplace_back(
//                result.at_pointer("/row_results/results/ml/prediction_probability").as_double());
//        }
//    }
//    BOOST_REQUIRE_EQUAL(numberExamples, predictions.size());
//
//    frame->resizeColumns(1, weights.size() + 1);
//
//    auto summarisation = classification->dataSummarization();
//
//    TDoubleVecVec newTrainingData;
//    newTrainingData.reserve(numberExamples +
//                            static_cast<std::size_t>(summarisation.manhattan()));
//    frame->readRows(1, 0, frame->numberRows(),
//                    [&](const TRowItr& beginRows, const TRowItr& endRows) {
//                        for (auto row = beginRows; row != endRows; ++row) {
//                            newTrainingData.push_back(TDoubleVec(row->numberColumns()));
//                            row->copyTo(newTrainingData.back().begin());
//                        }
//                    },
//                    &summarisation);
//    TSizeVec permutation{3, 0, 1, 2, 4};
//    newTrainingDataFrame->readRows(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
//        for (auto row = beginRows; row != endRows; ++row) {
//            newTrainingData.emplace_back();
//            newTrainingData.back().reserve(permutation.size());
//            for (auto i : permutation) {
//                newTrainingData.back().push_back((*row)[i]);
//            }
//        }
//    });
//
//    frame->resizeRows(0);
//    for (std::size_t i = 0; i < newTrainingData.size(); ++i) {
//        frame->writeRow([&](core::CDataFrame::TFloatVecItr column, std::int32_t& id) {
//            for (std::size_t j = 0; j < newTrainingData[i].size(); ++j, ++column) {
//                *column = newTrainingData[i][j];
//                id = static_cast<std::int32_t>(i);
//            }
//        });
//    }
//    frame->finishWritingRows();
//
//    core::CPackedBitVector newTrainingRowMask(
//        static_cast<std::size_t>(summarisation.manhattan()), false);
//    newTrainingRowMask.extend(true, numberExamples);
//
//    classification = maths::analytics::CBoostedTreeFactory::constructFromModel(
//                         std::move(classification))
//                         .newTrainingRowMask(newTrainingRowMask)
//                         .buildForTrainIncremental(*frame, weights.size());
//
//    classification->trainIncremental();
//    classification->predict();
//
//    auto prediction = predictions.begin();
//    frame->readRows(
//        1, 0, frame->numberRows(),
//        [&](const TRowItr& beginRows, const TRowItr& endRows) {
//            for (auto row = beginRows; row != endRows; ++row) {
//                double expectedPrediction{classification->prediction(*row)[0]};
//                // See testClassificationIncrementalTraining for an explanation.
//                BOOST_REQUIRE((std::fabs(*prediction - expectedPrediction) < 1e-6) ||
//                              (std::fabs(*prediction + expectedPrediction - 1.0) < 1e-6));
//                ++prediction;
//            }
//        },
//        &newTrainingRowMask);
//}

// TODO
//BOOST_AUTO_TEST_CASE(testParsingOfCategoricalFields) {
//
//    // Test that we correctly map categorical fields and handle the case that the
//    // cardinality is so high it overflows the max representable integer in a data
//    // frame.
//
//    std::stringstream output;
//    auto outputWriterFactory = [&output]() {
//        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
//    };
//
//    {
//        TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
//        auto spec = test::CDataFrameAnalysisSpecificationFactory{}
//                        .rows(1000)
//                        .memoryLimit(27000000)
//                        .predictionCategoricalFieldNames({"x1", "x2"})
//                        .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::regression(),
//                                        "x5", &frameAndDirectory);
//        api::CDataFrameAnalyzer analyzer{
//            std::move(spec), std::move(frameAndDirectory), outputWriterFactory};
//
//        TStrVec x[]{{"x11", "x12", "x13", "x14", "x15"},
//                    {"x21", "x22", "x23", "x24", "x25", "x26", "x27"}};
//
//        for (std::size_t i = 0; i < 10; ++i) {
//            analyzer.handleRecord({"x1", "x2", "x3", "x4", "x5", ".", "."},
//                                  {x[0][i % x[0].size()], x[1][i % x[1].size()],
//                                   std::to_string(i), std::to_string(i),
//                                   std::to_string(i), std::to_string(i), ""});
//        }
//        analyzer.receivedAllRows();
//
//        bool passed{true};
//
//        const core::CDataFrame& frame{analyzer.dataFrame()};
//        frame.readRows(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
//            std::size_t i{0};
//            for (auto row = beginRows; row != endRows; ++row, ++i) {
//                core::CFloatStorage expected[]{static_cast<double>(i % x[0].size()),
//                                               static_cast<double>(i % x[1].size())};
//                bool wasPassed{passed};
//                passed &= (expected[0] == (*row)[0]);
//                passed &= (expected[1] == (*row)[1]);
//                if (wasPassed && passed == false) {
//                    LOG_ERROR(<< "expected " << expected << ", got ["
//                              << (*row)[0] << ", " << (*row)[1] << "]");
//                }
//            }
//        });
//
//        BOOST_TEST_REQUIRE(passed);
//    }
//
//    LOG_DEBUG(<< "Test overflow");
//    {
//        std::size_t rows{core::CDataFrame::MAX_CATEGORICAL_CARDINALITY + 3};
//
//        TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
//        auto spec = test::CDataFrameAnalysisSpecificationFactory{}
//                        .rows(rows)
//                        .memoryLimit(8000000000)
//                        .predictionCategoricalFieldNames({"x1"})
//                        .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::regression(),
//                                        "x5", &frameAndDirectory);
//        api::CDataFrameAnalyzer analyzer{
//            std::move(spec), std::move(frameAndDirectory), outputWriterFactory};
//
//        TStrVec fieldNames{"x1", "x2", "x3", "x4", "x5", ".", "."};
//        TStrVec fieldValues{"", "", "", "", "", "", ""};
//        for (std::size_t i = 0; i < rows; ++i) {
//            std::fill_n(fieldValues.begin(), 6, std::to_string(i));
//            analyzer.handleRecord(fieldNames, fieldValues);
//        }
//        analyzer.receivedAllRows();
//
//        bool passed{true};
//        std::size_t i{0};
//
//        const core::CDataFrame& frame{analyzer.dataFrame()};
//        frame.readRows(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
//            for (auto row = beginRows; row != endRows; ++row, ++i) {
//                core::CFloatStorage expected{
//                    i < core::CDataFrame::MAX_CATEGORICAL_CARDINALITY
//                        ? static_cast<double>(i)
//                        : static_cast<double>(core::CDataFrame::MAX_CATEGORICAL_CARDINALITY)};
//                bool wasPassed{passed};
//                passed &= (expected == (*row)[0]);
//                if (wasPassed && passed == false) {
//                    LOG_ERROR(<< "expected " << expected << ", got " << (*row)[0]);
//                }
//            }
//        });
//
//        BOOST_TEST_REQUIRE(passed);
//    }
//}

// TODO
//BOOST_AUTO_TEST_CASE(testCategoricalFieldsEmptyAsMissing) {
//
//    // Test supplying an empty category is interpreted as if the value is missing.
//
//    auto eq = [](double expected) {
//        return [expected](double actual) { return expected == actual; };
//    };
//
//    auto missing = [](double actual) {
//        return maths::analytics::CDataFrameUtils::isMissing(actual);
//    };
//
//    auto assertRow = [&](const std::size_t row_i,
//                         const std::vector<std::function<bool(double)>>& matchers,
//                         const TRowRef& row) {
//        BOOST_REQUIRE_MESSAGE(matchers.size() == row.numberColumns(),
//                              "row " + std::to_string(row_i));
//        for (std::size_t i = 0; i < row.numberColumns(); ++i) {
//            BOOST_REQUIRE_MESSAGE(matchers[i](row[i]), "row " + std::to_string(row_i) + ", column " +
//                                                           std::to_string(i));
//        }
//    };
//
//    std::stringstream output;
//    auto outputWriterFactory = [&output]() {
//        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
//    };
//
//    std::string missingString{"foo"};
//
//    TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
//    auto spec = test::CDataFrameAnalysisSpecificationFactory{}
//                    .rows(1000)
//                    .memoryLimit(27000000)
//                    .predictionCategoricalFieldNames({"x1", "x2", "x5"})
//                    .missingString(missingString)
//                    .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::classification(),
//                                    "x5", &frameAndDirectory);
//    api::CDataFrameAnalyzer analyzer{std::move(spec), std::move(frameAndDirectory),
//                                     std::move(outputWriterFactory)};
//
//    TStrVec fieldNames{"x1", "x2", "x3", "x4", "x5", ".", "."};
//    analyzer.handleRecord(fieldNames, {"x11", "x21", "0", "0", "x51", "0", ""});
//    analyzer.handleRecord(fieldNames, {"x12", "x22", "1", "1", "x52", "1", ""});
//    analyzer.handleRecord(fieldNames, {"", "x23", "2", "2", "x51", "2", ""});
//    analyzer.handleRecord(fieldNames, {"x14", "x24", "3", "3", missingString, "3", ""});
//    analyzer.handleRecord(fieldNames, {"x15", "x25", "4", "4", "x51", "4", ""});
//    analyzer.handleRecord(fieldNames, {"x11", "x26", "5", "5", "x52", "5", ""});
//    analyzer.handleRecord(fieldNames, {"x12", "", "6", "6", missingString, "6", ""});
//    analyzer.handleRecord(fieldNames, {"x13", "x21", "7", "7", missingString, "7", ""});
//    analyzer.handleRecord(fieldNames, {"x14", "x22", "8", "8", "x51", "8", ""});
//    analyzer.handleRecord(fieldNames, {"", "x23", "9", "9", "x52", "9", ""});
//    analyzer.receivedAllRows();
//
//    const core::CDataFrame& frame{analyzer.dataFrame()};
//    frame.readRows(1, [&](const TRowItr& beginRows, const TRowItr& endRows) {
//        std::vector<TRowRef> rows;
//        std::copy(beginRows, endRows, std::back_inserter(rows));
//        BOOST_REQUIRE_EQUAL(std::size_t{10}, rows.size());
//        assertRow(0, {eq(0.0), eq(0.0), eq(0.0), eq(0.0), eq(0.0)}, rows[0]);
//        assertRow(1, {eq(1.0), eq(1.0), eq(1.0), eq(1.0), eq(1.0)}, rows[1]);
//        assertRow(2, {eq(2.0), eq(2.0), eq(2.0), eq(2.0), eq(0.0)}, rows[2]);
//        assertRow(3, {eq(3.0), eq(3.0), eq(3.0), eq(3.0), missing}, rows[3]);
//        assertRow(4, {eq(4.0), eq(4.0), eq(4.0), eq(4.0), eq(0.0)}, rows[4]);
//        assertRow(5, {eq(0.0), eq(5.0), eq(5.0), eq(5.0), eq(1.0)}, rows[5]);
//        assertRow(6, {eq(1.0), eq(6.0), eq(6.0), eq(6.0), missing}, rows[6]);
//        assertRow(7, {eq(5.0), eq(0.0), eq(7.0), eq(7.0), missing}, rows[7]);
//        assertRow(8, {eq(3.0), eq(1.0), eq(8.0), eq(8.0), eq(0.0)}, rows[8]);
//        assertRow(9, {eq(2.0), eq(2.0), eq(9.0), eq(9.0), eq(1.0)}, rows[9]);
//    });
//}

BOOST_AUTO_TEST_CASE(testNoRegressors) {

    // Check we catch an exit immediately if there are too few columns to run the analysis.

    TStrVec errors;
    auto errorHandler = [&errors](std::string error) { errors.push_back(error); };

    core::CLogger::CScopeSetFatalErrorHandler scope{errorHandler};
    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
    auto spec = test::CDataFrameAnalysisSpecificationFactory{}
                    .rows(1000)
                    .columns(1)
                    .memoryLimit(18000000)
                    .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::regression(),
                                    "x1", &frameAndDirectory);
    api::CDataFrameAnalyzer analyzer{std::move(spec), std::move(frameAndDirectory),
                                     std::move(outputWriterFactory)};

    TStrVec fieldNames{"x1", ".", "."};
    analyzer.handleRecord(fieldNames, {"0.0", "0", ""});
    analyzer.handleRecord(fieldNames, {"1.0", "1", ""});
    analyzer.handleRecord(fieldNames, {"2.0", "2", ""});
    analyzer.handleRecord(fieldNames, {"", "", "$"});

    LOG_DEBUG(<< "Errors = " << errors);

    BOOST_TEST_REQUIRE(errors.size() == 1);
    BOOST_REQUIRE_EQUAL(errors[0], "Input error: analysis need at least one regressor.");
}

BOOST_AUTO_TEST_CASE(testProgressMonitoring) {

    // Test we get 100% progress reported for all stages of the analysis.

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", "target", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "", "0", ""};
    TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
    auto spec = test::CDataFrameAnalysisSpecificationFactory{}
                    .rows(300)
                    .columns(6)
                    .memoryLimit(18000000)
                    .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::regression(),
                                    "target", &frameAndDirectory);
    api::CDataFrameAnalyzer analyzer{std::move(spec), std::move(frameAndDirectory),
                                     std::move(outputWriterFactory)};
    test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
        TLossFunctionType::E_MseRegression, fieldNames, fieldValues, analyzer, 300);
    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "", "$"});

    json::error_code ec;
    json::value results = json::parse(output.str(), ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    BOOST_TEST_REQUIRE(results.is_array());

    std::int64_t featureSelectionLastProgress{0};
    std::int64_t coarseParameterSearchLastProgress{0};
    std::int64_t fineTuneParametersLastProgress{0};
    std::int64_t finalTrainLastProgress{0};

    for (const auto& result : results.as_array()) {
        if (result.as_object().contains("phase_progress")) {
            LOG_DEBUG(<< result.as_object().at("phase_progress"));

            std::string phase{result.at_pointer("/phase_progress/phase").as_string()};
            std::int64_t progress{
                result.at_pointer("/phase_progress/progress_percent").as_int64()};
            if (phase == maths::analytics::CBoostedTreeFactory::FEATURE_SELECTION) {
                featureSelectionLastProgress = std::max(featureSelectionLastProgress, progress);
            } else if (phase == maths::analytics::CBoostedTreeFactory::COARSE_PARAMETER_SEARCH) {
                coarseParameterSearchLastProgress =
                    std::max(coarseParameterSearchLastProgress, progress);
            } else if (phase == maths::analytics::CBoostedTreeFactory::FINE_TUNING_PARAMETERS) {
                fineTuneParametersLastProgress =
                    std::max(fineTuneParametersLastProgress, progress);
            } else if (phase == maths::analytics::CBoostedTreeFactory::FINAL_TRAINING) {
                finalTrainLastProgress = std::max(finalTrainLastProgress, progress);
            }
        }
    }

    BOOST_REQUIRE_EQUAL(100, featureSelectionLastProgress);
    BOOST_REQUIRE_EQUAL(100, coarseParameterSearchLastProgress);
    BOOST_REQUIRE_EQUAL(100, fineTuneParametersLastProgress);
    BOOST_REQUIRE_EQUAL(100, finalTrainLastProgress);
}

// TODO
//BOOST_AUTO_TEST_CASE(testProgressMonitoringFromRestart) {
//
//    // Check our progress picks up where it left off if we restart an analysis
//    // from a checkpoint.
//
//    auto makeSpec = [&](TDataFrameUPtrTemporaryDirectoryPtrPr& frameAndDirectory,
//                        TPersisterSupplier* persisterSupplier,
//                        TRestoreSearcherSupplier* restorerSupplier) {
//        test::CDataFrameAnalysisSpecificationFactory specFactory;
//        return specFactory.rows(400)
//            .columns(6)
//            .memoryLimit(18000000)
//            .predictionPersisterSupplier(persisterSupplier)
//            .predictionRestoreSearcherSupplier(restorerSupplier)
//            .earlyStoppingEnabled(false)
//            .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::regression(),
//                            "target", &frameAndDirectory);
//    };
//
//    std::stringstream output;
//    auto outputWriterFactory = [&output]() {
//        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
//    };
//
//    auto persistenceStream = std::make_shared<std::ostringstream>();
//    TPersisterSupplier persisterSupplier{[&persistenceStream]() {
//        return std::make_unique<api::CSingleStreamDataAdder>(persistenceStream);
//    }};
//
//    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", "target", ".", "."};
//    TStrVec fieldValues{"", "", "", "", "", "", "0", ""};
//    TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
//    auto spec = makeSpec(frameAndDirectory, &persisterSupplier, nullptr);
//    api::CDataFrameAnalyzer analyzer{std::move(spec), std::move(frameAndDirectory),
//                                     outputWriterFactory};
//    test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
//        TLossFunctionType::E_MseRegression, fieldNames, fieldValues, analyzer, 400);
//    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "", "$"});
//
//    TStrVec persistedStates{
//        splitOnNull(std::stringstream{std::move(persistenceStream->str())})};
//
//    LOG_DEBUG(<< "# states = " << persistedStates.size());
//
//    output.str("");
//    persistenceStream->str("");
//
//    std::istringstream intermediateStateStream{
//        persistedStates[2 * persistedStates.size() / 3]};
//    TRestoreSearcherSupplier restoreSearcherSupplier{[&intermediateStateStream]() {
//        return std::make_unique<CTestDataSearcher>(intermediateStateStream.str());
//    }};
//
//    spec = makeSpec(frameAndDirectory, &persisterSupplier, &restoreSearcherSupplier);
//    api::CDataFrameAnalyzer restoredAnalyzer{
//        std::move(spec), std::move(frameAndDirectory), outputWriterFactory};
//    test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
//        TLossFunctionType::E_MseRegression, fieldNames, fieldValues, restoredAnalyzer, 400);
//    restoredAnalyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "", "$"});
//
//    json::error_code ec;
//    json::value results = json::parse(output.str(), ec);
//    BOOST_TEST_REQUIRE(ec.failed() == false);
//    BOOST_TEST_REQUIRE(results.is_array());
//
//    std::int64_t coarseParameterSearchLastProgress{0};
//    std::int64_t fineTuneParametersFirstProgress{100};
//    std::int64_t featureSelectionLastProgress{0};
//    std::int64_t fineTuneParametersLastProgress{0};
//    std::int64_t finalTrainLastProgress{0};
//
//    for (const auto& result : results.as_array()) {
//        if (result.as_object().contains("phase_progress")) {
//            LOG_DEBUG(<< result.at("phase_progress"));
//
//            std::string phase{result.at_pointer("phase_progress/phase").as_string()};
//            std::int64_t progress{result.at_pointer("phase_progress/progress_percent").as_int64()};
//            if (phase == maths::analytics::CBoostedTreeFactory::FEATURE_SELECTION) {
//                featureSelectionLastProgress = std::max(featureSelectionLastProgress, progress);
//            } else if (phase == maths::analytics::CBoostedTreeFactory::COARSE_PARAMETER_SEARCH) {
//                coarseParameterSearchLastProgress =
//                    std::max(coarseParameterSearchLastProgress, progress);
//            } else if (phase == maths::analytics::CBoostedTreeFactory::FINE_TUNING_PARAMETERS) {
//                if (progress > 0) {
//                    fineTuneParametersFirstProgress =
//                        std::min(fineTuneParametersFirstProgress, progress);
//                }
//                fineTuneParametersLastProgress =
//                    std::max(fineTuneParametersLastProgress, progress);
//            } else if (phase == maths::analytics::CBoostedTreeFactory::FINAL_TRAINING) {
//                finalTrainLastProgress = std::max(finalTrainLastProgress, progress);
//            }
//        }
//    }
//
//    BOOST_REQUIRE_EQUAL(100, featureSelectionLastProgress);
//    BOOST_REQUIRE_EQUAL(100, coarseParameterSearchLastProgress);
//    BOOST_TEST_REQUIRE(fineTuneParametersFirstProgress > 50);
//    BOOST_REQUIRE_EQUAL(100, fineTuneParametersLastProgress);
//    BOOST_REQUIRE_EQUAL(100, finalTrainLastProgress);
//}

// TODO
//BOOST_AUTO_TEST_CASE(testCreationForEncoding) {
//    // We perform encoding and training in two separate steps and make sure that
//    // no errors were logged.
//
//    std::size_t rowsEncode{1000};
//    std::size_t rowsTrain{100};
//
//    // keep track of logged messages in a separate stream
//    ml::core::CLogger& logger{ml::core::CLogger::instance()};
//    logger.reset();
//    logger.setLoggingLevel(ml::core::CLogger::E_Error);
//    auto logDataSPtr = boost::make_shared<std::stringstream>();
//    logger.reconfigure(logDataSPtr);
//
//    auto makeSpec = [&](std::size_t rows, TDataFrameUPtrTemporaryDirectoryPtrPr& frameAndDirectory,
//                        TPersisterSupplier* persisterSupplier,
//                        TRestoreSearcherSupplier* restorerSupplier,
//                        test::CDataFrameAnalysisSpecificationFactory::TTask task) {
//        test::CDataFrameAnalysisSpecificationFactory specFactory;
//        return specFactory.rows(rows)
//            .columns(6)
//            .memoryLimit(15000000)
//            .predictionMaximumNumberTrees(3)
//            .predictionPersisterSupplier(persisterSupplier)
//            .predictionRestoreSearcherSupplier(restorerSupplier)
//            .regressionLossFunction(TLossFunctionType::E_MseRegression)
//            .task(task)
//            .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::regression(),
//                            "target", &frameAndDirectory);
//    };
//
//    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", "target", ".", "."};
//    TStrVec fieldValues{"", "", "", "", "", "", "0", ""};
//    TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
//
//    std::stringstream output;
//    auto outputWriterFactory = [&output]() {
//        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
//    };
//
//    auto persistenceStream = std::make_shared<std::ostringstream>();
//    TPersisterSupplier persisterSupplier{[&persistenceStream]() {
//        return std::make_unique<api::CSingleStreamDataAdder>(persistenceStream);
//    }};
//    auto spec = makeSpec(rowsEncode, frameAndDirectory, &persisterSupplier, nullptr,
//                         test::CDataFrameAnalysisSpecificationFactory::TTask::E_Encode);
//    api::CDataFrameAnalyzer analyzer{std::move(spec), std::move(frameAndDirectory),
//                                     outputWriterFactory};
//    test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
//        TLossFunctionType::E_MseRegression, fieldNames, fieldValues, analyzer, rowsEncode);
//    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "", "$"});
//
//    TStrVec persistedStates{splitOnNull(std::stringstream{persistenceStream->str()})};
//
//    BOOST_REQUIRE(persistedStates.empty() == false);
//
//    // Pass persisted state as a state to restore from in the new analyzer.
//    std::istringstream lastStateStream{persistedStates[0]};
//    TRestoreSearcherSupplier restoreSearcherSupplier{[&lastStateStream]() {
//        return std::make_unique<CTestDataSearcher>(lastStateStream.str());
//    }};
//    spec = makeSpec(rowsTrain, frameAndDirectory, nullptr, &restoreSearcherSupplier,
//                    test::CDataFrameAnalysisSpecificationFactory::TTask::E_Train);
//    api::CDataFrameAnalyzer restoredAnalyzer{
//        std::move(spec), std::move(frameAndDirectory), outputWriterFactory};
//
//    test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
//        TLossFunctionType::E_MseRegression, fieldNames, fieldValues,
//        restoredAnalyzer, rowsTrain);
//    restoredAnalyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "", "$"});
//
//    // reset logger for next tests
//    logger.reset();
//
//    LOG_DEBUG(<< "messages = '" << logDataSPtr->str() << "'");
//
//    // Check that no error messages were logged
//    BOOST_TEST_REQUIRE(logDataSPtr->rdbuf()->in_avail() == 0);
//}

BOOST_AUTO_TEST_SUITE_END()
