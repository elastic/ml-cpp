/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDataFrame.h>

#include <maths/CBasicStatistics.h>
#include <maths/CDataFramePredictiveModel.h>
#include <maths/CSampling.h>
#include <maths/CTools.h>
#include <maths/CToolsDetail.h>
#include <maths/CTreeShapFeatureImportance.h>

#include <api/CDataFrameAnalyzer.h>
#include <api/CDataFrameTrainBoostedTreeClassifierRunner.h>
#include <api/CDataFrameTrainBoostedTreeRunner.h>
#include <api/CInferenceModelMetadata.h>

#include <test/CDataFrameAnalysisSpecificationFactory.h>
#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <memory>
#include <random>
#include <utility>

BOOST_AUTO_TEST_SUITE(CDataFrameAnalyzerFeatureImportanceTest)

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;
using TVector = maths::CDenseVector<double>;
using TStrVec = std::vector<std::string>;
using TRowItr = core::CDataFrame::TRowItr;
using TRowRef = core::CDataFrame::TRowRef;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
using TMemoryMappedMatrix = maths::CMemoryMappedDenseMatrix<double>;
using TDocumentStrPr = std::pair<rapidjson::Document, std::string>;

void setupLinearRegressionData(const TStrVec& fieldNames,
                               TStrVec& fieldValues,
                               api::CDataFrameAnalyzer& analyzer,
                               const TDoubleVec& weights,
                               const TDoubleVec& values,
                               double noiseVar = 0.0) {
    test::CRandomNumbers rng;
    auto target = [&weights, &rng, noiseVar](const TDoubleVec& regressors) {
        TDoubleVec result(1);
        rng.generateNormalSamples(0, noiseVar, 1, result);
        for (std::size_t i = 0; i < weights.size(); ++i) {
            result[0] += weights[i] * regressors[i];
        }
        return core::CStringUtils::typeToStringPrecise(result[0], core::CIEEE754::E_DoublePrecision);
    };

    for (std::size_t i = 0; i < values.size(); i += weights.size()) {
        TDoubleVec row(weights.size());
        for (std::size_t j = 0; j < weights.size(); ++j) {
            row[j] = values[i + j];
        }

        fieldValues[0] = target(row);
        for (std::size_t j = 0; j < row.size(); ++j) {
            fieldValues[j + 1] = core::CStringUtils::typeToStringPrecise(
                row[j], core::CIEEE754::E_DoublePrecision);
        }

        analyzer.handleRecord(fieldNames, fieldValues);
    }
}

void setupRegressionDataWithMissingFeatures(const TStrVec& fieldNames,
                                            TStrVec& fieldValues,
                                            api::CDataFrameAnalyzer& analyzer,
                                            std::size_t rows,
                                            std::size_t cols) {
    test::CRandomNumbers rng;
    auto target = [](const TDoubleVec& regressors) {
        double result{0.0};
        for (auto regressor : regressors) {
            result += regressor;
        }
        return core::CStringUtils::typeToStringPrecise(result, core::CIEEE754::E_DoublePrecision);
    };

    for (std::size_t i = 0; i < rows; ++i) {
        TDoubleVec regressors;
        rng.generateUniformSamples(0.0, 10.0, cols - 1, regressors);

        fieldValues[0] = target(regressors);
        for (std::size_t j = 0; j < regressors.size(); ++j) {
            if (regressors[j] <= 9.0) {
                fieldValues[j + 1] = core::CStringUtils::typeToStringPrecise(
                    regressors[j], core::CIEEE754::E_DoublePrecision);
            }
        }

        analyzer.handleRecord(fieldNames, fieldValues);
    }
}

void setupBinaryClassificationData(const TStrVec& fieldNames,
                                   TStrVec& fieldValues,
                                   api::CDataFrameAnalyzer& analyzer,
                                   const TDoubleVec& weights,
                                   const TDoubleVec& values) {
    TStrVec classes{"foo", "bar"};
    maths::CPRNG::CXorOShiro128Plus rng;
    std::uniform_real_distribution<double> u01;
    auto target = [&](const TDoubleVec& regressors) {
        double logOddsBar{0.0};
        for (std::size_t i = 0; i < weights.size(); ++i) {
            logOddsBar += weights[i] * regressors[i];
        }
        return classes[u01(rng) < maths::CTools::logisticFunction(logOddsBar)];
    };

    for (std::size_t i = 0; i < values.size(); i += weights.size()) {
        TDoubleVec row(weights.size());
        for (std::size_t j = 0; j < weights.size(); ++j) {
            row[j] = values[i + j];
        }

        fieldValues[0] = target(row);
        for (std::size_t j = 0; j < row.size(); ++j) {
            fieldValues[j + 1] = core::CStringUtils::typeToStringPrecise(
                row[j], core::CIEEE754::E_DoublePrecision);
        }

        analyzer.handleRecord(fieldNames, fieldValues);
    }
}

void setupMultiClassClassificationData(const TStrVec& fieldNames,
                                       TStrVec& fieldValues,
                                       api::CDataFrameAnalyzer& analyzer,
                                       const TDoubleVec& weights,
                                       const TDoubleVec& values) {
    TStrVec classes{"foo", "bar", "baz"};
    maths::CPRNG::CXorOShiro128Plus rng;
    std::uniform_real_distribution<double> u01;
    int numberFeatures{static_cast<int>(weights.size())};
    int numberClasses{static_cast<int>(classes.size())};
    TDoubleVec storage(numberClasses * numberFeatures);
    for (int i = 0; i < numberClasses; ++i) {
        for (int j = 0; j < numberFeatures; ++j) {
            storage[j * numberClasses + i] = static_cast<double>(i) * weights[j];
        }
    }
    auto probability = [&](const TDoubleVec& row) {
        TMemoryMappedMatrix W(&storage[0], numberClasses, numberFeatures);
        TVector x(numberFeatures);
        for (int i = 0; i < numberFeatures; ++i) {
            x(i) = row[i];
        }
        TVector result{W * x};
        maths::CTools::inplaceSoftmax(result);
        return result;
    };
    auto target = [&](const TDoubleVec& row) {
        TDoubleVec probabilities{probability(row).to<TDoubleVec>()};
        return classes[maths::CSampling::categoricalSample(rng, probabilities)];
    };

    for (std::size_t i = 0; i < values.size(); i += weights.size()) {
        TDoubleVec row(weights.size());
        for (std::size_t j = 0; j < weights.size(); ++j) {
            row[j] = values[i + j];
        }

        fieldValues[0] = target(row);
        for (std::size_t j = 0; j < row.size(); ++j) {
            fieldValues[j + 1] = core::CStringUtils::typeToStringPrecise(
                row[j], core::CIEEE754::E_DoublePrecision);
        }

        analyzer.handleRecord(fieldNames, fieldValues);
    }
}

struct SFixture {
    TDocumentStrPr runRegression(std::size_t shapValues,
                                 const TDoubleVec& weights,
                                 double noiseVar = 0.0) {
        auto outputWriterFactory = [&]() {
            return std::make_unique<core::CJsonOutputStreamWrapper>(s_Output);
        };
        test::CDataFrameAnalysisSpecificationFactory specFactory;
        api::CDataFrameAnalyzer analyzer{
            specFactory.rows(s_Rows)
                .memoryLimit(26000000)
                .predictionCategoricalFieldNames({"c1"})
                .predictionAlpha(s_Alpha)
                .predictionLambda(s_Lambda)
                .predictionGamma(s_Gamma)
                .predictionSoftTreeDepthLimit(s_SoftTreeDepthLimit)
                .predictionSoftTreeDepthTolerance(s_SoftTreeDepthTolerance)
                .predictionEta(s_Eta)
                .predictionMaximumNumberTrees(s_MaximumNumberTrees)
                .predictionFeatureBagFraction(s_FeatureBagFraction)
                .predictionNumberTopShapValues(shapValues)
                .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::regression(), "target"),
            outputWriterFactory};
        TStrVec fieldNames{"target", "c1", "c2", "c3", "c4", ".", "."};
        TStrVec fieldValues{"", "", "", "", "", "0", ""};
        test::CRandomNumbers rng;

        TDoubleVec values;
        rng.generateUniformSamples(-10.0, 10.0, weights.size() * s_Rows, values);

        // make the first column categorical
        for (auto it = values.begin(); it < values.end(); it += 4) {
            *it = (*it < 0) ? -10.0 : 10.0;
        }

        setupLinearRegressionData(fieldNames, fieldValues, analyzer, weights, values, noiseVar);

        analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

        LOG_DEBUG(<< "estimated memory usage = "
                  << core::CProgramCounters::counter(counter_t::E_DFTPMEstimatedPeakMemoryUsage));
        LOG_DEBUG(<< "peak memory = "
                  << core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage));
        LOG_DEBUG(<< "time to train = " << core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain)
                  << "ms");

        BOOST_TEST_REQUIRE(
            core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage) <
            core::CProgramCounters::counter(counter_t::E_DFTPMEstimatedPeakMemoryUsage));
        rapidjson::Document results;
        rapidjson::ParseResult ok(results.Parse(s_Output.str()));
        BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);
        return std::make_pair(std::move(results), s_Output.str());
    }

    TDocumentStrPr runBinaryClassification(std::size_t shapValues, const TDoubleVec& weights) {
        auto outputWriterFactory = [&]() {
            return std::make_unique<core::CJsonOutputStreamWrapper>(s_Output);
        };
        test::CDataFrameAnalysisSpecificationFactory specFactory;
        api::CDataFrameAnalyzer analyzer{
            specFactory.rows(s_Rows)
                .memoryLimit(26000000)
                .predictionCategoricalFieldNames({"target"})
                .predictionAlpha(s_Alpha)
                .predictionLambda(s_Lambda)
                .predictionGamma(s_Gamma)
                .predictionSoftTreeDepthLimit(s_SoftTreeDepthLimit)
                .predictionSoftTreeDepthTolerance(s_SoftTreeDepthTolerance)
                .predictionEta(s_Eta)
                .predictionMaximumNumberTrees(s_MaximumNumberTrees)
                .predictionFeatureBagFraction(s_FeatureBagFraction)
                .predictionNumberTopShapValues(shapValues)
                .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::classification(), "target"),
            outputWriterFactory};
        TStrVec fieldNames{"target", "c1", "c2", "c3", "c4", ".", "."};
        TStrVec fieldValues{"", "", "", "", "", "0", ""};
        test::CRandomNumbers rng;

        TDoubleVec values;
        rng.generateUniformSamples(-10.0, 10.0, weights.size() * s_Rows, values);

        setupBinaryClassificationData(fieldNames, fieldValues, analyzer, weights, values);

        analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

        LOG_DEBUG(<< "estimated memory usage = "
                  << core::CProgramCounters::counter(counter_t::E_DFTPMEstimatedPeakMemoryUsage));
        LOG_DEBUG(<< "peak memory = "
                  << core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage));
        LOG_DEBUG(<< "time to train = " << core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain)
                  << "ms");

        BOOST_TEST_REQUIRE(
            core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage) <
            core::CProgramCounters::counter(counter_t::E_DFTPMEstimatedPeakMemoryUsage));

        rapidjson::Document results;
        rapidjson::ParseResult ok(results.Parse(s_Output.str()));
        BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);
        return std::make_pair(std::move(results), s_Output.str());
    }

    TDocumentStrPr runMultiClassClassification(std::size_t shapValues,
                                               const TDoubleVec& weights) {
        auto outputWriterFactory = [&]() {
            return std::make_unique<core::CJsonOutputStreamWrapper>(s_Output);
        };
        test::CDataFrameAnalysisSpecificationFactory specFactory;
        api::CDataFrameAnalyzer analyzer{
            specFactory.rows(s_Rows)
                .memoryLimit(26000000)
                .predictionCategoricalFieldNames({"target"})
                .predictionAlpha(s_Alpha)
                .predictionLambda(s_Lambda)
                .predictionGamma(s_Gamma)
                .predictionSoftTreeDepthLimit(s_SoftTreeDepthLimit)
                .predictionSoftTreeDepthTolerance(s_SoftTreeDepthTolerance)
                .predictionEta(s_Eta)
                .predictionMaximumNumberTrees(s_MaximumNumberTrees)
                .predictionFeatureBagFraction(s_FeatureBagFraction)
                .predictionNumberTopShapValues(shapValues)
                .numberClasses(3)
                .numberTopClasses(3)
                .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::classification(), "target"),
            outputWriterFactory};
        TStrVec fieldNames{"target", "c1", "c2", "c3", "c4", ".", "."};
        TStrVec fieldValues{"", "", "", "", "", "0", ""};
        test::CRandomNumbers rng;

        TDoubleVec values;
        rng.generateUniformSamples(-10.0, 10.0, weights.size() * s_Rows, values);

        setupMultiClassClassificationData(fieldNames, fieldValues, analyzer, weights, values);

        analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

        LOG_DEBUG(<< "estimated memory usage = "
                  << core::CProgramCounters::counter(counter_t::E_DFTPMEstimatedPeakMemoryUsage));
        LOG_DEBUG(<< "peak memory = "
                  << core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage));
        LOG_DEBUG(<< "time to train = " << core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain)
                  << "ms");

        BOOST_TEST_REQUIRE(
            core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage) <
            core::CProgramCounters::counter(counter_t::E_DFTPMEstimatedPeakMemoryUsage));

        rapidjson::Document results;
        rapidjson::ParseResult ok(results.Parse(s_Output.str()));
        BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);
        return std::make_pair(std::move(results), s_Output.str());
    }

    rapidjson::Document runRegressionWithMissingFeatures(std::size_t shapValues) {
        auto outputWriterFactory = [&]() {
            return std::make_unique<core::CJsonOutputStreamWrapper>(s_Output);
        };
        test::CDataFrameAnalysisSpecificationFactory specFactory;
        api::CDataFrameAnalyzer analyzer{
            specFactory.rows(s_Rows)
                .memoryLimit(26000000)
                .predictionAlpha(s_Alpha)
                .predictionLambda(s_Lambda)
                .predictionGamma(s_Gamma)
                .predictionSoftTreeDepthLimit(s_SoftTreeDepthLimit)
                .predictionSoftTreeDepthTolerance(s_SoftTreeDepthTolerance)
                .predictionEta(s_Eta)
                .predictionMaximumNumberTrees(s_MaximumNumberTrees)
                .predictionFeatureBagFraction(s_FeatureBagFraction)
                .predictionNumberTopShapValues(shapValues)
                .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::regression(), "target"),
            outputWriterFactory};
        TStrVec fieldNames{"target", "c1", "c2", "c3", "c4", ".", "."};
        TStrVec fieldValues{"", "", "", "", "", "0", ""};

        setupRegressionDataWithMissingFeatures(fieldNames, fieldValues, analyzer, s_Rows, 5);

        analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

        LOG_DEBUG(<< "estimated memory usage = "
                  << core::CProgramCounters::counter(counter_t::E_DFTPMEstimatedPeakMemoryUsage));
        LOG_DEBUG(<< "peak memory = "
                  << core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage));
        LOG_DEBUG(<< "time to train = " << core::CProgramCounters::counter(counter_t::E_DFTPMTimeToTrain)
                  << "ms");

        BOOST_TEST_REQUIRE(
            core::CProgramCounters::counter(counter_t::E_DFTPMPeakMemoryUsage) <
            core::CProgramCounters::counter(counter_t::E_DFTPMEstimatedPeakMemoryUsage));

        rapidjson::Document results;
        rapidjson::ParseResult ok(results.Parse(s_Output.str()));
        BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);
        return results;
    }

    double s_Alpha{2.0};
    double s_Lambda{1.0};
    double s_Gamma{10.0};
    double s_SoftTreeDepthLimit{5.0};
    double s_SoftTreeDepthTolerance{0.1};
    double s_Eta{0.9};
    std::size_t s_MaximumNumberTrees{1};
    double s_FeatureBagFraction{1.0};

    int s_Rows{2000};
    std::stringstream s_Output;
};

template<typename RESULTS>
double readShapValue(const RESULTS& results, std::string shapField) {
    if (results["row_results"]["results"]["ml"].HasMember(
            api::CDataFrameTrainBoostedTreeRunner::FEATURE_IMPORTANCE_FIELD_NAME)) {
        for (const auto& shapResult :
             results["row_results"]["results"]["ml"][api::CDataFrameTrainBoostedTreeRunner::FEATURE_IMPORTANCE_FIELD_NAME]
                 .GetArray()) {
            if (shapResult[api::CDataFrameTrainBoostedTreeRunner::FEATURE_NAME_FIELD_NAME]
                    .GetString() == shapField) {
                return shapResult[api::CDataFrameTrainBoostedTreeRunner::IMPORTANCE_FIELD_NAME]
                    .GetDouble();
            }
        }
    }
    return 0.0;
}

template<typename RESULTS>
double readClassProbability(const RESULTS& results, std::string className) {
    if (results["row_results"]["results"]["ml"].HasMember("top_classes")) {
        for (const auto& classResult :
             results["row_results"]["results"]["ml"]["top_classes"].GetArray()) {
            if (classResult["class_name"].GetString() == className) {
                return classResult["class_probability"].GetDouble();
            }
        }
    }
    LOG_DEBUG(<< "Class probability not found");
    return 0.0;
}

template<typename RESULTS>
double readShapValue(const RESULTS& results, std::string shapField, std::string className) {
    if (results["row_results"]["results"]["ml"].HasMember(
            api::CDataFrameTrainBoostedTreeRunner::FEATURE_IMPORTANCE_FIELD_NAME)) {
        for (const auto& shapResult :
             results["row_results"]["results"]["ml"][api::CDataFrameTrainBoostedTreeRunner::FEATURE_IMPORTANCE_FIELD_NAME]
                 .GetArray()) {
            if (shapResult[api::CDataFrameTrainBoostedTreeRunner::FEATURE_NAME_FIELD_NAME]
                    .GetString() == shapField) {
                for (const auto& item :
                     shapResult[api::CDataFrameTrainBoostedTreeClassifierRunner::CLASSES_FIELD_NAME]
                         .GetArray()) {
                    if (item[api::CDataFrameTrainBoostedTreeClassifierRunner::CLASS_NAME_FIELD_NAME]
                            .GetString() == className) {
                        return item[api::CDataFrameTrainBoostedTreeRunner::IMPORTANCE_FIELD_NAME]
                            .GetDouble();
                    }
                }
            }
        }
    }
    return 0.0;
}

template<typename RESULTS>
double readTotalShapValue(const RESULTS& results, std::string shapField) {
    using TModelMetadata = api::CInferenceModelMetadata;
    if (results[TModelMetadata::JSON_MODEL_METADATA_TAG].HasMember(
            TModelMetadata::JSON_TOTAL_FEATURE_IMPORTANCE_TAG)) {
        for (const auto& shapResult :
             results[TModelMetadata::JSON_MODEL_METADATA_TAG][TModelMetadata::JSON_TOTAL_FEATURE_IMPORTANCE_TAG]
                 .GetArray()) {
            if (shapResult[TModelMetadata::JSON_FEATURE_NAME_TAG].GetString() == shapField) {
                return shapResult[TModelMetadata::JSON_IMPORTANCE_TAG][TModelMetadata::JSON_MEAN_MAGNITUDE_TAG]
                    .GetDouble();
            }
        }
    }
    return 0.0;
}

template<typename RESULTS>
double readTotalShapValue(const RESULTS& results, std::string shapField, std::string className) {
    using TModelMetadata = api::CInferenceModelMetadata;
    if (results[TModelMetadata::JSON_MODEL_METADATA_TAG].HasMember(
            TModelMetadata::JSON_TOTAL_FEATURE_IMPORTANCE_TAG)) {
        for (const auto& shapResult :
             results[TModelMetadata::JSON_MODEL_METADATA_TAG][TModelMetadata::JSON_TOTAL_FEATURE_IMPORTANCE_TAG]
                 .GetArray()) {
            if (shapResult[TModelMetadata::JSON_FEATURE_NAME_TAG].GetString() == shapField) {
                for (const auto& item :
                     shapResult[TModelMetadata::JSON_CLASSES_TAG].GetArray()) {
                    if (item[TModelMetadata::JSON_CLASS_NAME_TAG].GetString() == className) {
                        return item[TModelMetadata::JSON_IMPORTANCE_TAG][TModelMetadata::JSON_MEAN_MAGNITUDE_TAG]
                            .GetDouble();
                    }
                }
            }
        }
    }
    return 0.0;
}

template<typename RESULTS>
double readBaselineValue(const RESULTS& results) {
    using TModelMetadata = api::CInferenceModelMetadata;
    for (const auto& result : results.GetArray()) {
        if (result.HasMember(TModelMetadata::JSON_MODEL_METADATA_TAG) &&
            result[TModelMetadata::JSON_MODEL_METADATA_TAG].HasMember(
                TModelMetadata::JSON_FEATURE_IMPORTANCE_BASELINE_TAG)) {
            return result[TModelMetadata::JSON_MODEL_METADATA_TAG][TModelMetadata::JSON_FEATURE_IMPORTANCE_BASELINE_TAG]
                         [TModelMetadata::JSON_BASELINE_TAG]
                             .GetDouble();
        }
    }
    return 0.0;
}

template<typename RESULTS>
double readBaselineValue(const RESULTS& results, std::string className) {
    using TModelMetadata = api::CInferenceModelMetadata;
    for (const auto& result : results.GetArray()) {
        if (result.HasMember(TModelMetadata::JSON_MODEL_METADATA_TAG) &&
            result[TModelMetadata::JSON_MODEL_METADATA_TAG].HasMember(
                TModelMetadata::JSON_FEATURE_IMPORTANCE_BASELINE_TAG)) {
            for (const auto& item :
                 result[TModelMetadata::JSON_MODEL_METADATA_TAG][TModelMetadata::JSON_FEATURE_IMPORTANCE_BASELINE_TAG]
                       [TModelMetadata::JSON_CLASSES_TAG]
                           .GetArray()) {
                if (item[TModelMetadata::JSON_CLASS_NAME_TAG].GetString() == className) {
                    return item[TModelMetadata::JSON_BASELINE_TAG].GetDouble();
                }
            }
        }
    }
    return 0.0;
}
}

BOOST_FIXTURE_TEST_CASE(testRegressionFeatureImportanceAllShap, SFixture) {
    // Test that feature importance statistically correctly recognize the impact of regressors
    // in a linear model. In particular, that the ordering is as expected and for IID features
    // the significance is proportional to the multiplier. Also make sure that the SHAP values
    // are indeed a local approximation of the prediction.

    std::size_t topShapValues{5}; //Note, number of requested shap values is larger than the number of regressors
    TDoubleVec weights{50, 150, 50, -50};
    auto resultsPair{runRegression(topShapValues, weights)};
    auto results{std::move(resultsPair.first)};

    TMeanAccumulator baselineAccumulator;
    TMeanAccumulator c1TotalShapExpected;
    TMeanAccumulator c2TotalShapExpected;
    TMeanAccumulator c3TotalShapExpected;
    TMeanAccumulator c4TotalShapExpected;
    double c1Sum{0.0}, c2Sum{0.0}, c3Sum{0.0}, c4Sum{0.0};
    double c1TotalShapActual{0.0}, c2TotalShapActual{0.0},
        c3TotalShapActual{0.0}, c4TotalShapActual{0.0};
    bool hasTotalFeatureImportance{false};
    double baseline{readBaselineValue(results)};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            double c1{readShapValue(result, "c1")};
            double c2{readShapValue(result, "c2")};
            double c3{readShapValue(result, "c3")};
            double c4{readShapValue(result, "c4")};
            double prediction{
                result["row_results"]["results"]["ml"]["target_prediction"].GetDouble()};
            // make sure that the local approximation differs from a prediction by a numeric error
            BOOST_REQUIRE_SMALL(prediction - (baseline + c1 + c2 + c3 + c4), 1e-3);
            c1Sum += std::fabs(c1);
            c2Sum += std::fabs(c2);
            c3Sum += std::fabs(c3);
            c4Sum += std::fabs(c4);
            c1TotalShapExpected.add(std::fabs(c1));
            c2TotalShapExpected.add(std::fabs(c2));
            c3TotalShapExpected.add(std::fabs(c3));
            c4TotalShapExpected.add(std::fabs(c4));
            // assert that no SHAP value for the dependent variable is returned
            BOOST_REQUIRE_EQUAL(readShapValue(result, "target"), 0.0);
        } else if (result.HasMember("model_metadata")) {
            if (result["model_metadata"].HasMember("total_feature_importance")) {
                hasTotalFeatureImportance = true;
                c1TotalShapActual = readTotalShapValue(result, "c1");
                c2TotalShapActual = readTotalShapValue(result, "c2");
                c3TotalShapActual = readTotalShapValue(result, "c3");
                c4TotalShapActual = readTotalShapValue(result, "c4");
            }
            // TODO check that the total feature importance is calculated correctly
        }
    }

    // since target is generated using the linear model
    // 50 c1 + 150 c2 + 50 c3 - 50 c4, with c1 categorical {-10,10}
    // we expect c2 > c1 > c3 \approx c4
    BOOST_TEST_REQUIRE(c2Sum > c1Sum);
    // since c1 is categorical -10 or 10, it's influence is generally higher than that of c3 and c4 which are sampled
    // randomly on [-10, 10].
    BOOST_TEST_REQUIRE(c1Sum > c3Sum);
    BOOST_TEST_REQUIRE(c1Sum > c4Sum);
    BOOST_REQUIRE_CLOSE(weights[1] / weights[2], c2Sum / c3Sum, 10.0); // ratio within 10% of ratio of coefficients
    BOOST_REQUIRE_CLOSE(c3Sum, c4Sum, 5.0); // c3 and c4 within 5% of each other
    BOOST_TEST_REQUIRE(hasTotalFeatureImportance);

    if (c1TotalShapActual == 0 || c2TotalShapActual == 0 ||
        c3TotalShapActual == 0 || c4TotalShapActual == 0) {
        LOG_INFO(<< "Incorrect results, missing total shap values: "
                 << resultsPair.second);
    }
    BOOST_REQUIRE_CLOSE(c1TotalShapActual,
                        maths::CBasicStatistics::mean(c1TotalShapExpected), 1.0);
    BOOST_REQUIRE_CLOSE(c2TotalShapActual,
                        maths::CBasicStatistics::mean(c2TotalShapExpected), 1.0);
    BOOST_REQUIRE_CLOSE(c3TotalShapActual,
                        maths::CBasicStatistics::mean(c3TotalShapExpected), 1.0);
    BOOST_REQUIRE_CLOSE(c4TotalShapActual,
                        maths::CBasicStatistics::mean(c4TotalShapExpected), 1.0);
}

BOOST_FIXTURE_TEST_CASE(testRegressionFeatureImportanceNoImportance, SFixture) {
    // Test that feature importance calculates low SHAP values if regressors have no weight.
    // We also add high noise variance.
    std::size_t topShapValues{4};
    auto resultsPair{runRegression(topShapValues, {10.0, 0.0, 0.0, 0.0}, 10.0)};
    auto results{std::move(resultsPair.first)};

    TMeanAccumulator cNoImportanceMean;
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            double c1{readShapValue(result, "c1")};
            double prediction{
                result["row_results"]["results"]["ml"]["target_prediction"].GetDouble()};
            // c1 explains 94% of the prediction value, i.e. the difference from the prediction is less than 6%.
            BOOST_REQUIRE_CLOSE(c1, prediction, 6.0);
            for (const auto& feature : {"c2", "c3", "c4"}) {
                double c = readShapValue(result, feature);
                BOOST_REQUIRE_SMALL(c, 3.0);
                cNoImportanceMean.add(std::fabs(c));
            }
        }
    }

    BOOST_REQUIRE_SMALL(maths::CBasicStatistics::mean(cNoImportanceMean), 0.1);
}

BOOST_FIXTURE_TEST_CASE(testClassificationFeatureImportanceAllShap, SFixture) {
    // Test that feature importance works correctly for classification. In particular, test that
    // feature importance statistically correctly recognizes the impact of regressors if the
    // log-odds of the classes are generated by a linear model. Also make sure that the SHAP
    // values are indeed a local approximation of the predicted log-odds.

    std::size_t topShapValues{4};
    auto resultsPair{runBinaryClassification(topShapValues, {0.5, -0.7, 0.2, -0.2})};
    auto results{std::move(resultsPair.first)};
    TMeanAccumulator c1TotalShapExpected;
    TMeanAccumulator c2TotalShapExpected;
    TMeanAccumulator c3TotalShapExpected;
    TMeanAccumulator c4TotalShapExpected;
    double c1Sum{0.0}, c2Sum{0.0}, c3Sum{0.0}, c4Sum{0.0};
    double c1TotalShapActual[2], c2TotalShapActual[2], c3TotalShapActual[2],
        c4TotalShapActual[2];
    bool hasTotalFeatureImportance{false};
    double baselineFoo{readBaselineValue(results, "foo")};
    double baselineBar{readBaselineValue(results, "bar")};
    BOOST_TEST_REQUIRE(baselineFoo == -baselineBar);
    TStrVec classes{"foo", "bar"};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            std::string targetPrediction{
                result["row_results"]["results"]["ml"]["target_prediction"].GetString()};
            double predictionProbability{
                result["row_results"]["results"]["ml"]["prediction_probability"].GetDouble()};
            for (const auto& className : classes) {
                double c1{readShapValue(result, "c1", className)};
                double c2{readShapValue(result, "c2", className)};
                double c3{readShapValue(result, "c3", className)};
                double c4{readShapValue(result, "c4", className)};

                double logOdds{std::log(predictionProbability /
                                        (1.0 - predictionProbability + 1e-10))};
                logOdds = (className == targetPrediction) ? logOdds : -logOdds;
                double cSum{c1 + c2 + c3 + c4};
                double baseline{(className == "bar") ? baselineBar : baselineFoo};
                BOOST_REQUIRE_SMALL(logOdds - (baseline + cSum), 1e-3);
                if (className == targetPrediction) {
                    c1Sum += std::fabs(c1);
                    c2Sum += std::fabs(c2);
                    c3Sum += std::fabs(c3);
                    c4Sum += std::fabs(c4);
                    c1TotalShapExpected.add(std::fabs(c1));
                    c2TotalShapExpected.add(std::fabs(c2));
                    c3TotalShapExpected.add(std::fabs(c3));
                    c4TotalShapExpected.add(std::fabs(c4));
                }
            }
        } else if (result.HasMember("model_metadata")) {
            if (result["model_metadata"].HasMember("total_feature_importance")) {
                hasTotalFeatureImportance = true;
            }
            for (std::size_t i = 0; i < classes.size(); ++i) {
                c1TotalShapActual[i] = readTotalShapValue(result, "c1", classes[i]);
                c2TotalShapActual[i] = readTotalShapValue(result, "c2", classes[i]);
                c3TotalShapActual[i] = readTotalShapValue(result, "c3", classes[i]);
                c4TotalShapActual[i] = readTotalShapValue(result, "c4", classes[i]);
            }
        }
    }

    // since the target using a linear model
    // 0.5 c1 + 0.7 c2 + 0.25 c3 - 0.25 c4
    // to generate the log odds we expect c2 > c1 > c3 \approx c4
    BOOST_TEST_REQUIRE(c2Sum > c1Sum);
    BOOST_TEST_REQUIRE(c1Sum > c3Sum);
    BOOST_TEST_REQUIRE(c1Sum > c4Sum);
    BOOST_REQUIRE_CLOSE(c3Sum, c4Sum, 40.0); // c3 and c4 within 40% of each other
    BOOST_TEST_REQUIRE(hasTotalFeatureImportance);
    for (std::size_t i = 0; i < classes.size(); ++i) {
        if (c1TotalShapActual[i] == 0 || c2TotalShapActual[i] == 0 ||
            c3TotalShapActual[i] == 0 || c4TotalShapActual[i] == 0) {
            LOG_INFO(<< "Incorrect results, missing total shap values: "
                     << resultsPair.second);
        }
        BOOST_REQUIRE_CLOSE(c1TotalShapActual[i],
                            maths::CBasicStatistics::mean(c1TotalShapExpected), 1.0);
        BOOST_REQUIRE_CLOSE(c2TotalShapActual[i],
                            maths::CBasicStatistics::mean(c2TotalShapExpected), 1.0);
        BOOST_REQUIRE_CLOSE(c3TotalShapActual[i],
                            maths::CBasicStatistics::mean(c3TotalShapExpected), 1.0);
        BOOST_REQUIRE_CLOSE(c4TotalShapActual[i],
                            maths::CBasicStatistics::mean(c4TotalShapExpected), 1.0);
    }
}

BOOST_FIXTURE_TEST_CASE(testMultiClassClassificationFeatureImportanceAllShap, SFixture) {

    std::size_t topShapValues{4};
    auto resultsPair{runMultiClassClassification(topShapValues, {0.5, -0.7, 0.2, -0.2})};
    auto results{std::move(resultsPair.first)};
    TMeanAccumulatorVec c1TotalShapExpected(3);
    TMeanAccumulatorVec c2TotalShapExpected(3);
    TMeanAccumulatorVec c3TotalShapExpected(3);
    TMeanAccumulatorVec c4TotalShapExpected(3);
    double c1TotalShapActual[3];
    double c2TotalShapActual[3];
    double c3TotalShapActual[3];
    double c4TotalShapActual[3];
    bool hasTotalFeatureImportance{false};
    TStrVec classes{"foo", "bar", "baz"};
    TDoubleVec baselines;
    baselines.reserve(3);
    // get baselines
    for (const auto& className : classes) {
        baselines.push_back(readBaselineValue(results, className));
    }
    double localApproximations[3];
    double classProbabilities[3];
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            double c1{0.0}, c2{0.0}, c3{0.0}, c4{0.0};
            double denominator{0.0};
            for (std::size_t i = 0; i < classes.size(); ++i) {
                // class shap values should sum(abs()) to the overall feature importance
                double c1ClassName{readShapValue(result, "c1", classes[i])};
                c1 += std::abs(c1ClassName);
                c1TotalShapExpected[i].add(std::abs(c1ClassName));

                double c2ClassName{readShapValue(result, "c2", classes[i])};
                c2 += std::abs(c2ClassName);
                c2TotalShapExpected[i].add(std::abs(c2ClassName));

                double c3ClassName{readShapValue(result, "c3", classes[i])};
                c3 += std::abs(c3ClassName);
                c3TotalShapExpected[i].add(std::abs(c3ClassName));

                double c4ClassName{readShapValue(result, "c4", classes[i])};
                c4 += std::abs(c4ClassName);
                c4TotalShapExpected[i].add(std::abs(c4ClassName));

                double classProbability{readClassProbability(result, classes[i])};
                double localApproximation{baselines[i] + c1ClassName +
                                          c2ClassName + c3ClassName + c4ClassName};
                localApproximations[i] = localApproximation;
                classProbabilities[i] = classProbability;
                denominator += std::exp(localApproximation);
            }

            // Test that sum of feature importances is a local approximations of
            // prediction probabilities for all classes
            for (std::size_t i = 0; i < classes.size(); ++i) {
                BOOST_REQUIRE_CLOSE(classProbabilities[i],
                                    std::exp(localApproximations[i]) / denominator, 1.0);
            }

            // We should have at least one feature that is important
            BOOST_TEST_REQUIRE((c1 > 0.0 || c2 > 0.0 || c3 > 0.0 || c4 > 0.0));

        } else if (result.HasMember("model_metadata")) {
            if (result["model_metadata"].HasMember("total_feature_importance")) {
                hasTotalFeatureImportance = true;
            }
            for (std::size_t i = 0; i < classes.size(); ++i) {
                c1TotalShapActual[i] = readTotalShapValue(result, "c1", classes[i]);
                c2TotalShapActual[i] = readTotalShapValue(result, "c2", classes[i]);
                c3TotalShapActual[i] = readTotalShapValue(result, "c3", classes[i]);
                c4TotalShapActual[i] = readTotalShapValue(result, "c4", classes[i]);
            }
        }
    }
    BOOST_TEST_REQUIRE(hasTotalFeatureImportance);

    for (std::size_t i = 0; i < classes.size(); ++i) {
        if (c1TotalShapActual[i] == 0 || c2TotalShapActual[i] == 0 ||
            c3TotalShapActual[i] == 0 || c4TotalShapActual[i] == 0) {
            LOG_INFO(<< "Incorrect results, missing total shap values: "
                     << resultsPair.second);
        }
        BOOST_REQUIRE_CLOSE(c1TotalShapActual[i],
                            maths::CBasicStatistics::mean(c1TotalShapExpected[i]), 1.0);
        BOOST_REQUIRE_CLOSE(c2TotalShapActual[i],
                            maths::CBasicStatistics::mean(c2TotalShapExpected[i]), 1.0);
        BOOST_REQUIRE_CLOSE(c3TotalShapActual[i],
                            maths::CBasicStatistics::mean(c3TotalShapExpected[i]), 1.0);
        BOOST_REQUIRE_CLOSE(c4TotalShapActual[i],
                            maths::CBasicStatistics::mean(c4TotalShapExpected[i]), 1.0);
    }
}

BOOST_FIXTURE_TEST_CASE(testRegressionFeatureImportanceNoShap, SFixture) {
    // Test that if topShapValue is set to 0, no feature importance values are returned.
    std::size_t topShapValues{0};
    auto resultsPair{runRegression(topShapValues, {50.0, 150.0, 50.0, -50.0})};
    auto results{std::move(resultsPair.first)};

    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            BOOST_TEST_REQUIRE(result["row_results"]["results"]["ml"].HasMember(
                                   api::CDataFrameTrainBoostedTreeRunner::FEATURE_IMPORTANCE_FIELD_NAME) ==
                               false);
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testMissingFeatures, SFixture) {
    // Test that feature importance behaves correctly when some features are missing:
    // We randomly omit 10% of all data in a simple additive model target=c1+c2+c3+c4. Hence,
    // calculated feature importances should be very similar and the bias should be close
    // to 0.
    std::size_t topShapValues{4};
    auto results = runRegressionWithMissingFeatures(topShapValues);

    TMeanVarAccumulator bias;
    double c1Sum{0.0}, c2Sum{0.0}, c3Sum{0.0}, c4Sum{0.0};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            double c1{readShapValue(result, "c1")};
            double c2{readShapValue(result, "c2")};
            double c3{readShapValue(result, "c3")};
            double c4{readShapValue(result, "c4")};
            double prediction{
                result["row_results"]["results"]["ml"]["target_prediction"].GetDouble()};
            // the difference between the prediction and the sum of all SHAP values constitutes bias
            bias.add(prediction - (c1 + c2 + c3 + c4));
            c1Sum += std::fabs(c1);
            c2Sum += std::fabs(c2);
            c3Sum += std::fabs(c3);
            c4Sum += std::fabs(c4);
        }
    }

    BOOST_REQUIRE_CLOSE(c1Sum, c2Sum, 15.0); // c1 and c2 within 15% of each other
    BOOST_REQUIRE_CLOSE(c1Sum, c3Sum, 15.0); // c1 and c3 within 15% of each other
    BOOST_REQUIRE_CLOSE(c1Sum, c4Sum, 15.0); // c1 and c4 within 15% of each other
    // make sure the local approximation differs from the prediction always by the same bias (up to a numeric error)
    BOOST_REQUIRE_SMALL(maths::CBasicStatistics::variance(bias), 1e-6);
}

BOOST_AUTO_TEST_SUITE_END()
