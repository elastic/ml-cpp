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
    rapidjson::Document runRegression(std::size_t shapValues,
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
        return results;
    }

    rapidjson::Document runBinaryClassification(std::size_t shapValues,
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
        return results;
    }

    rapidjson::Document runMultiClassClassification(std::size_t shapValues,
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
        return results;
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
}

BOOST_FIXTURE_TEST_CASE(testRegressionFeatureImportanceAllShap, SFixture) {
    // Test that feature importance statistically correctly recognize the impact of regressors
    // in a linear model. In particular, that the ordering is as expected and for IID features
    // the significance is proportional to the multiplier. Also make sure that the SHAP values
    // are indeed a local approximation of the prediction up to the constant bias term.

    std::size_t topShapValues{5}; //Note, number of requested shap values is larger than the number of regressors
    TDoubleVec weights{50, 150, 50, -50};
    auto results{runRegression(topShapValues, weights)};

    TMeanVarAccumulator bias;
    TMeanAccumulator c1TotalShapExpected;
    TMeanAccumulator c2TotalShapExpected;
    TMeanAccumulator c3TotalShapExpected;
    TMeanAccumulator c4TotalShapExpected;
    double c1Sum{0.0}, c2Sum{0.0}, c3Sum{0.0}, c4Sum{0.0};
    double c1TotalShapActual{0.0}, c2TotalShapActual{0.0},
        c3TotalShapActual{0.0}, c4TotalShapActual{0.0};
    bool hasTotalFeatureImportance{false};
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
    // make sure the local approximation differs from the prediction always by the same bias (up to a numeric error)
    BOOST_REQUIRE_SMALL(maths::CBasicStatistics::variance(bias), 1e-6);
    BOOST_TEST_REQUIRE(hasTotalFeatureImportance);
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
    auto results = runRegression(topShapValues, {10.0, 0.0, 0.0, 0.0}, 10.0);

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
    // values are indeed a local approximation of the predicted log-odds up to the constant
    // bias term.

    std::size_t topShapValues{4};
    TMeanVarAccumulator bias;
    auto results{runBinaryClassification(topShapValues, {0.5, -0.7, 0.2, -0.2})};
    TMeanAccumulator c1TotalShapExpected;
    TMeanAccumulator c2TotalShapExpected;
    TMeanAccumulator c3TotalShapExpected;
    TMeanAccumulator c4TotalShapExpected;
    double c1Sum{0.0}, c2Sum{0.0}, c3Sum{0.0}, c4Sum{0.0};
    double c1FooTotalShapActual{0.0}, c2FooTotalShapActual{0.0},
        c3FooTotalShapActual{0.0}, c4FooTotalShapActual{0.0};
    double c1BarTotalShapActual{0.0}, c2BarTotalShapActual{0.0},
        c3BarTotalShapActual{0.0}, c4BarTotalShapActual{0.0};
    bool hasTotalFeatureImportance{false};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            std::string targetPrediction{
                result["row_results"]["results"]["ml"]["target_prediction"].GetString()};
            double c1{readShapValue(result, "c1", targetPrediction)};
            double c2{readShapValue(result, "c2", targetPrediction)};
            double c3{readShapValue(result, "c3", targetPrediction)};
            double c4{readShapValue(result, "c4", targetPrediction)};
            double predictionProbability{
                result["row_results"]["results"]["ml"]["prediction_probability"].GetDouble()};

            double logOdds{0.0};
            if (targetPrediction == "bar") {
                logOdds = std::log(predictionProbability /
                                   (1.0 - predictionProbability + 1e-10));
            } else if (targetPrediction == "foo") {
                logOdds = std::log((1.0 - predictionProbability) /
                                   (predictionProbability + 1e-10));
            } else {
                BOOST_TEST_FAIL("Unknown predicted class " + targetPrediction);
            }
            // the difference between the prediction and the sum of all SHAP values constitutes bias
            bias.add(logOdds - (c1 + c2 + c3 + c4));
            c1Sum += std::fabs(c1);
            c2Sum += std::fabs(c2);
            c3Sum += std::fabs(c3);
            c4Sum += std::fabs(c4);
            c1TotalShapExpected.add(std::fabs(c1));
            c2TotalShapExpected.add(std::fabs(c2));
            c3TotalShapExpected.add(std::fabs(c3));
            c4TotalShapExpected.add(std::fabs(c4));
        } else if (result.HasMember("model_metadata")) {
            if (result["model_metadata"].HasMember("total_feature_importance")) {
                hasTotalFeatureImportance = true;
            }
            c1FooTotalShapActual = readTotalShapValue(result, "c1", "foo");
            c2FooTotalShapActual = readTotalShapValue(result, "c2", "foo");
            c3FooTotalShapActual = readTotalShapValue(result, "c3", "foo");
            c4FooTotalShapActual = readTotalShapValue(result, "c4", "foo");
            c1BarTotalShapActual = readTotalShapValue(result, "c1", "bar");
            c2BarTotalShapActual = readTotalShapValue(result, "c2", "bar");
            c3BarTotalShapActual = readTotalShapValue(result, "c3", "bar");
            c4BarTotalShapActual = readTotalShapValue(result, "c4", "bar");
        }
    }

    // since the target using a linear model
    // 0.5 c1 + 0.7 c2 + 0.25 c3 - 0.25 c4
    // to generate the log odds we expect c2 > c1 > c3 \approx c4
    BOOST_TEST_REQUIRE(c2Sum > c1Sum);
    BOOST_TEST_REQUIRE(c1Sum > c3Sum);
    BOOST_TEST_REQUIRE(c1Sum > c4Sum);
    BOOST_REQUIRE_CLOSE(c3Sum, c4Sum, 40.0); // c3 and c4 within 40% of each other
    // make sure the local approximation differs from the prediction always by the same bias (up to a numeric error)
    BOOST_REQUIRE_SMALL(maths::CBasicStatistics::variance(bias), 1e-6);
    BOOST_TEST_REQUIRE(hasTotalFeatureImportance);
    BOOST_REQUIRE_CLOSE(c1FooTotalShapActual,
                        maths::CBasicStatistics::mean(c1TotalShapExpected), 1.0);
    BOOST_REQUIRE_CLOSE(c2FooTotalShapActual,
                        maths::CBasicStatistics::mean(c2TotalShapExpected), 1.0);
    BOOST_REQUIRE_CLOSE(c3FooTotalShapActual,
                        maths::CBasicStatistics::mean(c3TotalShapExpected), 1.0);
    BOOST_REQUIRE_CLOSE(c4FooTotalShapActual,
                        maths::CBasicStatistics::mean(c4TotalShapExpected), 1.0);
    BOOST_REQUIRE_CLOSE(c1BarTotalShapActual,
                        maths::CBasicStatistics::mean(c1TotalShapExpected), 1.0);
    BOOST_REQUIRE_CLOSE(c2BarTotalShapActual,
                        maths::CBasicStatistics::mean(c2TotalShapExpected), 1.0);
    BOOST_REQUIRE_CLOSE(c3BarTotalShapActual,
                        maths::CBasicStatistics::mean(c3TotalShapExpected), 1.0);
    BOOST_REQUIRE_CLOSE(c4BarTotalShapActual,
                        maths::CBasicStatistics::mean(c4TotalShapExpected), 1.0);
}

BOOST_FIXTURE_TEST_CASE(testMultiClassClassificationFeatureImportanceAllShap, SFixture) {

    std::size_t topShapValues{4};
    auto results{runMultiClassClassification(topShapValues, {0.5, -0.7, 0.2, -0.2})};
    TMeanAccumulator c1FooTotalShapExpected;
    TMeanAccumulator c2FooTotalShapExpected;
    TMeanAccumulator c3FooTotalShapExpected;
    TMeanAccumulator c4FooTotalShapExpected;
    TMeanAccumulator c1BarTotalShapExpected;
    TMeanAccumulator c2BarTotalShapExpected;
    TMeanAccumulator c3BarTotalShapExpected;
    TMeanAccumulator c4BarTotalShapExpected;
    TMeanAccumulator c1BazTotalShapExpected;
    TMeanAccumulator c2BazTotalShapExpected;
    TMeanAccumulator c3BazTotalShapExpected;
    TMeanAccumulator c4BazTotalShapExpected;
    double c1FooTotalShapActual{0.0}, c2FooTotalShapActual{0.0},
        c3FooTotalShapActual{0.0}, c4FooTotalShapActual{0.0};
    double c1BarTotalShapActual{0.0}, c2BarTotalShapActual{0.0},
        c3BarTotalShapActual{0.0}, c4BarTotalShapActual{0.0};
    double c1BazTotalShapActual{0.0}, c2BazTotalShapActual{0.0},
        c3BazTotalShapActual{0.0}, c4BazTotalShapActual{0.0};
    bool hasTotalFeatureImportance{false};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("row_results")) {
            // class shap values should sum(abs()) to the overall feature importance
            double c1f{readShapValue(result, "c1", "foo")};
            double c1bar{readShapValue(result, "c1", "bar")};
            double c1baz{readShapValue(result, "c1", "baz")};
            double c1{std::abs(c1f) + std::abs(c1bar) + std::abs(c1baz)};
            c1FooTotalShapExpected.add(std::fabs(c1f));
            c1BarTotalShapExpected.add(std::fabs(c1bar));
            c1BazTotalShapExpected.add(std::fabs(c1baz));

            double c2f{readShapValue(result, "c2", "foo")};
            double c2bar{readShapValue(result, "c2", "bar")};
            double c2baz{readShapValue(result, "c2", "baz")};
            double c2{std::abs(c2f) + std::abs(c2bar) + std::abs(c2baz)};
            c2FooTotalShapExpected.add(std::fabs(c2f));
            c2BarTotalShapExpected.add(std::fabs(c2bar));
            c2BazTotalShapExpected.add(std::fabs(c2baz));

            double c3f{readShapValue(result, "c3", "foo")};
            double c3bar{readShapValue(result, "c3", "bar")};
            double c3baz{readShapValue(result, "c3", "baz")};
            double c3{std::abs(c3f) + std::abs(c3bar) + std::abs(c3baz)};
            c3FooTotalShapExpected.add(std::fabs(c3f));
            c3BarTotalShapExpected.add(std::fabs(c3bar));
            c3BazTotalShapExpected.add(std::fabs(c3baz));

            double c4f{readShapValue(result, "c4", "foo")};
            double c4bar{readShapValue(result, "c4", "bar")};
            double c4baz{readShapValue(result, "c4", "baz")};
            double c4{std::abs(c4f) + std::abs(c4bar) + std::abs(c4baz)};
            c4FooTotalShapExpected.add(std::fabs(c4f));
            c4BarTotalShapExpected.add(std::fabs(c4bar));
            c4BazTotalShapExpected.add(std::fabs(c4baz));

            // We should have at least one feature that is important
            BOOST_TEST_REQUIRE((c1 > 0.0 || c2 > 0.0 || c3 > 0.0 || c4 > 0.0));
        } else if (result.HasMember("model_metadata")) {
            if (result["model_metadata"].HasMember("total_feature_importance")) {
                hasTotalFeatureImportance = true;
            }
            c1FooTotalShapActual = readTotalShapValue(result, "c1", "foo");
            c2FooTotalShapActual = readTotalShapValue(result, "c2", "foo");
            c3FooTotalShapActual = readTotalShapValue(result, "c3", "foo");
            c4FooTotalShapActual = readTotalShapValue(result, "c4", "foo");
            c1BarTotalShapActual = readTotalShapValue(result, "c1", "bar");
            c2BarTotalShapActual = readTotalShapValue(result, "c2", "bar");
            c3BarTotalShapActual = readTotalShapValue(result, "c3", "bar");
            c4BarTotalShapActual = readTotalShapValue(result, "c4", "bar");
            c1BazTotalShapActual = readTotalShapValue(result, "c1", "baz");
            c2BazTotalShapActual = readTotalShapValue(result, "c2", "baz");
            c3BazTotalShapActual = readTotalShapValue(result, "c3", "baz");
            c4BazTotalShapActual = readTotalShapValue(result, "c4", "baz");
        }
    }
    BOOST_TEST_REQUIRE(hasTotalFeatureImportance);
    BOOST_REQUIRE_CLOSE(c1FooTotalShapActual,
                        maths::CBasicStatistics::mean(c1FooTotalShapExpected), 1.0);
    BOOST_REQUIRE_CLOSE(c2FooTotalShapActual,
                        maths::CBasicStatistics::mean(c2FooTotalShapExpected), 1.0);
    BOOST_REQUIRE_CLOSE(c3FooTotalShapActual,
                        maths::CBasicStatistics::mean(c3FooTotalShapExpected), 1.0);
    BOOST_REQUIRE_CLOSE(c4FooTotalShapActual,
                        maths::CBasicStatistics::mean(c4FooTotalShapExpected), 1.0);
    BOOST_REQUIRE_CLOSE(c1BarTotalShapActual,
                        maths::CBasicStatistics::mean(c1BarTotalShapExpected), 1.0);
    BOOST_REQUIRE_CLOSE(c2BarTotalShapActual,
                        maths::CBasicStatistics::mean(c2BarTotalShapExpected), 1.0);
    BOOST_REQUIRE_CLOSE(c3BarTotalShapActual,
                        maths::CBasicStatistics::mean(c3BarTotalShapExpected), 1.0);
    BOOST_REQUIRE_CLOSE(c4BarTotalShapActual,
                        maths::CBasicStatistics::mean(c4BarTotalShapExpected), 1.0);
    BOOST_REQUIRE_CLOSE(c1BazTotalShapActual,
                        maths::CBasicStatistics::mean(c1BazTotalShapExpected), 1.0);
    BOOST_REQUIRE_CLOSE(c2BazTotalShapActual,
                        maths::CBasicStatistics::mean(c2BazTotalShapExpected), 1.0);
    BOOST_REQUIRE_CLOSE(c3BazTotalShapActual,
                        maths::CBasicStatistics::mean(c3BazTotalShapExpected), 1.0);
    BOOST_REQUIRE_CLOSE(c4BazTotalShapActual,
                        maths::CBasicStatistics::mean(c4BazTotalShapExpected), 1.0);
}

BOOST_FIXTURE_TEST_CASE(testRegressionFeatureImportanceNoShap, SFixture) {
    // Test that if topShapValue is set to 0, no feature importance values are returned.
    std::size_t topShapValues{0};
    auto results{runRegression(topShapValues, {50.0, 150.0, 50.0, -50.0})};

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
