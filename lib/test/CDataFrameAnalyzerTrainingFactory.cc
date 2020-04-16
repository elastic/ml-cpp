/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <test/CDataFrameAnalyzerTrainingFactory.h>

namespace ml {
namespace test {

void CDataFrameAnalyzerTrainingFactory::appendPrediction(core::CDataFrame&,
                                                         std::size_t,
                                                         const TDouble2Vec& prediction,
                                                         TDoubleVec& predictions) {
    predictions.push_back(prediction[0]);
}

void CDataFrameAnalyzerTrainingFactory::appendPrediction(core::CDataFrame& frame,
                                                         std::size_t target,
                                                         const TDouble2Vec& scores,
                                                         TStrVec& predictions) {
    std::size_t i(std::max_element(scores.begin(), scores.end()) - scores.begin());
    predictions.push_back(frame.categoricalColumnValues()[target][i]);
}

CDataFrameAnalyzerTrainingFactory::TDataFrameUPtr
CDataFrameAnalyzerTrainingFactory::setupLinearRegressionData(const TStrVec& fieldNames,
                                                             TStrVec& fieldValues,
                                                             api::CDataFrameAnalyzer& analyzer,
                                                             const TDoubleVec& weights,
                                                             const TDoubleVec& regressors,
                                                             TStrVec& targets,
                                                             TTargetTransformer targetTransformer) {

    auto target = [&weights, &targetTransformer](const TDoubleVec& regressors_) {
        double result{0.0};
        for (std::size_t i = 0; i < weights.size(); ++i) {
            result += weights[i] * regressors_[i];
        }
        result = targetTransformer(result);
        return core::CStringUtils::typeToStringPrecise(result, core::CIEEE754::E_DoublePrecision);
    };

    auto frame = core::makeMainStorageDataFrame(weights.size() + 1).first;

    for (std::size_t i = 0; i < regressors.size(); i += weights.size()) {
        TDoubleVec row(weights.size());
        for (std::size_t j = 0; j < weights.size(); ++j) {
            row[j] = regressors[i + j];
        }

        for (std::size_t j = 0; j < row.size(); ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                row[j], core::CIEEE754::E_DoublePrecision);
        }
        fieldValues[weights.size()] = target(row);
        targets.push_back(fieldValues[weights.size()]);

        analyzer.handleRecord(fieldNames, fieldValues);
        frame->parseAndWriteRow(
            core::CVectorRange<const TStrVec>(fieldValues, 0, weights.size() + 1));
    }

    frame->finishWritingRows();

    return frame;
}

CDataFrameAnalyzerTrainingFactory::TDataFrameUPtr
CDataFrameAnalyzerTrainingFactory::setupBinaryClassificationData(const TStrVec& fieldNames,
                                                                 TStrVec& fieldValues,
                                                                 api::CDataFrameAnalyzer& analyzer,
                                                                 const TDoubleVec& weights,
                                                                 const TDoubleVec& regressors,
                                                                 TStrVec& targets) {
    TStrVec classes{"foo", "bar"};
    auto target = [&weights, &classes](const TDoubleVec& regressors_) {
        double result{0.0};
        for (std::size_t i = 0; i < weights.size(); ++i) {
            result += weights[i] * regressors_[i];
        }
        return classes[result < 0.0 ? 0 : 1];
    };

    auto frame = core::makeMainStorageDataFrame(weights.size() + 1).first;
    TBoolVec categoricalFields(weights.size(), false);
    categoricalFields.push_back(true);
    frame->categoricalColumns(std::move(categoricalFields));

    for (std::size_t i = 0; i < regressors.size(); i += weights.size()) {
        TDoubleVec row(weights.size());
        for (std::size_t j = 0; j < weights.size(); ++j) {
            row[j] = regressors[i + j];
        }

        for (std::size_t j = 0; j < row.size() - 1; ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                row[j], core::CIEEE754::E_DoublePrecision);
        }
        fieldValues[weights.size()] = target(row);
        targets.push_back(fieldValues[weights.size()]);

        analyzer.handleRecord(fieldNames, fieldValues);
        frame->parseAndWriteRow(
            core::CVectorRange<const TStrVec>(fieldValues, 0, weights.size() + 1));
    }

    frame->finishWritingRows();

    return frame;
}
}
}
