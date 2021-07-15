/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CInferenceModelMetadata_h
#define INCLUDED_ml_api_CInferenceModelMetadata_h

#include <maths/CBasicStatistics.h>
#include <maths/CBoostedTree.h>
#include <maths/CLinearAlgebraEigen.h>

#include <api/CInferenceModelDefinition.h>
#include <api/ImportExport.h>

#include <boost/unordered_map.hpp>

#include <string>
#include <tuple>

namespace ml {
namespace api {

//! \brief Class controls the serialization of the model meta information
//! (such as totol feature importance) into JSON format.
class API_EXPORT CInferenceModelMetadata {
public:
    static const std::string JSON_ABSOLUTE_IMPORTANCE_TAG;
    static const std::string JSON_BASELINE_TAG;
    static const std::string JSON_CLASS_NAME_TAG;
    static const std::string JSON_CLASSES_TAG;
    static const std::string JSON_DATA_SUMMARIZATION_TAG;
    static const std::string JSON_FEATURE_IMPORTANCE_BASELINE_TAG;
    static const std::string JSON_FEATURE_NAME_TAG;
    static const std::string JSON_HYPERPARAMETERS_TAG;
    static const std::string JSON_HYPERPARAMETER_NAME_TAG;
    static const std::string JSON_HYPERPARAMETER_VALUE_TAG;
    static const std::string JSON_HYPERPARAMETER_SUPPLIED_TAG;
    static const std::string JSON_IMPORTANCE_TAG;
    static const std::string JSON_MAX_TAG;
    static const std::string JSON_MEAN_MAGNITUDE_TAG;
    static const std::string JSON_MIN_TAG;
    static const std::string JSON_MODEL_METADATA_TAG;
    static const std::string JSON_NUM_ROWS_TAG;
    static const std::string JSON_NUM_TRAINING_ROWS_TAG;
    static const std::string JSON_RELATIVE_IMPORTANCE_TAG;
    static const std::string JSON_TOTAL_FEATURE_IMPORTANCE_TAG;
    static const std::string JSON_TRAIN_PARAMETERS_TAG;

public:
    using TVector = maths::CDenseVector<double>;
    using TStrVec = std::vector<std::string>;
    using TRapidJsonWriter = core::CRapidJsonConcurrentLineWriter;
    using THyperparameterImportanceVec = maths::CBoostedTree::THyperparameterImportanceVec;
    using TPredictionFieldTypeResolverWriter =
        std::function<void(const std::string&, TRapidJsonWriter&)>;

public:
    //! Writes metadata using \p writer.
    void write(TRapidJsonWriter& writer) const;
    void columnNames(const TStrVec& columnNames);
    void classValues(const TStrVec& classValues);
    void predictionFieldTypeResolverWriter(const TPredictionFieldTypeResolverWriter& resolverWriter);
    static const std::string& typeString();
    //! Add importances \p values to the feature with index \p i to calculate total feature importance.
    //! Total feature importance is the mean of the magnitudes of importances for individual data points.
    void addToFeatureImportance(std::size_t i, const TVector& values);
    //! Set the feature importance baseline (the individual feature importances are additive corrections
    //! to the baseline value).
    void featureImportanceBaseline(TVector&& baseline);
    //! Set the hyperparameter importances.
    void hyperparameterImportance(const THyperparameterImportanceVec& hyperparameterImportance);
    //! Set the number of rows used to train the model.
    void numTrainingRows(std::size_t numRows);
    //! Set the fraction of data per fold used for training when tuning hyperparameters.
    void trainFractionPerFold(double fraction);
    //! Set the number of rows in the training data summarization.
    void numDataSummarizationRows(std::size_t numRows);

private:
    struct SHyperparameterImportance {
        enum EType { E_Double, E_Uint64 };
        std::string s_HyperparameterName;
        double s_Value;
        double s_AbsoluteImportance;
        double s_RelativeImportance;
        bool s_Supplied;
        EType s_Type;
    };

    using TMeanAccumulator =
        std::vector<maths::CBasicStatistics::SSampleMean<double>::TAccumulator>;
    using TMinMaxAccumulator = std::vector<maths::CBasicStatistics::CMinMax<double>>;
    using TSizeMeanAccumulatorUMap = boost::unordered_map<std::size_t, TMeanAccumulator>;
    using TSizeMinMaxAccumulatorUMap = boost::unordered_map<std::size_t, TMinMaxAccumulator>;
    using TOptionalVector = boost::optional<TVector>;
    using THyperparametersVec = std::vector<SHyperparameterImportance>;

private:
    void writeTotalFeatureImportance(TRapidJsonWriter& writer) const;
    void writeFeatureImportanceBaseline(TRapidJsonWriter& writer) const;
    void writeHyperparameterImportance(TRapidJsonWriter& writer) const;
    void writeTrainParameters(TRapidJsonWriter& writer) const;
    void writeDataSummarization(TRapidJsonWriter& writer) const;

private:
    TSizeMeanAccumulatorUMap m_TotalShapValuesMean;
    TSizeMinMaxAccumulatorUMap m_TotalShapValuesMinMax;
    TOptionalVector m_ShapBaseline;
    TStrVec m_ColumnNames;
    TStrVec m_ClassValues;
    TPredictionFieldTypeResolverWriter m_PredictionFieldTypeResolverWriter{
        [](const std::string& value, TRapidJsonWriter& writer) {
            writer.String(value);
        }};
    THyperparametersVec m_HyperparameterImportance;
    std::size_t m_NumDataSummarizationRows{0};
    std::size_t m_NumTrainingRows{0};
    double m_TrainFractionPerFold{0.0};
};
}
}

#endif // INCLUDED_ml_api_CInferenceModelMetadata_h
