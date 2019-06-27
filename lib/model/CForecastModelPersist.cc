/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CForecastModelPersist.h>

#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/RestoreMacros.h>

#include <maths/CModelStateSerialiser.h>
#include <maths/CRestoreParams.h>
#include <maths/MathsTypes.h>

#include <model/CAnomalyDetectorModelConfig.h>

#include <boost/bind.hpp>

namespace ml {
namespace model {

namespace {
const std::string FORECAST_MODEL_PERSIST_TAG("forecast_persist");
const std::string FEATURE_TAG("feature");
const std::string DATA_TYPE_TAG("datatype");
const std::string FIRST_DATA_TIME_TAG("first_data_time");
const std::string LAST_DATA_TIME_TAG("last_data_time");
const std::string MODEL_TAG("model");
const std::string BY_FIELD_VALUE_TAG("by_field_value");
}

CForecastModelPersist::CPersist::CPersist(const std::string& temporaryPath)
    : m_FileName(temporaryPath), m_OutStream(), m_ModelCount(0) {
    m_FileName /= boost::filesystem::unique_path("forecast-persist-%%%%-%%%%-%%%%-%%%%");
    m_OutStream.open(m_FileName.string());
    m_OutStream << "[";
}

void CForecastModelPersist::CPersist::addModel(const maths::CModel* model,
                                               core_t::TTime firstDataTime,
                                               core_t::TTime lastDataTime,
                                               const model_t::EFeature feature,
                                               const std::string& byFieldValue) {
    if (m_ModelCount++ > 0) {
        m_OutStream << ",";
    }

    auto persistOneModel = [&](core::CStatePersistInserter& inserter) {
        inserter.insertValue(FEATURE_TAG, feature);
        inserter.insertValue(DATA_TYPE_TAG, model->dataType());
        inserter.insertValue(BY_FIELD_VALUE_TAG, byFieldValue);
        inserter.insertValue(FIRST_DATA_TIME_TAG, firstDataTime);
        inserter.insertValue(LAST_DATA_TIME_TAG, lastDataTime);
        inserter.insertLevel(MODEL_TAG, std::bind<void>(maths::CModelStateSerialiser(),
                                                          std::cref(*model), std::placeholders::_1));
    };

    core::CJsonStatePersistInserter inserter(m_OutStream);
    inserter.insertLevel(FORECAST_MODEL_PERSIST_TAG, persistOneModel);
}

std::string CForecastModelPersist::CPersist::finalizePersistAndGetFile() {
    m_OutStream << "]";
    m_OutStream.close();

    return m_FileName.string();
}

CForecastModelPersist::CRestore::CRestore(const SModelParams& modelParams,
                                          double minimumSeasonalVarianceScale,
                                          const std::string& fileName)
    : m_ModelParams(modelParams),
      m_MinimumSeasonalVarianceScale(minimumSeasonalVarianceScale),
      m_InStream(fileName), m_RestoreTraverser(m_InStream) {
}

bool CForecastModelPersist::CRestore::nextModel(TMathsModelPtr& model,
                                                core_t::TTime& firstDataTime,
                                                core_t::TTime& lastDataTime,
                                                model_t::EFeature& feature,
                                                std::string& byFieldValue) {
    if (m_RestoreTraverser.isEof() || m_RestoreTraverser.name().empty()) {
        return false;
    }

    if (m_RestoreTraverser.name() != FORECAST_MODEL_PERSIST_TAG) {
        LOG_ERROR(<< "Failed to restore forecast model, unexpected tag");
        return false;
    }

    if (!m_RestoreTraverser.hasSubLevel()) {
        LOG_ERROR(<< "Failed to restore forecast model, unexpected format");
        return false;
    }

    auto restoreOneModel = [&](core::CStateRestoreTraverser& traverser,
                               TMathsModelPtr& model_) {
        model_.reset();
        byFieldValue.clear();

        bool restoredFeature = false;
        bool restoredDataType = false;
        maths_t::EDataType dataType{};

        do {
            const std::string& name = traverser.name();
            RESTORE_ENUM_CHECKED(FEATURE_TAG, feature, model_t::EFeature, restoredFeature)
            RESTORE_ENUM_CHECKED(DATA_TYPE_TAG, dataType, maths_t::EDataType, restoredDataType)
            RESTORE_BUILT_IN(BY_FIELD_VALUE_TAG, byFieldValue)
            RESTORE_BUILT_IN(FIRST_DATA_TIME_TAG, firstDataTime)
            RESTORE_BUILT_IN(LAST_DATA_TIME_TAG, lastDataTime)
            if (name == MODEL_TAG) {
                if (restoredDataType == false) {
                    LOG_ERROR(<< "Failed to restore forecast model, datatype missing");
                    return false;
                }

                maths::SModelRestoreParams params{
                    maths::CModelParams{
                        m_ModelParams.s_BucketLength, m_ModelParams.s_LearnRate,
                        m_ModelParams.s_DecayRate, m_MinimumSeasonalVarianceScale,
                        m_ModelParams.s_MinimumTimeToDetectChange,
                        m_ModelParams.s_MaximumTimeToTestForChange},
                    maths::STimeSeriesDecompositionRestoreParams{
                        m_ModelParams.s_DecayRate, m_ModelParams.s_BucketLength,
                        m_ModelParams.s_ComponentSize,
                        m_ModelParams.distributionRestoreParams(dataType)},
                    m_ModelParams.distributionRestoreParams(dataType)};

                auto serialiser_operator = [&params = static_cast<const maths::SModelRestoreParams&>(params), &model_](core::CStateRestoreTraverser& traverser){
                	return maths::CModelStateSerialiser()(params, model_, traverser);};
                if (traverser.traverseSubLevel(serialiser_operator) == false) {
                    LOG_ERROR(<< "Failed to restore forecast model, model missing");
                    return false;
                }
            }
        } while (traverser.next());

        // only the by_field_value can be empty
        if (model_ == nullptr || restoredFeature == false || restoredDataType == false) {
            LOG_ERROR(<< "Failed to restore forecast model, data missing");
            return false;
        }

        return true;
    };

    TMathsModelPtr originalModel;
    if (m_RestoreTraverser.traverseSubLevel(std::bind<bool>(
            restoreOneModel, std::placeholders::_1, std::ref(originalModel))) == false) {
        LOG_ERROR(<< "Failed to restore forecast model, internal error");
        return false;
    }

    model.reset(originalModel->cloneForForecast());
    m_RestoreTraverser.nextObject();

    return true;
}

} /* namespace model  */
} /* namespace ml */
