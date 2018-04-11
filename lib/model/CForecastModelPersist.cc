/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2018 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
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
static const std::string FORECAST_MODEL_PERSIST_TAG("forecast_persist");
static const std::string FEATURE_TAG("feature");
static const std::string DATA_TYPE_TAG("datatype");
static const std::string MODEL_TAG("model");
static const std::string BY_FIELD_VALUE_TAG("by_field_value");
}

CForecastModelPersist::CPersist::CPersist(const std::string& temporaryPath) : m_FileName(temporaryPath), m_OutStream(), m_ModelCount(0) {
    m_FileName /= boost::filesystem::unique_path("forecast-persist-%%%%-%%%%-%%%%-%%%%");
    m_OutStream.open(m_FileName.string());
    m_OutStream << "[";
}

void CForecastModelPersist::CPersist::addModel(const maths::CModel* model,
                                               const model_t::EFeature feature,
                                               const std::string& byFieldValue) {
    if (m_ModelCount++ > 0) {
        m_OutStream << ",";
    }

    core::CJsonStatePersistInserter inserter(m_OutStream);
    inserter.insertLevel(FORECAST_MODEL_PERSIST_TAG,
                         boost::bind<void>(CForecastModelPersist::CPersist::persistOneModel, _1, model, feature, byFieldValue));
}

void CForecastModelPersist::CPersist::persistOneModel(core::CStatePersistInserter& inserter,
                                                      const maths::CModel* model,
                                                      const model_t::EFeature feature,
                                                      const std::string& byFieldValue) {
    inserter.insertValue(FEATURE_TAG, feature);
    inserter.insertValue(DATA_TYPE_TAG, model->dataType());
    inserter.insertValue(BY_FIELD_VALUE_TAG, byFieldValue);
    inserter.insertLevel(MODEL_TAG, boost::bind<void>(maths::CModelStateSerialiser(), boost::cref(*model), _1));
}

const std::string& CForecastModelPersist::CPersist::finalizePersistAndGetFile() {
    m_OutStream << "]";
    m_OutStream.close();
    return m_FileName.string();
}

CForecastModelPersist::CRestore::CRestore(const SModelParams& modelParams, double minimumSeasonalVarianceScale, const std::string& fileName)
    : m_ModelParams(modelParams),
      m_MinimumSeasonalVarianceScale(minimumSeasonalVarianceScale),
      m_InStream(fileName),
      m_RestoreTraverser(m_InStream) {
}

bool CForecastModelPersist::CRestore::nextModel(TMathsModelPtr& model, model_t::EFeature& feature, std::string& byFieldValue) {
    if (m_RestoreTraverser.isEof() || m_RestoreTraverser.name().empty()) {
        return false;
    }

    if (m_RestoreTraverser.name() != FORECAST_MODEL_PERSIST_TAG) {
        LOG_ERROR("Failed to restore forecast model, unexpected tag");
        return false;
    }

    if (!m_RestoreTraverser.hasSubLevel()) {
        LOG_ERROR("Failed to restore forecast model, unexpected format");
        return false;
    }

    TMathsModelPtr originalModel;
    if (!m_RestoreTraverser.traverseSubLevel(boost::bind<bool>(CForecastModelPersist::CRestore::restoreOneModel,
                                                               _1,
                                                               boost::cref(m_ModelParams),
                                                               m_MinimumSeasonalVarianceScale,
                                                               boost::ref(originalModel),
                                                               boost::ref(feature),
                                                               boost::ref(byFieldValue)))) {
        LOG_ERROR("Failed to restore forecast model, internal error");
        return false;
    }

    model.reset(originalModel->cloneForForecast());
    m_RestoreTraverser.nextObject();

    return true;
}

bool CForecastModelPersist::CRestore::restoreOneModel(core::CStateRestoreTraverser& traverser,
                                                      const SModelParams modelParams,
                                                      double minimumSeasonalVarianceScale,
                                                      TMathsModelPtr& model,
                                                      model_t::EFeature& feature,
                                                      std::string& byFieldValue) {
    // reset all
    model.reset();
    bool restoredFeature = false;
    bool restoredDataType = false;
    byFieldValue.clear();
    maths_t::EDataType dataType;

    do {
        const std::string& name = traverser.name();
        RESTORE_ENUM_CHECKED(FEATURE_TAG, feature, model_t::EFeature, restoredFeature)
        RESTORE_ENUM_CHECKED(DATA_TYPE_TAG, dataType, maths_t::EDataType, restoredDataType)
        RESTORE_BUILT_IN(BY_FIELD_VALUE_TAG, byFieldValue)
        if (name == MODEL_TAG) {
            if (!restoredDataType) {
                LOG_ERROR("Failed to restore forecast model, datatype missing");
                return false;
            }

            maths::SModelRestoreParams params{
                maths::CModelParams(
                    modelParams.s_BucketLength, modelParams.s_LearnRate, modelParams.s_DecayRate, minimumSeasonalVarianceScale),
                maths::STimeSeriesDecompositionRestoreParams{
                    modelParams.s_DecayRate, modelParams.s_BucketLength, modelParams.s_ComponentSize},
                modelParams.distributionRestoreParams(dataType)};

            if (!traverser.traverseSubLevel(
                    boost::bind<bool>(maths::CModelStateSerialiser(), boost::cref(params), boost::ref(model), _1))) {
                LOG_ERROR("Failed to restore forecast model, model missing");
                return false;
            }
        }
    } while (traverser.next());

    // only the by_field_value can be empty
    if (!model || !restoredFeature || !restoredDataType) {
        LOG_ERROR("Failed to restore forecast model, data missing");
        return false;
    }

    return true;
}

} /* namespace model  */
} /* namespace ml */
