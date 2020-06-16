/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CModelSizeDefinition.h>

namespace ml {
namespace api {

namespace {
// clang-format off
const std::string JSON_ENSEMBLE_MODEL_SIZE_TAG{"ensemble_model_size"};
const std::string JSON_PREPROCESSORS_TAG{"preprocessors"};
const std::string JSON_TRAINED_MODEL_SIZE_TAG{"trained_model_size"};
// clang-format on
}

CModelSizeDefinition::CModelSizeDefinition(const CInferenceModelDefinition& inferenceModel)
    : m_TrainedModelSize{nullptr} {
    // parse preprocessing
    m_EncodingSizeItems.reserve(inferenceModel.preprocessors().size());
    for (const auto& preprocessor : inferenceModel.preprocessors()) {
        m_EncodingSizeItems.push_back(preprocessor->sizeInfo());
    }
    // parse trained models
    if (inferenceModel.trainedModel()) {
        inferenceModel.trainedModel()->sizeInfo().swap(m_TrainedModelSize);
    }
}

std::string CModelSizeDefinition::jsonString() {
    std::ostringstream stream;
    {
        core::CJsonOutputStreamWrapper wrapper{stream};
        CSerializableToJson::TRapidJsonWriter writer{wrapper};
        rapidjson::Value doc = writer.makeObject();
        this->addToDocument(doc, writer);
        writer.write(doc);
        stream.flush();
    }
    // string writer puts the json object in an array, so we strip the external brackets
    std::string jsonStr{stream.str()};
    std::string resultString(jsonStr, 1, jsonStr.size() - 2);
    return resultString;
}

void CModelSizeDefinition::addToDocument(rapidjson::Value& parentObject,
                                         TRapidJsonWriter& writer) const {
    // preprocessors
    rapidjson::Value preprocessingArray = writer.makeArray();
    for (const auto& encoding : m_EncodingSizeItems) {
        rapidjson::Value encodingValue = writer.makeObject();
        encoding->addToDocument(encodingValue, writer);
        rapidjson::Value encodingEnclosingObject = writer.makeObject();
        writer.addMember(encoding->typeString(), encodingValue, encodingEnclosingObject);
        preprocessingArray.PushBack(encodingEnclosingObject, writer.getRawAllocator());
    }
    writer.addMember(JSON_PREPROCESSORS_TAG, preprocessingArray, parentObject);
    rapidjson::Value trainedModelSizeObject = writer.makeObject();
    rapidjson::Value ensembleModelSizeObject = writer.makeObject();
    m_TrainedModelSize->addToDocument(ensembleModelSizeObject, writer);
    writer.addMember(JSON_ENSEMBLE_MODEL_SIZE_TAG, ensembleModelSizeObject, trainedModelSizeObject);
    writer.addMember(JSON_TRAINED_MODEL_SIZE_TAG, trainedModelSizeObject, parentObject);
}
}
}