/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CModelSizeDefinition.h>

namespace ml {
namespace api {

namespace {
const std::string JSON_PREPROCESSORS_TAG{"preprocessors"};
}

CModelSizeDefinition::CModelSizeDefinition(const CInferenceModelDefinition& inferenceModel) {
    // parse preprocessing
    for (const auto& preprocessor : inferenceModel.preprocessors()) {
        m_EncodingSizeItems.push_back(preprocessor->sizeInfo());
    }
    // parse trained models
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
}
}
}