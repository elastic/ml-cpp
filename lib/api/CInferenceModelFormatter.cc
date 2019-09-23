#include <api/CInferenceModelFormatter.h>

#include <core/CLogger.h>
#include <core/LogMacros.h>

#include <api/InferenceModelDefinition.hpp>

#include <rapidjson/writer.h>
#include <rapidjson/fwd.h>

ml::api::CInferenceModelFormatter::CInferenceModelFormatter(const std::string &str) : m_String{str}, m_Definition() {

}

ml::api::CInferenceModelFormatter::CInferenceModelFormatter(const rapidjson::Document &doc) {
    if (doc.IsObject()) {
        const auto &bestForest{doc["best_forest"]};
        for (auto forestIterator = bestForest.MemberBegin();
             forestIterator != bestForest.MemberEnd(); ++forestIterator) {
            std::string objectName{forestIterator->name.GetString()};
            if (objectName == "a") {

                for (auto treeIterator = forestIterator->value.MemberBegin();
                     treeIterator != forestIterator->value.MemberEnd(); ++treeIterator) {
                    std::string objectName2{treeIterator->name.GetString()};
                    if (objectName2 == "a") {
                        inference_model::TreeNode treeNode;
                        for (auto treeElementIteartor = treeIterator->value.MemberBegin();
                             treeElementIteartor != treeIterator->value.MemberEnd(); ++treeElementIteartor) {
                            std::string name{treeElementIteartor->name.GetString()};
                            std::string value{treeElementIteartor->name.GetString()};
                            if (name == "split_feature") {
                                treeNode.splitFeature = std::stoul(value);
                            }
                            else if (name == "assign_missing_to_left") {
                                treeNode.defaultLeft = (value == "true");
                            }
                            else if (name == "node_value") {
                                treeNode.nodeValue = std::stod(value);
                            }
                            else if (name == "split_index") {
                                treeNode.splitIndex = std::stoul(value);
                            }
                            else if (name == "split_value") {
                                treeNode.threshold = std::stod(value);
                            }
                            else if (name == "left_child") {
                                if (std::stoi(value) != -1) {
                                    treeNode.leftChild =std::make_shared<std::int64_t >(std::stoi(value));
                                }
                            }
                            else if (name == "right_child") {
                                if (std::stoi(value) != -1) {
                                    treeNode.rightChild = std::make_shared<std::int64_t >(std::stoi(value));
                                }
                            }


                            LOG_DEBUG(<< treeElementIteartor->name.GetString());
                        }
                    }
                }
            }

        }
    } else {
        LOG_ERROR(<< "Doc is not an object");
    }
}

std::string ml::api::CInferenceModelFormatter::toString() {
    nlohmann::json j;
    nlohmann::to_json(j, m_Definition);

    return nlohmann::to_string(j);
}
