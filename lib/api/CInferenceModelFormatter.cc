#include <api/CInferenceModelFormatter.h>

#include <core/CLogger.h>
#include <core/LogMacros.h>

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
            if ( objectName == "a") {

                for (auto treeIterator = forestIterator->value.MemberBegin();
                     treeIterator != forestIterator->value.MemberEnd(); ++treeIterator) {
                    std::string objectName2{treeIterator->name.GetString()};
                    if (objectName2 == "a") {
                        for (auto treeElementIteartor = treeIterator->value.MemberBegin();
                             treeElementIteartor != treeIterator->value.MemberEnd(); ++treeElementIteartor) {
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
