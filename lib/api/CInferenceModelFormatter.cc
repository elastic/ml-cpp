#include <api/CInferenceModelFormatter.h>

ml::api::CInferenceModelFormatter::CInferenceModelFormatter(const std::string &str): m_String{str}, m_Definition() {

}

ml::api::CInferenceModelFormatter::CInferenceModelFormatter(rapidjson::Document &&doc): m_JsonDoc{std::move(doc)} {

}

std::string ml::api::CInferenceModelFormatter::toString() {
    nlohmann::json j;
    nlohmann::to_json(j, m_Definition);

    return nlohmann::to_string(j);
}
