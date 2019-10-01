#include <api/CInferenceModelFormatter.h>

#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/LogMacros.h>
#include <core/RestoreMacros.h>

#include <api/CInferenceModelDefinition.h>

#include <rapidjson/writer.h>

namespace {
const std::string BEST_FOREST_TAG{"best_forest"};
const std::string ENCODER_TAG{"encoder_tag"};
}

ml::api::CInferenceModelFormatter::CInferenceModelFormatter(const std::string& str,
                                                            const TStrVec& fieldNames,
                                                            const TStrSizeUMapVec& categoricalValuesMap)
    : m_String{str}, m_Definition(), m_FieldNames{fieldNames} {
    std::stringstream strm;
    strm.str(str);
    m_Definition.fieldNames(fieldNames);
    m_Definition.categoryNameMap(categoricalValuesMap);
    core::CJsonStateRestoreTraverser traverser{strm};
    std::unique_ptr<CEnsemble> ensemble = std::make_unique<CEnsemble>();
    std::unique_ptr<maths::CDataFrameCategoryEncoder> encoder;
    do {
        const std::string& name = traverser.name();
        RESTORE_NO_ERROR(BEST_FOREST_TAG, traverser.traverseSubLevel(std::bind(
                                              &CEnsemble::acceptRestoreTraverser,
                                              ensemble.get(), std::placeholders::_1)))
        if (name == ENCODER_TAG) {
            encoder = std::make_unique<maths::CDataFrameCategoryEncoder>(traverser);
        }
    } while (traverser.next());
    m_Definition.encodings(encoder->encodings());
    m_Definition.trainedModel(std::move(ensemble));

    LOG_DEBUG(<< "JSON Inference Output: " << m_Definition.jsonString());
}

std::string ml::api::CInferenceModelFormatter::toString() {
    return m_Definition.jsonString();
}
