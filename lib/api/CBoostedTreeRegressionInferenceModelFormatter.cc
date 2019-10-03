#include <api/CBoostedTreeRegressionInferenceModelFormatter.h>

#include <core/CJsonStateRestoreTraverser.h>
#include <core/LogMacros.h>
#include <core/RestoreMacros.h>

#include <api/CInferenceModelDefinition.h>

#include <rapidjson/writer.h>

namespace {
const std::string BEST_FOREST_TAG{"best_forest"};
const std::string ENCODER_TAG{"encoder_tag"};
}

ml::api::CBoostedTreeRegressionInferenceModelFormatter::CBoostedTreeRegressionInferenceModelFormatter(
    const std::string& persistenceString,
    const TStrVec& fieldNames,
    const TStrSizeUMapVec& categoryNameMap)
    : m_Definition(fieldNames, categoryNameMap) {

    std::stringstream strm;
    strm.str(persistenceString);
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

    // set pre-processing
    m_Definition.encodings(encoder->encodings());

    // set trained model
    ensemble->aggregateOutput(std::make_unique<CWeightedSum>(ensemble->size(), 1.0));
    m_Definition.trainedModel(std::move(ensemble));
    m_Definition.trainedModel()->targetType(CTrainedModel::E_Regression);
}

std::string ml::api::CBoostedTreeRegressionInferenceModelFormatter::toString() {
    return m_Definition.jsonString();
}

const ml::api::CInferenceModelDefinition&
ml::api::CBoostedTreeRegressionInferenceModelFormatter::definition() const {
    return m_Definition;
}
