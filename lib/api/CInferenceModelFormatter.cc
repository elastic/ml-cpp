#include <api/CInferenceModelFormatter.h>

#include <core/CLogger.h>
#include <core/LogMacros.h>

#include <api/SInferenceModelDefinition.h>

#include <core/CJsonStateRestoreTraverser.h>
#include <core/CPersistUtils.h>
#include <core/RestoreMacros.h>
#include <rapidjson/fwd.h>
#include <rapidjson/writer.h>

namespace {
const std::string BEST_FOREST_TAG{"best_forest"};
}

ml::api::CInferenceModelFormatter::CInferenceModelFormatter(const std::string& str)
    : m_String{str}, m_Definition() {
    std::stringstream strm;
    strm.str(str);
    core::CJsonStateRestoreTraverser traverser{strm};
    std::unique_ptr<SEnsemble> ensemble = std::make_unique<SEnsemble>();
    do {
        const std::string& name = traverser.name();
//        if (name == BEST_FOREST_TAG) {
//            LOG_DEBUG(<< "Found name "<< name);
//            ensemble->acceptRestoreTraverser(traverser);
//            continue;
//        }
        //        RESTORE_NO_ERROR(BEST_FOREST_TAG,
        //                core::CPersistUtils::restore(BEST_FOREST_TAG, m_Definition.m_TrainedModel, traverser))
        RESTORE_NO_ERROR(BEST_FOREST_TAG, traverser.traverseSubLevel(std::bind(&SEnsemble::acceptRestoreTraverser,
                ensemble.get(), std::placeholders::_1)))
    } while (traverser.next());
    m_Definition.m_TrainedModel = std::move(ensemble);
}

ml::api::CInferenceModelFormatter::CInferenceModelFormatter(const rapidjson::Document& doc) {
    //    if (doc.IsObject()) {
    //        const auto &bestForest{doc["best_forest"]};
    //        for (auto forestIterator = bestForest.MemberBegin();
    //             forestIterator != bestForest.MemberEnd(); ++forestIterator) {
    //            std::string objectName{forestIterator->name.GetString()};
    //            if (objectName == "a") {
    //                ml::api::STree tree;
    //                for (auto
    //                treeIterator = forestIterator->value.MemberBegin();
    //                     treeIterator != forestIterator->value.MemberEnd(); ++treeIterator) {
    //                    std::string objectName2{treeIterator->name.GetString()};
    //                    if (objectName2 == "a") {
    //                        ml::api::STreeNode treeNode;
    //                        for (auto treeElementIteartor = treeIterator->value.MemberBegin();
    //                             treeElementIteartor != treeIterator->value.MemberEnd(); ++treeElementIteartor) {
    //                            std::string name{treeElementIteartor->name.GetString()};
    //                            std::string value{treeElementIteartor->name.GetString()};
    //                            if (name == "split_feature") {
    //                                treeNode.splitFeature = std::stoul(value);
    //                            }
    //                            else if (name == "assign_missing_to_left") {
    //                                treeNode.m_DefaultLeft = (value == "true");
    //                            }
    //                            else if (name == "node_value") {
    //                                treeNode.nodeValue = std::stod(value);
    //                            }
    //                            else if (name == "split_index") {
    //                                treeNode.splitIndex = std::stoul(value);
    //                            }
    //                            else if (name == "split_value") {
    //                                treeNode.threshold = std::stod(value);
    //                            }
    //                            else if (name == "left_child") {
    //                                if (std::stoi(value) != -1) {
    //                                    treeNode.leftChild =std::make_shared<std::int64_t >(std::stoi(value));
    //                                }
    //                            }
    //                            else if (name == "right_child") {
    //                                if (std::stoi(value) != -1) {
    //                                    treeNode.rightChild = std::make_shared<std::int64_t >(std::stoi(value));
    //                                }
    //                            }
    //                        }
    //                        tree.m_TreeStructure.push_back(treeNode);
    //                    }
    //                    m_Definition.m_TrainedModel.models->push_back(tree);
    //                }
    //            }
    //
    //        }
    //    } else {
    //        LOG_ERROR(<< "Doc is not an object");
    //    }
}

std::string ml::api::CInferenceModelFormatter::toString() {
}
