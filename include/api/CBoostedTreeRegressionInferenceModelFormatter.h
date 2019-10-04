/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CInferenceModelFormatter_h
#define INCLUDED_ml_api_CInferenceModelFormatter_h

#include <maths/CBoostedTree.h>

#include <api/CInferenceModelDefinition.h>

#include <rapidjson/document.h>

#include <string>

namespace ml {
namespace api {

class CBoostedTreeRegressionInferenceModelBuilder : public maths::CBoostedTree::Visitor {
public:
    using TDoubleVec = std::vector<double>;
    using TStringVec = std::vector<std::string>;
    using TStringSizeUMap = std::unordered_map<std::string, std::size_t>;
    using TStringSizeUMapVec = std::vector<TStringSizeUMap>;
    using TSizeStringUMap = std::unordered_map<std::size_t, std::string>;
    using TSizeStringUMapVec = std::vector<TSizeStringUMap>;

public:
    CBoostedTreeRegressionInferenceModelBuilder(const TStringVec& fieldNames,
                                                const TStringSizeUMapVec& categoryNameMap);
    void visit(const maths::CBoostedTree* tree) override;

    void visit(const maths::CBoostedTreeImpl* impl) override;

    void visit(const maths::CBoostedTreeNode* node) override;

    void addTree() override;

    void addNode(std::size_t splitFeature,
                 double splitValue,
                 bool assignMissingToLeft,
                 double nodeValue,
                 double gain,
                 maths::CBoostedTreeNode::TOptionalSize leftChild,
                 maths::CBoostedTreeNode::TOptionalSize rightChild) override;

    void addOneHotEncoding(std::size_t inputColumnIndex, double mic, std::size_t hotCategory) override;

    void addTargetMeanEncoding(std::size_t inputColumnIndex,
                               double mic,
                               const TDoubleVec& map,
                               double fallback) override;

    void addFrequencyEncoding(std::size_t inputColumnIndex,
                              double mic,
                              const TDoubleVec& map,
                              double fallback) override;

    CInferenceModelDefinition&& build();

private:
    using TOneHotEncodingUPtr = std::unique_ptr<COneHotEncoding>;
    using TOneHotEncodingUMap = std::unordered_map<std::string, TOneHotEncodingUPtr>;

private:
    std::map<std::string, double> encodingMap(std::size_t inputColumnIndex,
                                              const TDoubleVec& map_) {
        //        std::vector<std::pair<std::string, double>> map;
        //        map.reserve(map_.size());
        //        for (std::size_t i = 0; i < map_.size(); ++i) {
        //            map.emplace_back(m_CategoricalFieldValues[field][i], map_[i]);
        //        }
        //        std::sort(map.begin(), map.end());
        //        return map;

        std::map<std::string, double> map;
        //        map.reserve(map_.size());
        for (std::size_t categoryUInt = 0; categoryUInt < map_.size(); ++categoryUInt) {
            std::string category{m_ReverseCategoryNameMap[inputColumnIndex][categoryUInt]};
            map.emplace(category, map_[categoryUInt]);
        }
        return map;
    }

    void categoryNameMap(const CInferenceModelDefinition::TStrSizeUMapVec& categoryNameMap) {
        m_CategoryNameMap = categoryNameMap;
        m_ReverseCategoryNameMap.reserve(categoryNameMap.size());
        for (const auto& categoryNameMapping : categoryNameMap) {
            if (categoryNameMapping.empty() == false) {
                TSizeStringUMap map;
                for (const auto& categoryMappingPair : categoryNameMapping) {
                    map.emplace(categoryMappingPair.second, categoryMappingPair.first);
                }
                m_ReverseCategoryNameMap.emplace_back(std::move(map));
            } else {
                m_ReverseCategoryNameMap.emplace_back();
            }
        }
    }

private:
    CInferenceModelDefinition m_Definition;
    TStringSizeUMapVec m_CategoryNameMap;
    TSizeStringUMapVec m_ReverseCategoryNameMap;
    TOneHotEncodingUMap m_OneHotEncodingMaps;
};

//! \brief This class contains logic of creating an inference model definition
//! for boosted tree regression.
class CBoostedTreeRegressionInferenceModelFormatter {
public:
    using TStrVec = std::vector<std::string>;
    using TStrSizeUMap = std::unordered_map<std::string, std::size_t>;
    using TStrSizeUMapVec = std::vector<TStrSizeUMap>;

public:
    //! \param persistenceString string serialization of the persisted regression object.
    //! \param fieldNames names of the input fields.
    //! \param categoryNameMap mapping from string to sequence number for categorical fields.
    explicit CBoostedTreeRegressionInferenceModelFormatter(const std::string& persistenceString,
                                                           const TStrVec& fieldNames,
                                                           const TStrSizeUMapVec& categoryNameMap);

    //! Inference model definition object.
    const CInferenceModelDefinition& definition() const;

    //! Format inference model definition as a JSON string.
    std::string toString();

private:
    CInferenceModelDefinition m_Definition;
};
}
}

#endif // INCLUDED_ml_api_CInferenceModelFormatter_h
