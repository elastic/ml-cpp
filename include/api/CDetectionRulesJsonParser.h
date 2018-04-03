/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
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
#ifndef INCLUDED_ml_api_CDetectionRulesJsonParser_h
#define INCLUDED_ml_api_CDetectionRulesJsonParser_h

#include <core/CLogger.h>
#include <core/CPatternSet.h>

#include <api/ImportExport.h>

#include <model/CDetectionRule.h>

#include <rapidjson/document.h>

#include <boost/unordered_map.hpp>

#include <string>
#include <vector>

namespace ml {
namespace api {

//! \brief A parser to convert JSON detection rules into objects
class API_EXPORT CDetectionRulesJsonParser {
public:
    typedef std::vector<model::CDetectionRule> TDetectionRuleVec;
    typedef boost::unordered_map<std::string, core::CPatternSet> TStrPatternSetUMap;

public:
    //! Default constructor
    CDetectionRulesJsonParser(TStrPatternSetUMap& filtersByIdMap);

    //! Parses a string expected to contain a JSON array with
    //! detection rules and adds the rule objects into the given vector.
    bool parseRules(const std::string& json, TDetectionRuleVec& rules);

private:
    bool parseRuleConditions(const rapidjson::Value& ruleObject, model::CDetectionRule& rule);
    bool parseFilterId(const rapidjson::Value& conditionObject, model::CRuleCondition& ruleCondition);

    static bool hasStringMember(const rapidjson::Value& object, const std::string& name);
    static bool hasArrayMember(const rapidjson::Value& object, const std::string& name);
    static bool parseRuleActions(const rapidjson::Value& ruleObject, model::CDetectionRule& rule);
    static bool parseConditionsConnective(const rapidjson::Value& ruleObject, model::CDetectionRule& rule);
    static bool parseRuleConditionType(const rapidjson::Value& ruleConditionObject, model::CRuleCondition& ruleCondition);
    static bool parseCondition(const rapidjson::Value& ruleConditionObject, model::CRuleCondition& ruleCondition);
    static bool parseConditionOperator(const rapidjson::Value& conditionObject, model::CRuleCondition& ruleCondition);
    static bool parseConditionThreshold(const rapidjson::Value& conditionObject, model::CRuleCondition& ruleCondition);

private:
    //! The filters per id used by categorical rule conditions.
    TStrPatternSetUMap& m_FiltersByIdMap;
};
}
}
#endif // INCLUDED_ml_api_CDetectionRulesJsonParser_h
