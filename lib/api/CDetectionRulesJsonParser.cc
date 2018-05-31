/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CDetectionRulesJsonParser.h>

#include <core/CLogger.h>
#include <core/CStringUtils.h>

namespace ml {
namespace api {

namespace {

const std::string ACTIONS("actions");
const std::string ACTUAL("actual");
const std::string APPLIES_TO("applies_to");
const std::string CONDITION("condition");
const std::string CONDITIONS("conditions");
const std::string DIFF_FROM_TYPICAL("diff_from_typical");
const std::string EXCLUDE("exclude");
const std::string FILTER_ID("filter_id");
const std::string FILTER_TYPE("filter_type");
const std::string GT("gt");
const std::string GTE("gte");
const std::string INCLUDE("include");
const std::string LT("lt");
const std::string LTE("lte");
const std::string OPERATOR("operator");
const std::string SCOPE("scope");
const std::string SKIP_RESULT("skip_result");
const std::string SKIP_MODEL_UPDATE("skip_model_update");
const std::string TIME("time");
const std::string TYPICAL("typical");
const std::string VALUE("value");
}

CDetectionRulesJsonParser::CDetectionRulesJsonParser(TStrPatternSetUMap& filtersByIdMap)
    : m_FiltersByIdMap(filtersByIdMap) {
}

bool CDetectionRulesJsonParser::parseRules(const std::string& json, TDetectionRuleVec& rules) {
    LOG_DEBUG(<< "Parsing detection rules");

    rules.clear();
    rapidjson::Document doc;
    if (doc.Parse<0>(json.c_str()).HasParseError()) {
        LOG_ERROR(<< "An error occurred while parsing detection rules from JSON: "
                  << doc.GetParseError());
        return false;
    }

    if (!doc.IsArray()) {
        LOG_ERROR(<< "Could not parse detection rules from non-array JSON object: " << json);
        return false;
    }

    if (doc.Empty()) {
        return true;
    }

    rules.resize(doc.Size());

    for (unsigned int i = 0; i < doc.Size(); ++i) {
        if (!doc[i].IsObject()) {
            LOG_ERROR(<< "Could not parse detection rules: "
                      << "expected detection rules array to contain objects. JSON: " << json);
            rules.clear();
            return false;
        }

        model::CDetectionRule& rule = rules[i];

        rapidjson::Value& ruleObject = doc[i];

        if (ruleObject.HasMember(SCOPE.c_str()) == false &&
            ruleObject.HasMember(CONDITIONS.c_str()) == false) {
            LOG_ERROR(<< "At least one of 'scope' or 'conditions' must be specified");
            rules.clear();
            return false;
        }

        bool isValid = parseRuleActions(ruleObject, rule) &&
                       parseRuleScope(ruleObject, rule) &&
                       parseRuleConditions(ruleObject, rule);
        if (isValid == false) {
            LOG_ERROR(<< "Failed to parse detection rules from JSON: " << json);
            rules.clear();
            return false;
        }
    }

    return true;
}

bool CDetectionRulesJsonParser::hasStringMember(const rapidjson::Value& object,
                                                const std::string& name) {
    const char* nameAsCStr = name.c_str();
    return object.HasMember(nameAsCStr) && object[nameAsCStr].IsString();
}

bool CDetectionRulesJsonParser::hasArrayMember(const rapidjson::Value& object,
                                               const std::string& name) {
    const char* nameAsCStr = name.c_str();
    return object.HasMember(nameAsCStr) && object[nameAsCStr].IsArray();
}

bool CDetectionRulesJsonParser::hasDoubleMember(const rapidjson::Value& object,
                                                const std::string& name) {
    const char* nameAsCStr = name.c_str();
    return object.HasMember(nameAsCStr) && object[nameAsCStr].IsDouble();
}

bool CDetectionRulesJsonParser::parseRuleActions(const rapidjson::Value& ruleObject,
                                                 model::CDetectionRule& rule) {
    if (hasArrayMember(ruleObject, ACTIONS) == false) {
        LOG_ERROR(<< "Missing rule field: " << ACTIONS);
        return false;
    }

    const rapidjson::Value& array = ruleObject[ACTIONS.c_str()];
    if (array.Empty()) {
        LOG_ERROR(<< "At least one rule action is required");
        return false;
    }

    int action = 0;
    for (unsigned int i = 0; i < array.Size(); ++i) {
        model::CRuleCondition ruleCondition;
        const std::string& parsedAction = array[i].GetString();
        if (parsedAction == SKIP_RESULT) {
            action |= model::CDetectionRule::E_SkipResult;
        } else if (parsedAction == SKIP_MODEL_UPDATE) {
            action |= model::CDetectionRule::E_SkipModelUpdate;
        } else {
            LOG_ERROR(<< "Invalid rule action: " << parsedAction);
            return false;
        }
    }

    rule.action(action);
    return true;
}

bool CDetectionRulesJsonParser::parseRuleScope(const rapidjson::Value& ruleObject,
                                               model::CDetectionRule& rule) {

    if (ruleObject.HasMember(SCOPE.c_str()) == false) {
        return true;
    }

    const rapidjson::Value& scopeObject = ruleObject[SCOPE.c_str()];
    if (scopeObject.IsObject() == false) {
        LOG_ERROR(<< "Unexpected type for scope; object was expected");
        return false;
    }

    if (scopeObject.Empty()) {
        LOG_ERROR(<< "Scope must not be empty");
        return false;
    }

    for (auto& member : scopeObject.GetObject()) {
        if (member.value.IsObject() == false) {
            LOG_ERROR(<< "Unexpected type for scope member; object was expected");
            return false;
        }

        if (hasStringMember(member.value, FILTER_ID) == false) {
            LOG_ERROR(<< "Scope member is missing field: " << FILTER_ID);
            return false;
        }
        const std::string& filterId = member.value[FILTER_ID.c_str()].GetString();
        auto filterEntry = m_FiltersByIdMap.find(filterId);
        if (filterEntry == m_FiltersByIdMap.end()) {
            LOG_ERROR(<< "Filter with id [" << filterId << "] could not be found");
            return false;
        }

        if (hasStringMember(member.value, FILTER_TYPE) == false) {
            LOG_ERROR(<< "Scope member is missing field: " << FILTER_TYPE);
            return false;
        }

        const std::string& filterType = member.value[FILTER_TYPE.c_str()].GetString();
        if (filterType == INCLUDE) {
            rule.includeScope(member.name.GetString(), filterEntry->second);
        } else if (filterType == EXCLUDE) {
            rule.excludeScope(member.name.GetString(), filterEntry->second);
        } else {
            LOG_ERROR(<< "Invalid filter_type [" << filterType << "]");
            return false;
        }
    }
    return true;
}

bool CDetectionRulesJsonParser::parseRuleConditions(const rapidjson::Value& ruleObject,
                                                    model::CDetectionRule& rule) {
    if (ruleObject.HasMember(CONDITIONS.c_str()) == false) {
        return true;
    }

    if (hasArrayMember(ruleObject, CONDITIONS) == false) {
        LOG_ERROR(<< "Unexpected type for conditions; array was expected");
        return false;
    }

    const rapidjson::Value& array = ruleObject[CONDITIONS.c_str()];

    if (array.Empty()) {
        LOG_ERROR(<< "Conditions must not be empty");
        return false;
    }

    for (unsigned int i = 0; i < array.Size(); ++i) {
        model::CRuleCondition condition;
        const rapidjson::Value& conditionObject = array[i];

        if (conditionObject.IsObject() == false) {
            LOG_ERROR(<< "Unexpected condition type: array conditions is expected to contain objects");
            return false;
        }

        bool isValid = parseRuleAppliesTo(conditionObject, condition) &&
                       parseCondition(conditionObject, condition);
        if (isValid == false) {
            return false;
        }

        rule.addCondition(condition);
    }
    return true;
}

bool CDetectionRulesJsonParser::parseRuleAppliesTo(const rapidjson::Value& ruleConditionObject,
                                                   model::CRuleCondition& ruleCondition) {
    if (hasStringMember(ruleConditionObject, APPLIES_TO) == false) {
        LOG_ERROR(<< "Missing rule condition field: " << APPLIES_TO);
        return false;
    }

    const std::string& appliesTo = ruleConditionObject[APPLIES_TO.c_str()].GetString();
    if (appliesTo == ACTUAL) {
        ruleCondition.appliesTo(model::CRuleCondition::E_Actual);
    } else if (appliesTo == TYPICAL) {
        ruleCondition.appliesTo(model::CRuleCondition::E_Typical);
    } else if (appliesTo == DIFF_FROM_TYPICAL) {
        ruleCondition.appliesTo(model::CRuleCondition::E_DiffFromTypical);
    } else if (appliesTo == TIME) {
        ruleCondition.appliesTo(model::CRuleCondition::E_Time);
    } else {
        LOG_ERROR(<< "Invalid condition; unknown applies_to [" << appliesTo << "]");
        return false;
    }
    return true;
}

bool CDetectionRulesJsonParser::parseCondition(const rapidjson::Value& ruleConditionObject,
                                               model::CRuleCondition& ruleCondition) {
    if (ruleConditionObject.HasMember(CONDITION.c_str()) == false) {
        LOG_ERROR(<< "Missing ruleCondition field: " << CONDITION);
        return false;
    }
    const rapidjson::Value& conditionObject = ruleConditionObject[CONDITION.c_str()];
    if (!conditionObject.IsObject()) {
        LOG_ERROR(<< "Unexpected type for condition; object was expected");
        return false;
    }

    return parseConditionOperator(conditionObject, ruleCondition) &&
           parseConditionValue(conditionObject, ruleCondition);
}

bool CDetectionRulesJsonParser::parseConditionOperator(const rapidjson::Value& conditionObject,
                                                       model::CRuleCondition& ruleCondition) {
    if (hasStringMember(conditionObject, OPERATOR) == false) {
        LOG_ERROR(<< "Missing condition field: " << OPERATOR);
        return false;
    }

    const std::string& operatorString = conditionObject[OPERATOR.c_str()].GetString();
    if (operatorString == LT) {
        ruleCondition.condition().s_Op = model::CRuleCondition::E_LT;
    } else if (operatorString == LTE) {
        ruleCondition.condition().s_Op = model::CRuleCondition::E_LTE;
    } else if (operatorString == GT) {
        ruleCondition.condition().s_Op = model::CRuleCondition::E_GT;
    } else if (operatorString == GTE) {
        ruleCondition.condition().s_Op = model::CRuleCondition::E_GTE;
    } else {
        LOG_ERROR(<< "Invalid operator value: " << operatorString);
        return false;
    }
    return true;
}

bool CDetectionRulesJsonParser::parseConditionValue(const rapidjson::Value& conditionObject,
                                                    model::CRuleCondition& ruleCondition) {
    if (hasDoubleMember(conditionObject, VALUE) == false) {
        LOG_ERROR(<< "Missing condition field: " << VALUE);
        return false;
    }

    ruleCondition.condition().s_Threshold = conditionObject[VALUE.c_str()].GetDouble();
    return true;
}
}
}
