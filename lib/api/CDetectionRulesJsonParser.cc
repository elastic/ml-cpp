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
#include <api/CDetectionRulesJsonParser.h>

#include <core/CLogger.h>
#include <core/CStringUtils.h>

namespace ml {
namespace api {

namespace {
const std::string ACTIONS("actions");
const std::string FILTER_RESULTS("filter_results");
const std::string SKIP_SAMPLING("skip_sampling");
const std::string CONDITIONS_CONNECTIVE("conditions_connective");
const std::string AND("and");
const std::string OR("or");
const std::string CONDITIONS("conditions");
const std::string TARGET_FIELD_NAME("target_field_name");
const std::string TARGET_FIELD_VALUE("target_field_value");
const std::string TYPE("type");
const std::string CATEGORICAL("categorical");
const std::string CATEGORICAL_MATCH("categorical_match");
const std::string CATEGORICAL_COMPLEMENT("categorical_complement");
const std::string NUMERICAL_ACTUAL("numerical_actual");
const std::string NUMERICAL_TYPICAL("numerical_typical");
const std::string NUMERICAL_DIFF_ABS("numerical_diff_abs");
const std::string TIME("time");
const std::string CONDITION("condition");
const std::string OPERATOR("operator");
const std::string LT("lt");
const std::string LTE("lte");
const std::string GT("gt");
const std::string GTE("gte");
const std::string VALUE("value");
const std::string FIELD_NAME("field_name");
const std::string FIELD_VALUE("field_value");
const std::string FILTER_ID("filter_id");
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

        bool isValid = true;

        // Required fields
        isValid &= parseRuleActions(ruleObject, rule);
        isValid &= parseConditionsConnective(ruleObject, rule);
        isValid &= parseRuleConditions(ruleObject, rule);

        if (isValid == false) {
            LOG_ERROR(<< "Failed to parse detection rules from JSON: " << json);
            rules.clear();
            return false;
        }

        // Optional fields
        if (hasStringMember(ruleObject, TARGET_FIELD_NAME)) {
            rule.targetFieldName(ruleObject[TARGET_FIELD_NAME.c_str()].GetString());
        }
        if (hasStringMember(ruleObject, TARGET_FIELD_VALUE)) {
            rule.targetFieldValue(ruleObject[TARGET_FIELD_VALUE.c_str()].GetString());
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

bool CDetectionRulesJsonParser::parseRuleActions(const rapidjson::Value& ruleObject,
                                                 model::CDetectionRule& rule) {
    if (!hasArrayMember(ruleObject, ACTIONS)) {
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
        if (parsedAction == FILTER_RESULTS) {
            action |= model::CDetectionRule::E_FilterResults;
        } else if (parsedAction == SKIP_SAMPLING) {
            action |= model::CDetectionRule::E_SkipSampling;
        } else {
            LOG_ERROR(<< "Invalid rule action: " << parsedAction);
            return false;
        }
    }

    rule.action(action);
    return true;
}

bool CDetectionRulesJsonParser::parseConditionsConnective(const rapidjson::Value& ruleObject,
                                                          model::CDetectionRule& rule) {
    if (!hasStringMember(ruleObject, CONDITIONS_CONNECTIVE)) {
        LOG_ERROR(<< "Missing rule field: " << CONDITIONS_CONNECTIVE);
        return false;
    }

    const std::string& connective = ruleObject[CONDITIONS_CONNECTIVE.c_str()].GetString();
    if (connective == OR) {
        rule.conditionsConnective(model::CDetectionRule::E_Or);
    } else if (connective == AND) {
        rule.conditionsConnective(model::CDetectionRule::E_And);
    } else {
        LOG_ERROR(<< "Invalid conditionsConnective: " << connective);
        return false;
    }
    return true;
}

bool CDetectionRulesJsonParser::parseRuleConditions(const rapidjson::Value& ruleObject,
                                                    model::CDetectionRule& rule) {
    if (!hasArrayMember(ruleObject, CONDITIONS)) {
        LOG_ERROR(<< "Missing rule field: " << CONDITIONS);
        return false;
    }

    const rapidjson::Value& array = ruleObject[CONDITIONS.c_str()];
    if (array.Empty()) {
        LOG_ERROR(<< "At least one condition is required");
        return false;
    }

    for (unsigned int i = 0; i < array.Size(); ++i) {
        model::CRuleCondition condition;
        const rapidjson::Value& conditionObject = array[i];

        if (!conditionObject.IsObject()) {
            LOG_ERROR(<< "Unexpected condition type: array conditions is expected to contain objects");
            return false;
        }

        bool isValid = true;

        // Required fields
        isValid &= parseRuleConditionType(conditionObject, condition);
        if (condition.isNumerical()) {
            isValid &= parseCondition(conditionObject, condition);
        } else if (condition.isCategorical()) {
            isValid &= this->parseFilterId(conditionObject, condition);
        }

        if (isValid == false) {
            return false;
        }

        // Optional fields
        if (hasStringMember(conditionObject, FIELD_NAME)) {
            condition.fieldName(conditionObject[FIELD_NAME.c_str()].GetString());
        }
        if (hasStringMember(conditionObject, FIELD_VALUE)) {
            condition.fieldValue(conditionObject[FIELD_VALUE.c_str()].GetString());
        }

        rule.addCondition(condition);
    }
    return true;
}

bool CDetectionRulesJsonParser::parseFilterId(const rapidjson::Value& conditionObject,
                                              model::CRuleCondition& ruleCondition) {
    if (!hasStringMember(conditionObject, FILTER_ID)) {
        LOG_ERROR(<< "Missing condition field: " << FILTER_ID);
        return false;
    }
    const std::string& filterId = conditionObject[FILTER_ID.c_str()].GetString();
    auto filterEntry = m_FiltersByIdMap.find(filterId);
    if (filterEntry == m_FiltersByIdMap.end()) {
        LOG_ERROR(<< "Filter with id [" << filterId << "] could not be found");
        return false;
    }
    ruleCondition.valueFilter(filterEntry->second);
    return true;
}

bool CDetectionRulesJsonParser::parseRuleConditionType(const rapidjson::Value& ruleConditionObject,
                                                       model::CRuleCondition& ruleCondition) {
    if (!hasStringMember(ruleConditionObject, TYPE)) {
        LOG_ERROR(<< "Missing ruleCondition field: " << TYPE);
        return false;
    }

    const std::string& type = ruleConditionObject[TYPE.c_str()].GetString();
    if (type == CATEGORICAL_MATCH || type == CATEGORICAL) {
        ruleCondition.type(model::CRuleCondition::E_CategoricalMatch);
    } else if (type == CATEGORICAL_COMPLEMENT) {
        ruleCondition.type(model::CRuleCondition::E_CategoricalComplement);
    } else if (type == NUMERICAL_ACTUAL) {
        ruleCondition.type(model::CRuleCondition::E_NumericalActual);
    } else if (type == NUMERICAL_TYPICAL) {
        ruleCondition.type(model::CRuleCondition::E_NumericalTypical);
    } else if (type == NUMERICAL_DIFF_ABS) {
        ruleCondition.type(model::CRuleCondition::E_NumericalDiffAbs);
    } else if (type == TIME) {
        ruleCondition.type(model::CRuleCondition::E_Time);
    } else {
        LOG_ERROR(<< "Invalid conditionType: " << type);
        return false;
    }
    return true;
}

bool CDetectionRulesJsonParser::parseCondition(const rapidjson::Value& ruleConditionObject,
                                               model::CRuleCondition& ruleCondition) {
    if (!ruleConditionObject.HasMember(CONDITION.c_str())) {
        LOG_ERROR(<< "Missing ruleCondition field: " << CONDITION);
        return false;
    }
    const rapidjson::Value& conditionObject = ruleConditionObject[CONDITION.c_str()];
    if (!conditionObject.IsObject()) {
        LOG_ERROR(<< "Unexpected type for condition; object was expected");
        return false;
    }

    return parseConditionOperator(conditionObject, ruleCondition) &&
           parseConditionThreshold(conditionObject, ruleCondition);
}

bool CDetectionRulesJsonParser::parseConditionOperator(const rapidjson::Value& conditionObject,
                                                       model::CRuleCondition& ruleCondition) {
    if (!hasStringMember(conditionObject, OPERATOR)) {
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

bool CDetectionRulesJsonParser::parseConditionThreshold(const rapidjson::Value& conditionObject,
                                                        model::CRuleCondition& ruleCondition) {
    if (!hasStringMember(conditionObject, VALUE)) {
        LOG_ERROR(<< "Missing condition field: " << VALUE);
        return false;
    }

    const std::string valueString = conditionObject[VALUE.c_str()].GetString();
    if (core::CStringUtils::stringToType(
            valueString, ruleCondition.condition().s_Threshold) == false) {
        LOG_ERROR(<< "Invalid operator value: " << valueString);
        return false;
    }
    return true;
}
}
}
