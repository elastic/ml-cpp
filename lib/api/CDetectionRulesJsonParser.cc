/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */
#include <api/CDetectionRulesJsonParser.h>

#include <boost/json/object.hpp>
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
const std::string FORCE_TIME_SHIFT("force_time_shift");
const std::string GT("gt");
const std::string GTE("gte");
const std::string INCLUDE("include");
const std::string LT("lt");
const std::string LTE("lte");
const std::string OPERATOR("operator");
const std::string PARAMETERS("params");
const std::string SCOPE("scope");
const std::string SKIP_RESULT("skip_result");
const std::string SKIP_MODEL_UPDATE("skip_model_update");
const std::string TIME("time");
const std::string TIME_SHIFT_AMOUNT("time_shift_amount");
const std::string TYPICAL("typical");
const std::string VALUE("value");
}

CDetectionRulesJsonParser::CDetectionRulesJsonParser(const TStrPatternSetUMap& filtersByIdMap)
    : m_FiltersByIdMap(filtersByIdMap) {
}

bool CDetectionRulesJsonParser::parseRules(const json::value& value,
                                           TDetectionRuleVec& rules,
                                           std::string& errorString) {

    if (value.is_array() == false) {
        errorString = "Could not parse detection rules from non-array JSON object: ";
        return false;
    }
    json::array arr = value.as_array();
    if (arr.empty()) {
        return true;
    }

    rules.resize(arr.size());

    for (unsigned int i = 0; i < arr.size(); ++i) {
        if (!arr[i].is_object()) {
            errorString = "Could not parse detection rules: "
                          "expected detection rules array to contain objects. JSON: ";
            rules.clear();
            return false;
        }

        model::CDetectionRule& rule = rules[i];

        const json::object& ruleObject = arr[i].as_object();

        if (ruleObject.contains(SCOPE.c_str()) == false &&
            ruleObject.contains(CONDITIONS.c_str()) == false) {
            errorString = "At least one of 'scope' or 'conditions' must be specified in JSON: ";
            rules.clear();
            return false;
        }

        bool isValid = parseRuleActions(ruleObject, rule) &&
                       parseRuleScope(ruleObject, rule) &&
                       parseRuleConditions(ruleObject, rule) &&
                       parseParameters(ruleObject, rule, rule.action());
        if (isValid == false) {
            errorString = "Failed to parse detection rules from JSON: ";
            rules.clear();
            return false;
        }
    }

    return true;
}

bool CDetectionRulesJsonParser::parseRules(const std::string& json, TDetectionRuleVec& rules) {
    LOG_DEBUG(<< "Parsing detection rules");

    rules.clear();
    json::error_code ec;
    json::parser p;
    p.write(json, ec);
    if (ec) {
        LOG_ERROR(<< "An error occurred while parsing detection rules from JSON: "
                  << ec.message());
        return false;
    }
    json::value doc = p.release();

    std::string errorString;
    if (this->parseRules(doc, rules, errorString) == false) {
        LOG_ERROR(<< errorString << json);
        return false;
    }

    return true;
}

bool CDetectionRulesJsonParser::hasStringMember(const json::object& object,
                                                const std::string& name) {
    const char* nameAsCStr = name.c_str();
    return object.contains(nameAsCStr) && object.at(nameAsCStr).is_string();
}


bool CDetectionRulesJsonParser::hasObjectMember(const json::object& object,
                                               const std::string& name) {
    const char* nameAsCStr = name.c_str();
    return object.contains(nameAsCStr) && object.at(nameAsCStr).is_object();
}

bool CDetectionRulesJsonParser::hasArrayMember(const json::object& object,
                                               const std::string& name) {
    const char* nameAsCStr = name.c_str();
    return object.contains(nameAsCStr) && object.at(nameAsCStr).is_array();
}

bool CDetectionRulesJsonParser::hasDoubleMember(const json::object& object,
                                                const std::string& name) {
    const char* nameAsCStr = name.c_str();
    return object.contains(nameAsCStr) && object.at(nameAsCStr).is_double();
}

bool CDetectionRulesJsonParser::hasIntegerMember(const json::object& object,
                                               const std::string& name) {
    const char* nameAsCStr = name.c_str();
    return object.contains(nameAsCStr) && object.at(nameAsCStr).is_number();
}

bool CDetectionRulesJsonParser::parseRuleActions(const json::object& ruleObject,
                                                 model::CDetectionRule& rule) {
    if (hasArrayMember(ruleObject, ACTIONS) == false) {
        LOG_ERROR(<< "Missing rule field: " << ACTIONS);
        return false;
    }

    const json::array& array = ruleObject.at(ACTIONS).as_array();
    if (array.empty()) {
        LOG_ERROR(<< "At least one rule action is required");
        return false;
    }

    int action = 0;
    for (unsigned int i = 0; i < array.size(); ++i) {
        const std::string_view& parsedAction = array[i].as_string();
        if (parsedAction == SKIP_RESULT) {
            action |= model::CDetectionRule::E_SkipResult;
        } else if (parsedAction == SKIP_MODEL_UPDATE) {
            action |= model::CDetectionRule::E_SkipModelUpdate;
        } else if (parsedAction == FORCE_TIME_SHIFT ) {
            action |= model::CDetectionRule::E_TimeShift;
        }
         else {
            LOG_ERROR(<< "Invalid rule action: " << parsedAction);
            return false;
        }
    }

    rule.action(action);
    return true;
}

bool CDetectionRulesJsonParser::parseParameters(const json::object& ruleObject, 
        model::CDetectionRule& rule, int action) {
    if (ruleObject.contains(PARAMETERS.c_str()) == false) {
        // Parameters are only required if "force_time_shift action is specified"
        if ((action & model::CDetectionRule::E_TimeShift) == 0) {
            return true;
        } else {
            LOG_ERROR(<< "Missing rule field: " << PARAMETERS << " for the force_time_shift action");
            return false;
        
        }
    }
    const json::object& parametersObject = ruleObject.at(PARAMETERS).as_object();
    if (parametersObject.empty()) {
        LOG_ERROR(<< "Parameters must not be empty");
        return false;
    }
    // if force_time_shift action is specified, then parameters must contain force_time_shift key
    if ((action & model::CDetectionRule::E_TimeShift) != 0) {
        if (hasObjectMember(parametersObject, FORCE_TIME_SHIFT) == false) {
            LOG_ERROR(<< "Missing parameter field: " << FORCE_TIME_SHIFT);
            return false;
        }
        const json::object& forceTimeShiftObject = parametersObject.at(FORCE_TIME_SHIFT).as_object();
        if (forceTimeShiftObject.empty()) {
            LOG_ERROR(<< "Force time shift parameters must not be empty");
            return false;
        }
        if (hasIntegerMember(forceTimeShiftObject, TIME_SHIFT_AMOUNT) == false) {
            LOG_ERROR(<< "Missing parameter field: " << TIME_SHIFT_AMOUNT);
            return false;
        }
        rule.addTimeShift(forceTimeShiftObject.at(TIME_SHIFT_AMOUNT).to_number<long>());
    }
    return true;
    
}

bool CDetectionRulesJsonParser::parseRuleScope(const json::object& ruleObject,
                                               model::CDetectionRule& rule) {

    if (ruleObject.contains(SCOPE.c_str()) == false) {
        return true;
    }

    const json::value& scopeObject = ruleObject.at(SCOPE);
    if (scopeObject.is_object() == false) {
        LOG_ERROR(<< "Unexpected type for scope; object was expected");
        return false;
    }

    if (scopeObject.as_object().empty()) {
        LOG_ERROR(<< "Scope must not be empty");
        return false;
    }

    for (auto& member : scopeObject.as_object()) {
        if (member.value().is_object() == false) {
            LOG_ERROR(<< "Unexpected type for scope member; object was expected");
            return false;
        }

        if (hasStringMember(member.value().as_object(), FILTER_ID) == false) {
            LOG_ERROR(<< "Scope member is missing field: " << FILTER_ID);
            return false;
        }
        const std::string_view& filterId =
            member.value().as_object().at(FILTER_ID).as_string();
        auto filterEntry = m_FiltersByIdMap.find(std::string(filterId));
        if (filterEntry == m_FiltersByIdMap.end()) {
            LOG_ERROR(<< "Filter with id [" << filterId << "] could not be found");
            return false;
        }

        if (hasStringMember(member.value().as_object(), FILTER_TYPE) == false) {
            LOG_ERROR(<< "Scope member is missing field: " << FILTER_TYPE);
            return false;
        }

        const std::string_view& filterType =
            member.value().as_object().at(FILTER_TYPE).as_string();
        if (filterType == INCLUDE) {
            rule.includeScope(member.key(), filterEntry->second);
        } else if (filterType == EXCLUDE) {
            rule.excludeScope(member.key(), filterEntry->second);
        } else {
            LOG_ERROR(<< "Invalid filter_type [" << filterType << "]");
            return false;
        }
    }
    return true;
}

bool CDetectionRulesJsonParser::parseRuleConditions(const json::object& ruleObject,
                                                    model::CDetectionRule& rule) {
    if (ruleObject.contains(CONDITIONS) == false) {
        return true;
    }

    if (hasArrayMember(ruleObject, CONDITIONS) == false) {
        LOG_ERROR(<< "Unexpected type for conditions; array was expected");
        return false;
    }

    const json::array& array = ruleObject.at(CONDITIONS).as_array();

    if (array.empty()) {
        LOG_ERROR(<< "Conditions must not be empty");
        return false;
    }

    for (unsigned int i = 0; i < array.size(); ++i) {
        model::CRuleCondition condition;
        const json::value& conditionObject_ = array[i];

        if (conditionObject_.is_object() == false) {
            LOG_ERROR(<< "Unexpected condition type: array conditions is expected to contain objects");
            return false;
        }

        const json::object& conditionObject = conditionObject_.as_object();

        bool isValid = parseConditionAppliesTo(conditionObject, condition) &&
                       parseConditionOperator(conditionObject, condition) &&
                       parseConditionValue(conditionObject, condition);
        if (isValid == false) {
            return false;
        }

        rule.addCondition(condition);
    }
    return true;
}

bool CDetectionRulesJsonParser::parseConditionAppliesTo(const json::object& conditionObject,
                                                        model::CRuleCondition& condition) {
    if (hasStringMember(conditionObject, APPLIES_TO) == false) {
        LOG_ERROR(<< "Missing rule condition field: " << APPLIES_TO);
        return false;
    }

    const std::string_view& appliesTo = conditionObject.at(APPLIES_TO).as_string();
    if (appliesTo == ACTUAL) {
        condition.appliesTo(model::CRuleCondition::E_Actual);
    } else if (appliesTo == TYPICAL) {
        condition.appliesTo(model::CRuleCondition::E_Typical);
    } else if (appliesTo == DIFF_FROM_TYPICAL) {
        condition.appliesTo(model::CRuleCondition::E_DiffFromTypical);
    } else if (appliesTo == TIME) {
        condition.appliesTo(model::CRuleCondition::E_Time);
    } else {
        LOG_ERROR(<< "Invalid condition; unknown applies_to [" << appliesTo << "]");
        return false;
    }
    return true;
}

bool CDetectionRulesJsonParser::parseConditionOperator(const json::object& conditionObject,
                                                       model::CRuleCondition& condition) {
    if (hasStringMember(conditionObject, OPERATOR) == false) {
        LOG_ERROR(<< "Missing condition field: " << OPERATOR);
        return false;
    }

    const std::string_view& operatorString = conditionObject.at(OPERATOR).as_string();
    if (operatorString == LT) {
        condition.op(model::CRuleCondition::E_LT);
    } else if (operatorString == LTE) {
        condition.op(model::CRuleCondition::E_LTE);
    } else if (operatorString == GT) {
        condition.op(model::CRuleCondition::E_GT);
    } else if (operatorString == GTE) {
        condition.op(model::CRuleCondition::E_GTE);
    } else {
        LOG_ERROR(<< "Invalid operator value: " << operatorString);
        return false;
    }
    return true;
}

bool CDetectionRulesJsonParser::parseConditionValue(const json::object& conditionObject,
                                                    model::CRuleCondition& condition) {
    if (hasDoubleMember(conditionObject, VALUE) == false) {
        LOG_ERROR(<< "Missing condition field: " << VALUE);
        return false;
    }

    condition.value(conditionObject.at(VALUE).to_number<double>());
    return true;
}
}
}
