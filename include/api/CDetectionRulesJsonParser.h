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
    using TDetectionRuleVec = std::vector<model::CDetectionRule>;
    using TStrPatternSetUMap = boost::unordered_map<std::string, core::CPatternSet>;

public:
    //! Default constructor
    CDetectionRulesJsonParser(const TStrPatternSetUMap& filtersByIdMap);

    //! Parses a string expected to contain a JSON array with
    //! detection rules and adds the rule objects into the given vector.
    bool parseRules(const std::string& json, TDetectionRuleVec& rules);

    //! Parses a JSON value object expected to represent a JSON array of
    //! detection rules. Adds rule objects to the given vector. Any error messages
    //! are passed back in the error string.
    bool parseRules(const rapidjson::Value& value, TDetectionRuleVec& rules, std::string& errorString);

private:
    bool parseRuleScope(const rapidjson::Value& ruleObject, model::CDetectionRule& rule);
    bool parseRuleConditions(const rapidjson::Value& ruleObject, model::CDetectionRule& rule);

    static bool hasStringMember(const rapidjson::Value& object, const std::string& name);
    static bool hasArrayMember(const rapidjson::Value& object, const std::string& name);
    static bool hasDoubleMember(const rapidjson::Value& object, const std::string& name);
    static bool parseRuleActions(const rapidjson::Value& ruleObject,
                                 model::CDetectionRule& rule);
    static bool parseConditionsConnective(const rapidjson::Value& ruleObject,
                                          model::CDetectionRule& rule);
    static bool parseConditionAppliesTo(const rapidjson::Value& ruleConditionObject,
                                        model::CRuleCondition& ruleCondition);
    static bool parseConditionOperator(const rapidjson::Value& conditionObject,
                                       model::CRuleCondition& condition);
    static bool parseConditionValue(const rapidjson::Value& conditionObject,
                                    model::CRuleCondition& condition);

private:
    //! The filters per id used by categorical rule conditions.
    const TStrPatternSetUMap& m_FiltersByIdMap;
};
}
}
#endif // INCLUDED_ml_api_CDetectionRulesJsonParser_h
