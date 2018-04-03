/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CFieldConfig.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CoreTypes.h>
#include <core/CRegex.h>
#include <core/CStrCaseCmp.h>
#include <core/CStringUtils.h>

#include <model/CLimits.h>
#include <model/FunctionTypes.h>

#include <api/CDetectionRulesJsonParser.h>
#include <api/CFieldDataTyper.h>

#include <boost/bind.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/tokenizer.hpp>

#include <algorithm>
#include <fstream>
#include <ostream>
#include <sstream>
#include <utility>


namespace ml
{
namespace api
{


// Initialise statics
const std::string CFieldConfig::DETECTOR_PREFIX("detector.");
const std::string CFieldConfig::CATEGORIZATION_FILTER_PREFIX("categorizationfilter.");
const std::string CFieldConfig::INFLUENCER_PREFIX("influencer.");
const std::string CFieldConfig::FILTER_PREFIX("filter.");
const std::string CFieldConfig::SCHEDULED_EVENT_PREFIX("scheduledevent.");
const std::string CFieldConfig::CLAUSE_SUFFIX(".clause");
const std::string CFieldConfig::DESCRIPTION_SUFFIX(".description");
const std::string CFieldConfig::RULES_SUFFIX(".rules");
const std::string CFieldConfig::CATEGORIZATION_FIELD_OPTION("categorizationfield");
const std::string CFieldConfig::SUMMARY_COUNT_FIELD_OPTION("summarycountfield");
const char        CFieldConfig::SUFFIX_SEPARATOR('.');
const char        CFieldConfig::FIELDNAME_SEPARATOR('-');
const std::string CFieldConfig::IS_ENABLED_SUFFIX(".isEnabled");
const std::string CFieldConfig::BY_SUFFIX(".by");
const std::string CFieldConfig::OVER_SUFFIX(".over");
const std::string CFieldConfig::PARTITION_SUFFIX(".partition");
const std::string CFieldConfig::PARTITION_FIELD_OPTION("partitionfield");
const std::string CFieldConfig::USE_NULL_SUFFIX(".useNull");
const std::string CFieldConfig::USE_NULL_OPTION("usenull");
const std::string CFieldConfig::BY_TOKEN("by");
const std::string CFieldConfig::OVER_TOKEN("over");
const std::string CFieldConfig::COUNT_NAME("count");
const std::string CFieldConfig::INFLUENCER_FIELD_NAMES_OPTION("influencers");
const std::string CFieldConfig::INFLUENCER_FIELD_OPTION("influencerfield");
const std::string CFieldConfig::FALSE_VALUE(1, 'F');
const std::string CFieldConfig::TRUE_VALUE(1, 'T');
const std::string CFieldConfig::EMPTY_STRING;

// Event rate functions
const std::string CFieldConfig::FUNCTION_COUNT("count");
const std::string CFieldConfig::FUNCTION_COUNT_ABBREV("c");
const std::string CFieldConfig::FUNCTION_LOW_COUNT("low_count");
const std::string CFieldConfig::FUNCTION_LOW_COUNT_ABBREV("low_c");
const std::string CFieldConfig::FUNCTION_HIGH_COUNT("high_count");
const std::string CFieldConfig::FUNCTION_HIGH_COUNT_ABBREV("high_c");
const std::string CFieldConfig::FUNCTION_DISTINCT_COUNT("distinct_count");
const std::string CFieldConfig::FUNCTION_DISTINCT_COUNT_ABBREV("dc");
const std::string CFieldConfig::FUNCTION_LOW_DISTINCT_COUNT("low_distinct_count");
const std::string CFieldConfig::FUNCTION_LOW_DISTINCT_COUNT_ABBREV("low_dc");
const std::string CFieldConfig::FUNCTION_HIGH_DISTINCT_COUNT("high_distinct_count");
const std::string CFieldConfig::FUNCTION_HIGH_DISTINCT_COUNT_ABBREV("high_dc");
const std::string CFieldConfig::FUNCTION_NON_ZERO_COUNT("non_zero_count");
const std::string CFieldConfig::FUNCTION_NON_ZERO_COUNT_ABBREV("nzc");
const std::string CFieldConfig::FUNCTION_RARE_NON_ZERO_COUNT("rare_non_zero_count");
const std::string CFieldConfig::FUNCTION_RARE_NON_ZERO_COUNT_ABBREV("rnzc");
const std::string CFieldConfig::FUNCTION_RARE("rare");
// No abbreviation for "rare" as "r" is a little too obscure
const std::string CFieldConfig::FUNCTION_RARE_COUNT("rare_count");
// No abbreviation for "rare_count" as "rc" is sometimes used as an abbreviation
// for "return code"
const std::string CFieldConfig::FUNCTION_FREQ_RARE("freq_rare");
const std::string CFieldConfig::FUNCTION_FREQ_RARE_ABBREV("fr");
const std::string CFieldConfig::FUNCTION_FREQ_RARE_COUNT("freq_rare_count");
const std::string CFieldConfig::FUNCTION_FREQ_RARE_COUNT_ABBREV("frc");
const std::string CFieldConfig::FUNCTION_LOW_NON_ZERO_COUNT("low_non_zero_count");
const std::string CFieldConfig::FUNCTION_LOW_NON_ZERO_COUNT_ABBREV("low_nzc");
const std::string CFieldConfig::FUNCTION_HIGH_NON_ZERO_COUNT("high_non_zero_count");
const std::string CFieldConfig::FUNCTION_HIGH_NON_ZERO_COUNT_ABBREV("high_nzc");
const std::string CFieldConfig::FUNCTION_INFO_CONTENT("info_content");
const std::string CFieldConfig::FUNCTION_LOW_INFO_CONTENT("low_info_content");
const std::string CFieldConfig::FUNCTION_HIGH_INFO_CONTENT("high_info_content");

// Metric functions
const std::string CFieldConfig::FUNCTION_METRIC("metric");
const std::string CFieldConfig::FUNCTION_AVERAGE("avg");
const std::string CFieldConfig::FUNCTION_MEAN("mean");
const std::string CFieldConfig::FUNCTION_LOW_MEAN("low_mean");
const std::string CFieldConfig::FUNCTION_HIGH_MEAN("high_mean");
const std::string CFieldConfig::FUNCTION_LOW_AVERAGE("low_avg");
const std::string CFieldConfig::FUNCTION_HIGH_AVERAGE("high_avg");
const std::string CFieldConfig::FUNCTION_MEDIAN("median");
const std::string CFieldConfig::FUNCTION_LOW_MEDIAN("low_median");
const std::string CFieldConfig::FUNCTION_HIGH_MEDIAN("high_median");
const std::string CFieldConfig::FUNCTION_MIN("min");
const std::string CFieldConfig::FUNCTION_MAX("max");
const std::string CFieldConfig::FUNCTION_VARIANCE("varp");
const std::string CFieldConfig::FUNCTION_LOW_VARIANCE("low_varp");
const std::string CFieldConfig::FUNCTION_HIGH_VARIANCE("high_varp");
const std::string CFieldConfig::FUNCTION_SUM("sum");
const std::string CFieldConfig::FUNCTION_LOW_SUM("low_sum");
const std::string CFieldConfig::FUNCTION_HIGH_SUM("high_sum");
const std::string CFieldConfig::FUNCTION_NON_NULL_SUM("non_null_sum");
const std::string CFieldConfig::FUNCTION_NON_NULL_SUM_ABBREV("nns");
const std::string CFieldConfig::FUNCTION_LOW_NON_NULL_SUM("low_non_null_sum");
const std::string CFieldConfig::FUNCTION_LOW_NON_NULL_SUM_ABBREV("low_nns");
const std::string CFieldConfig::FUNCTION_HIGH_NON_NULL_SUM("high_non_null_sum");
const std::string CFieldConfig::FUNCTION_HIGH_NON_NULL_SUM_ABBREV("high_nns");
const std::string CFieldConfig::FUNCTION_TIME_OF_DAY("time_of_day");
const std::string CFieldConfig::FUNCTION_TIME_OF_WEEK("time_of_week");
const std::string CFieldConfig::FUNCTION_LAT_LONG("lat_long");
const std::string CFieldConfig::FUNCTION_MAX_VELOCITY("max_velocity");
const std::string CFieldConfig::FUNCTION_MIN_VELOCITY("min_velocity");
const std::string CFieldConfig::FUNCTION_MEAN_VELOCITY("mean_velocity");
const std::string CFieldConfig::FUNCTION_SUM_VELOCITY("sum_velocity");

const std::string CFieldConfig::EXCLUDE_FREQUENT_SUFFIX(".excludefrequent");
const std::string CFieldConfig::EXCLUDE_FREQUENT_OPTION("excludefrequent");
const std::string CFieldConfig::ALL_TOKEN("all");
const std::string CFieldConfig::NONE_TOKEN("none");

const std::string CFieldConfig::CLEAR("clear");

CFieldConfig::CFieldConfig(void)
{
}

CFieldConfig::CFieldConfig(const std::string &categorizationFieldName)
    : m_CategorizationFieldName(categorizationFieldName)
{
    this->seenField(categorizationFieldName);
}

CFieldConfig::CFieldConfig(const std::string &fieldName,
                           const std::string &byFieldName,
                           bool useNull,
                           const std::string &summaryCountFieldName)
    : m_SummaryCountFieldName(summaryCountFieldName)
{
    CFieldOptions options(fieldName,
                          1,
                          byFieldName,
                          false,
                          useNull);

    m_FieldOptions.insert(options);

    // For historical reasons, the only function name we interpret in this
    // constructor is "count" - every other word is considered to be a metric
    // field name
    if (fieldName != COUNT_NAME)
    {
        this->seenField(fieldName);
    }
    this->seenField(byFieldName);
}

CFieldConfig::CFieldConfig(const std::string &fieldName,
                           const std::string &byFieldName,
                           const std::string &partitionFieldName,
                           bool useNull)
{
    CFieldOptions options(fieldName,
                          1,
                          byFieldName,
                          partitionFieldName,
                          false,
                          false,
                          useNull);

    m_FieldOptions.insert(options);

    // For historical reasons, the only function name we interpret in this
    // constructor is "count" - every other word is considered to be a metric
    // field name
    if (fieldName != COUNT_NAME)
    {
        this->seenField(fieldName);
    }
    this->seenField(byFieldName);
    this->seenField(partitionFieldName);
}

bool CFieldConfig::initFromCmdLine(const std::string &configFile,
                                   const TStrVec &tokens)
{
    if (tokens.empty() && configFile.empty())
    {
        LOG_ERROR("Neither a fieldname clause nor a field config file was specified");
        return false;
    }

    if (tokens.empty())
    {
        return this->initFromFile(configFile);
    }

    if (!configFile.empty())
    {
        LOG_ERROR("Cannot specify both a fieldname clause and a field config file");
        return false;
    }

    return this->initFromClause(tokens);
}

bool CFieldConfig::initFromFile(const std::string &configFile)
{
    LOG_DEBUG("Reading config file " << configFile);

    m_FieldOptions.clear();
    m_FieldNameSuperset.clear();
    m_CategorizationFilters.clear();
    m_Influencers.clear();
    m_DetectorRules.clear();
    m_RuleFilters.clear();
    m_ScheduledEvents.clear();

    boost::property_tree::ptree propTree;
    try
    {
        std::ifstream strm(configFile.c_str());
        if (!strm.is_open())
        {
            LOG_ERROR("Error opening config file " << configFile);
            return false;
        }
        model::CLimits::skipUtf8Bom(strm);

        boost::property_tree::ini_parser::read_ini(strm, propTree);
    }
    catch (boost::property_tree::ptree_error &e)
    {
        LOG_ERROR("Error reading config file " << configFile << ": " << e.what());
        return false;
    }

    TIntSet handledConfigs;
    TIntSet handledScheduledEvents;

    for (boost::property_tree::ptree::iterator level1Iter = propTree.begin();
         level1Iter != propTree.end();
         ++level1Iter)
    {
        const std::string &level1Key = level1Iter->first;
        const std::string &value = level1Iter->second.data();
        if (level1Key.length() > DETECTOR_PREFIX.length() &&
            level1Key.compare(0, DETECTOR_PREFIX.length(), DETECTOR_PREFIX) == 0)
        {
            if (this->processDetector(propTree,
                                      level1Key,
                                      value,
                                      handledConfigs) == false)
            {
                LOG_ERROR("Error reading config file " << configFile);
                return false;
            }
        }
        else if (level1Key.length() > CATEGORIZATION_FILTER_PREFIX.length() &&
                 level1Key.compare(0, CATEGORIZATION_FILTER_PREFIX.length(), CATEGORIZATION_FILTER_PREFIX) == 0)
        {
            this->addCategorizationFilter(value);
        }
        else if (level1Key.length() > INFLUENCER_PREFIX.length() &&
                 level1Key.compare(0, INFLUENCER_PREFIX.length(), INFLUENCER_PREFIX) == 0)
        {
            this->addInfluencerFieldName(value);
        }
        else if (level1Key.length() > FILTER_PREFIX.length() &&
                 level1Key.compare(0, FILTER_PREFIX.length(), FILTER_PREFIX) == 0)
        {
            this->processFilter(level1Key, value);
        }
        else if (level1Key.length() > SCHEDULED_EVENT_PREFIX.length() &&
                 level1Key.compare(0, SCHEDULED_EVENT_PREFIX.length(), SCHEDULED_EVENT_PREFIX) == 0)
        {
            this->processScheduledEvent(propTree, level1Key, value, handledScheduledEvents);
        }
        else
        {
            LOG_ERROR("Invalid setting " << level1Key << " = " << value << " in config file " << configFile);
            return false;
        }
    }

    this->sortInfluencers();

    return true;
}

bool CFieldConfig::tokenise(const std::string &clause,
                            TStrVec &copyTokens)
{
    // Tokenise on spaces or commas. Double quotes are used
    // for quoting, and the escape character is a backslash.
    using TCharEscapedListSeparator = boost::escaped_list_separator<char>;
    TCharEscapedListSeparator els("\\",
                                  core::CStringUtils::WHITESPACE_CHARS + ',',
                                  "\"");
    try
    {
        using TCharEscapedListSeparatorTokenizer = boost::tokenizer<TCharEscapedListSeparator>;
        TCharEscapedListSeparatorTokenizer tokenizer(clause, els);

        for (TCharEscapedListSeparatorTokenizer::iterator iter = tokenizer.begin();
             iter != tokenizer.end();
             ++iter)
        {
            const std::string &token = *iter;
            if (token.empty())
            {
                // boost::escaped_list_separator creates empty tokens for
                // multiple adjacent separators, but we don't want these
                continue;
            }

            copyTokens.push_back(token);
            LOG_TRACE(token);
        }
    }
    catch (boost::escaped_list_error &e)
    {
        LOG_ERROR("Cannot parse clause " << clause << ": " << e.what());
        return false;
    }

    return true;
}

void CFieldConfig::retokenise(const TStrVec &tokens,
                              TStrVec &copyTokens)
{
    for (const auto &token : tokens)
    {
        size_t commaPos(token.find(','));
        if (commaPos == std::string::npos)
        {
            copyTokens.push_back(token);
        }
        else
        {
            size_t startPos(0);
            do
            {
                if (commaPos > startPos)
                {
                    copyTokens.resize(copyTokens.size() + 1);
                    copyTokens.back().assign(token,
                                             startPos,
                                             commaPos - startPos);
                }

                startPos = commaPos + 1;
                commaPos = token.find(',', startPos);
            }
            while (commaPos != std::string::npos);

            if (startPos < token.length())
            {
                copyTokens.resize(copyTokens.size() + 1);
                copyTokens.back().assign(token,
                                         startPos,
                                         token.length() - startPos);
            }
        }
    }

    for (const auto &copyToken : copyTokens)
    {
        LOG_DEBUG(copyToken);
    }
}

bool CFieldConfig::findLastByOverTokens(const TStrVec &copyTokens,
                                        std::size_t &lastByTokenIndex,
                                        std::size_t &lastOverTokenIndex)
{
    for (size_t index = 0; index < copyTokens.size(); ++index)
    {
        if (copyTokens[index].length() == BY_TOKEN.length() &&
            core::CStrCaseCmp::strCaseCmp(copyTokens[index].c_str(), BY_TOKEN.c_str()) == 0)
        {
            if (lastByTokenIndex != copyTokens.size())
            {
                LOG_ERROR("Multiple '" << copyTokens[lastByTokenIndex] <<
                          "' tokens in analysis clause - tokens " <<
                          core::CStringUtils::typeToString(1 + lastByTokenIndex) <<
                          " and " <<
                          core::CStringUtils::typeToString(1 + index));
                return false;
            }

            lastByTokenIndex = index;
        }

        if (copyTokens[index].length() == OVER_TOKEN.length() &&
            core::CStrCaseCmp::strCaseCmp(copyTokens[index].c_str(), OVER_TOKEN.c_str()) == 0)
        {
            if (lastOverTokenIndex != copyTokens.size())
            {
                LOG_ERROR("Multiple '" << copyTokens[lastOverTokenIndex] <<
                          "' tokens in analysis clause - tokens " <<
                          core::CStringUtils::typeToString(1 + lastOverTokenIndex) <<
                          " and " << core::CStringUtils::typeToString(1 + index));
                return false;
            }

            lastOverTokenIndex = index;
        }
    }
    return true;
}

bool CFieldConfig::validateByOverField(const TStrVec &copyTokens,
                                       const std::size_t thisIndex,
                                       const std::size_t otherIndex,
                                       const TStrVec &clashingNames,
                                       std::string &fieldName)
{
    if (thisIndex != copyTokens.size())
    {
        if (thisIndex == 0)
        {
            LOG_ERROR("Analysis clause begins with a '" <<
                      copyTokens[thisIndex] << "' token");
            return false;
        }

        if (thisIndex + 1 == copyTokens.size() ||
            thisIndex + 1 == otherIndex)
        {
            LOG_ERROR("No field name follows the '" <<
                      copyTokens[thisIndex] <<
                      "' token in the analysis clause");
            return false;
        }

        if (thisIndex + 2 < copyTokens.size() &&
            thisIndex + 2 < otherIndex)
        {
            LOG_ERROR("Only one field name may follow the '" <<
                      copyTokens[thisIndex] <<
                      "' token in the analysis clause");
            return false;
        }

        fieldName = copyTokens[thisIndex + 1];
        for (const auto &clashingName : clashingNames)
        {
            if (fieldName == clashingName)
            {
                LOG_ERROR("The '" << copyTokens[thisIndex] <<
                          "' field cannot be " << fieldName);
                return false;
            }
        }
        this->seenField(fieldName);
    }
    return true;
}

std::string CFieldConfig::findParameter(const std::string &parameter,
                                        TStrVec &copyTokens)
{
    for (TStrVecItr iter = copyTokens.begin(); iter != copyTokens.end(); ++iter)
    {
        const std::string &token = *iter;
        std::size_t equalPos = token.find('=');
        if (equalPos == parameter.length() &&
            core::CStrCaseCmp::strNCaseCmp(parameter.c_str(),
                                           token.c_str(),
                                           equalPos) == 0)
        {
            std::string value(token, equalPos + 1, token.length() - equalPos);
            LOG_TRACE("Found parameter " << parameter << " : " << value);
            copyTokens.erase(iter);
            return value;
        }
    }
    return std::string();
}

bool CFieldConfig::initFromClause(const TStrVec &tokens)
{
    m_FieldOptions.clear();
    m_FieldNameSuperset.clear();
    m_CategorizationFilters.clear();
    m_Influencers.clear();
    m_DetectorRules.clear();
    m_RuleFilters.clear();

    // The input tokens will have been split up by the command line parser, i.e.
    // tokenised on whitespace.  However, we needed them tokenised by whitespace
    // and/or commas, so here we split them again on the commas.
    TStrVec copyTokens;
    this->retokenise(tokens, copyTokens);
    if (copyTokens.empty())
    {
        LOG_ERROR("No fields specified for analysis");
        return false;
    }

    std::string defaultCategorizationFieldName;
    std::string summaryCountFieldName;

    if (this->parseClause(true,
                          0,
                          EMPTY_STRING,
                          copyTokens,
                          m_FieldOptions,
                          defaultCategorizationFieldName,
                          summaryCountFieldName) == false)
    {
        // parseClause() will have logged the problem
        return false;
    }

    if (!defaultCategorizationFieldName.empty())
    {
        m_CategorizationFieldName.swap(defaultCategorizationFieldName);
    }
    if (!summaryCountFieldName.empty())
    {
        m_SummaryCountFieldName.swap(summaryCountFieldName);
    }

    this->sortInfluencers();

    return true;
}

bool CFieldConfig::addOptions(const CFieldOptions &options)
{
    using TFieldOptionsMIndexItrBoolPr = std::pair<TFieldOptionsMIndexItr, bool>;
    TFieldOptionsMIndexItrBoolPr result(m_FieldOptions.insert(options));
    if (result.second == false)
    {
        LOG_ERROR("Duplicate config found: " << options << core_t::LINE_ENDING
             << "It clashes with config " << *result.first);
        return false;
    }

    this->seenField(options.fieldName());
    this->seenField(options.overFieldName());
    this->seenField(options.byFieldName());
    this->seenField(options.partitionFieldName());

    return true;
}

bool CFieldConfig::parseClause(bool allowMultipleFunctions,
                               int configKey,
                               const std::string &description,
                               TStrVec &copyTokens,
                               TFieldOptionsMIndex &optionsIndex,
                               std::string &categorizationFieldName,
                               std::string &summaryCountFieldName)
{
    std::string partitionFieldName = this->findParameter(PARTITION_FIELD_OPTION,
                                                         copyTokens);

    // Allow any number of influencerfield arguments
    std::string influencerFieldName = this->findParameter(INFLUENCER_FIELD_OPTION,
                                                          copyTokens);
    while (!influencerFieldName.empty())
    {
        this->addInfluencerFieldName(influencerFieldName);
        influencerFieldName = this->findParameter(INFLUENCER_FIELD_OPTION,
                                                  copyTokens);
    }

    categorizationFieldName =
        this->findParameter(CATEGORIZATION_FIELD_OPTION, copyTokens);

    if (!categorizationFieldName.empty())
    {
        this->seenField(categorizationFieldName);
    }

    summaryCountFieldName =
        this->findParameter(SUMMARY_COUNT_FIELD_OPTION, copyTokens);
    this->seenField(summaryCountFieldName);

    std::string useNullStr = this->findParameter(USE_NULL_OPTION, copyTokens);
    bool useNull(false);
    if (!useNullStr.empty() &&
        core::CStringUtils::stringToType(useNullStr, useNull) == false)
    {
        LOG_ERROR("Cannot convert usenull value to boolean: " << useNullStr);
        return false;
    }

    std::string excludeFrequentString = this->findParameter(EXCLUDE_FREQUENT_OPTION, copyTokens);

    // Check for 'by' and 'over' tokens (there should only be one but we have to
    // check all tokens so that we can report errors)
    size_t lastByTokenIndex(copyTokens.size());
    size_t lastOverTokenIndex(copyTokens.size());
    if (!this->findLastByOverTokens(copyTokens,
                                    lastByTokenIndex,
                                    lastOverTokenIndex))
    {
        return false;
    }

    TStrVec clashingNames;
    clashingNames.push_back(COUNT_NAME);
    clashingNames.push_back(partitionFieldName);
    std::string byFieldName;
    if (!this->validateByOverField(copyTokens,
                                   lastByTokenIndex,
                                   lastOverTokenIndex,
                                   clashingNames,
                                   byFieldName))
    {
        return false;
    }

    std::string overFieldName;
    clashingNames.push_back(byFieldName);
    if (!this->validateByOverField(copyTokens,
                                   lastOverTokenIndex,
                                   lastByTokenIndex,
                                   clashingNames,
                                   overFieldName))
    {
        return false;
    }

    bool isPopulation(!overFieldName.empty());
    bool hasByField(!byFieldName.empty());

    //! Validate the "excludefrequent" flag if it has been set
    bool byExcludeFrequent(false);
    bool overExcludeFrequent(false);
    if (this->decipherExcludeFrequentSetting(excludeFrequentString,
                                             hasByField,
                                             isPopulation,
                                             byExcludeFrequent,
                                             overExcludeFrequent) == false)
    {
        LOG_ERROR("Unknown setting for excludefrequent: " << excludeFrequentString);
        return false;
    }

    int tokenNum(0);
    size_t stop(std::min(lastByTokenIndex, lastOverTokenIndex));
    if (stop > 1 && !allowMultipleFunctions)
    {
        LOG_ERROR("Only one analysis function is allowed in this context but " <<
                  core::CStringUtils::typeToString(stop) <<
                  " were specified");
        return false;
    }

    for (size_t index = 0; index < stop; ++index)
    {
        model::function_t::EFunction function;
        std::string fieldName;
        if (this->parseFieldString(!summaryCountFieldName.empty(),
                                   isPopulation,
                                   hasByField,
                                   copyTokens[index],
                                   function,
                                   fieldName) == false)
        {
            LOG_ERROR("Failed to process token '" << copyTokens[index] << "'");

            // External error reporting is done within parseFieldString() so
            // don't do it again here

            return false;
        }

        CFieldOptions options(function,
                              fieldName,
                              allowMultipleFunctions ? ++tokenNum : configKey,
                              byFieldName,
                              overFieldName,
                              partitionFieldName,
                              byExcludeFrequent,
                              overExcludeFrequent,
                              useNull);
        if (!description.empty())
        {
            options.description(description);
        }

        using TFieldOptionsMIndexItrBoolPr = std::pair<TFieldOptionsMIndexItr, bool>;
        TFieldOptionsMIndexItrBoolPr result(optionsIndex.insert(options));
        if (result.second == false)
        {
            LOG_ERROR("Token " <<
                      core::CStringUtils::typeToString(options.configKey()) <<
                      " in the analysis clause is a duplicate of token " <<
                      result.first->configKey());
            return false;
        }

        this->seenField(fieldName);
    }

    this->seenField(partitionFieldName);

    return true;
}

bool CFieldConfig::parseRules(int detectorIndex, const std::string &rules)
{
    return parseRules(m_DetectorRules[detectorIndex], rules);
}

bool CFieldConfig::parseRules(TDetectionRuleVec &detectionRules, const std::string &rules)
{
    if (rules.empty())
    {
        return true;
    }

    CDetectionRulesJsonParser rulesParser(m_RuleFilters);
    return rulesParser.parseRules(rules, detectionRules);
}

const CFieldConfig::TFieldOptionsMIndex &CFieldConfig::fieldOptions(void) const
{
    return m_FieldOptions;
}

const std::string &CFieldConfig::categorizationFieldName(void) const
{
    return m_CategorizationFieldName;
}

const CFieldConfig::TStrPatternSetUMap &CFieldConfig::ruleFilters(void) const
{
    return m_RuleFilters;
}

const CFieldConfig::TStrVec &CFieldConfig::categorizationFilters(void) const
{
    return m_CategorizationFilters;
}

const std::string &CFieldConfig::summaryCountFieldName(void) const
{
    return m_SummaryCountFieldName;
}

bool CFieldConfig::havePartitionFields(void) const
{
    for (const auto &fieldOption : m_FieldOptions)
    {
        if (!fieldOption.partitionFieldName().empty())
        {
            return true;
        }
    }
    return false;
}

const CFieldConfig::TStrSet &CFieldConfig::fieldNameSuperset(void) const
{
    return m_FieldNameSuperset;
}

bool CFieldConfig::processDetector(const boost::property_tree::ptree &propTree,
                                   const std::string &key,
                                   const std::string &value,
                                   TIntSet &handledConfigs)
{
    // Drive the map population off the first setting in the file that is for a
    // particular detector

    // Here we pull out the "1" in "detector.1.clause"
    size_t sepPos(key.rfind(SUFFIX_SEPARATOR));
    if (sepPos == std::string::npos ||
        sepPos <= DETECTOR_PREFIX.length() ||
        sepPos == key.length() - 1)
    {
        LOG_ERROR("Unrecognised configuration option " << key << " = " << value);
        return false;
    }
    std::string configKeyString(key,
                                DETECTOR_PREFIX.length(),
                                sepPos - DETECTOR_PREFIX.length());
    int configKey;
    if (core::CStringUtils::stringToType(configKeyString, configKey) == false)
    {
        LOG_ERROR("Cannot convert config key to integer: " << configKeyString);
        return false;
    }

    // Check if we've already seen this key
    if (handledConfigs.insert(configKey).second == false)
    {
        // Not an error
        return true;
    }

    std::string description(propTree.get(boost::property_tree::ptree::path_type(DETECTOR_PREFIX + configKeyString + DESCRIPTION_SUFFIX, '\t'),
                                         EMPTY_STRING));

    std::string clause(propTree.get(boost::property_tree::ptree::path_type(DETECTOR_PREFIX + configKeyString + CLAUSE_SUFFIX, '\t'),
                                    EMPTY_STRING));

    std::string rules(propTree.get(boost::property_tree::ptree::path_type(DETECTOR_PREFIX + configKeyString + RULES_SUFFIX, '\t'),
                                    EMPTY_STRING));

    TStrVec tokens;
    if (this->tokenise(clause, tokens) == false)
    {
        // tokenise() will already have logged the error
        return false;
    }

    // This is an active configuration
    if (this->addActiveDetector(configKey,
                                description,
                                rules,
                                tokens) == false)
    {
        return false;
    }

    return true;
}

bool CFieldConfig::addActiveDetector(int configKey,
                                     const std::string &description,
                                     const std::string &rules,
                                     TStrVec &copyTokens)
{
    std::string categorizationFieldName;
    std::string summaryCountFieldName;

    if (this->parseClause(false,
                          configKey,
                          description,
                          copyTokens,
                          m_FieldOptions,
                          categorizationFieldName,
                          summaryCountFieldName) == false)
    {
        // parseClause() will have logged the error
        return false;
    }

    if (!categorizationFieldName.empty())
    {
        m_CategorizationFieldName.swap(categorizationFieldName);
    }
    if (!summaryCountFieldName.empty())
    {
        m_SummaryCountFieldName.swap(summaryCountFieldName);
    }

    TDetectionRuleVec &detectionRules = m_DetectorRules[configKey];
    if (this->parseRules(detectionRules, rules) == false)
    {
        // parseRules() will have logged the error
        return false;
    }

    return true;
}

bool CFieldConfig::parseFieldString(bool haveSummaryCountField,
                                    bool isPopulation,
                                    bool hasByField,
                                    const std::string &str,
                                    model::function_t::EFunction &function,
                                    std::string &fieldName)
{
    // Parse using a regex
    core::CRegex regex;

    // Match:
    // count
    // count()
    // rnzc(category)
    // dc(category)
    // metric - i.e. any name not recognised as a function name is assumed to be
    //          a metric field name
    // etc.
    std::string regexStr("([^()]+)(?:\\((.*)\\))?");
    if (!regex.init(regexStr))
    {
        LOG_FATAL("Unable to init regex " << regexStr);
        return false;
    }

    core::CRegex::TStrVec tokens;

    if (regex.tokenise(str, tokens) == false)
    {
        LOG_ERROR("Unable to parse a function from " << str);
        return false;
    }

    if (tokens.size() != 2)
    {
        LOG_INFO("Got wrong number of tokens:: " << tokens.size());
        return false;
    }

    // Overall string "x(y)" => outerToken is "x" and innerToken is "y"
    // Overall string "x"    => outerToken is "x" and innerToken is empty
    // Overall string "x()"  => outerToken is "x" and innerToken is empty
    const std::string &outerToken = tokens[0];
    const std::string &innerToken = tokens[1];

    // Some functions must take an argument, some mustn't and for the rest it's
    // optional.  Validate this based on the contents of these flags after
    // determining the function.  Similarly for by fields.
    bool argumentRequired(false);
    bool argumentInvalid(false);
    bool byFieldRequired(false);
    bool byFieldInvalid(false);

    if (outerToken == FUNCTION_COUNT ||
        outerToken == FUNCTION_COUNT_ABBREV)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationCount :
                   model::function_t::E_IndividualRareCount;
        argumentInvalid = true;
    }
    else if (outerToken == FUNCTION_DISTINCT_COUNT ||
             outerToken == FUNCTION_DISTINCT_COUNT_ABBREV)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationDistinctCount :
                   model::function_t::E_IndividualDistinctCount;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_LOW_DISTINCT_COUNT ||
             outerToken == FUNCTION_LOW_DISTINCT_COUNT_ABBREV)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationLowDistinctCount :
                   model::function_t::E_IndividualLowDistinctCount;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_HIGH_DISTINCT_COUNT ||
             outerToken == FUNCTION_HIGH_DISTINCT_COUNT_ABBREV)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationHighDistinctCount :
                   model::function_t::E_IndividualHighDistinctCount;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_NON_ZERO_COUNT ||
             outerToken == FUNCTION_NON_ZERO_COUNT_ABBREV)
    {
        function = model::function_t::E_IndividualNonZeroCount;
        argumentInvalid = true;
    }
    else if (outerToken == FUNCTION_RARE_NON_ZERO_COUNT ||
             outerToken == FUNCTION_RARE_NON_ZERO_COUNT_ABBREV)
    {
        function = model::function_t::E_IndividualRareNonZeroCount;
        argumentInvalid = true;
        byFieldRequired = true;
    }
    else if (outerToken == FUNCTION_RARE)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationRare :
                   model::function_t::E_IndividualRare;
        argumentInvalid = true;
        byFieldRequired = true;
    }
    else if (outerToken == FUNCTION_RARE_COUNT)
    {
        function = model::function_t::E_PopulationRareCount;
        argumentInvalid = true;
        byFieldRequired = true;
    }
    else if (outerToken == FUNCTION_LOW_COUNT ||
             outerToken == FUNCTION_LOW_COUNT_ABBREV)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationLowCounts :
                   model::function_t::E_IndividualLowCounts;
        argumentInvalid = true;
    }
    else if (outerToken == FUNCTION_HIGH_COUNT ||
             outerToken == FUNCTION_HIGH_COUNT_ABBREV)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationHighCounts :
                   model::function_t::E_IndividualHighCounts;
        argumentInvalid = true;
    }
    else if (outerToken == FUNCTION_LOW_NON_ZERO_COUNT ||
             outerToken == FUNCTION_LOW_NON_ZERO_COUNT_ABBREV)
    {
        function = model::function_t::E_IndividualLowNonZeroCount;
        argumentInvalid = true;
    }
    else if (outerToken == FUNCTION_HIGH_NON_ZERO_COUNT ||
             outerToken == FUNCTION_HIGH_NON_ZERO_COUNT_ABBREV)
    {
        function = model::function_t::E_IndividualHighNonZeroCount;
        argumentInvalid = true;
    }
    else if (outerToken == FUNCTION_FREQ_RARE ||
             outerToken == FUNCTION_FREQ_RARE_ABBREV)
    {
        function = model::function_t::E_PopulationFreqRare;
        argumentInvalid = true;
        byFieldRequired = true;
    }
    else if (outerToken == FUNCTION_FREQ_RARE_COUNT ||
             outerToken == FUNCTION_FREQ_RARE_COUNT_ABBREV)
    {
        function = model::function_t::E_PopulationFreqRareCount;
        argumentInvalid = true;
        byFieldRequired = true;
    }
    else if (outerToken == FUNCTION_INFO_CONTENT)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationInfoContent :
                   model::function_t::E_IndividualInfoContent;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_LOW_INFO_CONTENT)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationLowInfoContent :
                   model::function_t::E_IndividualLowInfoContent;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_HIGH_INFO_CONTENT)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationHighInfoContent :
                   model::function_t::E_IndividualHighInfoContent;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_METRIC)
    {
        if (haveSummaryCountField)
        {
            LOG_ERROR("Function " << outerToken <<
                      "() cannot be used with a summary count field");
            return false;
        }

        function = isPopulation ?
                   model::function_t::E_PopulationMetric :
                   model::function_t::E_IndividualMetric;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_AVERAGE ||
             outerToken == FUNCTION_MEAN)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationMetricMean :
                   model::function_t::E_IndividualMetricMean;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_LOW_AVERAGE ||
             outerToken == FUNCTION_LOW_MEAN)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationMetricLowMean :
                   model::function_t::E_IndividualMetricLowMean;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_HIGH_AVERAGE ||
             outerToken == FUNCTION_HIGH_MEAN)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationMetricHighMean :
                   model::function_t::E_IndividualMetricHighMean;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_MEDIAN)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationMetricMedian :
                   model::function_t::E_IndividualMetricMedian;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_LOW_MEDIAN)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationMetricLowMedian :
                   model::function_t::E_IndividualMetricLowMedian;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_HIGH_MEDIAN)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationMetricHighMedian :
                   model::function_t::E_IndividualMetricHighMedian;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_MIN)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationMetricMin :
                   model::function_t::E_IndividualMetricMin;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_MAX)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationMetricMax :
                   model::function_t::E_IndividualMetricMax;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_VARIANCE)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationMetricVariance :
                   model::function_t::E_IndividualMetricVariance;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_LOW_VARIANCE)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationMetricLowVariance :
                   model::function_t::E_IndividualMetricLowVariance;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_HIGH_VARIANCE)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationMetricHighVariance :
                   model::function_t::E_IndividualMetricHighVariance;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_SUM)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationMetricSum :
                   model::function_t::E_IndividualMetricSum;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_LOW_SUM)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationMetricLowSum :
                   model::function_t::E_IndividualMetricLowSum;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_HIGH_SUM)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationMetricHighSum :
                   model::function_t::E_IndividualMetricHighSum;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_NON_NULL_SUM ||
             outerToken == FUNCTION_NON_NULL_SUM_ABBREV)
    {
        function = model::function_t::E_IndividualMetricNonNullSum;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_LOW_NON_NULL_SUM ||
             outerToken == FUNCTION_LOW_NON_NULL_SUM_ABBREV)
    {
        function = model::function_t::E_IndividualMetricLowNonNullSum;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_HIGH_NON_NULL_SUM ||
             outerToken == FUNCTION_HIGH_NON_NULL_SUM_ABBREV)
    {
        function = model::function_t::E_IndividualMetricHighNonNullSum;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_TIME_OF_DAY)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationTimeOfDay :
                   model::function_t::E_IndividualTimeOfDay;
        argumentRequired = false;
        argumentInvalid = true;
    }
    else if (outerToken == FUNCTION_TIME_OF_WEEK)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationTimeOfWeek :
                   model::function_t::E_IndividualTimeOfWeek;
        argumentRequired = false;
        argumentInvalid = true;
    }
    else if (outerToken == FUNCTION_LAT_LONG)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationLatLong:
                   model::function_t::E_IndividualLatLong;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_MAX_VELOCITY)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationMaxVelocity:
                   model::function_t::E_IndividualMaxVelocity;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_MIN_VELOCITY)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationMinVelocity:
                   model::function_t::E_IndividualMinVelocity;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_MEAN_VELOCITY)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationMeanVelocity:
                   model::function_t::E_IndividualMeanVelocity;
        argumentRequired = true;
    }
    else if (outerToken == FUNCTION_SUM_VELOCITY)
    {
        function = isPopulation ?
                   model::function_t::E_PopulationSumVelocity:
                   model::function_t::E_IndividualSumVelocity;
        argumentRequired = true;
    }
    else
    {
        // We expect an individual metric here, but if the original string
        // contained brackets then there's probably been a typo because a metric
        // name should not be followed by brackets
        if (str.find('(') != std::string::npos)
        {
            LOG_ERROR(outerToken << "() is not a known function");
            return false;
        }

        if (haveSummaryCountField)
        {
            LOG_ERROR("Implicit function metric() cannot be "
                      "used with a summary count field");
            return false;
        }

        function = isPopulation ?
                   model::function_t::E_PopulationMetric :
                   model::function_t::E_IndividualMetric;

        // This is inconsistent notation, but kept for backwards compatibility
        fieldName = outerToken;

        return true;
    }

    // Validate
    if (model::function_t::isPopulation(function) && !isPopulation)
    {
        LOG_ERROR("Function " << outerToken <<
                  "() requires an 'over' field");
        return false;
    }

    if (isPopulation && !model::function_t::isPopulation(function))
    {
        LOG_ERROR("Function " << outerToken <<
                  "() cannot be used with an 'over' field");
        return false;
    }

    if (byFieldRequired && !hasByField)
    {
        LOG_ERROR("Function " << outerToken << "() requires a 'by' field");
        return false;
    }

    if (byFieldInvalid && hasByField)
    {
        LOG_ERROR("Function " << outerToken <<
                  "() cannot be used with a 'by' field");
        return false;
    }

    if (argumentRequired && innerToken.empty())
    {
        LOG_ERROR("Function " << outerToken << "() requires an argument");
        return false;
    }

    if (argumentInvalid && !innerToken.empty())
    {
        LOG_ERROR("Function " << outerToken << "() does not take an argument");
        return false;
    }

    fieldName = innerToken;

    return true;
}

void CFieldConfig::seenField(const std::string &fieldName)
{
    if (fieldName.empty())
    {
        return;
    }

    // Duplicates are expected here so don't log them
    m_FieldNameSuperset.insert(fieldName);
}

std::string CFieldConfig::debug(void) const
{
    std::ostringstream strm;

    bool needLineBreak(false);

    if (!m_FieldOptions.empty())
    {
        if (needLineBreak)
        {
            strm << core_t::LINE_ENDING;
        }
        this->debug(m_FieldOptions, strm);
    }

    return strm.str();
}

void CFieldConfig::debug(const TFieldOptionsMIndex &fieldOptions,
                         std::ostream &strm) const
{
    for (const auto &fieldOption : fieldOptions)
    {
        strm << fieldOption << '|';
    }
}

bool CFieldConfig::decipherExcludeFrequentSetting(const std::string &excludeFrequentString,
                                                  bool hasByField,
                                                  bool isPopulation,
                                                  bool &byExcludeFrequent,
                                                  bool &overExcludeFrequent)
{
    byExcludeFrequent = false;
    overExcludeFrequent = false;

    if (!excludeFrequentString.empty())
    {
        if (excludeFrequentString.length() == ALL_TOKEN.length() &&
            core::CStrCaseCmp::strCaseCmp(excludeFrequentString.c_str(), ALL_TOKEN.c_str()) == 0)
        {
            byExcludeFrequent = hasByField;
            overExcludeFrequent = isPopulation;
        }
        else if (excludeFrequentString.length() == BY_TOKEN.length() &&
                 core::CStrCaseCmp::strCaseCmp(excludeFrequentString.c_str(), BY_TOKEN.c_str()) == 0)
        {
            byExcludeFrequent = hasByField;
        }
        else if (excludeFrequentString.length() == OVER_TOKEN.length() &&
                 core::CStrCaseCmp::strCaseCmp(excludeFrequentString.c_str(), OVER_TOKEN.c_str()) == 0)
        {
            overExcludeFrequent = isPopulation;
        }
        else
        {
            if (excludeFrequentString.length() != NONE_TOKEN.length() ||
                core::CStrCaseCmp::strCaseCmp(excludeFrequentString.c_str(), NONE_TOKEN.c_str()) != 0)
            {
                LOG_ERROR("Unexpected excludeFrequent value = " << excludeFrequentString);
                return false;
            }
        }
    }

    return true;
}

void CFieldConfig::addInfluencerFieldsFromByOverPartitionFields(void)
{
    this->addInfluencerFieldsFromByOverPartitionFields(m_FieldOptions);
}

void CFieldConfig::addInfluencerFieldsFromByOverPartitionFields(const TFieldOptionsMIndex &fieldOptions)
{
    for (const auto &fieldOption : fieldOptions)
    {
        this->addInfluencerFieldName(fieldOption.byFieldName(), true);
        this->addInfluencerFieldName(fieldOption.overFieldName(), true);
        this->addInfluencerFieldName(fieldOption.partitionFieldName(), true);
    }
}

const CFieldConfig::TStrVec &CFieldConfig::influencerFieldNames(void) const
{
    return m_Influencers;
}

const CFieldConfig::TIntDetectionRuleVecUMap &CFieldConfig::detectionRules(void) const
{
    return m_DetectorRules;
}

const CFieldConfig::TStrDetectionRulePrVec &CFieldConfig::scheduledEvents(void) const
{
    return m_ScheduledEvents;
}

void CFieldConfig::influencerFieldNames(TStrVec influencers)
{
    LOG_DEBUG("Set influencers : " << core::CContainerPrinter::print(influencers));
    std::for_each(influencers.begin(),
                  influencers.end(),
                  boost::bind(&CFieldConfig::seenField, this, _1));
    m_Influencers.swap(influencers);
}

void CFieldConfig::addInfluencerFieldName(const std::string &influencer,
                                          bool quiet)
{
    if (influencer.empty())
    {
        if (!quiet)
        {
            LOG_WARN("Ignoring blank influencer field");
        }
        return;
    }

    if (std::find(m_Influencers.begin(),
                  m_Influencers.end(),
                  influencer) == m_Influencers.end())
    {
        LOG_TRACE("Add influencer : " << influencer);
        this->seenField(influencer);
        m_Influencers.push_back(influencer);
    }
}

void CFieldConfig::sortInfluencers(void)
{
    std::sort(m_Influencers.begin(), m_Influencers.end());
}

void CFieldConfig::addCategorizationFilter(const std::string &filter)
{
    if (filter.empty())
    {
        LOG_WARN("Ignoring blank categorization filter");
        return;
    }

    TStrVec tokens;
    this->tokenise(filter, tokens);

    if (tokens.size() != 1)
    {
        LOG_ERROR("Unexpected number of tokens: " << tokens.size()
                  << "; ignoring categorization filter: " << filter);
        return;
    }

    m_CategorizationFilters.push_back(tokens[0]);
}

bool CFieldConfig::processFilter(const std::string &key,
                                 const std::string &value)
{
    // expected format is filter.<filterId>=[json, array]
    size_t sepPos(key.find(SUFFIX_SEPARATOR));
    if (sepPos == std::string::npos)
    {
        LOG_ERROR("Unrecognised filter key: " + key);
        return false;
    }
    std::string filterId = key.substr(sepPos + 1);
    core::CPatternSet &filter = m_RuleFilters[filterId];
    return filter.initFromJson(value);
}

bool CFieldConfig::updateFilters(const boost::property_tree::ptree &propTree)
{
    for (const auto &filterEntry : propTree)
    {
        const std::string &key = filterEntry.first;
        const std::string &value = filterEntry.second.data();
        if (this->processFilter(key, value) == false)
        {
            return false;
        }
    }
    return true;
}

bool CFieldConfig::processScheduledEvent(const boost::property_tree::ptree &propTree,
                                         const std::string &key,
                                         const std::string &value,
                                         TIntSet &handledScheduledEvents)
{
    // Here we pull out the "1" in "scheduledevent.1.description"
    // description may contain a '.'
    size_t sepPos(key.find(SUFFIX_SEPARATOR, SCHEDULED_EVENT_PREFIX.length() + 1));
    if (sepPos == std::string::npos ||
        sepPos == key.length() - 1)
    {
        LOG_ERROR("Unrecognised configuration option " << key << " = " << value);
        return false;
    }

    std::string indexString(key, SCHEDULED_EVENT_PREFIX.length(), sepPos - SCHEDULED_EVENT_PREFIX.length());
    int indexKey;
    if (core::CStringUtils::stringToType(indexString, indexKey) == false)
    {
        LOG_ERROR("Cannot convert config key to integer: " << indexString);
        return false;
    }

    // Check if we've already seen this key
    if (handledScheduledEvents.insert(indexKey).second == false)
    {
        // Not an error
        return true;
    }

    std::string description(propTree.get(boost::property_tree::ptree::path_type(SCHEDULED_EVENT_PREFIX + indexString + DESCRIPTION_SUFFIX, '\t'),
                                         EMPTY_STRING));

    std::string rules(propTree.get(boost::property_tree::ptree::path_type(SCHEDULED_EVENT_PREFIX + indexString + RULES_SUFFIX, '\t'),
                                    EMPTY_STRING));

    TDetectionRuleVec detectionRules;
    if (this->parseRules(detectionRules, rules) == false)
    {
        // parseRules() will have logged the error
        return false;
    }

    if (detectionRules.size() != 1)
    {
        LOG_ERROR("Scheduled events must have exactly 1 rule");
        return false;
    }

    m_ScheduledEvents.emplace_back(description, detectionRules[0]);

    return true;
}

bool CFieldConfig::updateScheduledEvents(const boost::property_tree::ptree &propTree)
{
    m_ScheduledEvents.clear();

    bool isClear = propTree.get(CLEAR, false);
    if (isClear)
    {
        return true;
    }

    TIntSet handledScheduledEvents;

    for (const auto &scheduledEventEntry : propTree)
    {
        const std::string &key = scheduledEventEntry.first;
        const std::string &value = scheduledEventEntry.second.data();
        if (this->processScheduledEvent(propTree, key, value, handledScheduledEvents) == false)
        {
            return false;
        }
    }
    return true;
}

CFieldConfig::CFieldOptions::CFieldOptions(const std::string &fieldName,
                                           int configKey)
    : m_Function(fieldName == COUNT_NAME ?
                 model::function_t::E_IndividualRareCount :
                 model::function_t::E_IndividualMetric),
      m_FieldName(fieldName),
      m_ConfigKey(configKey),
      m_ByHasExcludeFrequent(false),
      m_OverHasExcludeFrequent(false),
      m_UseNull(true)
{
}

CFieldConfig::CFieldOptions::CFieldOptions(const std::string &fieldName,
                                           int configKey,
                                           const std::string &byFieldName,
                                           bool byHasExcludeFrequent,
                                           bool useNull)
    // For historical reasons, the only function name we interpret in this
    // constructor is "count" - every other word is considered to be a metric
    // field name.
    : m_Function(fieldName == COUNT_NAME ?
                 model::function_t::E_IndividualRareCount :
                 model::function_t::E_IndividualMetric),
      m_FieldName(fieldName == COUNT_NAME ?
                  EMPTY_STRING :
                  fieldName),
      m_ConfigKey(configKey),
      m_ByFieldName(byFieldName),
      m_ByHasExcludeFrequent(byHasExcludeFrequent),
      m_OverHasExcludeFrequent(false),
      m_UseNull(useNull)
{
}

CFieldConfig::CFieldOptions::CFieldOptions(const std::string &fieldName,
                                           int configKey,
                                           const std::string &byFieldName,
                                           const std::string &partitionFieldName,
                                           bool byHasExcludeFrequent,
                                           bool overHasExcludeFrequent,
                                           bool useNull)
    // For historical reasons, the only function name we interpret in this
    // constructor is "count" - every other word is considered to be a metric
    // field name.
    : m_Function(fieldName == COUNT_NAME ?
                 model::function_t::E_IndividualRareCount :
                 model::function_t::E_IndividualMetric),
      m_FieldName(fieldName == COUNT_NAME ?
                  EMPTY_STRING :
                  fieldName),
      m_ConfigKey(configKey),
      m_ByFieldName(byFieldName),
      m_PartitionFieldName(partitionFieldName),
      m_ByHasExcludeFrequent(byHasExcludeFrequent),
      m_OverHasExcludeFrequent(overHasExcludeFrequent),
      m_UseNull(useNull)
{
}

CFieldConfig::CFieldOptions::CFieldOptions(model::function_t::EFunction function,
                                           const std::string &fieldName,
                                           int configKey,
                                           const std::string &byFieldName,
                                           const std::string &overFieldName,
                                           const std::string &partitionFieldName,
                                           bool byHasExcludeFrequent,
                                           bool overHasExcludeFrequent,
                                           bool useNull)
    : m_Function(function),
      m_FieldName(fieldName),
      m_ConfigKey(configKey),
      m_ByFieldName(byFieldName),
      m_OverFieldName(overFieldName),
      m_PartitionFieldName(partitionFieldName),
      m_ByHasExcludeFrequent(byHasExcludeFrequent),
      m_OverHasExcludeFrequent(overHasExcludeFrequent),
      m_UseNull(useNull)
{
}

void CFieldConfig::CFieldOptions::description(std::string description)
{
    m_Description.swap(description);
}

const std::string &CFieldConfig::CFieldOptions::description(void) const
{
    return m_Description;
}

model::function_t::EFunction CFieldConfig::CFieldOptions::function(void) const
{
    return m_Function;
}

const std::string &CFieldConfig::CFieldOptions::fieldName(void) const
{
    return m_FieldName;
}

int CFieldConfig::CFieldOptions::configKey(void) const
{
    return m_ConfigKey;
}

const std::string &CFieldConfig::CFieldOptions::byFieldName(void) const
{
    return m_ByFieldName;
}

const std::string &CFieldConfig::CFieldOptions::overFieldName(void) const
{
    return m_OverFieldName;
}

const std::string &CFieldConfig::CFieldOptions::partitionFieldName(void) const
{
    return m_PartitionFieldName;
}

bool CFieldConfig::CFieldOptions::useNull(void) const
{
    return m_UseNull;
}

model_t::EExcludeFrequent CFieldConfig::CFieldOptions::excludeFrequent(void) const
{
    if (m_OverHasExcludeFrequent)
    {
        if (m_ByHasExcludeFrequent)
        {
            return model_t::E_XF_Both;
        }
        else
        {
            return model_t::E_XF_Over;
        }
    }
    else
    {
        if (m_ByHasExcludeFrequent)
        {
            return model_t::E_XF_By;
        }
    }
    return model_t::E_XF_None;
}

const std::string &CFieldConfig::CFieldOptions::terseFunctionName(void) const
{
    switch (m_Function)
    {
        case model::function_t::E_IndividualCount:
            // For backwards compatibility the "count" function name maps to
            // E_IndividualRareCount, not E_IndividualCount as you might think
            return EMPTY_STRING;
        case model::function_t::E_IndividualNonZeroCount:
            return FUNCTION_NON_ZERO_COUNT_ABBREV;
        case model::function_t::E_IndividualRareCount:
            // For backwards compatibility the "count" function name maps to
            // E_IndividualRareCount, not E_IndividualCount as you might think
            return FUNCTION_COUNT_ABBREV;
        case model::function_t::E_IndividualRareNonZeroCount:
            return FUNCTION_RARE_NON_ZERO_COUNT_ABBREV;
        case model::function_t::E_IndividualRare:
            return FUNCTION_RARE;
        case model::function_t::E_IndividualLowCounts:
            return FUNCTION_LOW_COUNT_ABBREV;
        case model::function_t::E_IndividualHighCounts:
            return FUNCTION_HIGH_COUNT_ABBREV;
        case model::function_t::E_IndividualLowNonZeroCount:
            return FUNCTION_LOW_NON_ZERO_COUNT;
        case model::function_t::E_IndividualHighNonZeroCount:
            return FUNCTION_HIGH_NON_ZERO_COUNT;
        case model::function_t::E_IndividualDistinctCount:
            return FUNCTION_DISTINCT_COUNT_ABBREV;
        case model::function_t::E_IndividualLowDistinctCount:
            return FUNCTION_LOW_DISTINCT_COUNT_ABBREV;
        case model::function_t::E_IndividualHighDistinctCount:
            return FUNCTION_HIGH_DISTINCT_COUNT_ABBREV;
        case model::function_t::E_IndividualInfoContent:
            return FUNCTION_INFO_CONTENT;
        case model::function_t::E_IndividualLowInfoContent:
            return FUNCTION_LOW_INFO_CONTENT;
        case model::function_t::E_IndividualHighInfoContent:
            return FUNCTION_HIGH_INFO_CONTENT;
        case model::function_t::E_IndividualTimeOfDay:
            return FUNCTION_TIME_OF_DAY;
        case model::function_t::E_IndividualTimeOfWeek:
            return FUNCTION_TIME_OF_WEEK;
        case model::function_t::E_IndividualMetric:
            return FUNCTION_METRIC;
        case model::function_t::E_IndividualMetricMean:
            return FUNCTION_AVERAGE;
        case model::function_t::E_IndividualMetricLowMean:
            return FUNCTION_LOW_MEAN;
        case model::function_t::E_IndividualMetricHighMean:
            return FUNCTION_HIGH_MEAN;
        case model::function_t::E_IndividualMetricMedian:
            return FUNCTION_MEDIAN;
        case model::function_t::E_IndividualMetricLowMedian:
            return FUNCTION_LOW_MEDIAN;
        case model::function_t::E_IndividualMetricHighMedian:
            return FUNCTION_HIGH_MEDIAN;
        case model::function_t::E_IndividualMetricMin:
            return FUNCTION_MIN;
        case model::function_t::E_IndividualMetricMax:
            return FUNCTION_MAX;
        case model::function_t::E_IndividualMetricVariance:
            return FUNCTION_VARIANCE;
        case model::function_t::E_IndividualMetricLowVariance:
            return FUNCTION_LOW_VARIANCE;
        case model::function_t::E_IndividualMetricHighVariance:
            return FUNCTION_HIGH_VARIANCE;
        case model::function_t::E_IndividualMetricSum:
            return FUNCTION_SUM;
        case model::function_t::E_IndividualMetricLowSum:
            return FUNCTION_LOW_SUM;
        case model::function_t::E_IndividualMetricHighSum:
            return FUNCTION_HIGH_SUM;
        case model::function_t::E_IndividualMetricNonNullSum:
            return FUNCTION_NON_NULL_SUM_ABBREV;
        case model::function_t::E_IndividualMetricLowNonNullSum:
            return FUNCTION_LOW_NON_NULL_SUM_ABBREV;
        case model::function_t::E_IndividualMetricHighNonNullSum:
            return FUNCTION_HIGH_NON_NULL_SUM_ABBREV;
        case model::function_t::E_IndividualLatLong:
            return FUNCTION_LAT_LONG;
        case model::function_t::E_IndividualMinVelocity:
            return FUNCTION_MIN_VELOCITY;
        case model::function_t::E_IndividualMaxVelocity:
            return FUNCTION_MAX_VELOCITY;
        case model::function_t::E_IndividualMeanVelocity:
            return FUNCTION_MEAN_VELOCITY;
        case model::function_t::E_IndividualSumVelocity:
            return FUNCTION_SUM_VELOCITY;
        case model::function_t::E_PopulationCount:
            return FUNCTION_COUNT_ABBREV;
        case model::function_t::E_PopulationDistinctCount:
            return FUNCTION_DISTINCT_COUNT_ABBREV;
        case model::function_t::E_PopulationLowDistinctCount:
            return FUNCTION_LOW_DISTINCT_COUNT_ABBREV;
        case model::function_t::E_PopulationHighDistinctCount:
            return FUNCTION_HIGH_DISTINCT_COUNT_ABBREV;
        case model::function_t::E_PopulationRare:
            return FUNCTION_RARE;
        case model::function_t::E_PopulationRareCount:
            return FUNCTION_RARE_COUNT;
        case model::function_t::E_PopulationFreqRare:
            return FUNCTION_FREQ_RARE_ABBREV;
        case model::function_t::E_PopulationFreqRareCount:
            return FUNCTION_FREQ_RARE_COUNT_ABBREV;
        case model::function_t::E_PopulationLowCounts:
            return FUNCTION_LOW_COUNT_ABBREV;
        case model::function_t::E_PopulationHighCounts:
            return FUNCTION_HIGH_COUNT_ABBREV;
        case model::function_t::E_PopulationInfoContent:
            return FUNCTION_INFO_CONTENT;
        case model::function_t::E_PopulationLowInfoContent:
            return FUNCTION_LOW_INFO_CONTENT;
        case model::function_t::E_PopulationHighInfoContent:
            return FUNCTION_HIGH_INFO_CONTENT;
        case model::function_t::E_PopulationTimeOfDay:
            return FUNCTION_TIME_OF_DAY;
        case model::function_t::E_PopulationTimeOfWeek:
            return FUNCTION_TIME_OF_WEEK;
        case model::function_t::E_PopulationMetric:
            return FUNCTION_METRIC;
        case model::function_t::E_PopulationMetricMean:
            return FUNCTION_AVERAGE;
        case model::function_t::E_PopulationMetricLowMean:
            return FUNCTION_LOW_MEAN;
        case model::function_t::E_PopulationMetricHighMean:
            return FUNCTION_HIGH_MEAN;
        case model::function_t::E_PopulationMetricMedian:
            return FUNCTION_MEDIAN;
        case model::function_t::E_PopulationMetricLowMedian:
            return FUNCTION_LOW_MEDIAN;
        case model::function_t::E_PopulationMetricHighMedian:
            return FUNCTION_HIGH_MEDIAN;
        case model::function_t::E_PopulationMetricMin:
            return FUNCTION_MIN;
        case model::function_t::E_PopulationMetricMax:
            return FUNCTION_MAX;
        case model::function_t::E_PopulationMetricSum:
            return FUNCTION_SUM;
        case model::function_t::E_PopulationMetricVariance:
            return FUNCTION_VARIANCE;
        case model::function_t::E_PopulationMetricLowVariance:
            return FUNCTION_LOW_VARIANCE;
        case model::function_t::E_PopulationMetricHighVariance:
            return FUNCTION_HIGH_VARIANCE;
        case model::function_t::E_PopulationMetricLowSum:
            return FUNCTION_LOW_SUM;
        case model::function_t::E_PopulationMetricHighSum:
            return FUNCTION_HIGH_SUM;
        case model::function_t::E_PopulationLatLong:
            return FUNCTION_LAT_LONG;
        case model::function_t::E_PopulationMinVelocity:
            return FUNCTION_MIN_VELOCITY;
        case model::function_t::E_PopulationMaxVelocity:
            return FUNCTION_MAX_VELOCITY;
        case model::function_t::E_PopulationMeanVelocity:
            return FUNCTION_MEAN_VELOCITY;
        case model::function_t::E_PopulationSumVelocity:
            return FUNCTION_SUM_VELOCITY;
        case model::function_t::E_PeersCount:
            return FUNCTION_COUNT_ABBREV;
        case model::function_t::E_PeersLowCounts:
            return FUNCTION_LOW_COUNT_ABBREV;
        case model::function_t::E_PeersHighCounts:
            return FUNCTION_HIGH_COUNT_ABBREV;
        case model::function_t::E_PeersDistinctCount:
            return FUNCTION_DISTINCT_COUNT_ABBREV;
        case model::function_t::E_PeersLowDistinctCount:
            return FUNCTION_LOW_DISTINCT_COUNT_ABBREV;
        case model::function_t::E_PeersHighDistinctCount:
            return FUNCTION_HIGH_DISTINCT_COUNT_ABBREV;
        case model::function_t::E_PeersInfoContent:
            return FUNCTION_INFO_CONTENT;
        case model::function_t::E_PeersLowInfoContent:
            return FUNCTION_LOW_INFO_CONTENT;
        case model::function_t::E_PeersHighInfoContent:
            return FUNCTION_HIGH_INFO_CONTENT;
        case model::function_t::E_PeersTimeOfDay:
            return FUNCTION_TIME_OF_DAY;
        case model::function_t::E_PeersTimeOfWeek:
            return FUNCTION_TIME_OF_WEEK;
    }

    LOG_ERROR("Unexpected function = " << m_Function);
    return EMPTY_STRING;
}

const std::string &CFieldConfig::CFieldOptions::verboseFunctionName(void) const
{
    switch (m_Function)
    {
        case model::function_t::E_IndividualCount:
            // For backwards compatibility the "count" function name maps to
            // E_IndividualRareCount, not E_IndividualCount as you might think
            return EMPTY_STRING;
        case model::function_t::E_IndividualNonZeroCount:
            return FUNCTION_NON_ZERO_COUNT;
        case model::function_t::E_IndividualRareCount:
            // For backwards compatibility the "count" function name maps to
            // E_IndividualRareCount, not E_IndividualCount as you might think
            return FUNCTION_COUNT;
        case model::function_t::E_IndividualRareNonZeroCount:
            return FUNCTION_RARE_NON_ZERO_COUNT;
        case model::function_t::E_IndividualRare:
            return FUNCTION_RARE;
        case model::function_t::E_IndividualLowCounts:
            return FUNCTION_LOW_COUNT;
        case model::function_t::E_IndividualHighCounts:
            return FUNCTION_HIGH_COUNT;
        case model::function_t::E_IndividualLowNonZeroCount:
            return FUNCTION_LOW_NON_ZERO_COUNT;
        case model::function_t::E_IndividualHighNonZeroCount:
            return FUNCTION_HIGH_NON_ZERO_COUNT;
        case model::function_t::E_IndividualDistinctCount:
            return FUNCTION_DISTINCT_COUNT;
        case model::function_t::E_IndividualLowDistinctCount:
            return FUNCTION_LOW_DISTINCT_COUNT;
        case model::function_t::E_IndividualHighDistinctCount:
            return FUNCTION_HIGH_DISTINCT_COUNT;
        case model::function_t::E_IndividualInfoContent:
            return FUNCTION_INFO_CONTENT;
        case model::function_t::E_IndividualLowInfoContent:
            return FUNCTION_LOW_INFO_CONTENT;
        case model::function_t::E_IndividualHighInfoContent:
            return FUNCTION_HIGH_INFO_CONTENT;
        case model::function_t::E_IndividualTimeOfDay:
            return FUNCTION_TIME_OF_DAY;
        case model::function_t::E_IndividualTimeOfWeek:
            return FUNCTION_TIME_OF_WEEK;
        case model::function_t::E_IndividualMetric:
            return FUNCTION_METRIC;
        case model::function_t::E_IndividualMetricMean:
            return FUNCTION_AVERAGE;
        case model::function_t::E_IndividualMetricLowMean:
            return FUNCTION_LOW_MEAN;
        case model::function_t::E_IndividualMetricHighMean:
            return FUNCTION_HIGH_MEAN;
        case model::function_t::E_IndividualMetricMedian:
            return FUNCTION_MEDIAN;
        case model::function_t::E_IndividualMetricLowMedian:
            return FUNCTION_LOW_MEDIAN;
        case model::function_t::E_IndividualMetricHighMedian:
            return FUNCTION_HIGH_MEDIAN;
        case model::function_t::E_IndividualMetricMin:
            return FUNCTION_MIN;
        case model::function_t::E_IndividualMetricMax:
            return FUNCTION_MAX;
        case model::function_t::E_IndividualMetricVariance:
            return FUNCTION_VARIANCE;
        case model::function_t::E_IndividualMetricLowVariance:
            return FUNCTION_LOW_VARIANCE;
        case model::function_t::E_IndividualMetricHighVariance:
            return FUNCTION_HIGH_VARIANCE;
        case model::function_t::E_IndividualMetricSum:
            return FUNCTION_SUM;
        case model::function_t::E_IndividualMetricLowSum:
            return FUNCTION_LOW_SUM;
        case model::function_t::E_IndividualMetricHighSum:
            return FUNCTION_HIGH_SUM;
        case model::function_t::E_IndividualMetricNonNullSum:
            return FUNCTION_NON_NULL_SUM;
        case model::function_t::E_IndividualMetricLowNonNullSum:
            return FUNCTION_LOW_NON_NULL_SUM;
        case model::function_t::E_IndividualMetricHighNonNullSum:
            return FUNCTION_HIGH_NON_NULL_SUM;
        case model::function_t::E_IndividualLatLong:
            return FUNCTION_LAT_LONG;
        case model::function_t::E_IndividualMaxVelocity:
            return FUNCTION_MAX_VELOCITY;
        case model::function_t::E_IndividualMinVelocity:
            return FUNCTION_MIN_VELOCITY;
        case model::function_t::E_IndividualMeanVelocity:
            return FUNCTION_MEAN_VELOCITY;
        case model::function_t::E_IndividualSumVelocity:
            return FUNCTION_SUM_VELOCITY;
        case model::function_t::E_PopulationCount:
            return FUNCTION_COUNT;
        case model::function_t::E_PopulationDistinctCount:
            return FUNCTION_DISTINCT_COUNT;
        case model::function_t::E_PopulationLowDistinctCount:
            return FUNCTION_LOW_DISTINCT_COUNT;
        case model::function_t::E_PopulationHighDistinctCount:
            return FUNCTION_HIGH_DISTINCT_COUNT;
        case model::function_t::E_PopulationRare:
            return FUNCTION_RARE;
        case model::function_t::E_PopulationRareCount:
            return FUNCTION_RARE_COUNT;
        case model::function_t::E_PopulationFreqRare:
            return FUNCTION_FREQ_RARE;
        case model::function_t::E_PopulationFreqRareCount:
            return FUNCTION_FREQ_RARE_COUNT;
        case model::function_t::E_PopulationLowCounts:
            return FUNCTION_LOW_COUNT;
        case model::function_t::E_PopulationHighCounts:
            return FUNCTION_HIGH_COUNT;
        case model::function_t::E_PopulationInfoContent:
            return FUNCTION_INFO_CONTENT;
        case model::function_t::E_PopulationLowInfoContent:
            return FUNCTION_LOW_INFO_CONTENT;
        case model::function_t::E_PopulationHighInfoContent:
            return FUNCTION_HIGH_INFO_CONTENT;
        case model::function_t::E_PopulationTimeOfDay:
            return FUNCTION_TIME_OF_DAY;
        case model::function_t::E_PopulationTimeOfWeek:
            return FUNCTION_TIME_OF_WEEK;
        case model::function_t::E_PopulationMetric:
            return FUNCTION_METRIC;
        case model::function_t::E_PopulationMetricMean:
            return FUNCTION_AVERAGE;
        case model::function_t::E_PopulationMetricLowMean:
            return FUNCTION_LOW_MEAN;
        case model::function_t::E_PopulationMetricHighMean:
            return FUNCTION_HIGH_MEAN;
        case model::function_t::E_PopulationMetricMedian:
            return FUNCTION_MEDIAN;
        case model::function_t::E_PopulationMetricLowMedian:
            return FUNCTION_LOW_MEDIAN;
        case model::function_t::E_PopulationMetricHighMedian:
            return FUNCTION_HIGH_MEDIAN;
        case model::function_t::E_PopulationMetricMin:
            return FUNCTION_MIN;
        case model::function_t::E_PopulationMetricMax:
            return FUNCTION_MAX;
        case model::function_t::E_PopulationMetricVariance:
            return FUNCTION_VARIANCE;
        case model::function_t::E_PopulationMetricLowVariance:
            return FUNCTION_LOW_VARIANCE;
        case model::function_t::E_PopulationMetricHighVariance:
            return FUNCTION_HIGH_VARIANCE;
        case model::function_t::E_PopulationMetricSum:
            return FUNCTION_SUM;
        case model::function_t::E_PopulationMetricLowSum:
            return FUNCTION_LOW_SUM;
        case model::function_t::E_PopulationMetricHighSum:
            return FUNCTION_HIGH_SUM;
        case model::function_t::E_PopulationLatLong:
            return FUNCTION_LAT_LONG;
        case model::function_t::E_PopulationMaxVelocity:
            return FUNCTION_MAX_VELOCITY;
        case model::function_t::E_PopulationMinVelocity:
            return FUNCTION_MIN_VELOCITY;
        case model::function_t::E_PopulationMeanVelocity:
            return FUNCTION_MEAN_VELOCITY;
        case model::function_t::E_PopulationSumVelocity:
            return FUNCTION_SUM_VELOCITY;
        case model::function_t::E_PeersCount:
            return FUNCTION_COUNT;
        case model::function_t::E_PeersLowCounts:
            return FUNCTION_LOW_COUNT;
        case model::function_t::E_PeersHighCounts:
            return FUNCTION_HIGH_COUNT;
        case model::function_t::E_PeersDistinctCount:
            return FUNCTION_DISTINCT_COUNT;
        case model::function_t::E_PeersLowDistinctCount:
            return FUNCTION_LOW_DISTINCT_COUNT;
        case model::function_t::E_PeersHighDistinctCount:
            return FUNCTION_HIGH_DISTINCT_COUNT;
        case model::function_t::E_PeersInfoContent:
            return FUNCTION_INFO_CONTENT;
        case model::function_t::E_PeersLowInfoContent:
            return FUNCTION_LOW_INFO_CONTENT;
        case model::function_t::E_PeersHighInfoContent:
            return FUNCTION_HIGH_INFO_CONTENT;
        case model::function_t::E_PeersTimeOfDay:
            return FUNCTION_TIME_OF_DAY;
        case model::function_t::E_PeersTimeOfWeek:
            return FUNCTION_TIME_OF_WEEK;
    }

    LOG_ERROR("Unexpected function = " << m_Function);
    return EMPTY_STRING;
}

std::ostream &CFieldConfig::CFieldOptions::debugPrintClause(std::ostream &strm) const
{
    strm << this->verboseFunctionName();
    if (!m_FieldName.empty())
    {
        strm << '(' << m_FieldName << ')';
    }
    bool considerUseNull(false);
    if (!m_ByFieldName.empty())
    {
        strm << ' ' << BY_TOKEN << ' ' << m_ByFieldName;
        considerUseNull = true;
    }
    if (!m_OverFieldName.empty())
    {
        strm << ' ' << OVER_TOKEN << ' ' << m_OverFieldName;
        considerUseNull = true;
    }
    if (!m_PartitionFieldName.empty())
    {
        strm << ' ' << PARTITION_FIELD_OPTION << '=' << m_PartitionFieldName;
    }
    if (m_UseNull && considerUseNull)
    {
        strm << ' ' << USE_NULL_OPTION << "=1";
    }
    if (m_OverHasExcludeFrequent)
    {
        if (m_ByHasExcludeFrequent)
        {
            strm << ' ' << EXCLUDE_FREQUENT_OPTION << '=' << ALL_TOKEN;
        }
        else
        {
            strm << ' ' << EXCLUDE_FREQUENT_OPTION << '=' << OVER_TOKEN;
        }
    }
    else
    {
        if (m_ByHasExcludeFrequent)
        {
            strm << ' ' << EXCLUDE_FREQUENT_OPTION << '=' << BY_TOKEN;
        }
    }

    return strm;
}

void CFieldConfig::CFieldOptions::swap(CFieldOptions &other)
{
    m_Description.swap(other.m_Description);
    std::swap(m_Function, other.m_Function);
    m_FieldName.swap(other.m_FieldName);
    std::swap(m_ConfigKey, other.m_ConfigKey);
    m_ByFieldName.swap(other.m_ByFieldName);
    m_OverFieldName.swap(other.m_OverFieldName);
    m_PartitionFieldName.swap(other.m_PartitionFieldName);
    std::swap(m_ByHasExcludeFrequent, other.m_ByHasExcludeFrequent);
    std::swap(m_OverHasExcludeFrequent, other.m_OverHasExcludeFrequent);
    std::swap(m_UseNull, other.m_UseNull);
}

void swap(CFieldConfig::CFieldOptions &lhs, CFieldConfig::CFieldOptions &rhs)
{
    lhs.swap(rhs);
}

std::ostream &operator<<(std::ostream &strm,
                         const CFieldConfig::CFieldOptions &options)
{
    options.debugPrintClause(strm);
    strm << " (config key: " << options.m_ConfigKey << " description: " << options.m_Description << ')';
    return strm;
}


}
}

