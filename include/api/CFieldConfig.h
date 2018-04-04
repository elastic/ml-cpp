/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CFieldConfig_h
#define INCLUDED_ml_api_CFieldConfig_h

#include <core/BoostMultiIndex.h>
#include <core/CPatternSet.h>

#include <model/CDetectionRule.h>
#include <model/FunctionTypes.h>

#include <api/ImportExport.h>

#include <boost/property_tree/ptree_fwd.hpp>
#include <boost/unordered_map.hpp>

#include <iosfwd>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

class CFieldConfigTest;

namespace ml {
namespace api {

//! \brief
//! Holds field configuration options.
//!
//! DESCRIPTION:\n
//! Holds configuration options that define how Ml custom
//! search commands should interpret fields in events.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This class is designed to be populated from either a clause
//! provided to a custom search command, e.g.:
//!
//! count BY status
//! ResponseTime BY Airline
//! count by DPT over SRC
//!
//! or from a properties config file (which is basically a
//! Windows .ini file with hash as the comment character instead
//! of semi-colon).
//!
//! Boost's property_tree package can parse such config files, as
//! it accepts either hash or semi-colon as comment characters.
//! Therefore, we use boost::property_tree::ini_parser to load the
//! config file.
//!
//! To decouple the public interface from the config file format,
//! the boost property_tree is copied into separate bespoke data
//! structures.
//!
//! In the config file approach, each detector is specified
//! using three settings:
//!
//! detector.1.clause = responsetime by airline
//! detector.1.description = Airline booking system performance
//!
//! The number after "detector" (1 in the example above) varies for
//! each different detector.  In addition, influencers are specified
//! separately, as these are the same for all detectors in a given
//! job:
//!
//! influencer.1 = client_ip
//! influencer.2 = uri_path
//!
//! Within the Ml model code "person field" corresponds to the
//! "by field" for individual models and "over field" for population
//! models.  For population models the "by field" is referred to as
//! the "attribute field" in model library code.
//!
class API_EXPORT CFieldConfig {
public:
    //! Prefix for detector settings
    static const std::string DETECTOR_PREFIX;

    //! Prefix for categorization filter settings
    static const std::string CATEGORIZATION_FILTER_PREFIX;

    //! Prefix for influencer settings
    static const std::string INFLUENCER_PREFIX;

    //! Prefix for filter setting
    static const std::string FILTER_PREFIX;

    //! Prefix for scheduled events
    static const std::string SCHEDULED_EVENT_PREFIX;

    //! Suffix for clause settings
    static const std::string CLAUSE_SUFFIX;

    //! Suffix for description settings
    static const std::string DESCRIPTION_SUFFIX;

    //! Suffix for detector rules
    static const std::string RULES_SUFFIX;

    //! Name of the "categorizationfield" option
    static const std::string CATEGORIZATION_FIELD_OPTION;

    //! Name of the "summarycountfield" option
    static const std::string SUMMARY_COUNT_FIELD_OPTION;

    //! Character to look for to distinguish setting names
    static const char SUFFIX_SEPARATOR;

    //! Character to look for to split field names out of complete config keys
    static const char FIELDNAME_SEPARATOR;

    //! Suffix applied to field names for the setting that indicates whether
    //! they're enabled
    static const std::string IS_ENABLED_SUFFIX;

    //! Suffix applied to field names for the setting that indicates whether
    //! they're metrics
    static const std::string BY_SUFFIX;

    //! Suffix applied to field names for the setting that indicates whether
    //! they're metrics
    static const std::string OVER_SUFFIX;

    //! Suffix applied to field names for the setting that indicates whether
    //! they're metrics
    static const std::string PARTITION_SUFFIX;

    //! Option to look for in the command line clause to indicate that the
    //! "partitionfield" parameter is specified (case-insensitive)
    static const std::string PARTITION_FIELD_OPTION;

    //! Suffix applied to field names for the setting that indicates whether
    //! empty/missing values of the "by" field should be ignored
    static const std::string USE_NULL_SUFFIX;

    //! Option to look for in the command line clause to indicate that the
    //! "usenull" parameter is specified (case-insensitive)
    static const std::string USE_NULL_OPTION;

    //! Token to look for in the command line clause to indicate that the
    //! "by" field follows
    static const std::string BY_TOKEN;

    //! Token to look for in the command line clause to indicate that the
    //! "over" field follows
    static const std::string OVER_TOKEN;

    //! Magic field name used to indicate that event rate should be
    //! analysed rather than a field value
    static const std::string COUNT_NAME;

    //! A default string value that our string utilities will convert to
    //! boolean false
    static const std::string FALSE_VALUE;

    //! A default string value that our string utilities will convert to
    //! boolean true
    static const std::string TRUE_VALUE;

    //! Token to look in the config file to indicate a list of field
    //! names that are used to indicate an influence pivot relationship
    static const std::string INFLUENCER_FIELD_NAMES_OPTION;

    //! Option to specify an influencer field
    static const std::string INFLUENCER_FIELD_OPTION;

    //! Strings that define the type of analysis to run
    static const std::string FUNCTION_COUNT;
    static const std::string FUNCTION_COUNT_ABBREV;
    static const std::string FUNCTION_LOW_COUNT;
    static const std::string FUNCTION_LOW_COUNT_ABBREV;
    static const std::string FUNCTION_HIGH_COUNT;
    static const std::string FUNCTION_HIGH_COUNT_ABBREV;
    static const std::string FUNCTION_DISTINCT_COUNT;
    static const std::string FUNCTION_DISTINCT_COUNT_ABBREV;
    static const std::string FUNCTION_LOW_DISTINCT_COUNT;
    static const std::string FUNCTION_LOW_DISTINCT_COUNT_ABBREV;
    static const std::string FUNCTION_HIGH_DISTINCT_COUNT;
    static const std::string FUNCTION_HIGH_DISTINCT_COUNT_ABBREV;
    static const std::string FUNCTION_NON_ZERO_COUNT;
    static const std::string FUNCTION_NON_ZERO_COUNT_ABBREV;
    static const std::string FUNCTION_RARE_NON_ZERO_COUNT;
    static const std::string FUNCTION_RARE_NON_ZERO_COUNT_ABBREV;
    static const std::string FUNCTION_RARE;
    static const std::string FUNCTION_RARE_COUNT;
    static const std::string FUNCTION_FREQ_RARE;
    static const std::string FUNCTION_FREQ_RARE_ABBREV;
    static const std::string FUNCTION_FREQ_RARE_COUNT;
    static const std::string FUNCTION_FREQ_RARE_COUNT_ABBREV;
    static const std::string FUNCTION_LOW_NON_ZERO_COUNT;
    static const std::string FUNCTION_LOW_NON_ZERO_COUNT_ABBREV;
    static const std::string FUNCTION_HIGH_NON_ZERO_COUNT;
    static const std::string FUNCTION_HIGH_NON_ZERO_COUNT_ABBREV;
    static const std::string FUNCTION_INFO_CONTENT;
    static const std::string FUNCTION_LOW_INFO_CONTENT;
    static const std::string FUNCTION_HIGH_INFO_CONTENT;
    static const std::string FUNCTION_METRIC;
    static const std::string FUNCTION_AVERAGE;
    static const std::string FUNCTION_MEAN;
    static const std::string FUNCTION_LOW_MEAN;
    static const std::string FUNCTION_HIGH_MEAN;
    static const std::string FUNCTION_LOW_AVERAGE;
    static const std::string FUNCTION_HIGH_AVERAGE;
    static const std::string FUNCTION_MEDIAN;
    static const std::string FUNCTION_LOW_MEDIAN;
    static const std::string FUNCTION_HIGH_MEDIAN;
    static const std::string FUNCTION_MIN;
    static const std::string FUNCTION_MAX;
    static const std::string FUNCTION_VARIANCE;
    static const std::string FUNCTION_LOW_VARIANCE;
    static const std::string FUNCTION_HIGH_VARIANCE;
    static const std::string FUNCTION_SUM;
    static const std::string FUNCTION_LOW_SUM;
    static const std::string FUNCTION_HIGH_SUM;
    static const std::string FUNCTION_NON_NULL_SUM;
    static const std::string FUNCTION_NON_NULL_SUM_ABBREV;
    static const std::string FUNCTION_LOW_NON_NULL_SUM;
    static const std::string FUNCTION_LOW_NON_NULL_SUM_ABBREV;
    static const std::string FUNCTION_HIGH_NON_NULL_SUM;
    static const std::string FUNCTION_HIGH_NON_NULL_SUM_ABBREV;
    static const std::string FUNCTION_TIME_OF_DAY;
    static const std::string FUNCTION_TIME_OF_WEEK;
    static const std::string FUNCTION_LAT_LONG;
    static const std::string FUNCTION_MAX_VELOCITY;
    static const std::string FUNCTION_MIN_VELOCITY;
    static const std::string FUNCTION_MEAN_VELOCITY;
    static const std::string FUNCTION_SUM_VELOCITY;

    //! String that defines whether to exclude frequent results
    static const std::string EXCLUDE_FREQUENT_SUFFIX;
    static const std::string EXCLUDE_FREQUENT_OPTION;
    static const std::string ALL_TOKEN;
    static const std::string NONE_TOKEN;

    static const std::string CLEAR;
    static const std::string EMPTY_STRING;

public:
    //! Class representing all the options associated with a field config.
    class API_EXPORT CFieldOptions {
    public:
        //! Construct with no "by" field nor "partition" field, deducing
        //! the function from the fieldName
        CFieldOptions(const std::string& fieldName, int configKey);

        //! Deduce the function from the fieldName
        CFieldOptions(const std::string& fieldName, int configKey, const std::string& byFieldName, bool byHasExcludeFrequent, bool useNull);

        //! Deduce the function from the fieldName
        CFieldOptions(const std::string& fieldName,
                      int configKey,
                      const std::string& byFieldName,
                      const std::string& partitionFieldName,
                      bool byHasExcludeFrequent,
                      bool overHasExcludeFrequent,
                      bool useNull);

        //! Specify everything
        CFieldOptions(model::function_t::EFunction function,
                      const std::string& fieldName,
                      int configKey,
                      const std::string& byFieldName,
                      const std::string& overFieldName,
                      const std::string& partitionFieldName,
                      bool byHasExcludeFrequent,
                      bool overHasExcludeFrequent,
                      bool useNull);

        //! Set description
        void description(std::string description);

        //! Accessors
        const std::string& description() const;
        model::function_t::EFunction function() const;
        const std::string& fieldName() const;
        int configKey() const;
        const std::string& byFieldName() const;
        const std::string& overFieldName() const;
        const std::string& partitionFieldName() const;
        bool useNull() const;
        ml::model_t::EExcludeFrequent excludeFrequent() const;

        //! Map back from the function enum to the shortest possible
        //! function name that could be used to specify the function
        const std::string& terseFunctionName() const;

        //! Map back from the function enum to the longest possible
        //! function name that could be used to specify the function
        const std::string& verboseFunctionName() const;

        //! Write the detector-specific parts of the configuration
        //! clause.  Note that this cannot include summarycountfield,
        //! influencerfield or categorizationfield as these are not
        //! detector-specific.
        std::ostream& debugPrintClause(std::ostream& strm) const;

        //! Efficient swap
        void swap(CFieldOptions& other);

    private:
        std::string m_Description;
        model::function_t::EFunction m_Function;
        std::string m_FieldName;
        int m_ConfigKey;
        std::string m_ByFieldName;
        std::string m_OverFieldName;
        std::string m_PartitionFieldName;
        bool m_ByHasExcludeFrequent;
        bool m_OverHasExcludeFrequent;
        bool m_UseNull;

        friend std::ostream& operator<<(std::ostream&, const CFieldOptions&);
    };

public:
    //! Key specifiers for the multi-index
    struct SUniqueKey {};
    struct SConfigKey {};

    //! Index of field names to field options.
    //! Uniqueness is enforced by config key and also by the combination of
    //! function, field name, by field name, over field name and
    //! partition field name.
    using TFieldOptionsMIndex = boost::multi_index::multi_index_container<
        CFieldOptions,
        boost::multi_index::indexed_by<
            boost::multi_index::ordered_unique<boost::multi_index::tag<SConfigKey>,
                                               BOOST_MULTI_INDEX_CONST_MEM_FUN(CFieldOptions, int, configKey)>,
            boost::multi_index::ordered_unique<
                boost::multi_index::tag<SUniqueKey>,
                boost::multi_index::composite_key<
                    CFieldOptions,
                    BOOST_MULTI_INDEX_CONST_MEM_FUN(CFieldOptions, model::function_t::EFunction, function),
                    BOOST_MULTI_INDEX_CONST_TYPE_CONST_MEM_FUN(CFieldOptions, std::string, fieldName),
                    BOOST_MULTI_INDEX_CONST_TYPE_CONST_MEM_FUN(CFieldOptions, std::string, byFieldName),
                    BOOST_MULTI_INDEX_CONST_TYPE_CONST_MEM_FUN(CFieldOptions, std::string, overFieldName),
                    BOOST_MULTI_INDEX_CONST_TYPE_CONST_MEM_FUN(CFieldOptions, std::string, partitionFieldName)>>>>;

    using TFieldOptionsMIndexItr = TFieldOptionsMIndex::iterator;
    using TFieldOptionsMIndexCItr = TFieldOptionsMIndex::const_iterator;

    //! Used to maintain a list of all unique config keys
    using TIntSet = std::set<int>;

    //! Used to return the superset of enabled field names
    using TStrSet = std::set<std::string>;

    //! Used to obtain command line clause tokens
    using TStrVec = std::vector<std::string>;
    using TStrVecItr = TStrVec::iterator;

    using TDetectionRuleVec = std::vector<model::CDetectionRule>;
    using TIntDetectionRuleVecUMap = boost::unordered_map<int, TDetectionRuleVec>;

    using TStrPatternSetUMap = boost::unordered_map<std::string, core::CPatternSet>;

    using TStrDetectionRulePr = std::pair<std::string, model::CDetectionRule>;
    using TStrDetectionRulePrVec = std::vector<TStrDetectionRulePr>;

public:
    //! Construct empty.  This call should generally be followed by a call to
    //! one of the init...() methods.
    CFieldConfig();

    //! Construct with just a categorization field.  (In the case of a
    //! categorization job, this is all that is needed for this config.)
    CFieldConfig(const std::string& categorizationFieldName);

    //! Construct with a single field.  (This constructor is largely for
    //! unit testing and backwards compatibility.)
    CFieldConfig(const std::string& fieldName,
                 const std::string& byFieldName,
                 bool useNull = false,
                 const std::string& summaryCountFieldName = EMPTY_STRING);

    //! Construct with a single field and a partition field.  (This
    //! constructor is only used for unit testing.)
    CFieldConfig(const std::string& fieldName, const std::string& byFieldName, const std::string& partitionFieldName, bool useNull);

    //! Initialise from command line options.  This method expects that only
    //! one of the config file and the tokens will have been specified.  If
    //! neither or both have been specified, this is reported as an error.
    bool initFromCmdLine(const std::string& configFile, const TStrVec& tokens);

    //! Initialise from a config file.
    bool initFromFile(const std::string& configFile);

    //! Initialise from a command line clause that has been tokenised by the
    //! command line parser (i.e. using whitespace).  The clause may have at
    //! most one "by" token.  If there is a "by" token, there must be
    //! exactly one token following it, and one or more tokens before it.
    bool initFromClause(const TStrVec& tokens);

    //! Add an extra set of field config options to the configuration.  It
    //! is not expected that this will be done once analysis has started.
    //! If options are added during the analysis then this class will remain
    //! consistent but it is likely to cause problems for other classes that
    //! do not expect this.  The likely use case for this function is for
    //! building config migration programs.
    bool addOptions(const CFieldOptions& options);

    //! Get the list of categorization filters
    const TStrVec& categorizationFilters() const;

    //! Get the field to use for summary counts.  If the returned string is
    //! empty then this implies that input has not been manually summarised.
    const std::string& summaryCountFieldName() const;

    //! Does any config have a non-empty partition field configured?
    //! (This is used by licensing.)
    bool havePartitionFields() const;

    //! Access the superset of all field names that are used by any detector.
    const TStrSet& fieldNameSuperset() const;

    //! Debug dump of fields
    std::string debug() const;

    //! Add influencer fields for all the by/over/partition fields of all
    //! existing configurations
    void addInfluencerFieldsFromByOverPartitionFields();

    //! Get the list of field names for pivoting the anomaly results
    const TStrVec& influencerFieldNames() const;

    //! Get the detector key to detection rules map
    const TIntDetectionRuleVecUMap& detectionRules() const;

    //! Get the scheduled events
    const TStrDetectionRulePrVec& scheduledEvents() const;

    //! Attempt to parse a detector's rules.
    bool parseRules(int detectorIndex, const std::string& rules);

    //! Process and store a rule filter
    bool processFilter(const std::string& key, const std::string& value);

    // //! Replaces filters with the ones in the given property tree
    bool updateFilters(const boost::property_tree::ptree& propTree);

    //! Replaces scheduled events with the ones in the given property tree
    bool updateScheduledEvents(const boost::property_tree::ptree& propTree);

    const TFieldOptionsMIndex& fieldOptions() const;

    const std::string& categorizationFieldName() const;

    const TStrPatternSetUMap& ruleFilters() const;

private:
    //! Parse detection rules into detectionRules
    bool parseRules(TDetectionRuleVec& detectionRules, const std::string& rules);

    //! Attempt to parse a single analysis clause.  This could have come
    //! from either the command line of a custom command or from one entry
    //! in a config file.  The supplied tokens must have been
    //! split using both whitespace and commas.
    bool parseClause(bool allowMultipleFunctions,
                     int configKey,
                     const std::string& description,
                     TStrVec& copyTokens,
                     TFieldOptionsMIndex& optionsIndex,
                     std::string& categorizationFieldName,
                     std::string& summaryCountFieldName);

    //! Helper method for initFromFile().  Because multiple config
    //! file settings are required to specify a single configuration, the
    //! config file is read using a mix of iteration and search.  We iterate
    //! to find the unique config keys, then search for all the settings
    //! that correspond to each particular config key.  Doing this
    //! simplifies the error reporting.
    bool
    processDetector(const boost::property_tree::ptree& propTree, const std::string& key, const std::string& value, TIntSet& handledConfigs);

    //! Add data structures relating to an active detector.
    bool addActiveDetector(int configKey, const std::string& description, const std::string& rules, TStrVec& copyTokens);

    //! Get a function name and field name from a field string
    static bool parseFieldString(bool haveSummaryCountField,
                                 bool isPopulation,
                                 bool hasByField,
                                 const std::string& str,
                                 model::function_t::EFunction& function,
                                 std::string& fieldName);

    //! Used to keep the field superset up-to-date
    void seenField(const std::string& fieldName);

    //! Split a config clause on whitespace and commas.
    bool tokenise(const std::string& clause, TStrVec& copyTokens);

    //! Split a config clause that has already been tokenised by a
    //! command-line processor to additionally be split at commas
    void retokenise(const TStrVec& tokens, TStrVec& copyTokens);

    //! Check that we have at most one "by" and one "over" token
    //! and report their positions in the token list
    bool findLastByOverTokens(const TStrVec& copyTokens, std::size_t& lastByTokenIndex, std::size_t& lastOverTokenIndex);

    //! Check that the "by" or "over" field is valid
    bool validateByOverField(const TStrVec& copyTokens,
                             const std::size_t thisIndex,
                             const std::size_t otherIndex,
                             const TStrVec& clashingNames,
                             std::string& fieldName);

    std::string findParameter(const std::string& parameter, TStrVec& copyTokens);

    //! How does a setting for excludefrequent map to the underlying boolean
    //! flags?
    static bool decipherExcludeFrequentSetting(const std::string& excludeFrequentString,
                                               bool hasByField,
                                               bool isPopulation,
                                               bool& byExcludeFrequent,
                                               bool& overExcludeFrequent);

    //! Store the list of influencer field names, if any
    void influencerFieldNames(TStrVec influencers);

    //! Add influencer fields for all the by/over/partition fields of all
    //! detectors.
    void addInfluencerFieldsFromByOverPartitionFields(const TFieldOptionsMIndex& fieldOptions);

    //! Store one influencer field name
    void addInfluencerFieldName(const std::string& influencer, bool quiet = false);

    //! Sort the influencers (so that downstream code doesn't have to worry
    //! about ordering changes when the overall set is unchanged)
    void sortInfluencers();

    //! Store one categorization filter
    void addCategorizationFilter(const std::string& filter);

    //! Process and store a scheduled event
    bool processScheduledEvent(const boost::property_tree::ptree& propTree,
                               const std::string& key,
                               const std::string& value,
                               TIntSet& handledScheduledEvents);

    //! Debug dump of field options
    void debug(const TFieldOptionsMIndex& fieldOptions, std::ostream& strm) const;

private:
    //! The fields options.
    TFieldOptionsMIndex m_FieldOptions;

    //! The superset of all field names that are used in the config.
    TStrSet m_FieldNameSuperset;

    //! The categorization field name.
    std::string m_CategorizationFieldName;

    //! The filters to be applied to values of the categorization field.
    TStrVec m_CategorizationFilters;

    //! The field names specified for "influence", if any have been provided
    TStrVec m_Influencers;

    //! The summary count field name.   If this is empty then it implies
    //! that input has not been manually summarised.
    std::string m_SummaryCountFieldName;

    //! The detection rules per detector index.
    TIntDetectionRuleVecUMap m_DetectorRules;

    //! The filters per id used by categorical rule conditions.
    TStrPatternSetUMap m_RuleFilters;

    //! The scheduled events (events apply to all detectors).
    //! Events consist of a description and a detection rule
    TStrDetectionRulePrVec m_ScheduledEvents;

    // For unit testing
    friend class ::CFieldConfigTest;
};

//! Efficient swap for field options
void swap(CFieldConfig::CFieldOptions& lhs, CFieldConfig::CFieldOptions& rhs);

//! Print field options
std::ostream& operator<<(std::ostream& strm, const CFieldConfig::CFieldOptions& options);
}
}

#endif // INCLUDED_ml_api_CFieldConfig_h
