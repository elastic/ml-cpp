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

#ifndef INCLUDED_ml_config_CAutoconfigurerParams_h
#define INCLUDED_ml_config_CAutoconfigurerParams_h

#include <core/CoreTypes.h>

#include <config/ConfigTypes.h>
#include <config/Constants.h>
#include <config/ImportExport.h>

#include <boost/optional.hpp>

#include <cstddef>
#include <string>
#include <vector>

#include <stdint.h>

namespace ml {
namespace config {

//! \brief The parameters which control auto-configuration.
//!
//! DESCRIPTION:\n
//! This wraps up the parameters used to control auto-configuration.
//! It also implements functionality to read the parameters from a
//! configuration file and provide sensible defaults when no file
//! is supplied.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Configuration of the auto-configuration analytics is via a config
//! file, which is basically a Windows .ini file with hash as the
//! comment character instead of semi-colon.
//!
//! Boost's property_tree package can parse such config files, as
//! it accepts either hash or semi-colon as comment characters.
//! Therefore, we use boost::property_tree::ini_parser to load the
//! config file.
//!
//! Because supplying parameters via the config file is optional the
//! boost property_tree is copied into separate member variables.
class CONFIG_EXPORT CAutoconfigurerParams {
    public:
        typedef std::vector<core_t::TTime>                      TTimeVec;
        typedef std::vector<std::size_t>                        TSizeVec;
        typedef std::vector<std::string>                        TStrVec;
        typedef boost::optional<TStrVec>                        TOptionalStrVec;
        typedef std::pair<std::string, config_t::EUserDataType> TStrUserDataTypePr;
        typedef std::vector<TStrUserDataTypePr>                 TStrUserDataTypePrVec;
        typedef boost::optional<config_t::EUserDataType>        TOptionalUserDataType;
        typedef std::vector<config_t::EFunctionCategory>        TFunctionCategoryVec;

    public:
        CAutoconfigurerParams(const std::string &timeFieldName,
                              const std::string &timeFieldFormat,
                              bool verbose,
                              bool writeDetectorConfigs);

        //! Initialize from the specified file.
        bool init(const std::string &file);

        //! Get the name of field holding the time.
        const std::string &timeFieldName(void) const;

        //! Get the time field format. Blank means seconds since the epoch, i.e.
        //! the time field can be converted to a time_t by simply converting the
        //! string to a number. Otherwise, it is assumed to be suitable for passing
        //! to strptime.
        const std::string &timeFieldFormat(void) const;

        //! Check if we should be outputting all detectors including those that
        //! have been discarded.
        bool verbose(void) const;

        //! Check if we should output the top detectors in JSON format.
        bool writeDetectorConfigs(void) const;

        //! Get the line ending to use when writing detectors in JSON format.
        const std::string &detectorConfigLineEnding(void) const;

        //! \name Scoping
        //@{
        //! Check that \p field is not one of the fields in the data set we've
        //! been told to ignore.
        bool fieldOfInterest(const std::string &field) const;

        //! Check if we can use \p argument for the argument of a function.
        bool canUseForFunctionArgument(const std::string &argument) const;

        //! Check if we can use \p by as a by field.
        bool canUseForByField(const std::string &by) const;

        //! Check if we can use \p over as an over field.
        bool canUseForOverField(const std::string &over) const;

        //! Check if we can use \p partition as a partition field.
        bool canUseForPartitionField(const std::string &partition) const;

        //! Get the function categories to configure.
        const TFunctionCategoryVec &functionsCategoriesToConfigure(void) const;
        //@}

        //! \name Statistics
        //@{
        //! The user specified field data types.
        TOptionalUserDataType dataType(const std::string &field) const;

        //! The minimum number of records to classify a field.
        uint64_t minimumExamplesToClassify(void) const;

        //! The minimum number of records to classify a field.
        std::size_t numberOfMostFrequentFieldsCounts(void) const;
        //@}

        //! \name General Configuration
        //@{
        //! The minimum number of records to classify a field.
        uint64_t minimumRecordsToAttemptConfig(void) const;

        //! The minimum permitted detector score.
        double minimumDetectorScore(void) const;
        //@}

        //! A number of by field values which is considered high so
        //! larger numbers will be penalized.
        std::size_t highNumberByFieldValues(void) const;

        //! The highest permitted number of by field values.
        std::size_t maximumNumberByFieldValues(void) const;

        //! A number of by field values for rare commands which is considered
        //! high so larger numbers will be penalized.
        std::size_t highNumberRareByFieldValues(void) const;

        //! The highest permitted number of by field values for rare commands.
        std::size_t maximumNumberRareByFieldValues(void) const;

        //! A number of partition field values which is considered high
        //! so larger numbers will be penalized.
        std::size_t highNumberPartitionFieldValues(void) const;

        //! The highest permitted number of partition field values.
        std::size_t maximumNumberPartitionFieldValues(void) const;

        //! A number of over field values which is considered small so
        //! that smaller numbers will be penalized.
        std::size_t lowNumberOverFieldValues(void) const;

        //! The lowest permitted number of over field values.
        std::size_t minimumNumberOverFieldValues(void) const;

        //! The factor, as a multiple of the lowest field value count, which is
        //! an upper bound for a field value being in the low frequency tail.
        double highCardinalityInTailFactor(void) const;

        //! The margin, as an increment on the lowest field value count, which
        //! is an upper bound for a field value being in the low frequency tail.
        uint64_t highCardinalityInTailIncrement(void) const;

        //! A proportion of records in the low frequency tail for rare analysis
        //! which is considered large so that larger proportions will be penalized.
        double highCardinalityHighTailFraction(void) const;

        //! The highest permitted proportion of records in the low frequency
        //! tail for rare analysis.
        double highCardinalityMaximumTailFraction(void) const;

        //! A fraction of populated buckets that is considered small for \p function
        //! and \p ignoreEmpty so that smaller proportions will be penalized.
        double lowPopulatedBucketFraction(config_t::EFunctionCategory function, bool ignoreEmpty) const;

        //! The smallest permitted fraction of populated buckets for \p function and
        //! \p ignoreEmpty.
        double minimumPopulatedBucketFraction(config_t::EFunctionCategory function, bool ignoreEmpty) const;

        //! A fraction of populated buckets that is considered high for \p function
        //! and \p ignoreEmpty so that higher fractions will be penalized.
        double highPopulatedBucketFraction(config_t::EFunctionCategory function, bool ignoreEmpty) const;

        //! The maximum permitted fraction of populated buckets for \p function and
        //! \p ignoreEmpty.
        double maximumPopulatedBucketFraction(config_t::EFunctionCategory function, bool ignoreEmpty) const;

        //! Get the candidate bucket lengths to test for each detector.
        const TTimeVec &candidateBucketLengths(void) const;

        //! Get a number of buckets considered small to configure a detector.
        double lowNumberOfBucketsForConfig(void) const;

        //! Get the lowest permitted number of buckets we'll use to configure
        //! a detector.
        double minimumNumberOfBucketsForConfig(void) const;

        //! Get the minimum possible proportion of values in the jitter interval
        //! surrounding the polling interval to classify data as polled.
        double polledDataMinimumMassAtInterval(void) const;

        //! Get the maximum amount that polled data times can jitter about the
        //! polling interval.
        double polledDataJitter(void) const;

        //! Get a coefficient of variation for a bucketed statistic which is
        //! considered low such that lower values are penalized.
        double lowCoefficientOfVariation(void) const;

        //! Get the minimum coefficient of variation for a bucketed statistic
        //! to be worthwhile modeling.
        double minimumCoefficientOfVariation(void) const;

        //! Get a low range for the category lengths to be a suitable argument
        //! for information content such that lower values are penalized.
        double lowLengthRangeForInfoContent(void) const;

        //! Get the minimum range for the category lengths to be a suitable
        //! argument for information content.
        double minimumLengthRangeForInfoContent(void) const;

        //! Get a low maximum category length for it to be a suitable argument
        //! for information content such that lower values are penalized.
        double lowMaximumLengthForInfoContent(void) const;

        //! Get the minimum category length for it to be a suitable argument
        //! for information content.
        double minimumMaximumLengthForInfoContent(void) const;

        //! Get a low empirical entropy for a field to be a suitable argument
        //! for information content such that lower values will be penalized.
        //!
        //! \note This is as a portion of the maximum possible entropy based
        //! on the distinct count.
        double lowEntropyForInfoContent(void) const;

        //! Get the minimum empirical entropy for a field to be a suitable
        //! argument for information content.
        //!
        //! \note This is as a portion of the maximum possible entropy based
        //! on the distinct count.
        double minimumEntropyForInfoContent(void) const;

        //! Get a low distinct count for a field to be a suitable argument
        //! for information content such that lower values will be penalized.
        double lowDistinctCountForInfoContent(void) const;

        //! Get the minimum distinct count for a field to be a suitable
        //! argument for information content.
        double minimumDistinctCountForInfoContent(void) const;

        //! Get the penalty indices for the candidate bucket length identified
        //! by \p bid.
        const TSizeVec &penaltyIndicesFor(std::size_t bid) const;

        //! Get the penalty indices for the function version which ignores empty
        //! value if \p value is true and considers them otherwise.
        const TSizeVec &penaltyIndicesFor(bool ignoreEmpty) const;

        //! Get the penalty index for the candidate bucket length identified
        //! by \p bid and the function version which ignores empty.
        std::size_t penaltyIndexFor(std::size_t bid, bool ignoreEmpty) const;

        //! Get a string describing all the parameters.
        std::string print(void) const;

    private:
        typedef std::vector<double>   TDoubleVec;
        typedef std::vector<TSizeVec> TSizeVecVec;

    private:
        //! Refresh the penalty indices.
        void refreshPenaltyIndices(void);

    private:
        //! The name of field holding the time.
        std::string           m_TimeFieldName;

        //! The time field format.
        std::string           m_TimeFieldFormat;

        //! If true then output information about all possible detectors including
        //! those that have been discarded.
        bool                  m_Verbose;

        //! If true then output the top detector for each candidate in JSON format.
        bool                  m_WriteDetectorConfigs;

        //! The line ending to use when writing detectors in JSON format.
        std::string           m_DetectorConfigLineEnding;

        //! \name Scoping
        //@{
        //! The only fields to use in auto-configuration.
        TOptionalStrVec       m_FieldsOfInterest;

        //! The only argument or partition fields to use in auto-configuration.
        TOptionalStrVec       m_FieldsToUseInAutoconfigureByRole[constants::NUMBER_FIELD_INDICES];

        //! The function categories to consider configuring.
        TFunctionCategoryVec  m_FunctionCategoriesToConfigure;
        //@}

        //! \name Statistics
        //@{
        //! The type of data in each field (numeric or categorical).
        TStrUserDataTypePrVec m_FieldDataTypes;

        //! The minimum number of records to use to classify a field.
        uint64_t              m_MinimumExamplesToClassify;

        //! The number of field values to count occurrences for in the categorical
        //! field statistics.
        std::size_t           m_NumberOfMostFrequentFieldsCounts;
        //@}

        //! \name General Configuration
        //@{
        //! The minimum number of records needed to attempt generating search
        //! configurations.
        uint64_t    m_MinimumRecordsToAttemptConfig;

        //! The minimum permitted detector score.
        double      m_MinimumDetectorScore;
        //@}

        //! \name Field Role Scoring
        //@{
        //! A number of by field values which is considered high.
        std::size_t m_HighNumberByFieldValues;

        //! The highest permitted number of by field values.
        std::size_t m_MaximumNumberByFieldValues;

        //! A number of by field values which is considered high for rare commands.
        std::size_t m_HighNumberRareByFieldValues;

        //! The highest permitted number of by field values for rare commands.
        std::size_t m_MaximumNumberRareByFieldValues;

        //! A number of partition field values which is considered high.
        std::size_t m_HighNumberPartitionFieldValues;

        //! The highest permitted number of partition field values.
        std::size_t m_MaximumNumberPartitionFieldValues;

        //! A number of over field values which is considered small.
        std::size_t m_LowNumberOverFieldValues;

        //! The lowest permitted number of over field values.
        std::size_t m_MinimumNumberOverFieldValues;
        //@}

        //! \name Detector Scoring
        //@{
        //! The factor for a field value being in the low frequency tail.
        double      m_HighCardinalityInTailFactor;

        //! The margin for a field value being in the low frequency tail.
        uint64_t    m_HighCardinalityInTailIncrement;

        //! A high proportion of records in the low frequency tail for rare
        //! analysis to be effective.
        double      m_HighCardinalityHighTailFraction;

        //! The maximum proportion of records in the low frequency tail
        //! for rare analysis.
        double      m_HighCardinalityMaximumTailFraction;

        //! The lower fractions for populated buckets which trigger a penalty.
        TDoubleVec  m_LowPopulatedBucketFractions;

        //! The minimum permitted fractions of populated buckets.
        TDoubleVec  m_MinimumPopulatedBucketFractions;

        //! The upper fractions for populated buckets which trigger a penalty.
        TDoubleVec  m_HighPopulatedBucketFractions;

        //! The maximum permitted fractions of populated buckets.
        TDoubleVec  m_MaximumPopulatedBucketFractions;

        //! The bucket lengths that can be selected in seconds.
        TTimeVec    m_CandidateBucketLengths;

        //! A low number of buckets for configuration.
        double      m_LowNumberOfBucketsForConfig;

        //! The lowest permitted number of buckets for configuration.
        double      m_MinimumNumberOfBucketsForConfig;

        //! The minimum proportion of regular data to classify a data set
        //! as polled.
        double      m_PolledDataMinimumMassAtInterval;

        //! The maximum amount that polled data times can jitter about the
        //! polling interval.
        double      m_PolledDataJitter;

        //! A coefficient of variation for a bucketed statistic which is
        //! considered low.
        double      m_LowCoefficientOfVariation;

        //! The minimum coefficient of variation for a bucketed statistic.
        double      m_MinimumCoefficientOfVariation;

        //! A low range for the category lengths to be a suitable argument
        //! for information content.
        double      m_LowLengthRangeForInfoContent;

        //! The minimum range for the category lengths to be a suitable
        //! argument for information content.
        double      m_MinimumLengthRangeForInfoContent;

        //! A low maximum category length for it to be a suitable argument
        //! for information content.
        double      m_LowMaximumLengthForInfoContent;

        //! The minimum category length for it to be a suitable argument
        //! for information content.
        double      m_MinimumMaximumLengthForInfoContent;

        //! A low empirical entropy for a field to be a suitable argument
        //! for information content.
        double      m_LowEntropyForInfoContent;

        //! The minimum empirical entropy for a field to be a suitable
        //! argument for information content.
        double      m_MinimumEntropyForInfoContent;

        //! A low distinct count for a field to be a suitable argument
        //! for information content.
        double      m_LowDistinctCountForInfoContent;

        //! The minimum distinct count for a field to be a suitable
        //! argument for information content.
        double      m_MinimumDistinctCountForInfoContent;
        //@}

        //! The penalty indices for each bucket length.
        TSizeVecVec m_BucketLengthPenaltyIndices;

        //! The penalty indices for function versions which do and don't
        //! consider empty buckets.
        TSizeVecVec m_IgnoreEmptyPenaltyIndices;
};

}
}

#endif // INCLUDED_ml_config_CAutoconfigurerParams_h
