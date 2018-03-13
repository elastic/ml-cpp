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

#ifndef INCLUDED_ml_config_ConfigTypes_h
#define INCLUDED_ml_config_ConfigTypes_h

#include <config/ImportExport.h>

#include <iosfwd>
#include <string>

namespace ml {
namespace config_t {

//! Enumeration of the user specified data types.
enum EUserDataType { E_UserCategorical, E_UserNumeric };

//! Get a string for the data type.
CONFIG_EXPORT
const std::string &print(EUserDataType type);

//! Write the data type to a stream.
CONFIG_EXPORT
std::ostream &operator<<(std::ostream &o, EUserDataType type);

//! Enumeration of the data types we understand.
enum EDataType {
    E_UndeterminedType,
    E_Binary,
    E_Categorical,
    E_PositiveInteger,
    E_Integer,
    E_PositiveReal,
    E_Real
};

//! Check if the type is categorical.
CONFIG_EXPORT
bool isCategorical(EDataType type);

//! Check if the type is numeric.
CONFIG_EXPORT
bool isNumeric(EDataType type);

//! Check if the type is integer.
CONFIG_EXPORT
bool isInteger(EDataType type);

//! Get a string for the data type.
CONFIG_EXPORT
const std::string &print(EDataType type);

//! Write the data type to a stream.
CONFIG_EXPORT
std::ostream &operator<<(std::ostream &o, EDataType type);

//! Enumeration of the top-level functions we'll consider configuring.
enum EFunctionCategory {
    E_Count,
    E_Rare,
    E_DistinctCount,
    E_InfoContent,
    E_Mean,
    E_Min,
    E_Max,
    E_Sum,
    E_Varp,
    E_Median
};

//! Check if the function takes an argument.
CONFIG_EXPORT
bool hasArgument(EFunctionCategory function);

//! Check if the function is count.
CONFIG_EXPORT
bool isCount(EFunctionCategory function);

//! Check if the function is rare.
CONFIG_EXPORT
bool isRare(EFunctionCategory function);

//! Check if the function is information content.
CONFIG_EXPORT
bool isInfoContent(EFunctionCategory function);

//! Check if the function applies to a metric valued field.
CONFIG_EXPORT
bool isMetric(EFunctionCategory function);

//! Check if the function has sided probability calculations.
CONFIG_EXPORT
bool hasSidedCalculation(EFunctionCategory function);

//! Check if the function has two versions, one which ignores empty
//! buckets and the other which counts them.
CONFIG_EXPORT
bool hasDoAndDontIgnoreEmptyVersions(EFunctionCategory function);

//! Get the prefix of the function corresponding to \p ignoreEmpty
//! and \p isPopulation.
CONFIG_EXPORT
const std::string &
ignoreEmptyVersionName(EFunctionCategory function, bool ignoreEmpty, bool isPopulation);

//! Get a string for the function function.
CONFIG_EXPORT
const std::string &print(EFunctionCategory function);

//! Write the function function to a stream.
CONFIG_EXPORT
std::ostream &operator<<(std::ostream &o, EFunctionCategory function);

//! Enumeration of the sensitivity of the anomaly detection to high,
//! low, or both tails.
enum ESide { E_HighSide, E_LowSide, E_TwoSide, E_UndeterminedSide };

//! Get a string for the side.
CONFIG_EXPORT
const std::string &print(ESide side);

//! Write the side to a stream.
CONFIG_EXPORT
std::ostream &operator<<(std::ostream &o, ESide side);
}
}

#endif// INCLUDED_ml_config_ConfigTypes_h
