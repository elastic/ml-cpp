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

#include <config/ConfigTypes.h>

#include <ostream>

namespace ml {
namespace config_t {
namespace {

const std::string USER_DATA_TYPE_NAMES[] = {
    std::string("categorical"),
    std::string("numeric")
};

const std::string DATA_TYPE_NAMES[] = {
    std::string("<undetermined>"),
    std::string("binary"),
    std::string("categorical"),
    std::string("positive integer"),
    std::string("integer"),
    std::string("positive real"),
    std::string("real")
};

const std::string FUNCTION_CATEGORY_NAMES[] = {
    std::string("count"),
    std::string("rare"),
    std::string("distinct_count"),
    std::string("info_content"),
    std::string("mean"),
    std::string("min"),
    std::string("max"),
    std::string("sum"),
    std::string("varp"),
    std::string("median")
};

const std::string IGNORE_EMPTY_VERSION_NAMES[][2] = {
    { std::string("n/a"),   std::string("n/a")            },
    { std::string("count"), std::string("non_zero_count") },
    { std::string("sum"),   std::string("non_null_sum")   }
};

const std::string SIDE_NAME[] = {
    std::string("high"),
    std::string("low"),
    std::string("both"),
    std::string("<unknown>")
};

}

const std::string &print(EUserDataType type) {
    return USER_DATA_TYPE_NAMES[type];
}

std::ostream &operator<<(std::ostream &o, EUserDataType type) {
    return o << USER_DATA_TYPE_NAMES[type];
}

bool isCategorical(EDataType type) {
    switch (type) {
        case E_Binary:
        case E_Categorical:
            return true;
        case E_UndeterminedType:
        case E_PositiveInteger:
        case E_Integer:
        case E_PositiveReal:
        case E_Real:
            break;
    }
    return false;
}

bool isNumeric(EDataType type) {
    switch (type) {
        case E_PositiveInteger:
        case E_Integer:
        case E_PositiveReal:
        case E_Real:
            return true;
        case E_UndeterminedType:
        case E_Binary:
        case E_Categorical:
            break;
    }
    return false;
}

bool isInteger(EDataType type) {
    switch (type) {
        case E_PositiveInteger:
        case E_Integer:
            return true;
        case E_UndeterminedType:
        case E_Binary:
        case E_Categorical:
        case E_PositiveReal:
        case E_Real:
            break;
    }
    return false;
}

const std::string &print(EDataType type) {
    return DATA_TYPE_NAMES[type];
}

std::ostream &operator<<(std::ostream &o, EDataType type) {
    return o << DATA_TYPE_NAMES[type];
}

bool hasArgument(EFunctionCategory function) {
    switch (function) {
        case E_Count:
        case E_Rare:
            return false;
        case E_DistinctCount:
        case E_InfoContent:
        case E_Mean:
        case E_Min:
        case E_Max:
        case E_Sum:
        case E_Varp:
        case E_Median:
            break;
    }
    return true;
}

bool isCount(EFunctionCategory function) {
    switch (function) {
        case E_Rare:
        case E_DistinctCount:
        case E_InfoContent:
        case E_Mean:
        case E_Min:
        case E_Max:
        case E_Sum:
        case E_Varp:
        case E_Median:
            return false;
        case E_Count:
            break;
    }
    return true;
}

bool isRare(EFunctionCategory function) {
    switch (function) {
        case E_Count:
        case E_DistinctCount:
        case E_InfoContent:
        case E_Mean:
        case E_Min:
        case E_Max:
        case E_Sum:
        case E_Varp:
        case E_Median:
            return false;
        case E_Rare:
            break;
    }
    return true;
}

bool isInfoContent(EFunctionCategory function) {
    switch (function) {
        case E_Count:
        case E_Rare:
        case E_DistinctCount:
        case E_Mean:
        case E_Min:
        case E_Max:
        case E_Sum:
        case E_Varp:
        case E_Median:
            return false;
        case E_InfoContent:
            break;
    }
    return true;
}

bool isMetric(EFunctionCategory function) {
    switch (function) {
        case E_Count:
        case E_Rare:
        case E_DistinctCount:
        case E_InfoContent:
            return false;
        case E_Mean:
        case E_Min:
        case E_Max:
        case E_Sum:
        case E_Varp:
        case E_Median:
            break;
    }
    return true;
}

bool hasSidedCalculation(EFunctionCategory function) {
    switch (function) {
        case E_Rare:
            return false;
        case E_Count:
        case E_DistinctCount:
        case E_InfoContent:
        case E_Mean:
        case E_Min:
        case E_Max:
        case E_Sum:
        case E_Varp:
        case E_Median:
            break;
    }
    return true;
}

bool hasDoAndDontIgnoreEmptyVersions(EFunctionCategory function) {
    switch (function) {
        case E_Rare:
        case E_DistinctCount:
        case E_InfoContent:
        case E_Mean:
        case E_Min:
        case E_Max:
        case E_Varp:
        case E_Median:
            return false;
        case E_Count:
        case E_Sum:
            break;
    }
    return true;
}

const std::string &ignoreEmptyVersionName(EFunctionCategory function,
                                          bool ignoreEmpty,
                                          bool isPopulation) {
    std::size_t index = 0u;
    switch (function) {
        case E_Count:
            index = 1u;
            break;
        case E_Sum:
            index = 2u;
            break;
        case E_Rare:
        case E_DistinctCount:
        case E_InfoContent:
        case E_Mean:
        case E_Min:
        case E_Max:
        case E_Varp:
        case E_Median:
            break;
    }
    return IGNORE_EMPTY_VERSION_NAMES[index][ignoreEmpty && !isPopulation];
}

const std::string &print(EFunctionCategory function) {
    return FUNCTION_CATEGORY_NAMES[function];
}

std::ostream &operator<<(std::ostream &o, EFunctionCategory function) {
    return o << FUNCTION_CATEGORY_NAMES[function];
}

const std::string &print(ESide side) {
    return SIDE_NAME[side];
}

std::ostream &operator<<(std::ostream &o, ESide side) {
    return o << SIDE_NAME[side];
}

}
}
