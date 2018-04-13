/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <config/CAutoconfigurerParams.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CNonCopyable.h>
#include <core/CStringUtils.h>

#include <maths/COrderings.h>

#include <config/Constants.h>
#include <config/CPenalty.h>

#include <boost/property_tree/ini_parser.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/range.hpp>

#include <fstream>
#include <functional>

namespace ml
{
namespace config
{
namespace
{
using TStrVec = std::vector<std::string>;

//! \brief A constraint which applies to a value of type T.
template<typename T>
class CConstraint
{
    public:
        virtual ~CConstraint() {}
        virtual bool operator()(const T &/*value*/) const { return true; }
        virtual bool operator()(const std::vector<T> &/*value*/) const { return true; }
        virtual std::string print() const = 0;
};

//! \brief Represents the fact that T is unconstrained.
template<typename T>
class CUnconstrained : public CConstraint<T>
{
    public:
        bool operator()(const T &/*value*/) const
        {
            return true;
        }
        std::string print() const
        {
            return "unconstrained";
        }
};

//! \brief A collection constraint which apply in conjunction to a value
//! of type T.
template<typename T>
class CConstraintConjunction : public CConstraint<T>
{
    public:
        using TConstraintCPtr = boost::shared_ptr<const CConstraint<T>>;

    public:
        CConstraintConjunction *addConstraint(const CConstraint<T> *constraint)
        {
            m_Constraints.push_back(TConstraintCPtr(constraint));
            return this;
        }
        bool operator()(const T &value) const
        {
            return this->evaluate(value);
        }
        bool operator()(const std::vector<T> &value) const
        {
            return this->evaluate(value);
        }
        std::string print() const
        {
            std::string result;
            if (m_Constraints.size() > 0)
            {
                result += m_Constraints[0]->print();
                for (std::size_t i = 1u; i < m_Constraints.size(); ++i)
                {
                    result += " && " + m_Constraints[i]->print();
                }
            }
            return result;
        }

    private:
        template<typename U>
        bool evaluate(const U &value) const
        {
            for (std::size_t i = 0u; i < m_Constraints.size(); ++i)
            {
                if (!(*m_Constraints[i])(value))
                {
                    return false;
                }
            }
            return true;
        }

    private:
        std::vector<TConstraintCPtr> m_Constraints;
};

//! \brief Less than.
template<typename T> class CLess : public std::less<T>
{
    public:
        std::string print() const { return "<"; }
};
//! \brief Less than or equal to.
template<typename T> class CLessEqual : public std::less_equal<T>
{
    public:
        std::string print() const { return "<="; }
};
//! \brief Greater than.
template<typename T> class CGreater : public std::greater<T>
{
    public:
        std::string print() const { return ">"; }
};
//! \brief Greater than or equal to.
template<typename T> class CGreaterEqual : public std::greater_equal<T>
{
    public:
        std::string print() const { return ">="; }
};
//! \brief The constraint that a value of type T is greater than another.
template<typename T, template<typename> class PREDICATE>
class CValueIs : public CConstraint<T>
{
    public:
        CValueIs(const T &rhs) : m_Rhs(&rhs) {}
        bool operator()(const T &lhs) const
        {
            return m_Pred(lhs, *m_Rhs);
        }
        std::string print() const
        {
            return m_Pred.print() + core::CStringUtils::typeToString(*m_Rhs);
        }

    private:
        const T *m_Rhs;
        PREDICATE<T> m_Pred;
};
//! \brief The constraint that a value of type T is greater than another.
template<typename T, template<typename> class PREDICATE>
class CVectorValueIs : public CConstraint<T>
{
    public:
        CVectorValueIs(const std::vector<T> &rhs) : m_Rhs(&rhs) {}
        bool operator()(const std::vector<T> &lhs) const
        {
            std::size_t n = std::min(lhs.size(), m_Rhs->size());
            for (std::size_t i = 0u; i < n; ++i)
            {
                if (!m_Pred(lhs[i], (*m_Rhs)[i]))
                {
                    return false;
                }
            }
            return true;
        }
        std::string print() const
        {
            return m_Pred.print() + core::CContainerPrinter::print(*m_Rhs);
        }

    private:
        const std::vector<T> *m_Rhs;
        PREDICATE<T> m_Pred;
};

//! \brief The constraint that a vector isn't empty.
template<typename T>
class CNotEmpty : public CConstraint<T>
{
    public:
        bool operator()(const std::vector<T> &value) const
        {
            return !value.empty();
        }
        std::string print() const
        {
            return "not empty";
        }
};

//! \brief The constraint that a vector has a fixed size.
template<typename T>
class CSizeIs : public CConstraint<T>
{
    public:
        CSizeIs(std::size_t size) : m_Size(size) {}
        bool operator()(const std::vector<T> &value) const
        {
            return value.size() == m_Size;
        }
        std::string print() const
        {
            return "size is " + core::CStringUtils::typeToString(m_Size);
        }

    private:
        std::size_t m_Size;
};


//! \brief Wrapper around parameters so we can process an array in init.
class CParameter : private core::CNonCopyable
{
    public:
        virtual ~CParameter() {}
        bool fromString(std::string value)
        {
            core::CStringUtils::trimWhitespace(value);
            value = core::CStringUtils::normaliseWhitespace(value);
            return this->fromStringImpl(value);
        }

    private:
        virtual bool fromStringImpl(const std::string &value) = 0;
};

//! \brief A parameter which is a built-in type.
template<typename T>
class CBuiltinParameter : public CParameter
{
    public:
        using TConstraintCPtr = boost::shared_ptr<const CConstraint<T>>;

    public:
        CBuiltinParameter(T &value) :
                m_Value(value),
                m_Constraint(new CUnconstrained<T>)
        {}
        CBuiltinParameter(T &value, const CConstraint<T> *constraint) :
                m_Value(value),
                m_Constraint(constraint)
        {}
        CBuiltinParameter(T &value, TConstraintCPtr constraint) :
                m_Value(value),
                m_Constraint(constraint)
        {}

    private:
        virtual bool fromStringImpl(const std::string &value)
        {
            if (boost::is_unsigned<T>::value && this->hasSign(value))
            {
                return false;
            }
            T value_;
            if (!core::CStringUtils::stringToType(value, value_))
            {
                return false;
            }
            if (!(*m_Constraint)(value_))
            {
                LOG_ERROR("'" << value_ << "' doesn't satisfy '" << m_Constraint->print() << "'");
                return false;
            }
            m_Value = value_;
            return true;
        }

        bool hasSign(const std::string &value) const
        {
            return value[0] == '-';
        }

    private:
        T &m_Value;
        TConstraintCPtr m_Constraint;
};

//! \brief A parameter which is a vector of a built-in type.
template<typename T>
class CBuiltinVectorParameter : public CParameter
{
    public:
        CBuiltinVectorParameter(std::vector<T> &value) :
                m_Value(value),
                m_Constraint(new CUnconstrained<T>)
        {}
        CBuiltinVectorParameter(std::vector<T> &value, const CConstraint<T> *constraint) :
                m_Value(value),
                m_Constraint(constraint)
        {}

    private:
        virtual bool fromStringImpl(const std::string &value)
        {
            std::string remainder;
            TStrVec tokens;
            core::CStringUtils::tokenise(std::string(" "), value, tokens, remainder);
            if (!remainder.empty())
            {
                tokens.push_back(remainder);
            }
            std::vector<T> value_(tokens.size());
            for (std::size_t i = 0u; i < tokens.size(); ++i)
            {
                CBuiltinParameter<T> param(value_[i], m_Constraint);
                if (!param.fromString(tokens[i]))
                {
                    return false;
                }
            }
            if (!(*m_Constraint)(value_))
            {
                LOG_ERROR("'" << core::CContainerPrinter::print(value_)
                          << "' doesn't satisfy '" << m_Constraint->print() << "'");
                return false;
            }
            m_Value.swap(value_);
            return true;
        }

    private:
        std::vector<T> &m_Value;
        boost::shared_ptr<const CConstraint<T>> m_Constraint;
};

//! \brief A parameter which is a vector of strings.
class COptionalStrVecParameter : public CParameter
{
    public:
        COptionalStrVecParameter(CAutoconfigurerParams::TOptionalStrVec &value) :
                m_Value(value),
                m_Constraint(new CUnconstrained<std::string>)
        {}
        COptionalStrVecParameter(CAutoconfigurerParams::TOptionalStrVec &value,
                                 const CConstraint<std::string> *constraint) :
                m_Value(value),
                m_Constraint(constraint)
        {}

        virtual bool fromStringImpl(const std::string &value)
        {
            std::string remainder;
            TStrVec value_;
            core::CStringUtils::tokenise(std::string(" "), value, value_, remainder);
            if (!remainder.empty())
            {
                value_.push_back(remainder);
            }
            if (!(*m_Constraint)(value_))
            {
                LOG_ERROR("'" << core::CContainerPrinter::print(value_)
                          << "' doesn't satisfy '" << m_Constraint->print() << "'");
                return false;
            }
            m_Value.reset(TStrVec());
            (*m_Value).swap(value_);
            return true;
        }

    private:
        CAutoconfigurerParams::TOptionalStrVec &m_Value;
        boost::shared_ptr<const CConstraint<std::string>> m_Constraint;
};

//! \brief The field data type parameter.
class CFieldDataTypeParameter : public CParameter
{
    public:
        CFieldDataTypeParameter(CAutoconfigurerParams::TStrUserDataTypePrVec &value) : m_Value(value) {}

    private:
        virtual bool fromStringImpl(const std::string &value)
        {
            std::string remainder;
            TStrVec tokens;
            core::CStringUtils::tokenise(std::string(" "), value, tokens, remainder);
            if (!remainder.empty())
            {
                tokens.push_back(remainder);
            }
            if (tokens.size() % 2 != 0)
            {
                LOG_ERROR("Unmatched field and type in '" << value << "'");
                return false;
            }

            CAutoconfigurerParams::TStrUserDataTypePrVec value_;
            value_.reserve(tokens.size());
            for (std::size_t i = 0u; i < tokens.size(); i += 2)
            {
                config_t::EUserDataType type;
                if (!this->fromString(core::CStringUtils::toLower(tokens[i+1]), type))
                {
                    LOG_ERROR("Couldn't interpret '" << tokens[i+1] << "' as a data type:"
                              << " ignoring field data type for '" << tokens[i] << "'");
                    continue;
                }
                value_.push_back(std::make_pair(tokens[i], type));
            }
            std::sort(value_.begin(), value_.end(), maths::COrderings::SFirstLess());
            m_Value.swap(value_);

            return true;
        }

        bool fromString(const std::string &value, config_t::EUserDataType &type) const
        {
            for (int i = config_t::E_UserCategorical; i <= config_t::E_UserNumeric; ++i)
            {
                type = static_cast<config_t::EUserDataType>(i);
                if (value == config_t::print(type))
                {
                    return true;
                }
            }
            return false;
        }

    private:
        CAutoconfigurerParams::TStrUserDataTypePrVec &m_Value;
};

//! \brief The function category parameter.
class CFunctionCategoryParameter : public CParameter
{
    public:
        CFunctionCategoryParameter(CAutoconfigurerParams::TFunctionCategoryVec &value) :
                m_Value(value),
                m_Constraint(new CUnconstrained<config_t::EFunctionCategory>)
        {}
        CFunctionCategoryParameter(CAutoconfigurerParams::TFunctionCategoryVec &value,
                                   const CConstraint<config_t::EFunctionCategory> *constraint) :
                m_Value(value),
                m_Constraint(constraint)
        {}

    private:
        virtual bool fromStringImpl(const std::string &value)
        {
            std::string remainder;
            TStrVec tokens;
            core::CStringUtils::tokenise(std::string(" "), value, tokens, remainder);
            if (!remainder.empty())
            {
                tokens.push_back(remainder);
            }
            std::sort(tokens.begin(), tokens.end());
            tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());

            CAutoconfigurerParams::TFunctionCategoryVec value_;
            value_.reserve(tokens.size());
            for (std::size_t i = 0u; i < tokens.size(); ++i)
            {
                config_t::EFunctionCategory function;
                if (!this->fromString(core::CStringUtils::toLower(tokens[i]), function))
                {
                    LOG_ERROR("Couldn't interpret '" << tokens[i] << "' as a function");
                    return false;
                }
                value_.push_back(function);
            }
            std::sort(value_.begin(), value_.end());
            if (!(*m_Constraint)(value_))
            {
                LOG_ERROR("'" << core::CContainerPrinter::print(value_)
                          << "' doesn't satisfy '" << m_Constraint->print() << "'");
                return false;
            }
            m_Value.swap(value_);

            return true;
        }

        bool fromString(const std::string &value, config_t::EFunctionCategory &function) const
        {
            for (int i = config_t::E_Count; i <= config_t::E_Median; ++i)
            {
                function = static_cast<config_t::EFunctionCategory>(i);
                if (value == config_t::print(function))
                {
                    return true;
                }
            }
            return false;
        }

    private:
        CAutoconfigurerParams::TFunctionCategoryVec &m_Value;
        boost::shared_ptr<const CConstraint<config_t::EFunctionCategory>> m_Constraint;
};

//! boost::ini_parser doesn't like UTF-8 ini files that begin with
//! byte order markers. This function advances the seek pointer of
//! the stream over a UTF-8 BOM, but only if one exists.
void skipUtf8Bom(std::ifstream &strm)
{
    if (strm.tellg() != std::streampos(0))
    {
        return;
    }
    std::ios_base::iostate origState(strm.rdstate());
    // The 3 bytes 0xEF, 0xBB, 0xBF form a UTF-8 byte order marker (BOM)
    if (strm.get() == 0xEF && strm.get() == 0xBB && strm.get() == 0xBF)
    {
        LOG_DEBUG("Skipping UTF-8 BOM");
        return;
    }
    // Set the stream state back to how it was originally so subsequent
    // code can report errors.
    strm.clear(origState);
    // There was no BOM, so seek back to the beginning of the file.
    strm.seekg(0);
}

//! Helper method for CAutoconfigurerParams::init() to extract parameter
//! value from the property file.
static bool processSetting(const boost::property_tree::ptree &propTree,
                           const std::string &iniPath,
                           CParameter &parameter)
{
    try
    {
        // This get() will throw an exception if the path isn't found
        std::string value = propTree.get<std::string>(iniPath);

        // Use our own string-to-type conversion, because what's built
        // into the boost::property_tree is too lax.
        if (!parameter.fromString(value))
        {
            LOG_ERROR("Invalid value for setting '" << iniPath << "' : " << value);
            return false;
        }
    }
    catch (boost::property_tree::ptree_error &)
    {
        LOG_INFO("Keeping default value for unspecified setting '" << iniPath << "'");
    }

    return true;
}

//! Check if value can be used for one of the detector fields.
bool canUse(const CAutoconfigurerParams::TOptionalStrVec &primary,
            const CAutoconfigurerParams::TOptionalStrVec &secondary,
            const std::string &value)
{
    if (primary)
    {
        return std::find(primary->begin(), primary->end(), value) != primary->end();
    }
    if (secondary)
    {
        return std::find(secondary->begin(), secondary->end(), value) != secondary->end();
    }
    return true;
}

const std::size_t MINIMUM_EXAMPLES_TO_CLASSIFY(1000);
const std::size_t MINIMUM_RECORDS_TO_ATTEMPT_CONFIG(10000);
const double MINIMUM_DETECTOR_SCORE(0.1);
const std::size_t NUMBER_OF_MOST_FREQUENT_FIELDS_COUNTS(10);
std::string DEFAULT_DETECTOR_CONFIG_LINE_ENDING("\n");
const config_t::EFunctionCategory FUNCTION_CATEGORIES[] =
    {
        config_t::E_Count,
        config_t::E_Rare,
        config_t::E_DistinctCount,
        config_t::E_InfoContent,
        config_t::E_Mean,
        config_t::E_Min,
        config_t::E_Max,
        config_t::E_Sum,
        config_t::E_Varp,
        config_t::E_Median
    };
const std::size_t HIGH_NUMBER_BY_FIELD_VALUES(500);
const std::size_t MAXIMUM_NUMBER_BY_FIELD_VALUES(1000);
const std::size_t HIGH_NUMBER_RARE_BY_FIELD_VALUES(50000);
const std::size_t MAXIMUM_NUMBER_RARE_BY_FIELD_VALUES(500000);
const std::size_t HIGH_NUMBER_PARTITION_FIELD_VALUES(500000);
const std::size_t MAXIMUM_NUMBER_PARTITION_FIELD_VALUES(5000000);
const std::size_t LOW_NUMBER_OVER_FIELD_VALUES(50);
const std::size_t MINIMUM_NUMBER_OVER_FIELD_VALUES(5);
const double HIGH_CARDINALITY_IN_TAIL_FACTOR(1.1);
const uint64_t HIGH_CARDINALITY_IN_TAIL_INCREMENT(10);
const double HIGH_CARDINALITY_HIGH_TAIL_FRACTION(0.005);
const double HIGH_CARDINALITY_MAXIMUM_TAIL_FRACTION(0.05);
const double LOW_POPULATED_BUCKET_FRACTIONS[] =
    {
        1.0 / 3.0, 1.0 / 50.0
    };
const double MINIMUM_POPULATED_BUCKET_FRACTIONS[] =
    {
        1.0 / 50.0, 1.0 / 500.0
    };
const double HIGH_POPULATED_BUCKET_FRACTIONS[] =
    {
        1.1, 1.0 / 10.0
    };
const double MAXIMUM_POPULATED_BUCKET_FRACTIONS[] =
    {
        1.2, 5.0 / 10.0
    };
const core_t::TTime CANDIDATE_BUCKET_LENGTHS[] =
    {
        60, 300, 600, 1800, 3600, 7200, 14400, constants::LONGEST_BUCKET_LENGTH
    };
const double LOW_NUMBER_OF_BUCKETS_FOR_CONFIG(500.0);
const double MINIMUM_NUMBER_OF_BUCKETS_FOR_CONFIG(50.0);
const double POLLED_DATA_MINIMUM_MASS_AT_INTERVAL(0.99);
const double POLLED_DATA_JITTER(0.01);
const double LOW_COEFFICIENT_OF_VARIATION(1e-3);
const double MINIMUM_COEFFICIENT_OF_VARIATION(1e-6);
const double LOW_LENGTH_RANGE_FOR_INFO_CONTENT(10.0);
const double MINIMUM_LENGTH_RANGE_FOR_INFO_CONTENT(1.0);
const double LOW_MAXIMUM_LENGTH_FOR_INFO_CONTENT(25.0);
const double MINIMUM_MAXIMUM_LENGTH_FOR_INFO_CONTENT(5.0);
const double LOW_ENTROPY_FOR_INFO_CONTENT(0.01);
const double MINIMUM_ENTROPY_FOR_INFO_CONTENT(1e-6);
const double LOW_DISTINCT_COUNT_FOR_INFO_CONTENT(500000.0);
const double MINIMUM_DISTINCT_COUNT_FOR_INFO_CONTENT(5000.0);

}

CAutoconfigurerParams::CAutoconfigurerParams(const std::string &timeFieldName,
                                             const std::string &timeFieldFormat,
                                             bool verbose,
                                             bool writeDetectorConfigs) :
        m_TimeFieldName(timeFieldName),
        m_TimeFieldFormat(timeFieldFormat),
        m_Verbose(verbose),
        m_WriteDetectorConfigs(writeDetectorConfigs),
        m_DetectorConfigLineEnding(DEFAULT_DETECTOR_CONFIG_LINE_ENDING),
        m_FunctionCategoriesToConfigure(boost::begin(FUNCTION_CATEGORIES), boost::end(FUNCTION_CATEGORIES)),
        m_MinimumExamplesToClassify(MINIMUM_EXAMPLES_TO_CLASSIFY),
        m_NumberOfMostFrequentFieldsCounts(NUMBER_OF_MOST_FREQUENT_FIELDS_COUNTS),
        m_MinimumRecordsToAttemptConfig(MINIMUM_RECORDS_TO_ATTEMPT_CONFIG),
        m_MinimumDetectorScore(MINIMUM_DETECTOR_SCORE),
        m_HighNumberByFieldValues(HIGH_NUMBER_BY_FIELD_VALUES),
        m_MaximumNumberByFieldValues(MAXIMUM_NUMBER_BY_FIELD_VALUES),
        m_HighNumberRareByFieldValues(HIGH_NUMBER_RARE_BY_FIELD_VALUES),
        m_MaximumNumberRareByFieldValues(MAXIMUM_NUMBER_RARE_BY_FIELD_VALUES),
        m_HighNumberPartitionFieldValues(HIGH_NUMBER_PARTITION_FIELD_VALUES),
        m_MaximumNumberPartitionFieldValues(MAXIMUM_NUMBER_PARTITION_FIELD_VALUES),
        m_LowNumberOverFieldValues(LOW_NUMBER_OVER_FIELD_VALUES),
        m_MinimumNumberOverFieldValues(MINIMUM_NUMBER_OVER_FIELD_VALUES),
        m_HighCardinalityInTailFactor(HIGH_CARDINALITY_IN_TAIL_FACTOR),
        m_HighCardinalityInTailIncrement(HIGH_CARDINALITY_IN_TAIL_INCREMENT),
        m_HighCardinalityHighTailFraction(HIGH_CARDINALITY_HIGH_TAIL_FRACTION),
        m_HighCardinalityMaximumTailFraction(HIGH_CARDINALITY_MAXIMUM_TAIL_FRACTION),
        m_LowPopulatedBucketFractions(boost::begin(LOW_POPULATED_BUCKET_FRACTIONS), boost::end(LOW_POPULATED_BUCKET_FRACTIONS)),
        m_MinimumPopulatedBucketFractions(boost::begin(MINIMUM_POPULATED_BUCKET_FRACTIONS), boost::end(MINIMUM_POPULATED_BUCKET_FRACTIONS)),
        m_HighPopulatedBucketFractions(boost::begin(HIGH_POPULATED_BUCKET_FRACTIONS), boost::end(HIGH_POPULATED_BUCKET_FRACTIONS)),
        m_MaximumPopulatedBucketFractions(boost::begin(MAXIMUM_POPULATED_BUCKET_FRACTIONS), boost::end(MAXIMUM_POPULATED_BUCKET_FRACTIONS)),
        m_CandidateBucketLengths(boost::begin(CANDIDATE_BUCKET_LENGTHS), boost::end(CANDIDATE_BUCKET_LENGTHS)),
        m_LowNumberOfBucketsForConfig(LOW_NUMBER_OF_BUCKETS_FOR_CONFIG),
        m_MinimumNumberOfBucketsForConfig(MINIMUM_NUMBER_OF_BUCKETS_FOR_CONFIG),
        m_PolledDataMinimumMassAtInterval(POLLED_DATA_MINIMUM_MASS_AT_INTERVAL),
        m_PolledDataJitter(POLLED_DATA_JITTER),
        m_LowCoefficientOfVariation(LOW_COEFFICIENT_OF_VARIATION),
        m_MinimumCoefficientOfVariation(MINIMUM_COEFFICIENT_OF_VARIATION),
        m_LowLengthRangeForInfoContent(LOW_LENGTH_RANGE_FOR_INFO_CONTENT),
        m_MinimumLengthRangeForInfoContent(MINIMUM_LENGTH_RANGE_FOR_INFO_CONTENT),
        m_LowMaximumLengthForInfoContent(LOW_MAXIMUM_LENGTH_FOR_INFO_CONTENT),
        m_MinimumMaximumLengthForInfoContent(MINIMUM_MAXIMUM_LENGTH_FOR_INFO_CONTENT),
        m_LowEntropyForInfoContent(LOW_ENTROPY_FOR_INFO_CONTENT),
        m_MinimumEntropyForInfoContent(MINIMUM_ENTROPY_FOR_INFO_CONTENT),
        m_LowDistinctCountForInfoContent(LOW_DISTINCT_COUNT_FOR_INFO_CONTENT),
        m_MinimumDistinctCountForInfoContent(MINIMUM_DISTINCT_COUNT_FOR_INFO_CONTENT)
{
    this->refreshPenaltyIndices();
}

bool CAutoconfigurerParams::init(const std::string &file)
{
    if (file.empty())
    {
        return true;
    }

    using TParameterPtr = boost::shared_ptr<CParameter>;

    boost::property_tree::ptree propTree;
    try
    {
        std::ifstream strm(file.c_str());
        if (!strm.is_open())
        {
            LOG_ERROR("Error opening file " << file);
            return false;
        }
        skipUtf8Bom(strm);
        boost::property_tree::ini_parser::read_ini(strm, propTree);
    }
    catch (boost::property_tree::ptree_error &e)
    {
        LOG_ERROR("Error reading file " << file << " : " << e.what());
        return false;
    }

    static const core_t::TTime ZERO_TIME = 0;
    static const double ZERO_DOUBLE = 0.0;
    static const double ONE_DOUBLE  = 1.0;
    static const std::string LABELS[] =
        {
            std::string("scope.fields_of_interest"),
            std::string("scope.permitted_argument_fields"),
            std::string("scope.permitted_by_fields"),
            std::string("scope.permitted_over_fields"),
            std::string("scope.permitted_partition_fields"),
            std::string("scope.functions_of_interest"),
            std::string("statistics.field_data_types"),
            std::string("statistics.minimum_examples_to_classify"),
            std::string("statistics.number_of_most_frequent_to_count"),
            std::string("configuration.minimum_records_to_attempt_config"),
            std::string("configuration.high_number_of_by_fields"),
            std::string("configuration.maximum_number_of_by_fields"),
            std::string("configuration.high_number_of_rare_by_fields"),
            std::string("configuration.maximum_number_of_rare_by_fields"),
            std::string("configuration.high_number_of_partition_fields"),
            std::string("configuration.maximum_of_number_partition_fields"),
            std::string("configuration.low_number_of_over_fields"),
            std::string("configuration.minimum_number_of_over_fields"),
            std::string("configuration.high_cardinality_in_tail_factor"),
            std::string("configuration.high_cardinality_in_tail_increment"),
            std::string("configuration.high_cardinality_high_tail_fraction"),
            std::string("configuration.high_cardinality_maximum_tail_fraction"),
            std::string("configuration.low_populated_bucket_ratio"),
            std::string("configuration.minimum_populated_bucket_ratio"),
            std::string("configuration.high_populated_bucket_ratio"),
            std::string("configuration.maximum_populated_bucket_ratio"),
            std::string("configuration.candidate_bucket_lengths"),
            std::string("configuration.low_number_buckets_for_config"),
            std::string("configuration.minimum_number_buckets_for_config"),
            std::string("configuration.polled_data_minimum_mass_at_interval"),
            std::string("configuration.polled_data_jitter"),
            std::string("configuration.low_coefficient_of_variation"),
            std::string("configuration.minimum_coefficient_of_variation"),
            std::string("configuration.low_length_range_for_info_content"),
            std::string("configuration.minimum_length_range_for_info_content"),
            std::string("configuration.low_maximum_length_for_info_content"),
            std::string("configuration.minimum_maximum_length_for_info_content"),
            std::string("configuration.low_entropy_for_info_content"),
            std::string("configuration.minimum_entropy_for_info_content"),
            std::string("configuration.low_distinct_count_for_info_content"),
            std::string("configuration.minimum_distinct_count_for_info_content")

        };
    TParameterPtr parameters[] =
        {
            TParameterPtr(new COptionalStrVecParameter(m_FieldsOfInterest, new CNotEmpty<std::string>)),
            TParameterPtr(new COptionalStrVecParameter(m_FieldsToUseInAutoconfigureByRole[constants::ARGUMENT_INDEX])),
            TParameterPtr(new COptionalStrVecParameter(m_FieldsToUseInAutoconfigureByRole[constants::BY_INDEX])),
            TParameterPtr(new COptionalStrVecParameter(m_FieldsToUseInAutoconfigureByRole[constants::OVER_INDEX])),
            TParameterPtr(new COptionalStrVecParameter(m_FieldsToUseInAutoconfigureByRole[constants::PARTITION_INDEX])),
            TParameterPtr(new CFunctionCategoryParameter(m_FunctionCategoriesToConfigure)),
            TParameterPtr(new CFieldDataTypeParameter(m_FieldDataTypes)),
            TParameterPtr(new CBuiltinParameter<uint64_t>(m_MinimumExamplesToClassify)),
            TParameterPtr(new CBuiltinParameter<std::size_t>(m_NumberOfMostFrequentFieldsCounts)),
            TParameterPtr(new CBuiltinParameter<uint64_t>(
                                  m_MinimumRecordsToAttemptConfig,
                                  new CValueIs<uint64_t, CGreater>(m_MinimumExamplesToClassify))),
            TParameterPtr(new CBuiltinParameter<std::size_t>(m_HighNumberByFieldValues)),
            TParameterPtr(new CBuiltinParameter<std::size_t>(
                                  m_MaximumNumberByFieldValues,
                                  new CValueIs<std::size_t, CGreaterEqual>(m_HighNumberByFieldValues))),
            TParameterPtr(new CBuiltinParameter<std::size_t>(m_HighNumberRareByFieldValues)),
            TParameterPtr(new CBuiltinParameter<std::size_t>(
                                  m_MaximumNumberRareByFieldValues,
                                  new CValueIs<std::size_t, CGreaterEqual>(m_HighNumberRareByFieldValues))),
            TParameterPtr(new CBuiltinParameter<std::size_t>(m_HighNumberPartitionFieldValues)),
            TParameterPtr(new CBuiltinParameter<std::size_t>(
                                  m_MaximumNumberPartitionFieldValues,
                                  new CValueIs<std::size_t, CGreaterEqual>(m_HighNumberPartitionFieldValues))),
            TParameterPtr(new CBuiltinParameter<std::size_t>(m_LowNumberOverFieldValues)),
            TParameterPtr(new CBuiltinParameter<std::size_t>(
                                  m_MinimumNumberOverFieldValues,
                                  new CValueIs<std::size_t, CLessEqual>(m_LowNumberOverFieldValues))),
            TParameterPtr(new CBuiltinParameter<double>(
                                  m_HighCardinalityInTailFactor,
                                  new CValueIs<double, CGreaterEqual>(ONE_DOUBLE))),
            TParameterPtr(new CBuiltinParameter<uint64_t>(m_HighCardinalityInTailIncrement)),
            TParameterPtr(new CBuiltinParameter<double>(
                                  m_HighCardinalityHighTailFraction,
                                 (new CConstraintConjunction<double>)
                                         ->addConstraint(new CValueIs<double, CGreaterEqual>(ZERO_DOUBLE))
                                         ->addConstraint(new CValueIs<double, CLessEqual>(ONE_DOUBLE)))),
            TParameterPtr(new CBuiltinParameter<double>(
                                  m_HighCardinalityMaximumTailFraction,
                                 (new CConstraintConjunction<double>)
                                         ->addConstraint(new CValueIs<double, CGreaterEqual>(m_HighCardinalityHighTailFraction))
                                         ->addConstraint(new CValueIs<double, CLessEqual>(ONE_DOUBLE)))),
            TParameterPtr(new CBuiltinVectorParameter<double>(
                                  m_LowPopulatedBucketFractions,
                                 (new CConstraintConjunction<double>)
                                         ->addConstraint(new CValueIs<double, CGreaterEqual>(ZERO_DOUBLE))
                                         ->addConstraint(new CValueIs<double, CLessEqual>(ONE_DOUBLE))
                                         ->addConstraint(new CSizeIs<double>(2)))),
            TParameterPtr(new CBuiltinVectorParameter<double>(
                                  m_MinimumPopulatedBucketFractions,
                                 (new CConstraintConjunction<double>)
                                         ->addConstraint(new CValueIs<double, CGreaterEqual>(ZERO_DOUBLE))
                                         ->addConstraint(new CVectorValueIs<double, CLessEqual>(m_LowPopulatedBucketFractions))
                                         ->addConstraint(new CSizeIs<double>(2)))),
            TParameterPtr(new CBuiltinParameter<double>(
                                  m_HighPopulatedBucketFractions[1],
                                 (new CConstraintConjunction<double>)
                                         ->addConstraint(new CValueIs<double, CGreaterEqual>(ZERO_DOUBLE))
                                         ->addConstraint(new CValueIs<double, CLessEqual>(ONE_DOUBLE)))),
            TParameterPtr(new CBuiltinParameter<double>(
                                  m_MaximumPopulatedBucketFractions[1],
                                 (new CConstraintConjunction<double>)
                                         ->addConstraint(new CVectorValueIs<double, CGreaterEqual>(m_HighPopulatedBucketFractions))
                                         ->addConstraint(new CValueIs<double, CLessEqual>(ONE_DOUBLE)))),
            TParameterPtr(new CBuiltinVectorParameter<core_t::TTime>(
                                  m_CandidateBucketLengths,
                                 (new CConstraintConjunction<core_t::TTime>)
                                         ->addConstraint(new CValueIs<core_t::TTime, CGreater>(ZERO_TIME))
                                         ->addConstraint(new CNotEmpty<core_t::TTime>))),
            TParameterPtr(new CBuiltinParameter<double>(
                                  m_LowNumberOfBucketsForConfig,
                                  new CValueIs<double, CGreater>(ZERO_DOUBLE))),
            TParameterPtr(new CBuiltinParameter<double>(
                                  m_MinimumNumberOfBucketsForConfig,
                                  (new CConstraintConjunction<double>)
                                         ->addConstraint(new CValueIs<double, CGreater>(ZERO_DOUBLE))
                                         ->addConstraint(new CValueIs<double, CLessEqual>(m_LowNumberOfBucketsForConfig)))),
            TParameterPtr(new CBuiltinParameter<double>(
                                  m_PolledDataMinimumMassAtInterval,
                                 (new CConstraintConjunction<double>)
                                         ->addConstraint(new CValueIs<double, CGreaterEqual>(ZERO_DOUBLE))
                                         ->addConstraint(new CValueIs<double, CLessEqual>(ONE_DOUBLE)))),
            TParameterPtr(new CBuiltinParameter<double>(
                                  m_PolledDataJitter,
                                 (new CConstraintConjunction<double>)
                                         ->addConstraint(new CValueIs<double, CGreaterEqual>(ZERO_DOUBLE))
                                         ->addConstraint(new CValueIs<double, CLessEqual>(ONE_DOUBLE)))),
            TParameterPtr(new CBuiltinParameter<double>(
                                  m_LowCoefficientOfVariation,
                                 (new CConstraintConjunction<double>)
                                         ->addConstraint(new CValueIs<double, CGreaterEqual>(ZERO_DOUBLE)))),
            TParameterPtr(new CBuiltinParameter<double>(
                                  m_MinimumCoefficientOfVariation,
                                 (new CConstraintConjunction<double>)
                                         ->addConstraint(new CValueIs<double, CGreaterEqual>(ZERO_DOUBLE))
                                         ->addConstraint(new CValueIs<double, CLessEqual>(m_LowCoefficientOfVariation)))),
            TParameterPtr(new CBuiltinParameter<double>(
                                  m_LowLengthRangeForInfoContent,
                                 (new CConstraintConjunction<double>)
                                         ->addConstraint(new CValueIs<double, CGreaterEqual>(ZERO_DOUBLE)))),
            TParameterPtr(new CBuiltinParameter<double>(
                                  m_MinimumLengthRangeForInfoContent,
                                 (new CConstraintConjunction<double>)
                                         ->addConstraint(new CValueIs<double, CGreaterEqual>(ZERO_DOUBLE))
                                         ->addConstraint(new CValueIs<double, CLessEqual>(m_LowLengthRangeForInfoContent)))),
            TParameterPtr(new CBuiltinParameter<double>(
                                  m_LowMaximumLengthForInfoContent,
                                 (new CConstraintConjunction<double>)
                                         ->addConstraint(new CValueIs<double, CGreaterEqual>(ZERO_DOUBLE)))),
            TParameterPtr(new CBuiltinParameter<double>(
                                  m_MinimumMaximumLengthForInfoContent,
                                 (new CConstraintConjunction<double>)
                                         ->addConstraint(new CValueIs<double, CGreaterEqual>(ZERO_DOUBLE))
                                         ->addConstraint(new CValueIs<double, CLessEqual>(m_LowMaximumLengthForInfoContent)))),
            TParameterPtr(new CBuiltinParameter<double>(
                                  m_LowEntropyForInfoContent,
                                 (new CConstraintConjunction<double>)
                                         ->addConstraint(new CValueIs<double, CGreaterEqual>(ZERO_DOUBLE))
                                         ->addConstraint(new CValueIs<double, CLessEqual>(ONE_DOUBLE)))),
            TParameterPtr(new CBuiltinParameter<double>(
                                  m_MinimumEntropyForInfoContent,
                                 (new CConstraintConjunction<double>)
                                         ->addConstraint(new CValueIs<double, CGreaterEqual>(ZERO_DOUBLE))
                                         ->addConstraint(new CValueIs<double, CLessEqual>(m_LowEntropyForInfoContent)))),
            TParameterPtr(new CBuiltinParameter<double>(
                                  m_LowDistinctCountForInfoContent,
                                 (new CConstraintConjunction<double>)
                                         ->addConstraint(new CValueIs<double, CGreaterEqual>(ZERO_DOUBLE)))),
            TParameterPtr(new CBuiltinParameter<double>(
                                  m_MinimumDistinctCountForInfoContent,
                                 (new CConstraintConjunction<double>)
                                         ->addConstraint(new CValueIs<double, CGreaterEqual>(ZERO_DOUBLE))
                                         ->addConstraint(new CValueIs<double, CLessEqual>(m_LowDistinctCountForInfoContent))))
        };

    bool result = true;
    for (std::size_t i = 0u; i < boost::size(LABELS); ++i)
    {
        if (processSetting(propTree, LABELS[i], *parameters[i]) == false)
        {
            result = false;
        }
    }
    if (!result)
    {
        LOG_ERROR("Error processing config file " << file);
    }
    this->refreshPenaltyIndices();
    return result;
}

const std::string &CAutoconfigurerParams::timeFieldName() const
{
    return m_TimeFieldName;
}

const std::string &CAutoconfigurerParams::timeFieldFormat() const
{
    return m_TimeFieldFormat;
}

bool CAutoconfigurerParams::verbose() const
{
    return m_Verbose;
}

bool CAutoconfigurerParams::writeDetectorConfigs() const
{
    return m_WriteDetectorConfigs;
}

const std::string &CAutoconfigurerParams::detectorConfigLineEnding() const
{
    return m_DetectorConfigLineEnding;
}

bool CAutoconfigurerParams::fieldOfInterest(const std::string &field) const
{
    if (m_FieldsOfInterest)
    {
        return std::find(m_FieldsOfInterest->begin(), m_FieldsOfInterest->end(), field) != m_FieldsOfInterest->end();
    }
    return true;
}

bool CAutoconfigurerParams::canUseForFunctionArgument(const std::string &argument) const
{
    return canUse(m_FieldsToUseInAutoconfigureByRole[constants::ARGUMENT_INDEX], m_FieldsOfInterest, argument);
}

bool CAutoconfigurerParams::canUseForByField(const std::string &by) const
{
    return canUse(m_FieldsToUseInAutoconfigureByRole[constants::BY_INDEX], m_FieldsOfInterest, by);
}

bool CAutoconfigurerParams::canUseForOverField(const std::string &over) const
{
    return canUse(m_FieldsToUseInAutoconfigureByRole[constants::OVER_INDEX], m_FieldsOfInterest, over);
}

bool CAutoconfigurerParams::canUseForPartitionField(const std::string &partition) const
{
    return canUse(m_FieldsToUseInAutoconfigureByRole[constants::PARTITION_INDEX], m_FieldsOfInterest, partition);
}

const CAutoconfigurerParams::TFunctionCategoryVec &CAutoconfigurerParams::functionsCategoriesToConfigure() const
{
    return m_FunctionCategoriesToConfigure;
}

CAutoconfigurerParams::TOptionalUserDataType CAutoconfigurerParams::dataType(const std::string &field) const
{
    TStrUserDataTypePrVec::const_iterator result =
            std::lower_bound(m_FieldDataTypes.begin(),
                             m_FieldDataTypes.end(),
                             field, maths::COrderings::SFirstLess());
    return result != m_FieldDataTypes.end() && result->first == field ?
           TOptionalUserDataType(result->second) : TOptionalUserDataType();
}

uint64_t CAutoconfigurerParams::minimumExamplesToClassify() const
{
    return m_MinimumExamplesToClassify;
}

std::size_t CAutoconfigurerParams::numberOfMostFrequentFieldsCounts() const
{
    return m_NumberOfMostFrequentFieldsCounts;
}

uint64_t CAutoconfigurerParams::minimumRecordsToAttemptConfig() const
{
    return m_MinimumRecordsToAttemptConfig;
}

double CAutoconfigurerParams::minimumDetectorScore() const
{
    return m_MinimumDetectorScore;
}

std::size_t CAutoconfigurerParams::highNumberByFieldValues() const
{
    return m_HighNumberByFieldValues;
}

std::size_t CAutoconfigurerParams::maximumNumberByFieldValues() const
{
    return m_MaximumNumberByFieldValues;
}

std::size_t CAutoconfigurerParams::highNumberRareByFieldValues() const
{
    return m_HighNumberRareByFieldValues;
}

std::size_t CAutoconfigurerParams::maximumNumberRareByFieldValues() const
{
    return m_MaximumNumberRareByFieldValues;
}

std::size_t CAutoconfigurerParams::highNumberPartitionFieldValues() const
{
    return m_HighNumberPartitionFieldValues;
}

std::size_t CAutoconfigurerParams::maximumNumberPartitionFieldValues() const
{
    return m_MaximumNumberPartitionFieldValues;
}

std::size_t CAutoconfigurerParams::lowNumberOverFieldValues() const
{
    return m_LowNumberOverFieldValues;
}

std::size_t CAutoconfigurerParams::minimumNumberOverFieldValues() const
{
    return m_MinimumNumberOverFieldValues;
}

double CAutoconfigurerParams::highCardinalityInTailFactor() const
{
    return m_HighCardinalityInTailFactor;
}

uint64_t CAutoconfigurerParams::highCardinalityInTailIncrement() const
{
    return m_HighCardinalityInTailIncrement;
}

double CAutoconfigurerParams::highCardinalityHighTailFraction() const
{
    return m_HighCardinalityHighTailFraction;
}

double CAutoconfigurerParams::highCardinalityMaximumTailFraction() const
{
    return m_HighCardinalityMaximumTailFraction;
}

double CAutoconfigurerParams::lowPopulatedBucketFraction(config_t::EFunctionCategory function, bool ignoreEmpty) const
{
    return m_LowPopulatedBucketFractions[config_t::hasDoAndDontIgnoreEmptyVersions(function) && ignoreEmpty];
}

double CAutoconfigurerParams::minimumPopulatedBucketFraction(config_t::EFunctionCategory function, bool ignoreEmpty) const
{
    return m_MinimumPopulatedBucketFractions[config_t::hasDoAndDontIgnoreEmptyVersions(function) && ignoreEmpty];
}

double CAutoconfigurerParams::highPopulatedBucketFraction(config_t::EFunctionCategory function, bool ignoreEmpty) const
{
    return m_HighPopulatedBucketFractions[config_t::hasDoAndDontIgnoreEmptyVersions(function) && ignoreEmpty];
}

double CAutoconfigurerParams::maximumPopulatedBucketFraction(config_t::EFunctionCategory function, bool ignoreEmpty) const
{
    return m_MaximumPopulatedBucketFractions[config_t::hasDoAndDontIgnoreEmptyVersions(function) && ignoreEmpty];
}

const CAutoconfigurerParams::TTimeVec &CAutoconfigurerParams::candidateBucketLengths() const
{
    return m_CandidateBucketLengths;
}

double CAutoconfigurerParams::lowNumberOfBucketsForConfig() const
{
    return m_LowNumberOfBucketsForConfig;
}

double CAutoconfigurerParams::minimumNumberOfBucketsForConfig() const
{
    return m_MinimumNumberOfBucketsForConfig;
}

double CAutoconfigurerParams::polledDataMinimumMassAtInterval() const
{
    return m_PolledDataMinimumMassAtInterval;
}

double CAutoconfigurerParams::polledDataJitter() const
{
    return m_PolledDataJitter;
}

double CAutoconfigurerParams::lowCoefficientOfVariation() const
{
    return m_LowCoefficientOfVariation;
}

double CAutoconfigurerParams::minimumCoefficientOfVariation() const
{
    return m_MinimumCoefficientOfVariation;
}

double CAutoconfigurerParams::lowLengthRangeForInfoContent() const
{
    return m_LowLengthRangeForInfoContent;
}

double CAutoconfigurerParams::minimumLengthRangeForInfoContent() const
{
    return m_MinimumLengthRangeForInfoContent;
}

double CAutoconfigurerParams::lowMaximumLengthForInfoContent() const
{
    return m_LowMaximumLengthForInfoContent;
}

double CAutoconfigurerParams::minimumMaximumLengthForInfoContent() const
{
    return m_MinimumMaximumLengthForInfoContent;
}

double CAutoconfigurerParams::lowEntropyForInfoContent() const
{
    return m_LowEntropyForInfoContent;
}

double CAutoconfigurerParams::minimumEntropyForInfoContent() const
{
    return m_MinimumEntropyForInfoContent;
}

double CAutoconfigurerParams::lowDistinctCountForInfoContent() const
{
    return m_LowDistinctCountForInfoContent;
}

double CAutoconfigurerParams::minimumDistinctCountForInfoContent() const
{
    return m_MinimumDistinctCountForInfoContent;
}

const CAutoconfigurerParams::TSizeVec &CAutoconfigurerParams::penaltyIndicesFor(std::size_t bid) const
{
    return m_BucketLengthPenaltyIndices[bid];
}

const CAutoconfigurerParams::TSizeVec &CAutoconfigurerParams::penaltyIndicesFor(bool ignoreEmpty) const
{
    return m_IgnoreEmptyPenaltyIndices[ignoreEmpty];
}

std::size_t CAutoconfigurerParams::penaltyIndexFor(std::size_t bid, bool ignoreEmpty) const
{
    TSizeVec result;
    std::set_intersection(this->penaltyIndicesFor(bid).begin(), this->penaltyIndicesFor(bid).end(),
                          this->penaltyIndicesFor(ignoreEmpty).begin(), this->penaltyIndicesFor(ignoreEmpty).end(),
                          std::back_inserter(result));
    return result[0];
}

std::string CAutoconfigurerParams::print() const
{
#define PRINT_STRING(field) result += "  "#field" = " + m_##field + "\n"
#define PRINT_VALUE(field) result += "  "#field" = " + core::CStringUtils::typeToString(m_##field) + "\n"
#define PRINT_CONTAINER(field) result += "  "#field" = " + core::CContainerPrinter::print(m_##field) + "\n"

    std::string result;
    PRINT_STRING(TimeFieldName);
    PRINT_STRING(TimeFieldFormat);
    PRINT_CONTAINER(FieldsOfInterest);
    PRINT_CONTAINER(FieldsToUseInAutoconfigureByRole[constants::ARGUMENT_INDEX]);
    PRINT_CONTAINER(FieldsToUseInAutoconfigureByRole[constants::BY_INDEX]);
    PRINT_CONTAINER(FieldsToUseInAutoconfigureByRole[constants::OVER_INDEX]);
    PRINT_CONTAINER(FieldsToUseInAutoconfigureByRole[constants::PARTITION_INDEX]);
    result += "  FunctionCategoriesToConfigure = [";
    if (m_FunctionCategoriesToConfigure.size() > 0)
    {
        result += config_t::print(m_FunctionCategoriesToConfigure[0]);
        for (std::size_t i = 1u; i < m_FunctionCategoriesToConfigure.size(); ++i)
        {
            result += ", " + config_t::print(m_FunctionCategoriesToConfigure[i]);
        }
    }
    result += "]\n";
    result += "  FieldDataType = [";
    if (m_FieldDataTypes.size() > 0)
    {
        result += "(" + m_FieldDataTypes[0].first + "," + config_t::print(m_FieldDataTypes[0].second) + ")";
        for (std::size_t i = 1u; i < m_FieldDataTypes.size(); ++i)
        {
            result += ", (" + m_FieldDataTypes[i].first + "," + config_t::print(m_FieldDataTypes[i].second) + ")";
        }
    }
    result += "]\n";
    PRINT_VALUE(MinimumExamplesToClassify);
    PRINT_VALUE(NumberOfMostFrequentFieldsCounts);
    PRINT_VALUE(MinimumRecordsToAttemptConfig);
    PRINT_VALUE(HighNumberByFieldValues);
    PRINT_VALUE(MaximumNumberByFieldValues);
    PRINT_VALUE(HighNumberRareByFieldValues);
    PRINT_VALUE(MaximumNumberRareByFieldValues);
    PRINT_VALUE(HighNumberPartitionFieldValues);
    PRINT_VALUE(MaximumNumberPartitionFieldValues);
    PRINT_VALUE(LowNumberOverFieldValues);
    PRINT_VALUE(MinimumNumberOverFieldValues);
    PRINT_VALUE(HighCardinalityInTailFactor);
    PRINT_VALUE(HighCardinalityInTailIncrement);
    PRINT_VALUE(HighCardinalityHighTailFraction);
    PRINT_VALUE(HighCardinalityMaximumTailFraction);
    PRINT_CONTAINER(LowPopulatedBucketFractions);
    PRINT_CONTAINER(MinimumPopulatedBucketFractions);
    PRINT_VALUE(HighPopulatedBucketFractions[1]);
    PRINT_VALUE(MaximumPopulatedBucketFractions[1]);
    PRINT_CONTAINER(CandidateBucketLengths);
    PRINT_VALUE(LowNumberOfBucketsForConfig);
    PRINT_VALUE(MinimumNumberOfBucketsForConfig);
    PRINT_VALUE(PolledDataMinimumMassAtInterval);
    PRINT_VALUE(PolledDataJitter);
    PRINT_VALUE(LowCoefficientOfVariation);
    PRINT_VALUE(MinimumCoefficientOfVariation);
    PRINT_VALUE(LowLengthRangeForInfoContent);
    PRINT_VALUE(MinimumLengthRangeForInfoContent);
    PRINT_VALUE(LowMaximumLengthForInfoContent);
    PRINT_VALUE(MinimumMaximumLengthForInfoContent);
    PRINT_VALUE(LowEntropyForInfoContent);
    PRINT_VALUE(MinimumEntropyForInfoContent);
    PRINT_VALUE(LowDistinctCountForInfoContent);
    PRINT_VALUE(MinimumDistinctCountForInfoContent);

    return result;
}

void CAutoconfigurerParams::refreshPenaltyIndices()
{
    m_BucketLengthPenaltyIndices.resize(m_CandidateBucketLengths.size(), TSizeVec(2));
    m_IgnoreEmptyPenaltyIndices.resize(2, TSizeVec(m_CandidateBucketLengths.size()));
    for (std::size_t i = 0u, n = m_CandidateBucketLengths.size(); i < m_CandidateBucketLengths.size(); ++i)
    {
        m_BucketLengthPenaltyIndices[i][0] = 0 + i;
        m_BucketLengthPenaltyIndices[i][1] = n + i;
        m_IgnoreEmptyPenaltyIndices[0][i]  = 0 + i;
        m_IgnoreEmptyPenaltyIndices[1][i]  = n + i;
    }
}

}
}
