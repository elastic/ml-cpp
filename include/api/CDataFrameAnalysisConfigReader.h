/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_api_CDataFrameAnalysisConfigReader_h
#define INCLUDED_ml_api_CDataFrameAnalysisConfigReader_h

#include <core/CLogger.h>

#include <api/ImportExport.h>

#include <rapidjson/document.h>

#include <map>
#include <string>
#include <type_traits>
#include <vector>

namespace ml {
namespace api {
class CDataFrameAnalysisParameters;

//! \brief Reads and validates parameters for a data frame analysis.
//!
//! DESCRIPTION:\n
//! This wraps up extracting parameter values from a JSON configuration object.
//! It supports predefining a set of expected parameters. It is expected that
//! there will be one static reader object per analysis type.
//!
//! The read method extracts all the parameters the JSON contains and returns
//! a collection to get them by name.
//!
//! Expected usage is:
//! \code
//! static const CDataFrameAnalysisConfigReader reader{[] {
//!     CDataFrameAnalysisConfigReader theReader;
//!     theReader.addParameter("foo", CDataFrameAnalysisConfigReader::E_OptionalParameter);
//!     theReader.addParameter("bar", CDataFrameAnalysisConfigReader::E_RequiredParameter);
//! }()};
//!
//! auto parameters = reader.read(json);
//! bool foo{parameters["foo"].fallback(true)};
//! double bar{parameters["bar"].as<double>()};
//! \endcode
//!
//! IMPLEMENTATION:\n
//! Not understanding a parameter is likely to result in the wrong analysis being
//! performed so any errors are treated as fatal causing the process to terminate
//! immediately. For the same reason this treats unexpected parameters as an error.
//! It is the calling code's responsibility to ask for the right type for a parameter.
//! If a parameter is known to exist, i.e. it is required, then it can be accessed
//! from the read value by calling as<T>(), for type T, otherwise use the fallback
//! method and provide the default value for the case it is missing.
class API_EXPORT CDataFrameAnalysisConfigReader {
public:
    using TStrIntMap = std::map<std::string, int>;

    enum ERequirement { E_OptionalParameter, E_RequiredParameter };

    //! \brief A single parameter which has been read.
    class API_EXPORT CParameter {
    public:
        explicit CParameter(const std::string& name) : m_Name{name} {}
        CParameter(const std::string& name,
                   const rapidjson::Value& value,
                   const TStrIntMap& permittedValues);

        //! Get the name of the parameter.
        const std::string& name() const { return m_Name; }
        //! Get the parameter of type T.
        template<typename T>
        T as() const {
            if (m_Value == nullptr) {
                HANDLE_FATAL(<< "Input error: expected value for '" << m_Name
                             << "'. Please report this problem.");
            }
            return this->fallback(T{});
        }
        //! Get the JSON object.
        const rapidjson::Value* jsonObject() { return m_Value; }
        //! Get a boolean parameter.
        bool fallback(bool value) const;
        //! Get an unsigned integer parameter.
        std::size_t fallback(std::size_t value) const;
        //! Get a floating point parameter.
        double fallback(double value) const;
        //! Get a string parameter.
        std::string fallback(const std::string& value) const;
        //! Get an enum point parameter.
        template<typename ENUM>
        ENUM fallback(ENUM value) const {
            static_assert(std::is_enum<ENUM>::value, "ENUM must be an enumeration");
            if (m_Value == nullptr) {
                return value;
            }
            if (m_Value->IsString() == false) {
                this->handleFatal();
                return value;
            }
            auto pos = m_PermittedValues->find(std::string{m_Value->GetString()});
            if (pos == m_PermittedValues->end()) {
                this->handleFatal();
                return value;
            }
            return static_cast<ENUM>(pos->second);
        }
        //! Get an array of objects of type T.
        template<typename T>
        std::vector<T> fallback(const std::vector<T>& value) const {
            if (m_Value == nullptr) {
                return value;
            }
            if (m_Value->IsArray() == false) {
                this->handleFatal();
                return value;
            }
            std::vector<T> result;
            result.reserve(m_Value->Size());
            CParameter element{m_Name, SArrayElementTag{}};
            for (std::size_t i = 0; i < m_Value->Size(); ++i) {
                element.m_Value = &(*m_Value)[static_cast<int>(i)];
                result.push_back(element.as<T>());
            }
            return result;
        }

    private:
        struct SArrayElementTag {};

    private:
        CParameter(const std::string& name, SArrayElementTag);
        void handleFatal() const;

    private:
        std::string m_Name;
        const rapidjson::Value* m_Value = nullptr;
        const TStrIntMap* m_PermittedValues = nullptr;
        bool m_ArrayElement = false;
    };

public:
    //! Register a parameter.
    //!
    //! \param[in] name The parameter name.
    //! \param[in] requirement Is the parameter required or optional.
    //! \param[in] permittedValues The permitted values for an enumeration.
    void addParameter(const std::string& name,
                      ERequirement requirement,
                      TStrIntMap permittedValues = TStrIntMap{});

    //! Extract the parameters from a JSON object.
    CDataFrameAnalysisParameters read(const rapidjson::Value& json) const;

private:
    //! Reads a parameter from the JSON configuration object.
    class API_EXPORT CParameterReader {
    public:
        CParameterReader(const std::string& name, ERequirement requirement, TStrIntMap permittedValues);

        const std::string& name() const { return m_Name; }
        bool required() const { return m_Requirement == E_RequiredParameter; }
        CParameter readFrom(const rapidjson::Value& json) const {
            return {m_Name, json[m_Name], m_PermittedValues};
        }

    private:
        std::string m_Name;
        ERequirement m_Requirement;
        TStrIntMap m_PermittedValues;
    };

private:
    std::vector<CParameterReader> m_ParameterReaders;
};

//! \brief A collection of all the data frame analysis parameters which have been read.
class API_EXPORT CDataFrameAnalysisParameters {
public:
    using TParameter = CDataFrameAnalysisConfigReader::CParameter;

    //! Add \p parameter.
    void add(const CDataFrameAnalysisConfigReader::CParameter& parameter) {
        m_ParameterValues.push_back(parameter);
    }

    //! Get the parameter called \p name.
    TParameter operator[](const std::string& name) const {
        for (const auto& value : m_ParameterValues) {
            if (name == value.name()) {
                return value;
            }
        }
        return TParameter{name};
    }

private:
    std::vector<TParameter> m_ParameterValues;
};
}
}

#endif // INCLUDED_ml_api_CDataFrameAnalysisConfigReader_h
