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

#include <api/CDataFrameAnalysisConfigReader.h>

#include <core/CLogger.h>

#include <boost/json.hpp>

#include <cstring>

namespace ml {
namespace api {
namespace {
std::string toString(const json::value& value) {
    return json::serialize(value);
}
}

void CDataFrameAnalysisConfigReader::addParameter(const std::string& name,
                                                  ERequirement requirement,
                                                  TStrIntMap permittedValues) {
    m_ParameterReaders.emplace_back(name, requirement, std::move(permittedValues));
}

CDataFrameAnalysisParameters CDataFrameAnalysisConfigReader::read(const json::value& json) const {

    CDataFrameAnalysisParameters result;

    if (json.is_object() == false) {
        HANDLE_FATAL(<< "Input error: expected JSON object but input was '"
                     << toString(json) << "'. Please report this problem.");
        return result;
    }

    const json::object& obj = json.as_object();

    for (const auto& reader : m_ParameterReaders) {
        if (obj.contains(reader.name())) {
            result.add(reader.readFrom(obj));
        } else if (reader.required()) {
            HANDLE_FATAL(<< "Input error: missing required parameter '"
                         << reader.name() << "'. Please report this problem.");
        } else {
            result.add(CParameter{reader.name()});
        }
    }

    // Check for any unrecognised fields: these might be typos.
    for (auto i = obj.begin(); i != obj.end(); ++i) {
        bool found{false};
        for (const auto& param : m_ParameterReaders) {
            if (i->key() == param.name()) {
                found = true;
                break;
            }
        }
        if (found == false) {
            HANDLE_FATAL(<< "Input error: unexpected parameter '" << i->key()
                         << "'. Please report this problem.");
        }
    }

    return result;
}

CDataFrameAnalysisConfigReader::CParameter::CParameter(const std::string& name,
                                                       const json::value& value,
                                                       const TStrIntMap& permittedValues)
    : m_Name{name}, m_Value{&value}, m_PermittedValues{&permittedValues} {
}

bool CDataFrameAnalysisConfigReader::CParameter::fallback(bool fallback) const {
    if (m_Value == nullptr) {
        return fallback;
    }
    if (m_Value->is_bool() == false) {
        this->handleFatal();
        return fallback;
    }
    return m_Value->as_bool();
}

std::size_t CDataFrameAnalysisConfigReader::CParameter::fallback(std::size_t fallback) const {
    if (m_Value == nullptr) {
        return fallback;
    }
    if (m_Value->is_int64() == false) {
        this->handleFatal();
        return fallback;
    }
    return m_Value->to_number<std::int64_t>();
}

std::ptrdiff_t CDataFrameAnalysisConfigReader::CParameter::fallback(std::ptrdiff_t fallback) const {
    if (m_Value == nullptr) {
        return fallback;
    }
    if (m_Value->is_int64() == false) {
        this->handleFatal();
        return fallback;
    }
    return m_Value->to_number<std::int64_t>();
}

double CDataFrameAnalysisConfigReader::CParameter::fallback(double fallback) const {
    if (m_Value == nullptr) {
        return fallback;
    }
    if (m_Value->is_int64()) {
        return static_cast<double>(m_Value->to_number<std::int64_t>());
    }
    if (m_Value->is_double() == false) {
        this->handleFatal();
        return fallback;
    }
    return m_Value->to_number<double>();
}

std::string CDataFrameAnalysisConfigReader::CParameter::fallback(const std::string& fallback) const {
    if (m_Value == nullptr) {
        return fallback;
    }
    if (m_Value->is_string() == false) {
        this->handleFatal();
        return fallback;
    }
    return std::string(m_Value->as_string());
}

std::pair<std::string, double> CDataFrameAnalysisConfigReader::CParameter::fallback(
    const std::string& name,
    const std::string& value,
    const std::pair<std::string, double>& fallback) const {
    if (m_Value == nullptr) {
        return fallback;
    }
    if (m_Value->is_object() == false) {
        this->handleFatal();
        return fallback;
    }
    const json::object& obj = m_Value->as_object();
    auto name_ = obj.find(name);
    auto value_ = obj.find(value);
    if (name_ == obj.end() || value_ == obj.end()) {
        this->handleFatal();
        return fallback;
    }
    if (name_->value().is_string() == false || value_->value().is_double() == false) {
        this->handleFatal();
        return fallback;
    }
    return {std::string(name_->value().as_string()), value_->value().to_number<double>()};
}

std::vector<std::pair<std::string, double>> CDataFrameAnalysisConfigReader::CParameter::fallback(
    const std::string& name,
    const std::string& value,
    const std::vector<std::pair<std::string, double>>& fallback) const {
    if (m_Value == nullptr) {
        return fallback;
    }
    if (m_Value->is_array() == false) {
        this->handleFatal();
        return fallback;
    }
    json::array arr = m_Value->as_array();
    std::vector<std::pair<std::string, double>> result;
    result.reserve(arr.size());
    CParameter element{m_Name, SArrayElementTag{}};
    for (std::size_t i = 0; i < arr.size(); ++i) {
        element.m_Value = &(arr)[static_cast<int>(i)];
        result.push_back(element.as(name, value));
    }
    return result;
}

CDataFrameAnalysisConfigReader::CParameter::CParameter(const std::string& name, SArrayElementTag)
    : m_Name{name}, m_ArrayElement{true} {
}

void CDataFrameAnalysisConfigReader::CParameter::handleFatal() const {
    HANDLE_FATAL(<< "Input error: bad value '" << toString(*m_Value) << "' for "
                 << (m_ArrayElement ? "element of '" : "'") << m_Name << "'.");
}

CDataFrameAnalysisConfigReader::CParameterReader::CParameterReader(const std::string& name,
                                                                   ERequirement requirement,
                                                                   TStrIntMap permittedValues)
    : m_Name{name}, m_Requirement{requirement}, m_PermittedValues{std::move(permittedValues)} {
}
}
}
