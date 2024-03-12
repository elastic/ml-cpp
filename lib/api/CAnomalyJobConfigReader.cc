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

#include <api/CAnomalyJobConfigReader.h>

#include <boost/json.hpp>

namespace ml {
namespace api {
namespace {
std::string toString(const json::value& value) {
    return json::serialize(value);
}
}

void CAnomalyJobConfigReader::addParameter(const std::string& name,
                                           ERequirement requirement,
                                           TStrIntMap permittedValues) {
    m_ParameterReaders.emplace_back(name, requirement, std::move(permittedValues));
}

CAnomalyJobParameters CAnomalyJobConfigReader::read(const json::value& json) const {
    if (json.is_object() == false) {
        throw CParseError("Input error: expected JSON object but input was '" +
                          toString(json) + "'. Please report this problem.");
    }

    const json::object& obj = json.as_object();
    CAnomalyJobParameters result;
    for (const auto& reader : m_ParameterReaders) {
        if (obj.contains(reader.name())) {
            result.add(reader.readFrom(obj));
        } else if (reader.required()) {
            throw CParseError("Input error: missing required parameter '" +
                              reader.name() + "'. Please report this problem.");
        } else {
            result.add(CParameter{reader.name()});
        }
    }

    return result;
}

CAnomalyJobConfigReader::CParameter::CParameter(const std::string& name,
                                                const json::value& value,
                                                const TStrIntMap& permittedValues)
    : m_Name{name}, m_Value{&value}, m_PermittedValues{&permittedValues} {
}

bool CAnomalyJobConfigReader::CParameter::fallback(bool value) const {
    if (m_Value == nullptr) {
        return value;
    }
    if (m_Value->is_bool() == false) {
        this->handleFatal();
    }
    return m_Value->as_bool();
}

int CAnomalyJobConfigReader::CParameter::fallback(int value) const {
    if (m_Value == nullptr) {
        return value;
    }
    if (m_Value->is_int64() == false) {
        this->handleFatal();
    }
    return m_Value->to_number<int>();
}

std::size_t CAnomalyJobConfigReader::CParameter::fallback(std::size_t value) const {
    if (m_Value == nullptr) {
        return value;
    }
    if (m_Value->is_int64() == false) {
        this->handleFatal();
    }
    return m_Value->to_number<std::int64_t>();
}

std::ptrdiff_t CAnomalyJobConfigReader::CParameter::fallback(std::ptrdiff_t value) const {
    if (m_Value == nullptr) {
        return value;
    }
    if (m_Value->is_int64() == false) {
        this->handleFatal();
    }
    return m_Value->to_number<std::int64_t>();
}

double CAnomalyJobConfigReader::CParameter::fallback(double value) const {
    if (m_Value == nullptr) {
        return value;
    }
    if (m_Value->is_int64()) {
        return static_cast<double>(m_Value->to_number<std::int64_t>());
    }
    if (m_Value->is_double() == false) {
        this->handleFatal();
    }
    return m_Value->to_number<double>();
}

std::string CAnomalyJobConfigReader::CParameter::fallback(const std::string& value) const {
    if (m_Value == nullptr) {
        return value;
    }
    if (m_Value->is_string() == false) {
        this->handleFatal();
    }

    return std::string(m_Value->as_string());
}

CAnomalyJobConfigReader::CParameter::CParameter(const std::string& name, SArrayElementTag)
    : m_Name{name}, m_ArrayElement{true} {
}

void CAnomalyJobConfigReader::CParameter::handleFatal() const {
    throw CParseError("Input error: bad value '" + toString(*m_Value) + "' for " +
                      (m_ArrayElement ? "element of '" : "'") + m_Name + "'.");
}

CAnomalyJobConfigReader::CParameterReader::CParameterReader(const std::string& name,
                                                            ERequirement requirement,
                                                            TStrIntMap permittedValues)
    : m_Name{name}, m_Requirement{requirement}, m_PermittedValues{std::move(permittedValues)} {
}
}
}
