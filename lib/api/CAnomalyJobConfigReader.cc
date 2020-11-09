/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CAnomalyJobConfigReader.h>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#ifdef Windows
// rapidjson::Writer<rapidjson::StringBuffer> gets instantiated in the core
// library, and on Windows it gets exported too, because
// CRapidJsonConcurrentLineWriter inherits from it and is also exported.
// To avoid breaching the one-definition rule we must reuse this exported
// instantiation, as deduplication of template instantiations doesn't work
// across DLLs.  To make this even more confusing, this is only strictly
// necessary when building without optimisation, because with optimisation
// enabled the instantiation in this library gets inlined to the extent that
// there are no clashing symbols.
template class CORE_EXPORT rapidjson::Writer<rapidjson::StringBuffer>;
#endif

namespace ml {
namespace api {
namespace {
std::string toString(const rapidjson::Value& value) {
    rapidjson::StringBuffer valueAsString;
    rapidjson::Writer<rapidjson::StringBuffer> writer(valueAsString);
    value.Accept(writer);
    return valueAsString.GetString();
}
}

void CAnomalyJobConfigReader::addParameter(const std::string& name,
                                           ERequirement requirement,
                                           TStrIntMap permittedValues) {
    m_ParameterReaders.emplace_back(name, requirement, std::move(permittedValues));
}

CAnomalyJobParameters CAnomalyJobConfigReader::read(const rapidjson::Value& json) const {
    if (json.IsObject() == false) {
        throw CParseError("Input error: expected JSON object but input was '" +
                          toString(json) + "'. Please report this problem.");
    }

    CAnomalyJobParameters result;
    for (const auto& reader : m_ParameterReaders) {
        if (json.HasMember(reader.name())) {
            result.add(reader.readFrom(json));
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
                                                const rapidjson::Value& value,
                                                const TStrIntMap& permittedValues)
    : m_Name{name}, m_Value{&value}, m_PermittedValues{&permittedValues} {
}

bool CAnomalyJobConfigReader::CParameter::fallback(bool value) const {
    if (m_Value == nullptr) {
        return value;
    }
    if (m_Value->IsBool() == false) {
        this->handleFatal();
        return value;
    }
    return m_Value->GetBool();
}

std::size_t CAnomalyJobConfigReader::CParameter::fallback(std::size_t value) const {
    if (m_Value == nullptr) {
        return value;
    }
    if (m_Value->IsUint64() == false) {
        this->handleFatal();
        return value;
    }
    return m_Value->GetUint64();
}

std::ptrdiff_t CAnomalyJobConfigReader::CParameter::fallback(std::ptrdiff_t value) const {
    if (m_Value == nullptr) {
        return value;
    }
    if (m_Value->IsInt64() == false) {
        this->handleFatal();
        return value;
    }
    return m_Value->GetInt64();
}

double CAnomalyJobConfigReader::CParameter::fallback(double value) const {
    if (m_Value == nullptr) {
        return value;
    }
    if (m_Value->IsInt64()) {
        return static_cast<double>(m_Value->GetInt64());
    }
    if (m_Value->IsDouble() == false) {
        this->handleFatal();
        return value;
    }
    return m_Value->GetDouble();
}

std::string CAnomalyJobConfigReader::CParameter::fallback(const std::string& value) const {
    if (m_Value == nullptr) {
        return value;
    }
    if (m_Value->IsString() == false) {
        this->handleFatal();
        return value;
    }
    return m_Value->GetString();
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
