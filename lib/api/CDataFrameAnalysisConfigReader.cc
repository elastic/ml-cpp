/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CDataFrameAnalysisConfigReader.h>

#include <core/CLogger.h>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <cstring>

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

void CDataFrameAnalysisConfigReader::addParameter(const std::string& name,
                                                  ERequirement requirement,
                                                  TStrIntMap permittedValues) {
    m_ParameterReaders.emplace_back(name, requirement, std::move(permittedValues));
}

CDataFrameAnalysisParameters
CDataFrameAnalysisConfigReader::read(const rapidjson::Value& json) const {

    CDataFrameAnalysisParameters result;

    if (json.IsObject() == false) {
        HANDLE_FATAL(<< "Input error: expected JSON object but input was '"
                     << toString(json) << "'. Please report this problem.");
        return result;
    }

    for (const auto& reader : m_ParameterReaders) {
        if (json.HasMember(reader.name())) {
            result.add(reader.readFrom(json));
        } else if (reader.required()) {
            HANDLE_FATAL(<< "Input error: missing required parameter '"
                         << reader.name() << "'. Please report this problem.");
        } else {
            result.add(CParameter{reader.name()});
        }
    }

    // Check for any unrecognised fields: these might be typos.
    for (auto i = json.MemberBegin(); i != json.MemberEnd(); ++i) {
        bool found{false};
        for (const auto& param : m_ParameterReaders) {
            if (i->name.GetString() == param.name()) {
                found = true;
                break;
            }
        }
        if (found == false) {
            HANDLE_FATAL(<< "Input error: unexpected parameter '"
                         << i->name.GetString() << "'. Please report this problem.");
        }
    }

    return result;
}

CDataFrameAnalysisConfigReader::CParameter::CParameter(const std::string& name,
                                                       const rapidjson::Value& value,
                                                       const TStrIntMap& permittedValues)
    : m_Name{name}, m_Value{&value}, m_PermittedValues{&permittedValues} {
}

bool CDataFrameAnalysisConfigReader::CParameter::fallback(bool fallback) const {
    if (m_Value == nullptr) {
        return fallback;
    }
    if (m_Value->IsBool() == false) {
        this->handleFatal();
        return fallback;
    }
    return m_Value->GetBool();
}

std::size_t CDataFrameAnalysisConfigReader::CParameter::fallback(std::size_t fallback) const {
    if (m_Value == nullptr) {
        return fallback;
    }
    if (m_Value->IsUint64() == false) {
        this->handleFatal();
        return fallback;
    }
    return m_Value->GetUint64();
}

std::ptrdiff_t CDataFrameAnalysisConfigReader::CParameter::fallback(std::ptrdiff_t fallback) const {
    if (m_Value == nullptr) {
        return fallback;
    }
    if (m_Value->IsInt64() == false) {
        this->handleFatal();
        return fallback;
    }
    return m_Value->GetInt64();
}

double CDataFrameAnalysisConfigReader::CParameter::fallback(double fallback) const {
    if (m_Value == nullptr) {
        return fallback;
    }
    if (m_Value->IsInt64()) {
        return static_cast<double>(m_Value->GetInt64());
    }
    if (m_Value->IsDouble() == false) {
        this->handleFatal();
        return fallback;
    }
    return m_Value->GetDouble();
}

std::string CDataFrameAnalysisConfigReader::CParameter::fallback(const std::string& fallback) const {
    if (m_Value == nullptr) {
        return fallback;
    }
    if (m_Value->IsString() == false) {
        this->handleFatal();
        return fallback;
    }
    return m_Value->GetString();
}

std::pair<std::string, double> CDataFrameAnalysisConfigReader::CParameter::fallback(
    const std::string& name,
    const std::string& value,
    const std::pair<std::string, double>& fallback) const {
    if (m_Value == nullptr) {
        return fallback;
    }
    if (m_Value->IsObject() == false) {
        this->handleFatal();
        return fallback;
    }
    auto name_ = m_Value->FindMember(name);
    auto value_ = m_Value->FindMember(value);
    if (name_ == m_Value->MemberEnd() || value_ == m_Value->MemberEnd()) {
        this->handleFatal();
        return fallback;
    }
    if (name_->value.IsString() == false || value_->value.IsDouble() == false) {
        this->handleFatal();
        return fallback;
    }
    return {name_->value.GetString(), value_->value.GetDouble()};
}

std::vector<std::pair<std::string, double>> CDataFrameAnalysisConfigReader::CParameter::fallback(
    const std::string& name,
    const std::string& value,
    const std::vector<std::pair<std::string, double>>& fallback) const {
    if (m_Value == nullptr) {
        return fallback;
    }
    if (m_Value->IsArray() == false) {
        this->handleFatal();
        return fallback;
    }
    std::vector<std::pair<std::string, double>> result;
    result.reserve(m_Value->Size());
    CParameter element{m_Name, SArrayElementTag{}};
    for (std::size_t i = 0; i < m_Value->Size(); ++i) {
        element.m_Value = &(*m_Value)[static_cast<int>(i)];
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
