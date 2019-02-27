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

void CDataFrameAnalysisConfigReader::addParameter(const char* name,
                                                  ERequirement requirement,
                                                  TStrIntMap permittedValues) {
    m_ParameterReaders.emplace_back(name, requirement, std::move(permittedValues));
}

CDataFrameAnalysisConfigReader::CParameters
CDataFrameAnalysisConfigReader::read(const rapidjson::Value& json) const {

    CParameters result;

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
            result.add(reader.name());
        }
    }

    // Check for any unrecognised fields: these might be typos.
    for (auto i = json.MemberBegin(); i != json.MemberEnd(); ++i) {
        bool found{false};
        for (const auto& param : m_ParameterReaders) {
            if (std::strcmp(i->name.GetString(), param.name()) == 0) {
                found = true;
                break;
            }
        }
        if (found == false) {
            HANDLE_FATAL(<< "Input error: unexpected parameter '"
                         << i->name.GetString() << "'. Please report this problem.")
        }
    }

    return result;
}

CDataFrameAnalysisConfigReader::CParameter::CParameter(const char* name,
                                                       const rapidjson::Value& value,
                                                       const TStrIntMap& permittedValues)
    : m_Name{name}, m_Value{&value}, m_PermittedValues{&permittedValues} {
}

bool CDataFrameAnalysisConfigReader::CParameter::fallback(bool value) const {
    if (m_Value == nullptr) {
        return value;
    }
    if (m_Value->IsBool() == false) {
        this->handleFatal();
    }
    return m_Value->GetBool();
}

std::size_t CDataFrameAnalysisConfigReader::CParameter::fallback(std::size_t value) const {
    if (m_Value == nullptr) {
        return value;
    }
    if (m_Value->IsUint() == false) {
        this->handleFatal();
    }
    return m_Value->GetUint();
}

double CDataFrameAnalysisConfigReader::CParameter::fallback(double value) const {
    if (m_Value == nullptr) {
        return value;
    }
    if (m_Value->IsInt64()) {
        return static_cast<double>(m_Value->GetInt64());
    }
    if (m_Value->IsDouble() == false) {
        this->handleFatal();
    }
    return m_Value->GetDouble();
}

std::string CDataFrameAnalysisConfigReader::CParameter::fallback(const std::string& value) const {
    if (m_Value == nullptr) {
        return value;
    }
    if (m_Value->IsString() == false) {
        this->handleFatal();
    }
    return m_Value->GetString();
}

void CDataFrameAnalysisConfigReader::CParameter::handleFatal() const {
    HANDLE_FATAL(<< "Input error: bad value '" << toString(*m_Value)
                 << "' for '" << m_Name << "'.");
}

CDataFrameAnalysisConfigReader::CParameter
    CDataFrameAnalysisConfigReader::CParameters::operator[](const char* name) const {
    for (const auto& value : m_ParameterValues) {
        if (std::strcmp(name, value.name()) == 0) {
            return value;
        }
    }
    return {name};
}

CDataFrameAnalysisConfigReader::CParameterReader::CParameterReader(const char* name,
                                                                   ERequirement requirement,
                                                                   TStrIntMap permittedValues)
    : m_Name{name}, m_Requirement{requirement}, m_PermittedValues{std::move(permittedValues)} {
}
}
}
