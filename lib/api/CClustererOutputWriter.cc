/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CClustererOutputWriter.h>

#include <core/CRapidJsonLineWriter.h>

#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>

#include <boost/unordered_map.hpp>

namespace ml {
namespace api {

struct CClustererOutputWriter::SState {
    using TGenericLineWriter = core::CRapidJsonLineWriter<rapidjson::OStreamWrapper>;

    SState(std::ostream& stream)
        : s_WriterStream(stream), s_Writer(s_WriterStream) {}

    //! JSON writer ostream wrapper.
    rapidjson::OStreamWrapper s_WriterStream;

    //! The JSON writer.
    TGenericLineWriter s_Writer;

    //! The result document which is being built.
    rapidjson::Document s_Result;
};

CClustererOutputWriter::CClustererOutputWriter(std::ostream& stream)
    : m_State(new SState(stream)) {
}

void CClustererOutputWriter::startResult(void) {
    m_State->s_Result.SetObject();
}

void CClustererOutputWriter::addMember(const std::string& name, bool value) {
    rapidjson::Document& result = m_State->s_Result;
    result.AddMember(rapidjson::StringRef(name), value, result.GetAllocator());
}

void CClustererOutputWriter::addMember(const std::string& name, double value) {
    rapidjson::Document& result = m_State->s_Result;
    result.AddMember(rapidjson::StringRef(name), value, result.GetAllocator());
}

void CClustererOutputWriter::addMember(const std::string& name, std::size_t value) {
    rapidjson::Document& result = m_State->s_Result;
    result.AddMember(rapidjson::StringRef(name), static_cast<uint64_t>(value),
                     result.GetAllocator());
}

void CClustererOutputWriter::addMember(const std::string& name, const std::string& value) {
    rapidjson::Document& result = m_State->s_Result;
    result.AddMember(rapidjson::StringRef(name), rapidjson::StringRef(value),
                     result.GetAllocator());
}

void CClustererOutputWriter::addMember(const std::string& name, const TDoubleVec& values) {
    rapidjson::Document& result = m_State->s_Result;
    rapidjson::Value v;
    v.SetArray();
    v.Reserve(static_cast<unsigned>(values.size()), result.GetAllocator());
    for (const auto& value : values) {
        v.PushBack(value, result.GetAllocator());
    }
    result.AddMember(rapidjson::StringRef(name), v, result.GetAllocator());
}

void CClustererOutputWriter::addMember(const std::string& name, const TStrVec& values) {
    rapidjson::Document& result = m_State->s_Result;
    rapidjson::Value v;
    v.SetArray();
    v.Reserve(static_cast<unsigned>(values.size()), result.GetAllocator());
    for (const auto& value : values) {
        v.PushBack(rapidjson::StringRef(value), result.GetAllocator());
    }
    result.AddMember(rapidjson::StringRef(name), v, result.GetAllocator());
}

void CClustererOutputWriter::addMember(const std::string& name, const TStrDoubleUMap& values) {
    rapidjson::Document& result = m_State->s_Result;
    rapidjson::Value v;
    v.SetObject();
    for (const auto& value : values) {
        v.AddMember(rapidjson::StringRef(value.first), value.second, result.GetAllocator());
    }
    result.AddMember(rapidjson::StringRef(name), v, result.GetAllocator());
}

void CClustererOutputWriter::writeResult(void) {
    m_State->s_Result.Accept(m_State->s_Writer);
}

bool CClustererOutputWriter::fieldNames(const TStrVec& /*fieldNames*/,
                                        const TStrVec& /*extraFieldNames*/) {
    return true;
}

bool CClustererOutputWriter::writeRow(const TStrStrUMap& /*dataRowFields*/,
                                      const TStrStrUMap& /*overrideDataRowFields*/) {
    return true;
}
}
}
