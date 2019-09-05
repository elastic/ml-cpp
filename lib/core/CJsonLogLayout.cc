/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CJsonLogLayout.h>

#include <core/CLogger.h>
#include <core/CProcess.h>
#include <core/CProgName.h>

#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/filesystem.hpp>
#include <boost/log/attributes/current_thread_id.hpp>
#include <boost/log/attributes/value_extraction.hpp>
#include <boost/log/expressions/message.hpp>
#include <boost/log/utility/formatting_ostream.hpp>

#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <cstdint>
#include <sstream>
#include <string>

namespace ml {
namespace core {

namespace {
const std::string LOGGER_NAME{"logger"};
// For consistency with the way we used log4cxx the "logger" is the name of the
// program
const std::string LOGGER{CProgName::progName()};
const std::string TIMESTAMP_NAME{"timestamp"};
const std::string LEVEL_NAME{"level"};
const std::string PID_NAME{"pid"};
// Cast this to int64_t as the type varies between int32_t and uint32_t on
// different platforms and int64_t covers both
const int64_t PID{static_cast<std::int64_t>(ml::core::CProcess::instance().id())};
const std::string THREAD_NAME{"thread"};
const std::string MESSAGE_NAME{"message"};
const std::string CLASS_NAME{"class"};
const std::string METHOD_NAME{"method"};
const std::string FILE_NAME{"file"};
const std::string LINE_NAME{"line"};
const std::string PROPERTIES_NAME{"properties"};
const boost::posix_time::ptime EPOCH{boost::gregorian::date(1970, 1, 1)};
}

CJsonLogLayout::CJsonLogLayout() : m_LocationInfo(true) {
}

void CJsonLogLayout::locationInfo(bool locationInfo) {
    m_LocationInfo = locationInfo;
}

bool CJsonLogLayout::locationInfo() const {
    return m_LocationInfo;
}

void CJsonLogLayout::operator()(const boost::log::record_view& rec,
                                boost::log::formatting_ostream& strm) const {
    using TStringBufferWriter = rapidjson::Writer<rapidjson::StringBuffer>;
    rapidjson::StringBuffer buffer;
    TStringBufferWriter writer{buffer};

    writer.StartObject();

    writer.String(LOGGER_NAME);
    writer.String(LOGGER);

    writer.String(TIMESTAMP_NAME);
    const boost::posix_time::ptime& timeStamp{
        boost::log::extract<boost::posix_time::ptime>("TimeStamp", rec).get()};
    writer.Int64((timeStamp - EPOCH).total_milliseconds());

    writer.String(LEVEL_NAME);
    CLogger::ELevel level{boost::log::extract<CLogger::ELevel>("Severity", rec).get()};
    writer.String(CLogger::levelToString(level));

    writer.String(PID_NAME);
    writer.Int64(PID);

    writer.String(THREAD_NAME);
    const boost::log::attributes::current_thread_id::value_type threadId{
        boost::log::extract<boost::log::attributes::current_thread_id::value_type>("ThreadID", rec)
            .get()};
    std::ostringstream oss;
    oss << threadId;
    writer.String(oss.str());

    writer.String(MESSAGE_NAME);
    const std::string& message{rec[boost::log::expressions::smessage].get()};
    writer.String(message);

    /* TODO
    if (m_LocationInfo) {
        const spi::LocationInfo& locInfo = rec->getLocationInformation();

        const std::string& className = locInfo.getClassName();
        if (!className.empty()) {
            writer.String(CLASS_NAME);
            writer.String(className);
        }

        const std::string& methodName = locInfo.getMethodName();
        if (!methodName.empty()) {
            writer.String(METHOD_NAME);
            writer.String(methodName);
        }

        writer.String(FILE_NAME);
        writer.String(CJsonLogLayout::cropPath(locInfo.getFileName()));

        writer.String(LINE_NAME);
        writer.Int(locInfo.getLineNumber());
    }
*/

    writer.EndObject();

    strm << buffer.GetString();
}

std::string CJsonLogLayout::cropPath(const std::string& filename) {
    boost::filesystem::path p(filename);
    return p.filename().string();
}
}
}
