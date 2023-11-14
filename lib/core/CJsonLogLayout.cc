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

#include <boost/json/src.hpp>

#include <cstdint>
#include <sstream>

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
// Cast this to std::int64_t as the type varies between std::int32_t and std::uint32_t on
// different platforms and std::int64_t covers both
const std::int64_t PID{static_cast<std::int64_t>(ml::core::CProcess::instance().id())};
const std::string THREAD_NAME{"thread"};
const std::string MESSAGE_NAME{"message"};
const std::string CLASS_NAME{"class"};
const std::string METHOD_NAME{"method"};
const std::string FILE_NAME{"file"};
const std::string LINE_NAME{"line"};
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
    namespace json = boost::json;

    json::object writer;
    writer[LOGGER_NAME] = LOGGER;

    const auto& timeStamp = boost::log::extract<boost::posix_time::ptime>(
                                boost::log::aux::default_attribute_names::timestamp(), rec)
                                .get();
    writer[TIMESTAMP_NAME] = (timeStamp - EPOCH).total_milliseconds();

    auto level = boost::log::extract<CLogger::ELevel>(
                     boost::log::aux::default_attribute_names::severity(), rec)
                     .get();
    writer[LEVEL_NAME] = CLogger::levelToString(level);

    writer[PID_NAME] = PID;

    auto threadId = boost::log::extract<boost::log::attributes::current_thread_id::value_type>(
                        boost::log::aux::default_attribute_names::thread_id(), rec)
                        .get();
    std::ostringstream oss;
    oss << threadId;
    writer[THREAD_NAME] = oss.str();

    writer[MESSAGE_NAME] = rec[boost::log::expressions::smessage].get();

    if (m_LocationInfo) {
        std::string className;
        std::string methodName;
        std::tie(className, methodName) = extractClassAndMethod(
            boost::log::extract<std::string>(CLogger::instance().functionAttributeName(), rec)
                .get());

        if (className.empty() == false) {
            writer[CLASS_NAME] = className;
        }

        writer[METHOD_NAME] = methodName;

        const auto& fullFileName = boost::log::extract<std::string>(
                                       CLogger::instance().fileAttributeName(), rec)
                                       .get();
        writer[FILE_NAME] = CJsonLogLayout::cropPath(fullFileName);

        writer[LINE_NAME] = boost::log::extract<int>(CLogger::instance().lineAttributeName(), rec).get();
    }

    strm << writer;
}

std::string CJsonLogLayout::cropPath(const std::string& filename) {
    boost::filesystem::path p(filename);
    return p.filename().string();
}

CJsonLogLayout::TStrStrPr CJsonLogLayout::extractClassAndMethod(std::string prettyFunctionSig) {

    std::size_t argsStartPos{prettyFunctionSig.find('(')};
    if (argsStartPos != std::string::npos) {
        prettyFunctionSig.erase(argsStartPos);
    }
    std::size_t returnTypeSeparatorPos{prettyFunctionSig.rfind(' ')};
    if (returnTypeSeparatorPos != std::string::npos) {
        prettyFunctionSig.erase(0, returnTypeSeparatorPos + 1);
    }
    std::size_t lastScopeOperatorPos{prettyFunctionSig.rfind("::")};

    if (lastScopeOperatorPos == std::string::npos) {
        // prettyFunctionSig has been cut down by the time we get here
        return {std::string(), prettyFunctionSig};
    }

    return {prettyFunctionSig.substr(0, lastScopeOperatorPos),
            prettyFunctionSig.substr(lastScopeOperatorPos + 2)};
}
}
}
