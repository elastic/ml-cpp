/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CCommandParser.h"

#include <core/CLogger.h>

#include <iostream>

#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace ml {
namespace torch {

namespace {
void debug(const rapidjson::Document& doc) {
	rapidjson::StringBuffer buffer;
	rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
	doc.Accept(writer);
	LOG_INFO(<< buffer.GetString());
}
}	

const std::string CCommandParser::REQUEST_ID{"request_id"};
const std::string CCommandParser::TOKENS{"tokens"};

CCommandParser::CCommandParser(std::istream& strmIn) : m_StrmIn(strmIn) {

}


bool CCommandParser::ioLoop(const TRequestHandlerFunc& requestHandler) const {

	rapidjson::IStreamWrapper isw(m_StrmIn);

	while (true) {
		rapidjson::Document doc;
		rapidjson::ParseResult parseResult = doc.ParseStream<rapidjson::kParseStopWhenDoneFlag>(isw);
		
		if (static_cast<bool>(parseResult) == false) {
			if (m_StrmIn.eof()) {								
				break;
			} 

			LOG_ERROR(<< "Error parsing command from JSON: " << rapidjson::GetParseError_En(parseResult.Code())
				<< ". At offset: " << parseResult.Offset());

			return false;
		}


		if (validateJson(doc) == false) {
			continue;
		}

		
		debug(doc);
		requestHandler(jsonToRequest(doc));
	}
	
	return true;	
}

bool CCommandParser::validateJson(const rapidjson::Document& doc) const {
	if (doc.HasMember(REQUEST_ID) == false) {
		LOG_ERROR(<< "Malformed command request: missing field [" << REQUEST_ID << "]");
		return false;
	}

	if (doc.HasMember(TOKENS) == false) {
		LOG_ERROR(<< "Malformed command request: missing field [" << TOKENS << "]");
		return false;
	}

	const rapidjson::Value& tokens = doc[TOKENS];
	if (tokens.IsArray() == false) {
		LOG_ERROR(<< "Malformed command request: expected an array [" << TOKENS << "]");
		return false;
	}

	return true;
}

CCommandParser::SRequest CCommandParser::jsonToRequest(const rapidjson::Document& doc) const {
	std::vector<std::uint32_t> tokens;
	const rapidjson::Value& arr = doc[TOKENS];
	for (auto itr = arr.Begin(); itr != arr.End(); ++itr) {
		tokens.push_back(itr->GetUint());
	}
	return {doc[REQUEST_ID].GetString(), tokens};
}

}
}
