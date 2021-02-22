/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_torch_CCommandParser_h
#define INCLUDED_ml_torch_CCommandParser_h

#include <iosfwd>
#include <functional>
#include <string>
#include <vector>

#include <rapidjson/document.h>


namespace ml {
namespace torch {


//! \brief
//! Reads JSON documents from a stream emitting a request for 
//! each parsed document.
//!
//! DESCRIPTION:\n
//! 
//!
//! IMPLEMENTATION DECISIONS:\n
//! RapidJSON will natively parse a stream of rootless JSON documents
//! given the correct parse flags. The documents may be separated by 
//!	whitespace but no other delineator is allowed.
//! 
//! The input stream is held by reference.  They must outlive objects of
//! this class, which, in practice, means that the CIoManager object managing
//! them must outlive this object.
//!
class CCommandParser {	
public:

	static const std::string REQUEST_ID;
	static const std::string TOKENS;
	static const std::string VAR_ARG_PREFIX;

	using TUint32Vec = std::vector<std::uint32_t>;
	using TUint32VecVec = std::vector<TUint32Vec>;

	struct SRequest {
		std::string s_RequestId;
		TUint32Vec s_Tokens;
		TUint32VecVec s_SecondaryArguments;
	};

	using TRequestHandlerFunc = std::function<bool(SRequest&)>;


public:
    CCommandParser(std::istream& strmIn);

    //! Pass input to the processor until it's consumed as much as it can.
    bool ioLoop(const TRequestHandlerFunc& requestHandler) const;

    CCommandParser(const CCommandParser&) = delete;
    CCommandParser& operator=(const CCommandParser&) = delete;

private:
	bool validateJson(const rapidjson::Document& doc) const;
	SRequest jsonToRequest(const rapidjson::Document& doc) const;
private:
	//! 
    std::istream& m_StrmIn;
};

}
}

#endif // INCLUDED_ml_torch_CCommandParser_h
