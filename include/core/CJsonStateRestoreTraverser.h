/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CJsonStateRestoreTraverser_h
#define INCLUDED_ml_core_CJsonStateRestoreTraverser_h

#include <core/CNonCopyable.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStringUtils.h>
#include <core/ImportExport.h>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/reader.h>

#include <iosfwd>

namespace ml {
namespace core {

//! \brief
//! For restoring state in JSON format.
//!
//! DESCRIPTION:\n
//! Concrete implementation of the CStateRestoreTraverser interface
//! that restores state in JSON format.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Input is streaming rather than building up an in-memory JSON
//! document.
//!
//! Unlike the CRapidXmlStatePersistInserter, there is no possibility
//! of including attributes on the root node (because JSON does not
//! have attributes).  This may complicate code that needs to be 100%
//! JSON/XML agnostic.
//!
class CORE_EXPORT CJsonStateRestoreTraverser : public CStateRestoreTraverser {
public:
    CJsonStateRestoreTraverser(std::istream& inputStream);

    //! Navigate to the next element at the current level, or return false
    //! if there isn't one
    virtual bool next();

    //! Go to the start of the next object
    //! Stops at the first '}' character so this will not
    //! work with nested objects
    bool nextObject();

    //! Does the current element have a sub-level?
    virtual bool hasSubLevel() const;

    //! Get the name of the current element - the returned reference is only
    //! valid for as long as the traverser is pointing at the same element
    virtual const std::string& name() const;

    //! Get the value of the current element - the returned reference is
    //! only valid for as long as the traverser is pointing at the same
    //! element
    virtual const std::string& value() const;

    //! Is the traverser at the end of the inputstream?
    virtual bool isEof() const;

protected:
    //! Navigate to the start of the sub-level of the current element, or
    //! return false if there isn't one
    virtual bool descend();

    //! Navigate to the element of the level above from which descend() was
    //! called, or return false if there isn't a level above
    virtual bool ascend();

    //! Print debug
    void debug() const;

private:
    //! Accessors for alternating state variables
    size_t currentLevel() const;
    bool currentIsEndOfLevel() const;
    const std::string& currentName() const;
    const std::string& currentValue() const;
    size_t nextLevel() const;
    bool nextIsEndOfLevel() const;
    const std::string& nextName() const;
    const std::string& nextValue() const;

    //! Start off the parsing process
    bool start();

    //! Get the next token
    bool advance();

    //! Log an error that the JSON parser has detected
    void logError();

    //! Continue parsing the JSON structure
    bool parseNext(bool remember);

    //! Skip the (JSON) array until it ends
    bool skipArray();

private:
    //! <a href="http://rapidjson.org/classrapidjson_1_1_handler.html">Handler</a>
    //! for events fired by rapidjson during parsing.
    struct SRapidJsonHandler final {
        SRapidJsonHandler();

        bool Null();
        bool Bool(bool b);
        bool Int(int i);
        bool Uint(unsigned u);
        bool Int64(int64_t i);
        bool Uint64(uint64_t u);
        bool Double(double d);
        bool RawNumber(const char*, rapidjson::SizeType, bool);
        bool String(const char* str, rapidjson::SizeType length, bool);
        bool StartObject();
        bool Key(const char* str, rapidjson::SizeType length, bool);
        bool EndObject(rapidjson::SizeType);
        bool StartArray();
        bool EndArray(rapidjson::SizeType);

        enum ETokenType {
            E_TokenNull = 0,
            E_TokenKey = 1,
            E_TokenBool = 2,
            E_TokenInt = 3,
            E_TokenUInt = 4,
            E_TokenInt64 = 5,
            E_TokenUInt64 = 6,
            E_TokenDouble = 7,
            E_TokenString = 8,
            E_TokenObjectStart = 9,
            E_TokenObjectEnd = 10,
            E_TokenArrayStart = 11,
            E_TokenArrayEnd = 12
        };

        ETokenType s_Type;

        size_t s_Level[2];
        bool s_IsEndOfLevel[2];
        std::string s_Name[2];
        std::string s_Value[2];

        //! Setting m_NextIndex = (1 - m_NextIndex) advances the
        //! stored details.
        size_t s_NextIndex;

        bool s_RememberValue;
    };

    //! JSON reader istream wrapper
    rapidjson::IStreamWrapper m_ReadStream;

    //! JSON reader
    rapidjson::Reader m_Reader;

    SRapidJsonHandler m_Handler;

    //! Flag to indicate whether we've started parsing
    bool m_Started;

    //! Which level within the JSON structure do we want to be getting
    //! values from?
    size_t m_DesiredLevel;

    //! If the first token is an '[' then we are parsing an array of objects
    bool m_IsArrayOfObjects;
};
}
}

#endif // INCLUDED_ml_core_CJsonStateRestoreTraverser_h
