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
#ifndef INCLUDED_ml_core_CJsonStateRestoreTraverser_h
#define INCLUDED_ml_core_CJsonStateRestoreTraverser_h

#include <core/CBoostJsonUnbufferedIStreamWrapper.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStringUtils.h>
#include <core/ImportExport.h>

#include <boost/json.hpp>

#include <iosfwd>
#include <string_view>

namespace json = boost::json;

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
    bool next() override;

    //! Go to the start of the next object
    //! Stops at the first '}' character so this will not
    //! work with nested objects
    bool nextObject();

    //! Does the current element have a sub-level?
    bool hasSubLevel() const override;

    //! Get the name of the current element - the returned reference is only
    //! valid for as long as the traverser is pointing at the same element
    const std::string& name() const override;

    //! Get the value of the current element - the returned reference is
    //! only valid for as long as the traverser is pointing at the same
    //! element
    const std::string& value() const override;

    //! Is the traverser at the end of the inputstream?
    bool isEof() const override;

protected:
    //! Navigate to the start of the sub-level of the current element, or
    //! return false if there isn't one
    bool descend() override;

    //! Navigate to the element of the level above from which descend() was
    //! called, or return false if there isn't a level above
    bool ascend() override;

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
    //! <a https://www.boost.org/doc/libs/1_83_0/libs/json/doc/html/json/ref/boost__json__basic_parser.html#json.ref.boost__json__basic_parser.handler0">Handler</a>
    //! for events fired by rapidjson during parsing.
    struct SBoostJsonHandler final {
        SBoostJsonHandler();

        /// The maximum number of elements allowed in an array
        static constexpr std::size_t max_array_size = -1;

        /// The maximum number of elements allowed in an object
        static constexpr std::size_t max_object_size = -1;

        /// The maximum number of characters allowed in a string
        static constexpr std::size_t max_string_size = -1;

        /// The maximum number of characters allowed in a key
        static constexpr std::size_t max_key_size = -1;

        /// Called once when the JSON parsing begins.
        ///
        /// @return `true` on success.
        /// @param ec Set to the error, if any occurred.
        ///
        bool on_document_begin( json::error_code& ec );

        /// Called when the JSON parsing is done.
        ///
        /// @return `true` on success.
        /// @param ec Set to the error, if any occurred.
        ///
        bool on_document_end( json::error_code& ec ) {
            return ec ? false : true;
        }

        /// Called when the beginning of an array is encountered.
        ///
        /// @return `true` on success.
        /// @param ec Set to the error, if any occurred.
        ///
        bool on_array_begin( json::error_code& ec );

        /// Called when the end of the current array is encountered.
        ///
        /// @return `true` on success.
        /// @param n The number of elements in the array.
        /// @param ec Set to the error, if any occurred.
        ///
        bool on_array_end( std::size_t n, json::error_code& ec );

        /// Called when the beginning of an object is encountered.
        ///
        /// @return `true` on success.
        /// @param ec Set to the error, if any occurred.
        ///
        bool on_object_begin( json::error_code& ec );

        /// Called when the end of the current object is encountered.
        ///
        /// @return `true` on success.
        /// @param n The number of elements in the object.
        /// @param ec Set to the error, if any occurred.
        ///
        bool on_object_end( std::size_t n, json::error_code& ec );

        /// Called with characters corresponding to part of the current string.
        ///
        /// @return `true` on success.
        /// @param s The partial characters
        /// @param n The total size of the string thus far
        /// @param ec Set to the error, if any occurred.
        ///
        bool on_string_part( std::string_view s, std::size_t n, json::error_code& ec );

        /// Called with the last characters corresponding to the current string.
        ///
        /// @return `true` on success.
        /// @param s The remaining characters
        /// @param n The total size of the string
        /// @param ec Set to the error, if any occurred.
        ///
        bool on_string( std::string_view s, std::size_t n, json::error_code& ec );

        /// Called with characters corresponding to part of the current key.
        ///
        /// @return `true` on success.
        /// @param s The partial characters
        /// @param n The total size of the key thus far
        /// @param ec Set to the error, if any occurred.
        ///
        bool on_key_part( std::string_view s, std::size_t n, json::error_code& ec );

        /// Called with the last characters corresponding to the current key.
        ///
        /// @return `true` on success.
        /// @param s The remaining characters
        /// @param n The total size of the key
        /// @param ec Set to the error, if any occurred.
        ///
        bool on_key( std::string_view s, std::size_t n, json::error_code& ec );

        /// Called with the characters corresponding to part of the current number.
        ///
        /// @return `true` on success.
        /// @param s The partial characters
        /// @param ec Set to the error, if any occurred.
        ///
        bool on_number_part( std::string_view s, json::error_code& ec ) {
            return true;
        }

        /// Called when a signed integer is parsed.
        ///
        /// @return `true` on success.
        /// @param i The value
        /// @param s The remaining characters
        /// @param ec Set to the error, if any occurred.
        ///
        bool on_int64( int64_t i, std::string_view s, json::error_code& ec );

        /// Called when an unsigend integer is parsed.
        ///
        /// @return `true` on success.
        /// @param u The value
        /// @param s The remaining characters
        /// @param ec Set to the error, if any occurred.
        ///
        bool on_uint64( uint64_t u, std::string_view s, json::error_code& ec );

        /// Called when a double is parsed.
        ///
        /// @return `true` on success.
        /// @param d The value
        /// @param s The remaining characters
        /// @param ec Set to the error, if any occurred.
        ///
        bool on_double( double d, std::string_view s, json::error_code& ec );

        /// Called when a boolean is parsed.
        ///
        /// @return `true` on success.
        /// @param b The value
        /// @param s The remaining characters
        /// @param ec Set to the error, if any occurred.
        ///
        bool on_bool( bool b, json::error_code& ec );

        /// Called when a null is parsed.
        ///
        /// @return `true` on success.
        /// @param ec Set to the error, if any occurred.
        ///
        bool on_null( json::error_code& ec );

        /// Called with characters corresponding to part of the current comment.
        ///
        /// @return `true` on success.
        /// @param s The partial characters.
        /// @param ec Set to the error, if any occurred.
        ///
        bool on_comment_part( std::string_view s, json::error_code& ec ) {
            return true;
        }

        /// Called with the last characters corresponding to the current comment.
        ///
        /// @return `true` on success.
        /// @param s The remaining characters
        /// @param ec Set to the error, if any occurred.
        ///
        bool on_comment( std::string_view s, json::error_code& ec ) {
            return true;
        }

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
            E_TokenArrayEnd = 12,
            E_TokenKeyPart = 13,
            E_TokenStringPart = 14,
            E_TokenComma,
            E_TokenColon
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
        bool m_NewToken{true};
    };

    //! JSON reader istream wrapper
//    core::CBoostJsonUnbufferedIStreamWrapper m_ReadStream;
    std::istream& m_ReadStream;

    //! JSON reader
    json::basic_parser<SBoostJsonHandler> m_Reader;

    SBoostJsonHandler& m_Handler;

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
