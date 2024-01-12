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

#ifndef INCLUDED_ml_core_CBoostJsonWriterBase_h
#define INCLUDED_ml_core_CBoostJsonWriterBase_h

#include <core/CBoostJsonPoolAllocator.h>
#include <core/CFunctional.h>
#include <core/CLogger.h>
#include <core/CTimeUtils.h>
#include <core/CoreTypes.h>
#include <core/ImportExport.h>

#include <boost/iterator/indirect_iterator.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <regex>
#include <stack>

namespace json = boost::json;

namespace ml {
namespace core {
//! \brief
//! A Json writer with fixed length allocator pool
//! With utility functions for adding fields to JSON objects.
//!
//! DESCRIPTION:\n
//! Wraps up the code needed to add various types of values to JSON
//! objects. 
//!
//! IMPLEMENTATION DECISIONS:\n
//! Empty string fields are not written to the output unless specifically
//! requested.
//!
template<typename OUTPUT_STREAM>
class CBoostJsonWriterBase {
public:
    using TTimeVec = std::vector<core_t::TTime>;
    using TStrVec = std::vector<std::string>;
    using TDoubleVec = std::vector<double>;
    using TDoubleDoublePr = std::pair<double, double>;
    using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
    using TDoubleDoubleDoublePrPr = std::pair<double, TDoubleDoublePr>;
    using TDoubleDoubleDoublePrPrVec = std::vector<TDoubleDoubleDoublePrPr>;
    using TStrUSet = boost::unordered_set<std::string>;
    using TDocument = boost::json::object;
    using TValue = boost::json::value;
    using TValuePtr = std::shared_ptr<boost::json::value>;
    using TDocumentWeakPtr = std::weak_ptr<TDocument>;
    using TDocumentPtr = std::shared_ptr<TDocument>;
    using TPoolAllocatorPtr = std::shared_ptr<CBoostJsonPoolAllocator>;
    using TPoolAllocatorPtrStack = std::stack<TPoolAllocatorPtr>;
    using TStrPoolAllocatorPtrMap = boost::unordered_map<std::string, TPoolAllocatorPtr>;
    using TStrPoolAllocatorPtrMapItr = TStrPoolAllocatorPtrMap::iterator;
    using TStrPoolAllocatorPtrMapItrBoolPr = std::pair<TStrPoolAllocatorPtrMapItr, bool>;

public:
    explicit CBoostJsonWriterBase(OUTPUT_STREAM& os)
        : m_Os(&os)/*, m_Values{json::storage_ptr(), m_Buffer, sizeof(m_Buffer)}*/ {

        // push a default boost::json allocator onto our stack
        m_JsonPoolAllocators.push(std::make_shared<CBoostJsonPoolAllocator>());
    }

    CBoostJsonWriterBase() : m_Os(nullptr) {

        // push a default boost::json allocator onto our stack
        m_JsonPoolAllocators.push(std::make_shared<CBoostJsonPoolAllocator>());
    }

    void Reset(OUTPUT_STREAM& os) {
        m_Os = &os;
    }

    // No need for an explicit destructor here as the allocators clear themselves
    // on destruction.
    virtual ~CBoostJsonWriterBase() = default;

    //! Push a named allocator on to the stack
    //! Look in the cache for the allocator - creating it if not present
    void pushAllocator(const std::string& allocatorName) {
        TPoolAllocatorPtr& ptr = m_AllocatorCache[allocatorName];
        if (ptr == nullptr) {
            ptr = std::make_shared<CBoostJsonPoolAllocator>();
        }
        m_JsonPoolAllocators.push(ptr);
    }

    //! Remove the last pushed allocator from the stack
    void popAllocator() {
        if (!m_JsonPoolAllocators.empty()) {
            TPoolAllocatorPtr allocator = m_JsonPoolAllocators.top();
            if (allocator) {
                allocator->clear();
            }
            m_JsonPoolAllocators.pop();
        }
    }

    //! Get a valid allocator from the stack
    //! If no valid allocator can be found then store and return a freshly minted one
    std::shared_ptr<CBoostJsonPoolAllocator> getAllocator() const {
        TPoolAllocatorPtr allocator;
        CBoostJsonPoolAllocator* rawAllocator = nullptr;
        while (!m_JsonPoolAllocators.empty()) {
            allocator = m_JsonPoolAllocators.top();

            if (allocator && (rawAllocator = allocator.get())) {
                break;
            } else {
                LOG_ERROR(<< "Invalid JSON memory allocator encountered. Removing.");
                m_JsonPoolAllocators.pop();
            }
        }

        // shouldn't ever happen as it indicates that the default allocator is invalid
        if (rawAllocator == nullptr) {
            LOG_ERROR(<< "No viable JSON memory allocator encountered. Recreating.");
            allocator = std::make_shared<CBoostJsonPoolAllocator>();
            m_JsonPoolAllocators.push(allocator);
        }

        return allocator;
    }

    boost::json::memory_resource& getRawAllocator() const {
        return this->getAllocator()->get();
    }

    bool IsComplete() const {
        return m_Levels.top() == 0;
    }

    bool WriteRawValue(const std::string& rawValue) {
        m_JsonStrm << (m_Levels.top() == 0 ? "" : ",");
        m_JsonStrm << rawValue;
        return true;
    }

    virtual bool StartDocument() {
        m_JsonStrm.str("");
        m_JsonStrm << "{";
//        m_Values.reset();
        m_Levels.push(0);
        m_ContainerType.push(E_Object);
        return true;
    }

    virtual bool StartObject() {
        if (m_ContainerType.top() == E_Array) {
            m_JsonStrm << (m_Levels.top() == 0 ? "" : ",");
        }
        m_JsonStrm << "{";
        m_Levels.top()++;
        m_Levels.push(0);
        m_ContainerType.push(E_Object);
        return true;
    }

    virtual bool EndObject(std::size_t memberCount = 0) {
        m_JsonStrm << "}";
//        m_Values.push_object(m_Levels.top());
        m_Levels.pop();
        m_ContainerType.pop();
        return true;
    }

    bool m_IsArrayDoc{false};

    virtual bool StartArray() {
        if (m_Levels.empty() == false) {
            if (m_ContainerType.top() == E_Array) {
                m_JsonStrm << (m_Levels.top() == 0 ? "" : ",");
            }
            m_Levels.top()++;
        } else {
            m_IsArrayDoc = true;
        }
        m_JsonStrm << "[";
        m_Levels.push(0);
        m_ContainerType.push(E_Array);
        return true;
    }

    virtual bool EndArray() {
        m_JsonStrm << "]";
//        m_Values.push_array(m_Levels.top());
        m_Levels.pop();
        m_ContainerType.pop();
        return true;
    }

    virtual bool Bool(bool boolVal) {
        if (m_ContainerType.top() == E_Array) {
            m_JsonStrm << (m_Levels.top() == 0 ? "" : ",");
        }

        m_Levels.top()++;

        m_JsonStrm << std::boolalpha << boolVal;
//        m_Values.push_bool(boolVal);
        return true;
    }

    virtual bool Null() {
        if (m_ContainerType.top() == E_Array) {
            m_JsonStrm << (m_Levels.top() == 0 ? "" : ",");
        }

        m_Levels.top()++;

        m_JsonStrm << "null";
        return true;
    }

    virtual bool Int(std::int64_t intVal) {
        if (m_ContainerType.top() == E_Array) {
            m_JsonStrm << (m_Levels.top() == 0 ? "" : ",");
        }
        m_Levels.top()++;

        m_JsonStrm << intVal;
//        m_Values.push_int64(intVal);
        return true;
    }

    virtual bool Int64(std::int64_t int64Val) {
        if (m_ContainerType.top() == E_Array) {
            m_JsonStrm << (m_Levels.top() == 0 ? "" : ",");
        }
        m_Levels.top()++;

        m_JsonStrm << int64Val;
//        m_Values.push_int64(int64Val);
        return true;
    }

    virtual bool Uint(std::uint64_t uintVal) {
        if (m_ContainerType.top() == E_Array) {
            m_JsonStrm << (m_Levels.top() == 0 ? "" : ",");
        }
        m_Levels.top()++;

        m_JsonStrm << uintVal;
//        m_Values.push_uint64(uintVal);
        return true;
    }

    virtual bool Uint64(std::uint64_t uintVal) {
        if (m_ContainerType.top() == E_Array) {
            m_JsonStrm << (m_Levels.top() == 0 ? "" : ",");
        }
        m_Levels.top()++;

        m_JsonStrm << uintVal;
//        m_Values.push_uint64(uintVal);
        return true;
    }

    virtual bool String(const std::string& str) {
        if (m_ContainerType.top() == E_Array) {
            m_JsonStrm << (m_Levels.top() == 0 ? "" : ",");
        }
        m_Levels.top()++;

        m_JsonStrm << "\"" << str << "\"";
//        m_Values.push_string(str);
        return true;
    }

    virtual bool String(const std::string_view & str) {
        if (m_ContainerType.top() == E_Array) {
            m_JsonStrm << (m_Levels.top() == 0 ? "" : ",");
        }
        m_Levels.top()++;

        m_JsonStrm << "\"" << str << "\"";
//        m_Values.push_string(str);
        return true;
    }

    virtual bool Key(const std::string& key) {
        m_JsonStrm << (m_Levels.top() == 0 ? "" : ",");
        m_JsonStrm << "\"" << key << "\":";
//        m_Values.push_key(key);
        return true;
    }

    virtual bool Double(double d) {
        if (m_ContainerType.top() == E_Array) {
            m_JsonStrm << (m_Levels.top() == 0 ? "" : ",");
        }
        // rewrite NaN and Infinity to 0
        if (std::isfinite(d) == false) {
            d = 0.0;
//            m_JsonStrm << CStringUtils::typeToStringPretty(0);
//            m_Values.push_double(0);
        }
        m_Levels.top()++;
//        m_JsonStrm << std::showpoint << std::scientific << std::setprecision(0) << d;
        m_JsonStrm << CStringUtils::typeToStringPretty(d);
//        m_Values.push_double(d);
        return true;
    }

    void Flush() {
//        this->write();
        this->flush();
    }

    //! Writes an epoch second timestamp as an epoch millis timestamp
    virtual bool Time(core_t::TTime t) {
        if (m_ContainerType.top() == E_Array) {
            m_JsonStrm << (m_Levels.top() == 0 ? "" : ",");
        }
        m_Levels.top()++;

        m_JsonStrm << CTimeUtils::toEpochMs(t);
//        m_Values.push_int64(CTimeUtils::toEpochMs(t));
        return true;
    }

    //! Push a constant string into a supplied boost::json array value
    //! \p[in] value constant string
    //! \p[out] obj boost::json array to contain the \p value
    //! \p name must outlive \p obj or memory corruption will occur.
    void pushBack(const char* value, json::array& obj) const {
        obj.push_back(value);
    }

    //! Push a generic boost::json value object into a supplied boost::json object value
    //! \p[in] value generic boost::json value object
    //! \p[out] obj boost::json value to contain the \p value
    //! \p name must outlive \p obj or memory corruption will occur.
    template<typename T>
    void pushBack(T&& value, TValue& obj) const {
        obj.as_array().push_back(value);
    }

    //! Push a generic boost::json value object into a supplied boost::json object value
    //! \p[in] value generic boost::json value object
    //! \p[out] obj shared pointer to a boost::json value to contain the \p value
    //! \p name must outlive \p obj or memory corruption will occur.
    template<typename T>
    void pushBack(T&& value, const TValuePtr& obj) const {
        obj->as_array().push_back(value);
    }

    //! Add an array of doubles to an object.
    //! \p fieldName must outlive \p obj or memory corruption will occur.
    template<typename CONTAINER>
    void addDoubleArrayFieldToObj(const std::string& fieldName,
                                  const CONTAINER& values,
                                  TDocument& obj) const {
        TValue array = this->makeArray(values.size());

        bool considerLogging(true);
        for (const auto& value : values) {
            this->checkArrayNumberFinite(value, fieldName, considerLogging);
            if (std::isfinite(value) == false) { // TODO tidy into virtual method allowing behaviour override
                this->pushBack(0, array);
            } else {
                this->pushBack(value, array);
            }
        }

        this->addMember(fieldName, array, obj);
    }

    //! write the boost::json value document to the output stream
    //! \p[in] doc boost::json document value to write out
    virtual bool write(const json::value& doc)  = 0;
    virtual bool write(const std::string& docStr)  = 0;
    virtual bool write()  = 0;


    std::string writeDocToString()  {
//        json::value doc = m_Values.release();
//        return this->writeDocToString(doc);
        return m_JsonStrm.str();
    }

    std::string writeDocToString(const TValue& doc) const {
        LOG_INFO(<< "doc: " << doc);
        if (m_Key.size() > 0) {
            LOG_INFO(<< "m_Key: " << m_Key);

            if (doc.as_object().contains(m_Key)) {
                LOG_INFO(<< "Found key: " << m_Key << " in doc " << doc);
                if (doc.at(m_Key).is_string()) {
                    json::value jinnerval = doc.at(m_Key).as_string();
                    std::string innerval = json::serialize(jinnerval);
                    LOG_INFO(<< "innerval: " << innerval);

                    std::string cooked = std::regex_replace(innerval, std::regex( R"(\\\\)" ), R"(\)");
                    cooked = std::regex_replace(cooked, std::regex( "\\\\\"" ), R"(")");
                    cooked = std::regex_replace(cooked, std::regex( "^\"" ), R"()");
                    cooked = std::regex_replace(cooked, std::regex( R"("$)" ), R"()");

                    LOG_INFO(<< "cooked: " << cooked);

                    json::value jv = json::parse(cooked);
                    const_cast<json::value&>(doc).as_object()[m_Key] = jv;
                }
            } else {
                LOG_INFO(<< "Did not find key: " << m_Key << " in doc " << doc);
            }
        }

        json::serialize_options opt;
        opt.allow_infinity_and_nan = false;
        std::string ret = json::serialize(doc, opt);
        LOG_DEBUG(<< "Returning: " << ret);
        return ret;
    }

    //! Return a new boost::json document
    TDocument makeDoc() const {
        TDocument newDoc(&this->getRawAllocator());
        return newDoc;
    }

    //! Return a weak pointer to a new boost::json document
    //! This is a convenience function to simplify the (temporary)
    //! storage of newly created documents in containers.
    //! Note: Be aware that the lifetime of the document
    //! should not exceed that of the writer lest the document
    //! be invalidated.
    std::weak_ptr<json::object> makeStorableDoc() const {
        return this->getAllocator()->makeStorableDoc();
    }

    //! Return a new boost::json array
    json::array makeArray(size_t length = 0) const {
        json::array array;
        if (length > 0) {
            array.reserve(length);
        }
        return array;
    }

    //! Return a new boost::json object
    json::object makeObject() const {
        return boost::json::object();
    }

    //! Adds a generic boost::json value field to an object.
    //! \p[in] name field name
    //! \p[in] value generic boost::json value
    //! \p[out] obj shared pointer to boost::json object to contain the \p name \p value pair
    TValuePtr addMember(const std::string& name, TValue& value, const TValuePtr& obj) const {
        obj->as_object()[name] = value;
        return obj;
    }

    TDocument & addMember(const std::string& name, const TValue& value, TDocument& obj) const {
        obj[name] = value;
        return obj;
    }

    TDocument & addMember(const std::string& name, TDocument value, TDocument& obj) const {
        obj[name] = value;
        return obj;
    }

    TDocument & addMember(const std::string& name, const json::array& value, TDocument& obj) const {
        obj[name] = value;
        return obj;
    }

    //! Adds a copy of a string field to an object.
    //! \p[in] name field name
    //! \p[in] value string field to be copied
    //! \p[out] obj shared pointer to boost::json object to contain the \p name \p value pair
    TValuePtr addMember(const std::string& name,
                        const std::string& value,
                        const TValuePtr& obj) const {
        obj->as_object()[name] = value;
        return obj;
    }

    TValue addMember(const std::string& name,
                        const std::string& value,
                        TValue& obj) const {
        obj.as_object()[name] = value;
        return obj;
    }

    //! Adds a string field as a reference to an object (use for adding constant strings).
    //! \p[in] name field name
    //! \p[in] value string field
    //! \p[out] obj shared pointer to boost::json object to contain the \p name \p value pair
    TValuePtr addMemberRef(const std::string& name,
                           const std::string& value,
                           const TValuePtr& obj) const {
        obj->as_object()[name] = value;
        return obj;
    }

//    //! Adds a generic boost::json value field to an object.
//    //! \p[in] name field name
//    //! \p[in] value generic boost::json value
//    //! \p[out] obj boost::json object to contain the \p name \p value pair
//    void addMember(const std::string& name, TValue& value, TValue& obj) const {
//        obj.as_object()[name] = value;
//    }

    //! Adds a copy of a string field to an object.
    //! \p[in] name field name
    //! \p[in] value string field to be copied
    //! \p[out] obj boost::json object to contain the \p name \p value pair
    void addMember(const std::string& name, const std::string& value, TDocument & obj) const {
        obj[name] = value;
    }

    //! Adds a string field as a reference to an object (use for adding constant strings).
    //! \p[in] name field name
    //! \p[in] value string field
    //! \p[out] obj boost::json object to contain the \p name \p value pair
    void addMemberRef(const std::string& name, const std::string& value, TDocument & obj) const {
        obj[name] = value;
    }

    //! Adds a copy of a string field with the name fieldname to an object.
    //! \p fieldName must outlive \p obj or memory corruption will occur.
    void addStringFieldCopyToObj(const std::string& fieldName,
                                 const std::string& value,
                                 TDocument& obj,
                                 bool allowEmptyString = false) const {
        // Don't add empty strings unless explicitly told to
        if (!allowEmptyString && value.empty()) {
            return;
        }

        this->addMember(fieldName, value, obj);
    }

    //! Adds a reference to a string field with the name fieldname to an object.
    //! \p fieldName AND \p value must outlive \p obj or memory corruption will occur.
    //! This is an optimized version of addStringFieldToObj() avoiding
    //! the string copy for the value. Use with care.
    void addStringFieldReferenceToObj(const std::string& fieldName,
                                      const std::string& value,
                                      TDocument & obj,
                                      bool allowEmptyString = false) const {
        // Don't add empty strings unless explicitly told to
        if (!allowEmptyString && value.empty()) {
            return;
        }

        this->addMemberRef(fieldName, value, obj);
    }

    //! Adds a time field with the name fieldname to an object.
    //! Automatically turns time from 'seconds_since_epoch' into 'milliseconds_since_epoch'
    //! \p fieldName must outlive \p obj or memory corruption will occur.
    void addTimeFieldToObj(const std::string& fieldName, core_t::TTime value, TDocument & obj) const {
        TValue v(CTimeUtils::toEpochMs(value));
        this->addMember(fieldName, v, obj);
    }

    //! Adds a double field with the name fieldname to an object.
    //! \p fieldName must outlive \p obj or memory corruption will occur.
    void addDoubleFieldToObj(const std::string& fieldName, double value, TDocument& obj) const {
        TValue v;
        if (std::isfinite(value) == false) {
            LOG_ERROR(<< "Adding " << value << " to the \"" << fieldName
                      << "\" field of a JSON document");
            // Don't return - make a best effort to add the value
            // Mimic rapidjson behaviour and convert infinity value to 0
            v = TValue(0);
        } else {
            v = TValue(value);
        }
        this->addMember(fieldName, v, obj);
    }

    //! Adds a bool field with the name fieldname to an object.
    //! \p fieldName must outlive \p obj or memory corruption will occur.
    void addBoolFieldToObj(const std::string& fieldName, bool value, TDocument& obj) const {
        TValue v(value);
        this->addMember(fieldName, v, obj);
    }

    //! Adds a signed integer field with the name fieldname to an object.
    //! \p fieldName must outlive \p obj or memory corruption will occur.
    void addIntFieldToObj(const std::string& fieldName, std::int64_t value, TDocument& obj) const {
        TValue v(value);
        this->addMember(fieldName, v, obj);
    }

    //! Adds an unsigned integer field with the name fieldname to an object.
    //! \p fieldName must outlive \p obj or memory corruption will occur.
    void addUIntFieldToObj(const std::string& fieldName, std::uint64_t value, TDocument & obj) const {
        TValue v(value);
        this->addMember(fieldName, v, obj);
    }

    //! Add an array of strings to an object.
    //! \p fieldName must outlive \p obj or memory corruption will occur.
    void addStringArrayFieldToObj(const std::string& fieldName,
                                  const TStrVec& values,
                                  TDocument & obj) const {
        this->addMember(fieldName, json::value_from(values), obj);
//        this->addArrayToObj(fieldName, values.begin(), values.end(), obj);
    }

    void addStringArrayFieldToObj(const std::string& fieldName,
                                  const json::array& values,
                                  TDocument & obj) const {
        this->addMember(fieldName, values, obj);
    }

    //! Add an array of strings to an object.
    //! \p fieldName must outlive \p obj or memory corruption will occur.
    void addStringArrayFieldToObj(const std::string& fieldName,
                                  const TStrUSet& values,
                                  TValue& obj) const {
        using TStrCPtrVec = std::vector<const std::string*>;

        TStrCPtrVec ordered;
        ordered.reserve(values.size());
        for (const auto& value : values) {
            ordered.push_back(&value);
        }
        std::sort(ordered.begin(), ordered.end(),
                  CFunctional::SDereference<std::less<std::string>>());

        addArrayToObj(fieldName,
                      boost::iterators::make_indirect_iterator(ordered.begin()),
                      boost::iterators::make_indirect_iterator(ordered.end()), obj);
    }

    //! Add an array of pair double, pair double double to an object.
    //! \p fieldName must outlive \p obj or memory corruption will occur.
    void addDoubleDoubleDoublePrPrArrayFieldToObj(const std::string& fieldName,
                                                  const TDoubleDoubleDoublePrPrVec& values,
                                                  TValue& obj) const {
        TValue array = this->makeArray(values.size());

        bool considerLogging(true);
        for (const auto& value : values) {
            double firstVal = value.first;
            this->checkArrayNumberFinite(firstVal, fieldName, considerLogging);
            this->pushBack(firstVal, array);
            double secondFirstVal = value.second.first;
            this->checkArrayNumberFinite(secondFirstVal, fieldName, considerLogging);
            this->pushBack(secondFirstVal, array);
            double secondSecondVal = value.second.second;
            this->checkArrayNumberFinite(secondSecondVal, fieldName, considerLogging);
            this->pushBack(secondSecondVal, array);
        }

        this->addMember(fieldName, array, obj);
    }

    //! Add an array of pair double double to an object.
    //! \p fieldName must outlive \p obj or memory corruption will occur.
    void addDoubleDoublePrArrayFieldToObj(const std::string& firstFieldName,
                                          const std::string& secondFieldName,
                                          const TDoubleDoublePrVec& values,
                                          TValue& obj) const {
        TValue firstArray = this->makeArray(values.size());
        TValue secondArray = this->makeArray(values.size());

        bool considerLoggingFirst(true);
        bool considerLoggingSecond(true);
        for (const auto& value : values) {
            double firstVal = value.first;
            this->checkArrayNumberFinite(firstVal, firstFieldName, considerLoggingFirst);
            this->pushBack(firstVal, firstArray);
            double secondVal = value.second;
            this->checkArrayNumberFinite(secondVal, secondFieldName, considerLoggingSecond);
            this->pushBack(secondVal, secondArray);
        }

        this->addMember(firstFieldName, firstArray, obj);
        this->addMember(secondFieldName, secondArray, obj);
    }

    //! Add an array of TTimes to an object.
    //! \p fieldName must outlive \p obj or memory corruption will occur.
    //! Note: The time values are adjusted to be in standard Java format
    //!i.e. milliseconds since epoch
    void addTimeArrayFieldToObj(const std::string& fieldName,
                                const TTimeVec& values,
                                TDocument& obj) const {
        TValue array = this->makeArray(values.size());

        for (const auto& value : values) {
            this->pushBack(CTimeUtils::toEpochMs(value), array);
        }

        this->addMember(fieldName, array, obj);
    }

    //! Checks if the \p obj has a member named \p fieldName and
    //! removes it if it does.
    void removeMemberIfPresent(const std::string& fieldName, TDocument & obj) const {
        auto pos = obj.find(fieldName);
        if (pos != obj.end()) {
            obj.erase(pos);
        }
    }

    virtual void put(char c) {
//        m_Os->put(c); // TODO
    }

    bool topLevel() {
        if (m_IsArrayDoc) {
            return m_Levels.top() == 0;
        }
        return m_Levels.empty();
    }

    virtual void flush() {
        m_JsonStrm.flush();
    }

    void setKey(const std::string& key) {
        m_Key = key;
    }

protected:

    OUTPUT_STREAM& outputStream() {
        return *m_Os;
    }

private:
    //! Log a message if we're trying to add nan/infinity to a JSON array
    template<typename NUMBER>
    void checkArrayNumberFinite(NUMBER val, const std::string& fieldName, bool& considerLogging) const {
        if (considerLogging && (std::isfinite(val) == false)) {
            LOG_ERROR(<< "Adding " << val << " to the \"" << fieldName
                      << "\" array in a JSON document");
            // Don't return - make a best effort to add the value
            // Some writers derived from this class may defend themselves by converting to 0
            considerLogging = false;
        }
    }

    //! Convert \p value to a RapidJSON value.
    TValue asRapidJsonValue(const std::string& value) const {
        return {value, this->getRawAllocator()};
    }

    //! Convert the range [\p begin, \p end) to a RapidJSON array and add to \p obj.
    template<typename ITR>
    void addArrayToObj(const std::string& fieldName, ITR begin, ITR end, TValue& obj) const {
        json::array array = this->makeArray(std::distance(begin, end)).as_array();
        for (/**/; begin != end; ++begin) {
            this->pushBack(asRapidJsonValue(*begin), array);
        }
        this->addMember(fieldName, array, obj);
    }

    void reset(OUTPUT_STREAM& os) {
        m_Os = &os;
    }

private:

    OUTPUT_STREAM* m_Os;

    unsigned char m_Buffer[4096];

    std::stack<std::size_t> m_Levels;

    enum E_ContainerType {
        E_Object = 0,
        E_Array = 1
    };
    std::stack<E_ContainerType> m_ContainerType;

    //! cache allocators for potential reuse
    TStrPoolAllocatorPtrMap m_AllocatorCache;

    //! Allow for different batches of documents to use independent allocators
    mutable TPoolAllocatorPtrStack m_JsonPoolAllocators;

    std::string m_Key;

protected:
    std::ostringstream m_JsonStrm;
//    json::value_stack m_Values;
    std::size_t m_DocumentCount{0};
};
}
}

#endif /*  INCLUDED_ml_core_CBoostJsonWriterBase_h */
