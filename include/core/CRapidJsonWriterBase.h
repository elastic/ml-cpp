/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_core_CRapidJsonWriterBase_h
#define INCLUDED_ml_core_CRapidJsonWriterBase_h

#include <core/CFunctional.h>
#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CRapidJsonPoolAllocator.h>
#include <core/CTimeUtils.h>
#include <core/CoreTypes.h>
#include <core/ImportExport.h>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <boost/iterator/indirect_iterator.hpp>
#include <boost/make_shared.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/unordered/unordered_map.hpp>
#include <boost/unordered/unordered_set.hpp>

#include <stack>

namespace ml {
namespace core {
//! \brief
//! A Json writer with fixed length allocator pool
//! With utility functions for adding fields to JSON objects.
//!
//! DESCRIPTION:\n
//! Wraps up the code needed to add various types of values to JSON
//! objects.  Note that if a JSON document is an object then these methods
//! can be used to add fields to the RapidJSON document object too.
//!
//!
//! IMPLEMENTATION DECISIONS:\n
//! Templatized on the actual rapidjson writer type - defaults to rapidjson::Writer
//!
//! Field names are not copied - the field name strings MUST outlive the
//! JSON document they are being added to, or else memory corruption will
//! occur.
//!
//! Empty string fields are not written to the output unless specifically
//! requested.
//!
//! Memory for values added to the output documents is allocated from a pool (to
//! reduce allocation cost and memory fragmentation).  The user of this class
//! is responsible for managing this pool.
//!
template<typename OUTPUT_STREAM,
         typename SOURCE_ENCODING = rapidjson::UTF8<>,
         typename TARGET_ENCODING = rapidjson::UTF8<>,
         typename STACK_ALLOCATOR = rapidjson::CrtAllocator,
         unsigned WRITE_FLAGS = rapidjson::kWriteDefaultFlags,
         template<typename, typename, typename, typename, unsigned> class JSON_WRITER = rapidjson::Writer>
class CRapidJsonWriterBase : public JSON_WRITER<OUTPUT_STREAM, SOURCE_ENCODING, TARGET_ENCODING, STACK_ALLOCATOR, WRITE_FLAGS> {
public:
    typedef std::vector<core_t::TTime> TTimeVec;
    typedef std::vector<std::string> TStrVec;
    typedef std::vector<double> TDoubleVec;
    typedef std::pair<double, double> TDoubleDoublePr;
    typedef std::vector<TDoubleDoublePr> TDoubleDoublePrVec;
    typedef std::pair<double, TDoubleDoublePr> TDoubleDoubleDoublePrPr;
    typedef std::vector<TDoubleDoubleDoublePrPr> TDoubleDoubleDoublePrPrVec;
    typedef boost::unordered_set<std::string> TStrUSet;
    typedef rapidjson::Document TDocument;
    typedef rapidjson::Value TValue;
    typedef boost::weak_ptr<TDocument> TDocumentWeakPtr;
    typedef boost::shared_ptr<TValue> TValuePtr;

    typedef boost::shared_ptr<CRapidJsonPoolAllocator> TPoolAllocatorPtr;
    typedef std::stack<TPoolAllocatorPtr> TPoolAllocatorPtrStack;
    typedef boost::unordered_map<std::string, TPoolAllocatorPtr> TStrPoolAllocatorPtrMap;
    typedef TStrPoolAllocatorPtrMap::iterator TStrPoolAllocatorPtrMapItr;
    typedef std::pair<TStrPoolAllocatorPtrMapItr, bool> TStrPoolAllocatorPtrMapItrBoolPr;

public:
    using TRapidJsonWriterBase = JSON_WRITER<OUTPUT_STREAM, SOURCE_ENCODING, TARGET_ENCODING, STACK_ALLOCATOR, WRITE_FLAGS>;

    //! Instances of this class may very well be long lived, potentially for the lifetime of the application.
    //! Over the course of that lifetime resources will accumulate in the underlying rapidjson memory
    //! allocator. To prevent excessive memory expansion these resources will need to be cleaned regularly.
    //!
    //! In preference to clients of this class explicitly clearing the allocator a helper/wrapper class -
    //! \p CScopedRapidJsonPoolAllocator - is provided. This helper has an RAII style interface that clears the
    //! allocator when it goes out of scope which requires that the writer provides the push/popAllocator
    //! functions.  The intent of this approach is to make it possible to use one or two separate allocators
    //! for the writer at nested scope.
    //!
    //! Note that allocators are not destroyed by the pop operation, they persist for the lifetime of the
    //! writer in a cache for swift retrieval.
    CRapidJsonWriterBase(OUTPUT_STREAM& os) : TRapidJsonWriterBase(os) {
        // push a default rapidjson allocator onto our stack
        m_JsonPoolAllocators.push(boost::make_shared<CRapidJsonPoolAllocator>());
    }

    CRapidJsonWriterBase() : TRapidJsonWriterBase() {
        // push a default rapidjson allocator onto our stack
        m_JsonPoolAllocators.push(boost::make_shared<CRapidJsonPoolAllocator>());
    }

    // No need for an explicit destructor here as the allocators clear themselves
    // on destruction.
    virtual ~CRapidJsonWriterBase() = default;

    //! Push a named allocator on to the stack
    //! Look in the cache for the allocator - creating it if not present
    void pushAllocator(const std::string& allocatorName) {
        TPoolAllocatorPtr& ptr = m_AllocatorCache[allocatorName];
        if (ptr == nullptr) {
            ptr = boost::make_shared<CRapidJsonPoolAllocator>();
        }
        m_JsonPoolAllocators.push(ptr);
    }

    //! Clear and remove the last pushed allocator from the stack
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
    boost::shared_ptr<CRapidJsonPoolAllocator> getAllocator() const {
        TPoolAllocatorPtr allocator;
        CRapidJsonPoolAllocator* rawAllocator = nullptr;
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
        if (!rawAllocator) {
            LOG_ERROR(<< "No viable JSON memory allocator encountered. Recreating.");
            allocator = boost::make_shared<CRapidJsonPoolAllocator>();
            m_JsonPoolAllocators.push(allocator);
        }

        return allocator;
    }

    rapidjson::MemoryPoolAllocator<>& getRawAllocator() const { return this->getAllocator()->get(); }

    bool Double(double d) {
        // rewrite NaN and Infinity to 0
        if (!(boost::math::isfinite)(d)) {
            return TRapidJsonWriterBase::Int(0);
        }

        return TRapidJsonWriterBase::Double(d);
    }

    //! Writes an epoch second timestamp as an epoch millis timestamp
    bool Time(core_t::TTime t) { return this->Int64(CTimeUtils::toEpochMs(t)); }

    //! Push a constant string into a supplied rapidjson object value
    //! \p[in] value constant string
    //! \p[out] obj rapidjson value to contain the \p value
    //! \p name must outlive \p obj or memory corruption will occur.
    void pushBack(const char* value, TValue& obj) const { obj.PushBack(rapidjson::StringRef(value), this->getRawAllocator()); }

    //! Push a generic rapidjson value object into a supplied rapidjson object value
    //! \p[in] value generic rapidjson value object
    //! \p[out] obj rapidjson value to contain the \p value
    //! \p name must outlive \p obj or memory corruption will occur.
    template<typename T>
    void pushBack(T&& value, TValue& obj) const {
        obj.PushBack(value, this->getRawAllocator());
    }

    //! Push a generic rapidjson value object into a supplied rapidjson object value
    //! \p[in] value generic rapidjson value object
    //! \p[out] obj shared pointer to a rapidjson value to contain the \p value
    //! \p name must outlive \p obj or memory corruption will occur.
    template<typename T>
    void pushBack(T&& value, const TValuePtr& obj) const {
        obj->PushBack(value, this->getRawAllocator());
    }

    //! Add an array of doubles to an object.
    //! \p fieldName must outlive \p obj or memory corruption will occur.
    template<typename CONTAINER>
    void addDoubleArrayFieldToObj(const std::string& fieldName, const CONTAINER& values, TValue& obj) const {
        TValue array = this->makeArray(values.size());

        bool considerLogging(true);
        for (const auto& value : values) {
            this->checkArrayNumberFinite(value, fieldName, considerLogging);
            this->pushBack(value, array);
        }

        this->addMember(fieldName, array, obj);
    }

    //! write the rapidjson value document to the output stream
    //! \p[in] doc rapidjson document value to write out
    virtual void write(TValue& doc) { doc.Accept(*this); }

    //! Return a new rapidjson document
    TDocument makeDoc() const {
        TDocument newDoc(&this->getRawAllocator());
        newDoc.SetObject();
        return newDoc;
    }

    //! Return a weak pointer to a new rapidjson document
    //! This is a convenience function to simplify the (temporary)
    //! storage of newly created documents in containers.
    //! Note: Be aware that the lifetime of the document
    //! should not exceed that of the writer lest the document
    //! be invalidated.
    TDocumentWeakPtr makeStorableDoc() const { return this->getAllocator()->makeStorableDoc(); }

    //! Return a new rapidjson array
    TValue makeArray(size_t length = 0) const {
        TValue array(rapidjson::kArrayType);
        if (length > 0) {
            array.Reserve(static_cast<rapidjson::SizeType>(length), this->getRawAllocator());
        }
        return array;
    }

    //! Return a new rapidjson object
    TValue makeObject() const {
        TValue obj(rapidjson::kObjectType);
        return obj;
    }

    //! Adds a generic rapidjson value field to an object.
    //! \p[in] name field name
    //! \p[in] value generic rapidjson value
    //! \p[out] obj shared pointer to rapidjson object to contain the \p name \p value pair
    TValuePtr addMember(const std::string& name, TValue& value, const TValuePtr& obj) const {
        obj->AddMember(rapidjson::StringRef(name), value, this->getRawAllocator());
        return obj;
    }

    //! Adds a copy of a string field to an object.
    //! \p[in] name field name
    //! \p[in] value string field to be copied
    //! \p[out] obj shared pointer to rapidjson object to contain the \p name \p value pair
    TValuePtr addMember(const std::string& name, const std::string& value, const TValuePtr& obj) const {
        TValue v(value, this->getRawAllocator());
        obj->AddMember(rapidjson::StringRef(name), v, this->getRawAllocator());
        return obj;
    }

    //! Adds a string field as a reference to an object (use for adding constant strings).
    //! \p[in] name field name
    //! \p[in] value string field
    //! \p[out] obj shared pointer to rapidjson object to contain the \p name \p value pair
    TValuePtr addMemberRef(const std::string& name, const std::string& value, const TValuePtr& obj) const {
        obj->AddMember(rapidjson::StringRef(name), rapidjson::StringRef(value), this->getRawAllocator());
        return obj;
    }

    //! Adds a generic rapidjson value field to an object.
    //! \p[in] name field name
    //! \p[in] value generic rapidjson value
    //! \p[out] obj rapidjson object to contain the \p name \p value pair
    void addMember(const std::string& name, TValue& value, TValue& obj) const {
        obj.AddMember(rapidjson::StringRef(name), value, this->getRawAllocator());
    }

    //! Adds a copy of a string field to an object.
    //! \p[in] name field name
    //! \p[in] value string field to be copied
    //! \p[out] obj rapidjson object to contain the \p name \p value pair
    void addMember(const std::string& name, const std::string& value, TValue& obj) const {
        TValue v(value, this->getRawAllocator());
        obj.AddMember(rapidjson::StringRef(name), v, this->getRawAllocator());
    }

    //! Adds a string field as a reference to an object (use for adding constant strings).
    //! \p[in] name field name
    //! \p[in] value string field
    //! \p[out] obj rapidjson object to contain the \p name \p value pair
    void addMemberRef(const std::string& name, const std::string& value, TValue& obj) const {
        obj.AddMember(rapidjson::StringRef(name), rapidjson::StringRef(value), this->getRawAllocator());
    }

    //! Adds a copy of a string field with the name fieldname to an object.
    //! \p fieldName must outlive \p obj or memory corruption will occur.
    void addStringFieldCopyToObj(const std::string& fieldName, const std::string& value, TValue& obj, bool allowEmptyString = false) const {
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
    void
    addStringFieldReferenceToObj(const std::string& fieldName, const std::string& value, TValue& obj, bool allowEmptyString = false) const {
        // Don't add empty strings unless explicitly told to
        if (!allowEmptyString && value.empty()) {
            return;
        }

        this->addMemberRef(fieldName, value, obj);
    }

    //! Adds a time field with the name fieldname to an object.
    //! Automatically turns time from 'seconds_since_epoch' into 'milliseconds_since_epoch'
    //! \p fieldName must outlive \p obj or memory corruption will occur.
    void addTimeFieldToObj(const std::string& fieldName, core_t::TTime value, TValue& obj) const {
        TValue v(CTimeUtils::toEpochMs(value));
        this->addMember(fieldName, v, obj);
    }

    //! Adds a double field with the name fieldname to an object.
    //! \p fieldName must outlive \p obj or memory corruption will occur.
    void addDoubleFieldToObj(const std::string& fieldName, double value, TValue& obj) const {
        if (!(boost::math::isfinite)(value)) {
            LOG_ERROR(<< "Adding " << value << " to the \"" << fieldName << "\" field of a JSON document");
            // Don't return - make a best effort to add the value
            // Some writers derived from this class may defend themselves by converting to 0
        }
        TValue v(value);
        this->addMember(fieldName, v, obj);
    }

    //! Adds a bool field with the name fieldname to an object.
    //! \p fieldName must outlive \p obj or memory corruption will occur.
    void addBoolFieldToObj(const std::string& fieldName, bool value, TValue& obj) const {
        TValue v(value);
        this->addMember(fieldName, v, obj);
    }

    //! Adds a signed integer field with the name fieldname to an object.
    //! \p fieldName must outlive \p obj or memory corruption will occur.
    void addIntFieldToObj(const std::string& fieldName, int64_t value, TValue& obj) const {
        TValue v(value);
        this->addMember(fieldName, v, obj);
    }

    //! Adds an unsigned integer field with the name fieldname to an object.
    //! \p fieldName must outlive \p obj or memory corruption will occur.
    void addUIntFieldToObj(const std::string& fieldName, uint64_t value, TValue& obj) const {
        TValue v(value);
        this->addMember(fieldName, v, obj);
    }

    //! Add an array of strings to an object.
    //! \p fieldName must outlive \p obj or memory corruption will occur.
    void addStringArrayFieldToObj(const std::string& fieldName, const TStrVec& values, TValue& obj) const {
        this->addArrayToObj(fieldName, values.begin(), values.end(), obj);
    }

    //! Add an array of strings to an object.
    //! \p fieldName must outlive \p obj or memory corruption will occur.
    void addStringArrayFieldToObj(const std::string& fieldName, const TStrUSet& values, TValue& obj) const {
        using TStrCPtrVec = std::vector<const std::string*>;

        TStrCPtrVec ordered;
        ordered.reserve(values.size());
        for (const auto& value : values) {
            ordered.push_back(&value);
        }
        std::sort(ordered.begin(), ordered.end(), CFunctional::SDereference<std::less<std::string>>());

        addArrayToObj(fieldName,
                      boost::iterators::make_indirect_iterator(ordered.begin()),
                      boost::iterators::make_indirect_iterator(ordered.end()),
                      obj);
    }

    //! Add an array of pair double, pair double double to an object.
    //! \p fieldName must outlive \p obj or memory corruption will occur.
    void
    addDoubleDoubleDoublePrPrArrayFieldToObj(const std::string& fieldName, const TDoubleDoubleDoublePrPrVec& values, TValue& obj) const {
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
    void addTimeArrayFieldToObj(const std::string& fieldName, const TTimeVec& values, TValue& obj) const {
        TValue array = this->makeArray(values.size());

        for (const auto& value : values) {
            this->pushBack(CTimeUtils::toEpochMs(value), array);
        }

        this->addMember(fieldName, array, obj);
    }

    //! Checks if the \p obj has a member named \p fieldName and
    //! removes it if it does.
    void removeMemberIfPresent(const std::string& fieldName, TValue& obj) const {
        if (obj.HasMember(fieldName)) {
            obj.RemoveMember(fieldName);
        }
    }

private:
    //! Log a message if we're trying to add nan/infinity to a JSON array
    template<typename NUMBER>
    void checkArrayNumberFinite(NUMBER val, const std::string& fieldName, bool& considerLogging) const {
        if (considerLogging && !(boost::math::isfinite)(val)) {
            LOG_ERROR(<< "Adding " << val << " to the \"" << fieldName << "\" array in a JSON document");
            // Don't return - make a best effort to add the value
            // Some writers derived from this class may defend themselves by converting to 0
            considerLogging = false;
        }
    }

    //! Convert \p value to a RapidJSON value.
    TValue asRapidJsonValue(const std::string& value) const { return {value, this->getRawAllocator()}; }

    //! Convert the range [\p begin, \p end) to a RapidJSON array and add to \p obj.
    template<typename ITR>
    void addArrayToObj(const std::string& fieldName, ITR begin, ITR end, TValue& obj) const {
        TValue array = this->makeArray(std::distance(begin, end));
        for (/**/; begin != end; ++begin) {
            this->pushBack(asRapidJsonValue(*begin), array);
        }
        this->addMember(fieldName, array, obj);
    }

private:
    //! cache allocators for potential reuse
    TStrPoolAllocatorPtrMap m_AllocatorCache;

    //! Allow for different batches of documents to use independent allocators
    mutable TPoolAllocatorPtrStack m_JsonPoolAllocators;
};
}
}

#endif /*  INCLUDED_ml_core_CRapidJsonWriterBase_h */
