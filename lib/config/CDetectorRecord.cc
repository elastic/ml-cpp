/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#include <config/CDetectorRecord.h>

#include <core/CContainerPrinter.h>
#include <core/CHashing.h>
#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <config/CDetectorSpecification.h>
#include <config/Constants.h>

#include <boost/range.hpp>

namespace ml {
namespace config {
namespace {

using TField = const CDetectorSpecification::TOptionalStr& (CDetectorSpecification::*)() const;
const TField FIELDS[] = {
    &CDetectorSpecification::argumentField,
    &CDetectorSpecification::byField,
    &CDetectorSpecification::overField,
    &CDetectorSpecification::partitionField,
};

const std::string NULL_STRING("null");

//! Print the string \p ptr or null if it is zero.
const std::string& extract(const std::string* ptr) {
    return ptr ? *ptr : NULL_STRING;
}

const core::CHashing::CMurmurHash2String HASHER;
}

CDetectorRecord::CDetectorRecord(core_t::TTime time,
                                 config_t::EFunctionCategory function,
                                 const TStrCPtrAry& fieldNames,
                                 const TStrCPtrAry& fieldValues,
                                 const TSizeAry& hashedFieldValues)
    : m_Time(time), m_Function(function), m_FieldNames(fieldNames), m_FieldValues(fieldValues), m_HashedFieldValues(hashedFieldValues) {
}

core_t::TTime CDetectorRecord::time() const {
    return m_Time;
}

config_t::EFunctionCategory CDetectorRecord::function() const {
    return m_Function;
}

const std::string* CDetectorRecord::argumentFieldName() const {
    return m_FieldNames[constants::ARGUMENT_INDEX];
}

const std::string* CDetectorRecord::byFieldName() const {
    return m_FieldNames[constants::BY_INDEX];
}

const std::string* CDetectorRecord::overFieldName() const {
    return m_FieldNames[constants::OVER_INDEX];
}

const std::string* CDetectorRecord::partitionFieldName() const {
    return m_FieldNames[constants::PARTITION_INDEX];
}

const std::string* CDetectorRecord::argumentFieldValue() const {
    return m_FieldValues[constants::ARGUMENT_INDEX];
}

const std::string* CDetectorRecord::byFieldValue() const {
    return m_FieldValues[constants::BY_INDEX];
}

const std::string* CDetectorRecord::overFieldValue() const {
    return m_FieldValues[constants::OVER_INDEX];
}

const std::string* CDetectorRecord::partitionFieldValue() const {
    return m_FieldValues[constants::PARTITION_INDEX];
}

std::size_t CDetectorRecord::argumentFieldValueHash() const {
    return m_HashedFieldValues[constants::ARGUMENT_INDEX];
}

std::size_t CDetectorRecord::byFieldValueHash() const {
    return m_HashedFieldValues[constants::BY_INDEX];
}

std::size_t CDetectorRecord::overFieldValueHash() const {
    return m_HashedFieldValues[constants::OVER_INDEX];
}

std::size_t CDetectorRecord::partitionFieldValueHash() const {
    return m_HashedFieldValues[constants::PARTITION_INDEX];
}

std::string CDetectorRecord::print() const {
    return core::CStringUtils::typeToString(m_Time) + ' ' + extract(this->argumentFieldValue()) + ' ' + extract(this->byFieldValue()) +
           ' ' + extract(this->overFieldValue()) + ' ' + extract(this->partitionFieldValue());
}

void CDetectorRecordDirectAddressTable::build(const TDetectorSpecificationVec& specs) {
    using TStrSizeUMap = boost::unordered_map<std::string, std::size_t>;
    using TStrSizeUMapCItr = TStrSizeUMap::const_iterator;

    this->clear();

    TStrSizeUMap uniques;
    size_t size = 0u;
    for (std::size_t i = 0u; i < specs.size(); ++i) {
        for (std::size_t j = 0u; j < boost::size(FIELDS); ++j) {
            if (const CDetectorSpecification::TOptionalStr& field = ((specs[i]).*FIELDS[j])()) {
                uniques.emplace(*field, uniques.size());
            }
        }
        size = std::max(size, specs[i].id());
    }

    m_FieldSchema.resize(uniques.size());
    for (TStrSizeUMapCItr i = uniques.begin(); i != uniques.end(); ++i) {
        m_FieldSchema[i->second] = std::make_pair(i->first, i->second);
    }
    m_FieldValueTable.resize(m_FieldSchema.size() + 1, 0);
    m_HashedFieldValueTable.resize(m_FieldSchema.size() + 1, HASHER(NULL_STRING));
    LOG_TRACE("field schema = " << core::CContainerPrinter::print(m_FieldSchema));

    m_DetectorFieldSchema.resize(size + 1);
    for (std::size_t i = 0u; i < specs.size(); ++i) {
        TSizeAry entry;
        for (std::size_t j = 0u; j < boost::size(FIELDS); ++j) {
            const CDetectorSpecification::TOptionalStr& field = ((specs[i]).*FIELDS[j])();
            entry[constants::CFieldIndices::ALL[j]] = field ? uniques[*field] : m_FieldSchema.size();
        }
        LOG_TRACE("Fields for " << specs[i].description() << " = " << core::CContainerPrinter::print(entry));
        m_DetectorFieldSchema[specs[i].id()] = entry;
    }
}

void CDetectorRecordDirectAddressTable::detectorRecords(core_t::TTime time,
                                                        const TStrStrUMap& fieldValues,
                                                        const TDetectorSpecificationVec& specs,
                                                        TDetectorRecordVec& result) {
    result.clear();

    if (specs.empty()) {
        return;
    }

    using TStrStrUMapCItr = TStrStrUMap::const_iterator;

    std::size_t size = 0u;
    for (std::size_t i = 0u; i < specs.size(); ++i) {
        size = std::max(size, specs[i].id());
    }

    for (std::size_t i = 0u; i < m_FieldSchema.size(); ++i) {
        TStrStrUMapCItr j = fieldValues.find(m_FieldSchema[i].first);
        m_FieldValueTable[i] = j != fieldValues.end() ? &j->second : 0;
        m_HashedFieldValueTable[i] = HASHER(m_FieldValueTable[i] ? *m_FieldValueTable[i] : NULL_STRING);
    }

    CDetectorRecord::TStrCPtrAry ni;
    CDetectorRecord::TStrCPtrAry vi;
    CDetectorRecord::TSizeAry hi;
    result.resize(size + 1, CDetectorRecord(time, config_t::E_Count, ni, vi, hi));
    for (std::size_t i = 0u; i < specs.size(); ++i) {
        std::size_t id = specs[i].id();
        const TSizeAry& schema = m_DetectorFieldSchema[id];
        config_t::EFunctionCategory function = specs[i].function();
        for (std::size_t j = 0u; j < TSizeAry::size(); ++j) {
            ni[j] = (specs[i].*FIELDS[j])().get_ptr();
            vi[j] = m_FieldValueTable[schema[j]];
            hi[j] = m_HashedFieldValueTable[schema[j]];
        }
        result[id] = CDetectorRecord(time, function, ni, vi, hi);
    }
}

void CDetectorRecordDirectAddressTable::clear() {
    m_FieldSchema.clear();
    m_DetectorFieldSchema.clear();
    m_FieldValueTable.clear();
}
}
}
