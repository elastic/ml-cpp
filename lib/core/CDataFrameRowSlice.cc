/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDataFrameRowSlice.h>

#include <core/CBase64Filter.h>
#include <core/CDataFrame.h>
#include <core/CHashing.h>
#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CompressUtils.h>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/make_unique.hpp>

#include <memory>

namespace ml {
namespace core {
using TFloatVec = std::vector<CFloatStorage>;
using TFloatVecCItr = TFloatVec::const_iterator;

namespace {
using namespace data_frame_row_slice_detail;

//! \brief A handle for reading CRawDataFrameRowSlice objects.
//!
//! DESCRIPTION:\n
//! This stores a reference to the underlying memory. This is primarily
//! intended to stop duplication of the underlying data frame state.
//! Together with our memory mapped vector type this means algorithms
//! on top of a raw data frame can work entirely in terms of the raw
//! values stored in the data frame.
class CORE_EXPORT CMainMemoryDataFrameRowSliceHandle : public CDataFrameRowSliceHandleImpl {
public:
    CMainMemoryDataFrameRowSliceHandle(const TFloatVec& values)
        : m_Values{values} {}
    virtual TImplPtr clone() const {
        return boost::make_unique<CMainMemoryDataFrameRowSliceHandle>(m_Values);
    }
    virtual const TFloatVec& values() const { return m_Values; }
    virtual bool bad() const { return false; }

private:
    using TFloatVecCRef = std::reference_wrapper<const TFloatVec>;

private:
    //! A reference to the data frame slice.
    TFloatVecCRef m_Values;
};

//! \brief A handle for reading CRawDataFrameRowSlice objects.
//!
//! DESCRIPTION:\n
//! This stores a copy of values since these are created on-the-fly when
//! the slice is inflated.
class CORE_EXPORT COnDiskDataFrameRowSliceHandle : public CDataFrameRowSliceHandleImpl {
public:
    COnDiskDataFrameRowSliceHandle(TFloatVec values)
        : m_Values{std::move(values)} {}
    virtual TImplPtr clone() const {
        return boost::make_unique<COnDiskDataFrameRowSliceHandle>(m_Values);
    }
    virtual const TFloatVec& values() const { return m_Values; }
    virtual bool bad() const { return false; }

private:
    //! A copy of the values in the data frame slice.
    TFloatVec m_Values;
};

//! \brief An implementation of a bad handle.
//!
//! DESCRIPTION:\n
//! This is used to signal that there is a problem accessing the slice.
class CORE_EXPORT CBadDataFrameRowSliceHandle : public CDataFrameRowSliceHandleImpl {
public:
    virtual TImplPtr clone() const {
        return boost::make_unique<CBadDataFrameRowSliceHandle>();
    }
    virtual const TFloatVec& values() const { return m_Empty; }
    virtual bool bad() const { return true; }

private:
    //! Stub for the values.
    TFloatVec m_Empty;
};
}

//////// CDataFrameRowSliceHandle ////////

CDataFrameRowSliceHandle::CDataFrameRowSliceHandle(TImplPtr impl)
    : m_Impl{std::move(impl)} {
}

CDataFrameRowSliceHandle::CDataFrameRowSliceHandle(const CDataFrameRowSliceHandle& other)
    : m_Impl{other.m_Impl->clone()} {
}

CDataFrameRowSliceHandle::CDataFrameRowSliceHandle(CDataFrameRowSliceHandle&& other)
    : m_Impl{std::move(other.m_Impl)} {
}

CDataFrameRowSliceHandle& CDataFrameRowSliceHandle::
operator=(const CDataFrameRowSliceHandle& other) {
    m_Impl = other.m_Impl->clone();
    return *this;
}

CDataFrameRowSliceHandle& CDataFrameRowSliceHandle::operator=(CDataFrameRowSliceHandle&& other) {
    m_Impl = std::move(other.m_Impl);
    return *this;
}

bool CDataFrameRowSliceHandle::bad() const {
    return m_Impl->bad();
}

std::size_t CDataFrameRowSliceHandle::size() const {
    return m_Impl->values().size();
}

TFloatVecCItr CDataFrameRowSliceHandle::begin() const {
    return m_Impl->values().begin();
}

TFloatVecCItr CDataFrameRowSliceHandle::end() const {
    return m_Impl->values().end();
}

//////// CMainMemoryDataFrameRowSlice ////////

CMainMemoryDataFrameRowSlice::CMainMemoryDataFrameRowSlice(std::size_t firstRow, TFloatVec state)
    : m_FirstRow{firstRow}, m_State{std::move(state)} {
    LOG_TRACE(<< "slice size = " << m_State.size() << " capacity = " << m_State.capacity());
    m_State.shrink_to_fit();
}

CMainMemoryDataFrameRowSlice::TSizeHandlePr CMainMemoryDataFrameRowSlice::read() const {
    return {m_FirstRow, {boost::make_unique<CMainMemoryDataFrameRowSliceHandle>(m_State)}};
}

std::size_t CMainMemoryDataFrameRowSlice::staticSize() const {
    return sizeof(*this);
}

std::size_t CMainMemoryDataFrameRowSlice::memoryUsage() const {
    return CMemory::dynamicSize(m_State);
}

//////// COnDiskDataFrameRowSlice ////////

namespace {

//! Check if there is \p minimumSpace disk space available.
bool sufficientDiskSpaceAvailable(const boost::filesystem::path& path, std::size_t minimumSpace) {
    boost::system::error_code errorCode;
    auto spaceInfo = boost::filesystem::space(path, errorCode);
    if (errorCode) {
        LOG_ERROR(<< "Failed to retrieve disk information for " << path
                  << " error " << errorCode.message());
        return false;
    }
    if (spaceInfo.available < minimumSpace) {
        LOG_ERROR(<< "Insufficient space have " << spaceInfo.available
                  << " and need " << minimumSpace);
        return false;
    }
    return true;
}

//! Checksum \p slice.
uint64_t checksum(const TFloatVec& slice) {
    return CHashing::murmurHash64(
        slice.data(), static_cast<int>(sizeof(CFloatStorage) * slice.size()), 0);
}
}

COnDiskDataFrameRowSlice::COnDiskDataFrameRowSlice(const TTemporaryDirectoryPtr& directory,
                                                   std::size_t firstRow,
                                                   TFloatVec state)
    : m_StateIsBad{directory->bad()}, m_FirstRow{firstRow}, m_NumberRows{state.size()},
      m_Directory{directory}, m_FileName{directory->name()}, m_Checksum{checksum(state)} {
    m_FileName /= boost::filesystem::unique_path(
        "rows-" + std::to_string(firstRow) + "-%%%%-%%%%-%%%%-%%%%");

    if (m_StateIsBad == false) {
        std::size_t bytes{sizeof(CFloatStorage) * m_NumberRows};
        LOG_TRACE(<< "bytes = " << bytes);

        std::ofstream file{m_FileName.string(), std::ios_base::out | std::ios_base::binary};
        file.write(reinterpret_cast<const char*>(state.data()), bytes);
    }
}

COnDiskDataFrameRowSlice::TSizeHandlePr COnDiskDataFrameRowSlice::read() const {

    if (m_StateIsBad) {
        LOG_ERROR(<< "Bad row slice 'rows-" << m_FirstRow << "'");
        return {0, {boost::make_unique<CBadDataFrameRowSliceHandle>()}};
    }

    std::size_t bytes{sizeof(CFloatStorage) * m_NumberRows};

    std::ifstream file{m_FileName.string(), std::ios_base::in | std::ios_base::binary};
    TFloatVec result(m_NumberRows);
    file.read(reinterpret_cast<char*>(result.data()), bytes);

    LOG_TRACE(<< "state = " << result[0] << "," << result[1] << ",...");

    if (file.bad()) {
        LOG_ERROR(<< "Failed to read 'rows-" << m_FirstRow << "'");
        m_StateIsBad = true;
        return {0, {boost::make_unique<CBadDataFrameRowSliceHandle>()}};
    }

    if (checksum(result) != m_Checksum) {
        LOG_ERROR(<< "Corrupt 'rows-" << m_FirstRow << "'");
        m_StateIsBad = true;
        return {0, {boost::make_unique<CBadDataFrameRowSliceHandle>()}};
    }

    return {m_FirstRow,
            {boost::make_unique<COnDiskDataFrameRowSliceHandle>(std::move(result))}};
}

std::size_t COnDiskDataFrameRowSlice::staticSize() const {
    return sizeof(*this);
}

std::size_t COnDiskDataFrameRowSlice::memoryUsage() const {
    return CMemory::dynamicSize(m_Directory) + CMemory::dynamicSize(m_FileName.string());
}

COnDiskDataFrameRowSlice::CTemporaryDirectory::CTemporaryDirectory(const std::string& name,
                                                                   std::size_t minimumSpace)
    : m_Name{name} {
    m_Name /= boost::filesystem::unique_path("dataframe-%%%%-%%%%-%%%%-%%%%");
    LOG_TRACE(<< "Trying to create directory '" << m_Name << "'");

    boost::system::error_code errorCode;
    boost::filesystem::create_directories(m_Name, errorCode);
    if (errorCode) {
        LOG_ERROR(<< "Failed to create temporary data from: '" << m_Name
                  << "' error " << errorCode.message());
        m_StateIsBad = true;
    }

    if (m_StateIsBad == false) {
        m_StateIsBad = (sufficientDiskSpaceAvailable(m_Name, minimumSpace) == false);
    }

    if (m_StateIsBad == false) {
        LOG_TRACE(<< "Created '" << m_Name << "'");
    }
}

COnDiskDataFrameRowSlice::CTemporaryDirectory::~CTemporaryDirectory() {
    boost::system::error_code errorCode;
    boost::filesystem::remove_all(m_Name, errorCode);
    if (errorCode) {
        LOG_ERROR(<< "Failed to cleanup temporary data from: '" << m_Name
                  << "' error " << errorCode.message());
    }
}

const std::string& COnDiskDataFrameRowSlice::CTemporaryDirectory::name() const {
    return m_Name.string();
}

bool COnDiskDataFrameRowSlice::CTemporaryDirectory::bad() const {
    return m_StateIsBad;
}
}
}
