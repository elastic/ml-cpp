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
class CMainMemoryDataFrameRowSliceHandle : public CDataFrameRowSliceHandleImpl {
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
    TFloatVecCRef m_Values;
};

//! \brief A handle for reading CRawDataFrameRowSlice objects.
//!
//! DESCRIPTION:\n
//! This stores a copy of values since these are created on-the-fly when
//! the slice is inflated.
class COnDiskDataFrameRowSliceHandle : public CDataFrameRowSliceHandleImpl {
public:
    COnDiskDataFrameRowSliceHandle(TFloatVec values)
        : m_Values{std::move(values)} {}
    virtual TImplPtr clone() const {
        return boost::make_unique<COnDiskDataFrameRowSliceHandle>(m_Values);
    }
    virtual const TFloatVec& values() const { return m_Values; }
    virtual bool bad() const { return false; }

private:
    TFloatVec m_Values;
};

//! \brief An implementation of a bad handle.
//!
//! DESCRIPTION:\n
//! This is used to signal that there is a problem accessing the slice.
class CBadDataFrameRowSliceHandle : public CDataFrameRowSliceHandleImpl {
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

std::size_t CDataFrameRowSliceHandle::size() const {
    return m_Impl->values().size();
}

TFloatVecCItr CDataFrameRowSliceHandle::begin() const {
    return m_Impl->values().begin();
}

TFloatVecCItr CDataFrameRowSliceHandle::end() const {
    return m_Impl->values().end();
}

bool CDataFrameRowSliceHandle::bad() const {
    return m_Impl->bad();
}

//////// CMainMemoryDataFrameRowSlice ////////

CMainMemoryDataFrameRowSlice::CMainMemoryDataFrameRowSlice(std::size_t firstRow, TFloatVec state)
    : m_FirstRow{firstRow}, m_State{std::move(state)} {
    LOG_TRACE(<< "slice size = " << m_State.size() << " capacity = " << m_State.capacity());
    m_State.shrink_to_fit();
}

bool CMainMemoryDataFrameRowSlice::reserve(std::size_t numberColumns, std::size_t extraColumns) {
    // "Reserve" space at the end of each row for extraColumns extra columns.
    // Padding is inserted into the underlying vector which is skipped over
    // by the CRowConstIterator object.

    std::size_t numberRows{m_State.size() / numberColumns};
    std::size_t newNumberColumns{numberColumns + extraColumns};
    try {
        TFloatVec state(m_State.size() + numberRows * extraColumns);
        for (auto i = m_State.begin(), j = state.begin(); i != m_State.end();
             i += numberColumns, j += newNumberColumns) {
            std::copy(i, i + numberColumns, j);
        }
        std::swap(state, m_State);
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to reserve " << extraColumns
                  << " extra columns: caught '" << e.what() << "'");
        return false;
    }
    return true;
}

CMainMemoryDataFrameRowSlice::TSizeHandlePr CMainMemoryDataFrameRowSlice::read() {
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
    : m_StateIsBad{directory->bad()}, m_FirstRow{firstRow}, m_Capacity{state.size()},
      m_Directory{directory}, m_FileName{directory->name()}, m_Checksum{0} {

    if (m_StateIsBad == false) {
        m_FileName /= boost::filesystem::unique_path(
            "rows-" + std::to_string(firstRow) + "-%%%%-%%%%-%%%%-%%%%");
        this->writeToDisk(state);
    }
}

bool COnDiskDataFrameRowSlice::reserve(std::size_t numberColumns, std::size_t extraColumns) {
    // "Reserve" space at the end of each row for extraColumns extra columns.
    // Padding is inserted into the underlying vector which is skipped over
    // by the CRowConstIterator object.

    if (m_StateIsBad) {
        return false;
    }

    try {
        TFloatVec oldState(m_Capacity);
        if (this->readFromDisk(oldState) == false) {
            LOG_ERROR(<< "Failed to read from row " << m_FirstRow);
            m_StateIsBad = true;
            return false;
        }

        std::size_t numberRows{oldState.size() / numberColumns};

        if (sufficientDiskSpaceAvailable(m_Directory->name(),
                                         numberRows * extraColumns) == false) {
            LOG_INFO(<< "Insufficient disk space to reserve " << extraColumns << " extra columns");
            m_StateIsBad = true;
            return false;
        }

        std::size_t newNumberColumns{numberColumns + extraColumns};
        TFloatVec newState(numberRows * newNumberColumns, 0.0);
        for (auto i = oldState.begin(), j = newState.begin();
             i != oldState.end(); i += numberColumns, j += newNumberColumns) {
            std::copy(i, i + numberColumns, j);
        }

        this->writeToDisk(newState);

    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to reserve " << extraColumns
                  << " extra columns: caught '" << e.what() << "'");
        return false;
    }

    return true;
}

COnDiskDataFrameRowSlice::TSizeHandlePr COnDiskDataFrameRowSlice::read() {

    if (m_StateIsBad) {
        LOG_ERROR(<< "Bad row slice 'rows-" << m_FirstRow << "'");
        return {0, {boost::make_unique<CBadDataFrameRowSliceHandle>()}};
    }

    TFloatVec result;

    try {
        if (this->readFromDisk(result) == false) {
            LOG_ERROR(<< "Failed to read from row " << m_FirstRow);
            m_StateIsBad = true;
            return {0, {boost::make_unique<CBadDataFrameRowSliceHandle>()}};
        }

        if (checksum(result) != m_Checksum) {
            LOG_ERROR(<< "Corrupt from row " << m_FirstRow);
            m_StateIsBad = true;
            return {0, {boost::make_unique<CBadDataFrameRowSliceHandle>()}};
        }
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Caught '" << e.what() << "' while reading from row " << m_FirstRow);
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

void COnDiskDataFrameRowSlice::writeToDisk(const TFloatVec& state) {
    m_Capacity = state.size();
    m_Checksum = checksum(state);
    LOG_TRACE(<< "Checksum = " << m_Checksum);

    std::size_t bytes{sizeof(CFloatStorage) * state.size()};
    LOG_TRACE(<< "bytes = " << bytes);

    std::ofstream file{m_FileName.string(), std::ios_base::trunc | std::ios_base::binary};
    file.write(reinterpret_cast<const char*>(state.data()), bytes);
}

bool COnDiskDataFrameRowSlice::readFromDisk(TFloatVec& result) const {
    result.resize(m_Capacity);

    std::size_t bytes{sizeof(CFloatStorage) * m_Capacity};
    LOG_TRACE(<< "bytes = " << bytes);

    std::ifstream file{m_FileName.string(), std::ios_base::binary};
    file.read(reinterpret_cast<char*>(result.data()), bytes);
    return file.bad() == false;
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
        if (m_StateIsBad) {
            LOG_INFO(<< "Insufficient disk space to create data frame");
        }
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

bool COnDiskDataFrameRowSlice::CTemporaryDirectory::sufficientSpaceAvailable(std::size_t minimumSpace) const {
    return sufficientDiskSpaceAvailable(m_Name, minimumSpace);
}

bool COnDiskDataFrameRowSlice::CTemporaryDirectory::bad() const {
    return m_StateIsBad;
}
}
}
