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

#include <boost/filesystem.hpp>

#include <memory>
#include <vector>

namespace ml {
namespace core {
using TFloatVec = std::vector<CFloatStorage>;
using TFloatVecItr = TFloatVec::iterator;
using TInt32Vec = std::vector<std::int32_t>;
using TInt32VecCItr = TInt32Vec::const_iterator;

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
    CMainMemoryDataFrameRowSliceHandle(TFloatVec& rows, const TInt32Vec& docIds)
        : m_Rows{rows}, m_DocIds{docIds} {}
    virtual TImplPtr clone() const {
        return std::make_unique<CMainMemoryDataFrameRowSliceHandle>(m_Rows, m_DocIds);
    }
    virtual bool inMainMemory() const { return true; }
    virtual TFloatVec& rows() const { return m_Rows; }
    virtual const TInt32Vec& docIds() const { return m_DocIds; }
    virtual bool bad() const { return false; }

private:
    using TFloatVecRef = std::reference_wrapper<TFloatVec>;
    using TInt32VecCRef = std::reference_wrapper<const TInt32Vec>;

private:
    TFloatVecRef m_Rows;
    TInt32VecCRef m_DocIds;
};

//! \brief A handle for reading CRawDataFrameRowSlice objects.
//!
//! DESCRIPTION:\n
//! This stores a copy of values since these are created on-the-fly when
//! the slice is inflated.
class COnDiskDataFrameRowSliceHandle : public CDataFrameRowSliceHandleImpl {
public:
    COnDiskDataFrameRowSliceHandle(TFloatVec rows, TInt32Vec docIds)
        : m_Rows{std::move(rows)}, m_DocIds{std::move(docIds)} {}
    virtual TImplPtr clone() const {
        return std::make_unique<COnDiskDataFrameRowSliceHandle>(m_Rows, m_DocIds);
    }
    virtual TFloatVec& rows() const { return m_Rows; }
    virtual const TInt32Vec& docIds() const { return m_DocIds; }
    virtual bool bad() const { return false; }

private:
    mutable TFloatVec m_Rows;
    TInt32Vec m_DocIds;
};

//! \brief The implementation of a bad handle.
//!
//! DESCRIPTION:\n
//! This is used to signal that there is a problem accessing the slice.
class CBadDataFrameRowSliceHandle : public CDataFrameRowSliceHandleImpl {
public:
    virtual TImplPtr clone() const {
        return std::make_unique<CBadDataFrameRowSliceHandle>();
    }
    virtual TFloatVec& rows() const { return m_EmptyRows; }
    virtual const TInt32Vec& docIds() const { return m_EmptyDocIds; }
    virtual bool bad() const { return true; }

private:
    //! Stub for the rows.
    mutable TFloatVec m_EmptyRows;
    //! Stub for the row document ids.
    TInt32Vec m_EmptyDocIds;
};

//! Checksum \p vec.
template<typename T>
std::uint64_t computeChecksum(const std::vector<T>& vec) {
    return CHashing::murmurHash64(vec.data(), static_cast<int>(sizeof(T) * vec.size()), 0);
}

//! Checksum \p rows and \p docIds.
std::uint64_t computeChecksum(const TFloatVec& rows, const TInt32Vec& docIds) {
    return CHashing::hashCombine(computeChecksum(rows), computeChecksum(docIds));
}
}

//////// CDataFrameRowSliceHandle ////////

CDataFrameRowSliceHandle::CDataFrameRowSliceHandle(TImplPtr impl)
    : m_Impl{std::move(impl)} {
}

CDataFrameRowSliceHandle::CDataFrameRowSliceHandle(const CDataFrameRowSliceHandle& other)
    : m_Impl{other.m_Impl != nullptr ? other.m_Impl->clone() : nullptr} {
}

CDataFrameRowSliceHandle::CDataFrameRowSliceHandle(CDataFrameRowSliceHandle&& other)
    : m_Impl{std::move(other.m_Impl)} {
}

CDataFrameRowSliceHandle& CDataFrameRowSliceHandle::
operator=(const CDataFrameRowSliceHandle& other) {
    if (other.m_Impl != nullptr) {
        other.m_Impl->clone();
    }
    return *this;
}

CDataFrameRowSliceHandle& CDataFrameRowSliceHandle::operator=(CDataFrameRowSliceHandle&& other) {
    m_Impl = std::move(other.m_Impl);
    return *this;
}

std::size_t CDataFrameRowSliceHandle::size() const {
    return m_Impl->rows().size();
}

TFloatVecItr CDataFrameRowSliceHandle::beginRows() const {
    return m_Impl->rows().begin();
}

TFloatVecItr CDataFrameRowSliceHandle::endRows() const {
    return m_Impl->rows().end();
}

TInt32VecCItr CDataFrameRowSliceHandle::beginDocIds() const {
    return m_Impl->docIds().begin();
}

TInt32VecCItr CDataFrameRowSliceHandle::endDocIds() const {
    return m_Impl->docIds().end();
}

const TFloatVec& CDataFrameRowSliceHandle::rows() const {
    return m_Impl->rows();
}

const CDataFrameRowSliceHandle::TInt32Vec& CDataFrameRowSliceHandle::docIds() const {
    return m_Impl->docIds();
}

bool CDataFrameRowSliceHandle::bad() const {
    return m_Impl->bad();
}

//////// CMainMemoryDataFrameRowSlice ////////

CMainMemoryDataFrameRowSlice::CMainMemoryDataFrameRowSlice(std::size_t firstRow,
                                                           TFloatVec rows,
                                                           TInt32Vec docIds)
    : m_FirstRow{firstRow}, m_Rows{std::move(rows)}, m_DocIds(docIds) {
    LOG_TRACE(<< "slice size = " << m_Rows.size() << " capacity = " << m_Rows.capacity());
    m_Rows.shrink_to_fit();
    m_DocIds.shrink_to_fit();
}

bool CMainMemoryDataFrameRowSlice::reserve(std::size_t numberColumns, std::size_t extraColumns) {
    // "Reserve" space at the end of each row for extraColumns extra columns.
    // Padding is inserted into the underlying vector which is skipped over
    // by the CRowConstIterator object.

    std::size_t numberRows{m_Rows.size() / numberColumns};
    std::size_t newNumberColumns{numberColumns + extraColumns};
    try {
        TFloatVec state(m_Rows.size() + numberRows * extraColumns);
        for (auto i = m_Rows.begin(), j = state.begin(); i != m_Rows.end();
             i += numberColumns, j += newNumberColumns) {
            std::copy(i, i + numberColumns, j);
        }
        std::swap(state, m_Rows);
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to reserve " << extraColumns
                  << " extra columns: caught '" << e.what() << "'");
        return false;
    }
    return true;
}

CMainMemoryDataFrameRowSlice::TSizeHandlePr CMainMemoryDataFrameRowSlice::read() {
    return {m_FirstRow,
            {std::make_unique<CMainMemoryDataFrameRowSliceHandle>(m_Rows, m_DocIds)}};
}

void CMainMemoryDataFrameRowSlice::write(const TFloatVec&, const TInt32Vec&) {
    // Nothing to do.
}

std::size_t CMainMemoryDataFrameRowSlice::staticSize() const {
    return sizeof(*this);
}

std::size_t CMainMemoryDataFrameRowSlice::memoryUsage() const {
    return CMemory::dynamicSize(m_Rows) + CMemory::dynamicSize(m_DocIds);
}

std::uint64_t CMainMemoryDataFrameRowSlice::checksum() const {
    return computeChecksum(m_Rows, m_DocIds);
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
}

COnDiskDataFrameRowSlice::COnDiskDataFrameRowSlice(const TTemporaryDirectoryPtr& directory,
                                                   std::size_t firstRow,
                                                   TFloatVec rows,
                                                   TInt32Vec docIds)
    : m_StateIsBad{directory->bad()}, m_FirstRow{firstRow},
      m_RowsCapacity{rows.size()}, m_DocIdsCapacity{docIds.size()},
      m_Directory{directory}, m_FileName{directory->name()}, m_Checksum{0} {

    if (m_StateIsBad == false) {
        m_FileName /= boost::filesystem::unique_path(
            "rows-" + std::to_string(firstRow) + "-%%%%-%%%%-%%%%-%%%%");
        this->writeToDisk(rows, docIds);
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
        TFloatVec oldRows(m_RowsCapacity);
        TInt32Vec docIds(m_DocIdsCapacity);
        if (this->readFromDisk(oldRows, docIds) == false) {
            LOG_ERROR(<< "Failed to read from row " << m_FirstRow);
            m_StateIsBad = true;
            return false;
        }

        std::size_t numberRows{oldRows.size() / numberColumns};

        if (sufficientDiskSpaceAvailable(m_Directory->name(),
                                         numberRows * extraColumns) == false) {
            LOG_INFO(<< "Insufficient disk space to reserve " << extraColumns << " extra columns");
            m_StateIsBad = true;
            return false;
        }

        std::size_t newNumberColumns{numberColumns + extraColumns};
        TFloatVec newRows(numberRows * newNumberColumns, 0.0);
        for (auto i = oldRows.begin(), j = newRows.begin(); i != oldRows.end();
             i += numberColumns, j += newNumberColumns) {
            std::copy(i, i + numberColumns, j);
        }

        this->writeToDisk(newRows, docIds);

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
        return {0, {std::make_unique<CBadDataFrameRowSliceHandle>()}};
    }

    TFloatVec rows;
    TInt32Vec docIds;

    try {
        if (this->readFromDisk(rows, docIds) == false) {
            LOG_ERROR(<< "Failed to read from row " << m_FirstRow);
            m_StateIsBad = true;
            return {0, {std::make_unique<CBadDataFrameRowSliceHandle>()}};
        }

        if (computeChecksum(rows, docIds) != m_Checksum) {
            LOG_ERROR(<< "Corrupt from row " << m_FirstRow);
            m_StateIsBad = true;
            return {0, {std::make_unique<CBadDataFrameRowSliceHandle>()}};
        }
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Caught '" << e.what() << "' while reading from row " << m_FirstRow);
        m_StateIsBad = true;
        return {0, {std::make_unique<CBadDataFrameRowSliceHandle>()}};
    }

    return {m_FirstRow,
            {std::make_unique<COnDiskDataFrameRowSliceHandle>(std::move(rows),
                                                              std::move(docIds))}};
}

void COnDiskDataFrameRowSlice::write(const TFloatVec& rows, const TInt32Vec& docIds) {
    if (m_StateIsBad == false) {
        this->writeToDisk(rows, docIds);
    }
}

std::size_t COnDiskDataFrameRowSlice::staticSize() const {
    return sizeof(*this);
}

std::size_t COnDiskDataFrameRowSlice::memoryUsage() const {
    return CMemory::dynamicSize(m_Directory) + CMemory::dynamicSize(m_FileName.string());
}

void COnDiskDataFrameRowSlice::writeToDisk(const TFloatVec& rows, const TInt32Vec& docIds) {
    m_RowsCapacity = rows.size();
    m_DocIdsCapacity = docIds.size();
    m_Checksum = CHashing::hashCombine(computeChecksum(rows), computeChecksum(docIds));
    LOG_TRACE(<< "Checksum = " << m_Checksum);

    std::size_t rowBytes{sizeof(CFloatStorage) * rows.size()};
    std::size_t docIdBytes{sizeof(std::int32_t) * docIds.size()};
    LOG_TRACE(<< "row bytes = " << rowBytes);
    LOG_TRACE(<< "doc ids bytes = " << docIdBytes);

    std::ofstream file{m_FileName.string(), std::ios_base::trunc | std::ios_base::binary};
    file.write(reinterpret_cast<const char*>(rows.data()), rowBytes);
    file.write(reinterpret_cast<const char*>(docIds.data()), docIdBytes);
}

std::uint64_t COnDiskDataFrameRowSlice::checksum() const {
    return m_Checksum;
}

bool COnDiskDataFrameRowSlice::readFromDisk(TFloatVec& rows, TInt32Vec& docIds) const {
    rows.resize(m_RowsCapacity);
    docIds.resize(m_DocIdsCapacity);

    std::size_t rowsBytes{sizeof(CFloatStorage) * m_RowsCapacity};
    std::size_t docIdsBytes{sizeof(std::int32_t) * m_DocIdsCapacity};
    LOG_TRACE(<< "row bytes = " << rowBytes);
    LOG_TRACE(<< "doc ids bytes = " << docIdBytes);

    std::ifstream file{m_FileName.string(), std::ios_base::binary};
    file.read(reinterpret_cast<char*>(rows.data()), rowsBytes);
    file.read(reinterpret_cast<char*>(docIds.data()), docIdsBytes);
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
