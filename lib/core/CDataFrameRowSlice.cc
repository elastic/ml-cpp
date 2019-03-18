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
class CMainMemoryDataFrameRowSliceHandle final : public CDataFrameRowSliceHandleImpl {
public:
    CMainMemoryDataFrameRowSliceHandle(std::size_t firstRow, TFloatVec& rows, const TInt32Vec& docHashes)
        : m_FirstRow{firstRow}, m_Rows{rows}, m_DocHashes{docHashes} {}
    TImplPtr clone() const override {
        return std::make_unique<CMainMemoryDataFrameRowSliceHandle>(m_FirstRow, m_Rows,
                                                                    m_DocHashes);
    }
    std::size_t indexOfFirstRow() const override { return m_FirstRow; }
    TFloatVec& rows() const override { return m_Rows; }
    const TInt32Vec& docHashes() const override { return m_DocHashes; }
    bool bad() const override { return false; }

private:
    using TFloatVecRef = std::reference_wrapper<TFloatVec>;
    using TInt32VecCRef = std::reference_wrapper<const TInt32Vec>;

private:
    std::size_t m_FirstRow;
    TFloatVecRef m_Rows;
    TInt32VecCRef m_DocHashes;
};

//! \brief A handle for reading CRawDataFrameRowSlice objects.
//!
//! DESCRIPTION:\n
//! This stores a copy of values since these are created on-the-fly when
//! the slice is inflated.
class COnDiskDataFrameRowSliceHandle final : public CDataFrameRowSliceHandleImpl {
public:
    COnDiskDataFrameRowSliceHandle(std::size_t firstRow, TFloatVec rows, TInt32Vec docHashes)
        : m_FirstRow{firstRow}, m_Rows{std::move(rows)}, m_DocHashes{std::move(docHashes)} {}
    TImplPtr clone() const override {
        return std::make_unique<COnDiskDataFrameRowSliceHandle>(m_FirstRow, m_Rows, m_DocHashes);
    }
    std::size_t indexOfFirstRow() const override { return m_FirstRow; }
    TFloatVec& rows() const override { return m_Rows; }
    const TInt32Vec& docHashes() const override { return m_DocHashes; }
    bool bad() const override { return false; }

private:
    std::size_t m_FirstRow;
    mutable TFloatVec m_Rows;
    TInt32Vec m_DocHashes;
};

//! \brief The implementation of a bad handle.
//!
//! DESCRIPTION:\n
//! This is used to signal that there is a problem accessing the slice.
class CBadDataFrameRowSliceHandle final : public CDataFrameRowSliceHandleImpl {
public:
    TImplPtr clone() const override {
        return std::make_unique<CBadDataFrameRowSliceHandle>();
    }
    std::size_t indexOfFirstRow() const override { return 0; }
    TFloatVec& rows() const override { return m_EmptyRows; }
    const TInt32Vec& docHashes() const override { return m_EmptyDocHashes; }
    bool bad() const override { return true; }

private:
    //! Stub for the rows.
    mutable TFloatVec m_EmptyRows;
    //! Stub for the row document ids.
    TInt32Vec m_EmptyDocHashes;
};

//! Checksum \p vec.
template<typename T>
std::uint64_t computeChecksum(const std::vector<T>& vec) {
    return CHashing::murmurHash64(vec.data(), static_cast<int>(sizeof(T) * vec.size()), 0);
}

//! Checksum \p rows and \p docHashes.
std::uint64_t computeChecksum(const TFloatVec& rows, const TInt32Vec& docHashes) {
    return CHashing::hashCombine(computeChecksum(rows), computeChecksum(docHashes));
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

std::size_t CDataFrameRowSliceHandle::indexOfFirstRow() const {
    return m_Impl->indexOfFirstRow();
}

TFloatVecItr CDataFrameRowSliceHandle::beginRows() const {
    return m_Impl->rows().begin();
}

TFloatVecItr CDataFrameRowSliceHandle::endRows() const {
    return m_Impl->rows().end();
}

TInt32VecCItr CDataFrameRowSliceHandle::beginDocHashes() const {
    return m_Impl->docHashes().begin();
}

TInt32VecCItr CDataFrameRowSliceHandle::endDocHashes() const {
    return m_Impl->docHashes().end();
}

const TFloatVec& CDataFrameRowSliceHandle::rows() const {
    return m_Impl->rows();
}

const CDataFrameRowSliceHandle::TInt32Vec& CDataFrameRowSliceHandle::docHashes() const {
    return m_Impl->docHashes();
}

bool CDataFrameRowSliceHandle::bad() const {
    return m_Impl->bad();
}

//////// CMainMemoryDataFrameRowSlice ////////

CMainMemoryDataFrameRowSlice::CMainMemoryDataFrameRowSlice(std::size_t firstRow,
                                                           TFloatVec rows,
                                                           TInt32Vec docHashes)
    : m_FirstRow{firstRow}, m_Rows{std::move(rows)}, m_DocHashes(docHashes) {
    LOG_TRACE(<< "slice size = " << m_Rows.size() << " capacity = " << m_Rows.capacity());
    m_Rows.shrink_to_fit();
    m_DocHashes.shrink_to_fit();
}

void CMainMemoryDataFrameRowSlice::reserve(std::size_t numberColumns, std::size_t extraColumns) {
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
        HANDLE_FATAL(<< "Environment error: failed to reserve " << extraColumns << " extra columns: caught '"
                     << e.what() << "'. The process is likely out of memory.");
    }
}

std::size_t CMainMemoryDataFrameRowSlice::indexOfFirstRow() const {
    return m_FirstRow;
}

std::size_t CMainMemoryDataFrameRowSlice::indexOfLastRow(std::size_t rowCapacity) const {
    return m_FirstRow + m_Rows.size() / rowCapacity - 1;
}

CDataFrameRowSliceHandle CMainMemoryDataFrameRowSlice::read() {
    return {std::make_unique<CMainMemoryDataFrameRowSliceHandle>(m_FirstRow, m_Rows, m_DocHashes)};
}

void CMainMemoryDataFrameRowSlice::write(const TFloatVec&, const TInt32Vec&) {
    // Nothing to do.
}

std::size_t CMainMemoryDataFrameRowSlice::staticSize() const {
    return sizeof(*this);
}

std::size_t CMainMemoryDataFrameRowSlice::memoryUsage() const {
    return CMemory::dynamicSize(m_Rows) + CMemory::dynamicSize(m_DocHashes);
}

std::uint64_t CMainMemoryDataFrameRowSlice::checksum() const {
    return computeChecksum(m_Rows, m_DocHashes);
}

//////// CTemporaryDirectory ////////

namespace {

//! Check if there is \p minimumSpace disk space available.
void sufficientDiskSpaceAvailable(const boost::filesystem::path& path, std::size_t minimumSpace) {
    boost::system::error_code errorCode;
    auto spaceInfo = boost::filesystem::space(path, errorCode);
    if (errorCode) {
        HANDLE_FATAL(<< "Environment error: failed to retrieve disk information for '"
                     << path << "' error '" << errorCode.message() << "'.");
    }
    if (spaceInfo.available < minimumSpace) {
        HANDLE_FATAL(<< "Environment error: insufficient disk space have '"
                     << spaceInfo.available << "' and need '" << minimumSpace << "'.");
    }
}
}

CTemporaryDirectory::CTemporaryDirectory(const std::string& name, std::size_t minimumSpace)
    : m_Name{name} {
    m_Name /= boost::filesystem::unique_path("dataframe-%%%%-%%%%-%%%%-%%%%");
    LOG_TRACE(<< "Trying to create directory '" << m_Name << "'");

    boost::system::error_code errorCode;
    boost::filesystem::create_directories(m_Name, errorCode);
    if (errorCode) {
        HANDLE_FATAL(<< "Environment error: failed to create temporary directory from: '"
                     << m_Name << "' error '" << errorCode.message() << "'");
    }

    sufficientDiskSpaceAvailable(m_Name, minimumSpace);

    LOG_TRACE(<< "Created '" << m_Name << "'");
}

CTemporaryDirectory::~CTemporaryDirectory() {
    this->removeAll();
}

std::string CTemporaryDirectory::name() const {
    return m_Name.string();
}

void CTemporaryDirectory::removeAll() {
    boost::system::error_code errorCode;
    boost::filesystem::remove_all(m_Name, errorCode);
    if (errorCode) {
        LOG_WARN(<< "Failed to cleanup temporary data from: '" << m_Name
                 << "' error '" << errorCode.message() << "'.");
    }
}

//////// COnDiskDataFrameRowSlice ////////

COnDiskDataFrameRowSlice::COnDiskDataFrameRowSlice(const TTemporaryDirectoryPtr& directory,
                                                   std::size_t firstRow,
                                                   TFloatVec rows,
                                                   TInt32Vec docHashes)
    : m_FirstRow{firstRow}, m_RowsCapacity{rows.size()},
      m_DocHashesCapacity{docHashes.size()}, m_Directory{directory},
      m_FileName{directory->name()}, m_Checksum{0} {

    m_FileName /= boost::filesystem::unique_path(
        "rows-" + std::to_string(firstRow) + "-%%%%-%%%%-%%%%-%%%%");
    this->writeToDisk(rows, docHashes);
}

void COnDiskDataFrameRowSlice::reserve(std::size_t numberColumns, std::size_t extraColumns) {
    // "Reserve" space at the end of each row for extraColumns extra columns.
    // Padding is inserted into the underlying vector which is skipped over
    // by the CRowConstIterator object.

    try {
        TFloatVec oldRows(m_RowsCapacity);
        TInt32Vec docHashes(m_DocHashesCapacity);
        if (this->readFromDisk(oldRows, docHashes) == false) {
            HANDLE_FATAL(<< "Environment error: failed to read from row "
                         << m_FirstRow << ".");
        }

        std::size_t numberRows{oldRows.size() / numberColumns};

        sufficientDiskSpaceAvailable(m_Directory->name(), numberRows * extraColumns);

        std::size_t newNumberColumns{numberColumns + extraColumns};
        TFloatVec newRows(numberRows * newNumberColumns, 0.0);
        for (auto i = oldRows.begin(), j = newRows.begin(); i != oldRows.end();
             i += numberColumns, j += newNumberColumns) {
            std::copy(i, i + numberColumns, j);
        }

        this->writeToDisk(newRows, docHashes);

    } catch (const std::exception& e) {
        HANDLE_FATAL(<< "Environment error: failed to reserve " << extraColumns
                     << " extra columns: caught '" << e.what() << "'.");
    }
}

std::size_t COnDiskDataFrameRowSlice::indexOfFirstRow() const {
    return m_FirstRow;
}

std::size_t COnDiskDataFrameRowSlice::indexOfLastRow(std::size_t rowCapacity) const {
    return m_FirstRow + m_RowsCapacity / rowCapacity - 1;
}

CDataFrameRowSliceHandle COnDiskDataFrameRowSlice::read() {
    LOG_TRACE(<< "Reading slice starting at row " << m_FirstRow);

    TFloatVec rows;
    TInt32Vec docHashes;

    try {
        if (this->readFromDisk(rows, docHashes) == false) {
            HANDLE_FATAL(<< "Environment error: failed to read from row "
                         << m_FirstRow << ".");
        }

        if (computeChecksum(rows, docHashes) != m_Checksum) {
            HANDLE_FATAL(<< "Environment error: corrupt from row " << m_FirstRow << ".");
        }
    } catch (const std::exception& e) {
        HANDLE_FATAL(<< "Environment error: caught '" << e.what()
                     << "' while reading from row " << m_FirstRow << ".");
    }

    return {std::make_unique<COnDiskDataFrameRowSliceHandle>(
        m_FirstRow, std::move(rows), std::move(docHashes))};
}

void COnDiskDataFrameRowSlice::write(const TFloatVec& rows, const TInt32Vec& docHashes) {
    this->writeToDisk(rows, docHashes);
}

std::size_t COnDiskDataFrameRowSlice::staticSize() const {
    return sizeof(*this);
}

std::size_t COnDiskDataFrameRowSlice::memoryUsage() const {
    return CMemory::dynamicSize(m_Directory) + CMemory::dynamicSize(m_FileName.string());
}

void COnDiskDataFrameRowSlice::writeToDisk(const TFloatVec& rows, const TInt32Vec& docHashes) {
    LOG_TRACE(<< "Writing slice starting at row " << m_FirstRow);

    m_RowsCapacity = rows.size();
    m_DocHashesCapacity = docHashes.size();
    m_Checksum = computeChecksum(rows, docHashes);
    LOG_TRACE(<< "Checksum = " << m_Checksum);

    std::size_t rowsBytes{sizeof(CFloatStorage) * rows.size()};
    std::size_t docHashesBytes{sizeof(std::int32_t) * docHashes.size()};
    LOG_TRACE(<< "rows bytes = " << rowsBytes);
    LOG_TRACE(<< "doc hashes bytes = " << docHashesBytes);

    std::ofstream file{m_FileName.string(), std::ios_base::trunc | std::ios_base::binary};
    file.write(reinterpret_cast<const char*>(rows.data()), rowsBytes);
    file.write(reinterpret_cast<const char*>(docHashes.data()), docHashesBytes);
}

std::uint64_t COnDiskDataFrameRowSlice::checksum() const {
    return m_Checksum;
}

bool COnDiskDataFrameRowSlice::readFromDisk(TFloatVec& rows, TInt32Vec& docHashes) const {
    rows.resize(m_RowsCapacity);
    docHashes.resize(m_DocHashesCapacity);

    std::size_t rowsBytes{sizeof(CFloatStorage) * m_RowsCapacity};
    std::size_t docHashesBytes{sizeof(std::int32_t) * m_DocHashesCapacity};
    LOG_TRACE(<< "rows bytes = " << rowsBytes);
    LOG_TRACE(<< "doc hashes bytes = " << docHashesBytes);

    std::ifstream file{m_FileName.string(), std::ios_base::binary};
    file.read(reinterpret_cast<char*>(rows.data()), rowsBytes);
    file.read(reinterpret_cast<char*>(docHashes.data()), docHashesBytes);
    return file.bad() == false;
}
}
}
