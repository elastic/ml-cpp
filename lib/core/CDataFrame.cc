/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDataFrame.h>

#include <core/CDataFrameRowSlice.h>
#include <core/CHashing.h>
#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/Concurrency.h>

#include <algorithm>
#include <future>
#include <memory>

namespace ml {
namespace core {
namespace data_frame_detail {

CRowRef::CRowRef(std::size_t index, TFloatVecItr beginColumns, TFloatVecItr endColumns, std::int32_t docHash)
    : m_Index{index}, m_BeginColumns{beginColumns}, m_EndColumns{endColumns}, m_DocHash{docHash} {
}

CFloatStorage CRowRef::operator[](std::size_t i) const {
    return *(m_BeginColumns + i);
}

std::size_t CRowRef::index() const {
    return m_Index;
}

std::size_t CRowRef::numberColumns() const {
    return std::distance(m_BeginColumns, m_EndColumns);
}

void CRowRef::writeColumn(std::size_t column, double value) const {
    m_BeginColumns[column] = value;
}

CFloatStorage* CRowRef::data() const {
    return &(*m_BeginColumns);
}

std::int32_t CRowRef::docHash() const {
    return m_DocHash;
}

CRowIterator::CRowIterator(std::size_t numberColumns,
                           std::size_t rowCapacity,
                           std::size_t index,
                           TFloatVecItr rowItr,
                           TInt32VecCItr docHashItr)
    : m_NumberColumns{numberColumns},
      m_RowCapacity{rowCapacity}, m_Index{index}, m_RowItr{rowItr}, m_DocHashItr{docHashItr} {
}

bool CRowIterator::operator==(const CRowIterator& rhs) const {
    return m_RowItr == rhs.m_RowItr && m_DocHashItr == rhs.m_DocHashItr;
}

bool CRowIterator::operator!=(const CRowIterator& rhs) const {
    return m_RowItr != rhs.m_RowItr || m_DocHashItr != rhs.m_DocHashItr;
}

CRowRef CRowIterator::operator*() const {
    return CRowRef{m_Index, m_RowItr, m_RowItr + m_NumberColumns, *m_DocHashItr};
}

CRowPtr CRowIterator::operator->() const {
    return CRowPtr{m_Index, m_RowItr, m_RowItr + m_NumberColumns, *m_DocHashItr};
}

CRowIterator& CRowIterator::operator++() {
    ++m_Index;
    m_RowItr += m_RowCapacity;
    ++m_DocHashItr;
    return *this;
}

CRowIterator CRowIterator::operator++(int) {
    CRowIterator result{*this};
    ++m_Index;
    m_RowItr += m_RowCapacity;
    ++m_DocHashItr;
    return result;
}
}
using namespace data_frame_detail;

namespace {
//! Compute the default slice capacity in rows.
std::size_t computeSliceCapacity(std::size_t numberColumns) {
    // TODO This probably needs some careful tuning, which I haven't performed
    // yet, and probably needs to be different for different storage strategies.
    return std::max(1048576 / sizeof(CFloatStorage) / numberColumns, std::size_t(100));
}
}

CDataFrame::CDataFrame(bool inMainMemory,
                       std::size_t numberColumns,
                       std::size_t sliceCapacityInRows,
                       EReadWriteToStorage readAndWriteToStoreSyncStrategy,
                       const TWriteSliceToStoreFunc& writeSliceToStore)
    : m_InMainMemory{inMainMemory}, m_NumberColumns{numberColumns},
      m_RowCapacity{numberColumns}, m_SliceCapacityInRows{sliceCapacityInRows},
      m_ReadAndWriteToStoreSyncStrategy{readAndWriteToStoreSyncStrategy}, m_WriteSliceToStore{writeSliceToStore} {
}

CDataFrame::CDataFrame(bool inMainMemory,
                       std::size_t numberColumns,
                       EReadWriteToStorage readAndWriteToStoreSyncStrategy,
                       const TWriteSliceToStoreFunc& writeSliceToStore)
    : CDataFrame{inMainMemory, numberColumns, computeSliceCapacity(numberColumns),
                 readAndWriteToStoreSyncStrategy, writeSliceToStore} {
}

CDataFrame::~CDataFrame() = default;

bool CDataFrame::inMainMemory() const {
    return m_InMainMemory;
}

std::size_t CDataFrame::numberRows() const {
    return m_NumberRows;
}

std::size_t CDataFrame::numberColumns() const {
    return m_NumberColumns;
}

void CDataFrame::reserve(std::size_t numberThreads, std::size_t rowCapacity) {
    if (m_RowCapacity >= rowCapacity) {
        return;
    }

    m_RowCapacity = rowCapacity;

    parallel_for_each(numberThreads, m_Slices.begin(), m_Slices.end(), [this](TRowSlicePtr& slice) {
        slice->reserve(m_NumberColumns, m_RowCapacity - m_NumberColumns);
    });
}

void CDataFrame::resizeColumns(std::size_t numberThreads, std::size_t numberColumns) {
    this->reserve(numberThreads, numberColumns);
    m_NumberColumns = numberColumns;
}

CDataFrame::TRowFuncVecBoolPr CDataFrame::readRows(std::size_t numberThreads,
                                                   std::size_t beginRows,
                                                   std::size_t endRows,
                                                   TRowFunc reader) const {

    beginRows = std::min(beginRows, m_NumberRows);
    endRows = std::min(endRows, m_NumberRows);

    if (beginRows >= endRows) {
        return {{std::move(reader)}, true};
    }

    return numberThreads > 1
               ? this->parallelApplyToAllRows(numberThreads, beginRows, endRows,
                                              std::move(reader), false)
               : this->sequentialApplyToAllRows(beginRows, endRows, std::move(reader), false);
}

bool CDataFrame::writeColumns(std::size_t numberThreads,
                              std::size_t beginRows,
                              std::size_t endRows,
                              TRowFunc writer) {

    beginRows = std::min(beginRows, m_NumberRows);
    endRows = std::min(endRows, m_NumberRows);

    if (beginRows >= endRows) {
        return true;
    }

    bool successful;
    std::tie(std::ignore, successful) =
        numberThreads > 1
            ? this->parallelApplyToAllRows(numberThreads, beginRows, endRows,
                                           std::move(writer), true)
            : this->sequentialApplyToAllRows(beginRows, endRows, std::move(writer), true);

    return successful;
}

void CDataFrame::writeRow(const TWriteFunc& writeRow) {
    if (m_Writer == nullptr) {
        m_Writer = std::make_unique<CDataFrameRowSliceWriter>(
            m_NumberRows, m_RowCapacity, m_SliceCapacityInRows,
            m_ReadAndWriteToStoreSyncStrategy, m_WriteSliceToStore);
    }
    (*m_Writer)(writeRow);
}

void CDataFrame::finishWritingRows() {
    // Get any slices which have been written, append and clear the writer.

    if (m_Writer != nullptr) {
        TRowSlicePtrVec slices;
        std::tie(m_NumberRows, slices) = m_Writer->finishWritingRows();
        m_Writer.reset();

        m_Slices.reserve(m_Slices.size() + slices.size());
        for (auto& slice : slices) {
            m_Slices.push_back(std::move(slice));
        }
        LOG_TRACE(<< "# slices = " << m_Slices.size());
    }
}

std::size_t CDataFrame::memoryUsage() const {
    return CMemory::dynamicSize(m_Slices) + CMemory::dynamicSize(m_Writer);
}

std::uint64_t CDataFrame::checksum() const {
    std::vector<std::uint64_t> checksums(m_Slices.size(), 0);
    parallel_for_each(0, m_Slices.size(), [&](std::size_t index) {
        checksums[index] = m_Slices[index]->checksum();
    });

    std::uint64_t result{0};
    for (auto checksum : checksums) {
        result = CHashing::hashCombine(result, checksum);
    }
    return result;
}

std::size_t CDataFrame::estimateMemoryUsage(bool inMainMemory,
                                            std::size_t numberRows,
                                            std::size_t numberColumns) {
    return inMainMemory ? numberRows * numberColumns * sizeof(float) : 0;
}

CDataFrame::TRowFuncVecBoolPr
CDataFrame::parallelApplyToAllRows(std::size_t numberThreads,
                                   std::size_t beginRows,
                                   std::size_t endRows,
                                   TRowFunc func,
                                   bool commitResult) const {

    // If we're reading in parallel then we don't want to interleave
    // reads from storage and applying the function because we're
    // already fully balancing our work across the slices.

    std::atomic_bool successful{true};
    CDataFrameRowSliceHandle readSlice;

    auto results = parallel_for_each(
        numberThreads, this->beginSlices(beginRows), this->endSlices(endRows),
        bindRetrievableState(
            [=, &successful](TRowFunc& func_, const TRowSlicePtr& slice) mutable {
                if (successful.load() == false) {
                    return;
                }
                readSlice = slice->read();
                if (readSlice.bad()) {
                    successful.store(false);
                    return;
                }
                this->applyToRowsOfOneSlice(func_, beginRows, endRows, readSlice);
                if (commitResult) {
                    slice->write(readSlice.rows(), readSlice.docHashes());
                }
            },
            std::move(func)));

    TRowFuncVec functions;
    functions.reserve(results.size());
    for (auto& result : results) {
        functions.emplace_back(std::move(result.s_FunctionState));
    }

    return {std::move(functions), successful.load()};
}

CDataFrame::TRowFuncVecBoolPr CDataFrame::sequentialApplyToAllRows(std::size_t beginRows,
                                                                   std::size_t endRows,
                                                                   TRowFunc func,
                                                                   bool commitResult) const {

    CDataFrameRowSliceHandle readSlice;

    switch (m_ReadAndWriteToStoreSyncStrategy) {
    case CDataFrame::EReadWriteToStorage::E_Async: {
        // The slices get read from storage on the thread executing this
        // function each slice is then concurrently read by the callback
        // on a worker thread.

        std::future<void> backgroundApply;

        // We need to wait and this isn't guaranteed by the future destructor.
        CWaitIfValidWhenExitingScope<void> waitFor(backgroundApply);

        for (auto slice = this->beginSlices(beginRows), endSlices = this->endSlices(endRows);
             slice != endSlices; ++slice) {

            readSlice = (*slice)->read();
            if (readSlice.bad()) {
                return {{std::move(func)}, false};
            }

            // We wait here so at most one slice is copied into memory.
            wait_for_valid(backgroundApply);

            backgroundApply = async(
                defaultAsyncExecutor(),
                [ =, &func, readSlice_ = std::move(readSlice) ] {
                    this->applyToRowsOfOneSlice(func, beginRows, endRows, readSlice_);
                    if (commitResult) {
                        (*slice)->write(readSlice_.rows(), readSlice_.docHashes());
                    }
                });
        }
        break;
    }
    case CDataFrame::EReadWriteToStorage::E_Sync:
        for (auto slice = this->beginSlices(beginRows), endSlices = this->endSlices(endRows);
             slice != endSlices; ++slice) {

            readSlice = (*slice)->read();
            if (readSlice.bad()) {
                return {{std::move(func)}, false};
            }
            this->applyToRowsOfOneSlice(func, beginRows, endRows, readSlice);
            if (commitResult) {
                (*slice)->write(readSlice.rows(), readSlice.docHashes());
            }
        }
        break;
    }

    return {{std::move(func)}, true};
}

void CDataFrame::applyToRowsOfOneSlice(TRowFunc& func,
                                       std::size_t beginRows,
                                       std::size_t endRows,
                                       const CDataFrameRowSliceHandle& slice) const {
    std::size_t firstRowToRead{std::max(slice.indexOfFirstRow(), beginRows)};
    std::size_t numberRowsInSlice{slice.size() / m_RowCapacity};
    std::size_t endRowsToRead{std::min(slice.indexOfFirstRow() + numberRowsInSlice, endRows)};

    LOG_TRACE(<< "Applying function to rows [" << firstRowToRead << ","
              << endRowsToRead << ")");

    std::size_t offsetOfFirstRowToRead{firstRowToRead - slice.indexOfFirstRow()};
    std::size_t offsetOfEndRowsToRead{endRowsToRead - slice.indexOfFirstRow()};

    std::size_t beginRowData{offsetOfFirstRowToRead * m_RowCapacity};
    std::size_t endRowData{offsetOfEndRowsToRead * m_RowCapacity};

    func(CRowIterator{m_NumberColumns, m_RowCapacity, firstRowToRead,
                      slice.beginRows() + beginRowData,
                      slice.beginDocHashes() + offsetOfFirstRowToRead},
         CRowIterator{m_NumberColumns, m_RowCapacity, endRowsToRead,
                      slice.beginRows() + endRowData,
                      slice.beginDocHashes() + offsetOfEndRowsToRead});
}

CDataFrame::TRowSlicePtrVecCItr CDataFrame::beginSlices(std::size_t beginRows) const {
    return std::upper_bound(m_Slices.begin(), m_Slices.end(), beginRows,
                            [](std::size_t row, const TRowSlicePtr& slice) {
                                return row < slice->indexOfFirstRow();
                            }) -
           1;
}

CDataFrame::TRowSlicePtrVecCItr CDataFrame::endSlices(std::size_t endRows) const {
    return std::upper_bound(m_Slices.begin(), m_Slices.end(), endRows,
                            [](std::size_t row, const TRowSlicePtr& slice) {
                                return row < slice->indexOfFirstRow();
                            });
}

CDataFrame::CDataFrameRowSliceWriter::CDataFrameRowSliceWriter(
    std::size_t numberRows,
    std::size_t rowCapacity,
    std::size_t sliceCapacityInRows,
    EReadWriteToStorage writeToStoreSyncStrategy,
    TWriteSliceToStoreFunc writeSliceToStore)
    : m_NumberRows{numberRows}, m_RowCapacity{rowCapacity}, m_SliceCapacityInRows{sliceCapacityInRows},
      m_WriteToStoreSyncStrategy{writeToStoreSyncStrategy}, m_WriteSliceToStore{writeSliceToStore} {
    m_RowsOfSliceBeingWritten.reserve(m_SliceCapacityInRows * m_RowCapacity);
    m_DocHashesOfSliceBeingWritten.reserve(m_SliceCapacityInRows);
}

void CDataFrame::CDataFrameRowSliceWriter::operator()(const TWriteFunc& writeRow) {
    // Write the next row at the end of the current slice being written
    // and if the slice is full pass to the thread storing slices.

    std::size_t end{m_RowsOfSliceBeingWritten.size()};

    m_RowsOfSliceBeingWritten.resize(end + m_RowCapacity);
    m_DocHashesOfSliceBeingWritten.emplace_back();
    writeRow(m_RowsOfSliceBeingWritten.begin() + end,
             m_DocHashesOfSliceBeingWritten.back());
    ++m_NumberRows;

    if (m_DocHashesOfSliceBeingWritten.size() == m_SliceCapacityInRows) {
        std::size_t firstRow{m_NumberRows - m_SliceCapacityInRows};
        LOG_TRACE(<< "Storing slice [" << firstRow << "," << m_NumberRows << ")");

        switch (m_WriteToStoreSyncStrategy) {
        case EReadWriteToStorage::E_Async: {
            if (m_SliceWrittenAsyncToStore.valid()) {
                m_SlicesWrittenToStore.push_back(m_SliceWrittenAsyncToStore.get());
            }
            m_SliceWrittenAsyncToStore =
                async(defaultAsyncExecutor(), m_WriteSliceToStore, firstRow,
                      std::move(m_RowsOfSliceBeingWritten),
                      std::move(m_DocHashesOfSliceBeingWritten));
            break;
        }
        case EReadWriteToStorage::E_Sync:
            m_SlicesWrittenToStore.push_back(
                m_WriteSliceToStore(firstRow, std::move(m_RowsOfSliceBeingWritten),
                                    std::move(m_DocHashesOfSliceBeingWritten)));
            break;
        }
        m_RowsOfSliceBeingWritten.clear();
        m_DocHashesOfSliceBeingWritten.clear();
        m_RowsOfSliceBeingWritten.reserve(m_SliceCapacityInRows * m_RowCapacity);
        m_DocHashesOfSliceBeingWritten.reserve(m_SliceCapacityInRows);
    }
}

CDataFrame::TSizeDataFrameRowSlicePtrVecPr
CDataFrame::CDataFrameRowSliceWriter::finishWritingRows() {
    if (m_SliceWrittenAsyncToStore.valid()) {
        m_SlicesWrittenToStore.push_back(m_SliceWrittenAsyncToStore.get());
    }

    if (m_DocHashesOfSliceBeingWritten.size() > 0) {
        std::size_t firstRow{m_NumberRows - m_RowsOfSliceBeingWritten.size() / m_RowCapacity};
        LOG_TRACE(<< "Last slice [" << std::to_string(firstRow) << ","
                  << std::to_string(m_NumberRows) + ")");
        m_SlicesWrittenToStore.push_back(
            m_WriteSliceToStore(firstRow, std::move(m_RowsOfSliceBeingWritten),
                                std::move(m_DocHashesOfSliceBeingWritten)));
    }

    return {m_NumberRows, std::move(m_SlicesWrittenToStore)};
}

std::pair<std::unique_ptr<CDataFrame>, std::shared_ptr<CTemporaryDirectory>>
makeMainStorageDataFrame(std::size_t numberColumns,
                         boost::optional<std::size_t> sliceCapacity,
                         CDataFrame::EReadWriteToStorage readWriteToStoreSyncStrategy) {
    auto writer = [](std::size_t firstRow, TFloatVec rows, TInt32Vec docHashes) {
        return std::make_unique<CMainMemoryDataFrameRowSlice>(
            firstRow, std::move(rows), std::move(docHashes));
    };

    if (sliceCapacity != boost::none) {
        return {std::make_unique<CDataFrame>(true, numberColumns, *sliceCapacity,
                                             readWriteToStoreSyncStrategy, writer),
                nullptr};
    }

    return {std::make_unique<CDataFrame>(true, numberColumns,
                                         readWriteToStoreSyncStrategy, writer),
            nullptr};
}

std::pair<std::unique_ptr<CDataFrame>, std::shared_ptr<CTemporaryDirectory>>
makeDiskStorageDataFrame(const std::string& rootDirectory,
                         std::size_t numberColumns,
                         std::size_t numberRows,
                         boost::optional<std::size_t> sliceCapacity,
                         CDataFrame::EReadWriteToStorage readWriteToStoreSyncStrategy) {
    std::size_t minimumSpace{2 * numberRows * numberColumns * sizeof(CFloatStorage)};

    auto directory = std::make_shared<CTemporaryDirectory>(rootDirectory, minimumSpace);

    // Note the writer lambda holding a reference to the directory shared
    // pointer is copied to the data frame. So this isn't destroyed, and
    // the folder cleaned up, until the data frame itself is destroyed.

    auto writer = [directory](std::size_t firstRow, TFloatVec rows, TInt32Vec docHashes) {
        return std::make_unique<COnDiskDataFrameRowSlice>(
            directory, firstRow, std::move(rows), std::move(docHashes));
    };

    if (sliceCapacity != boost::none) {
        return {std::make_unique<CDataFrame>(false, numberColumns, *sliceCapacity,
                                             readWriteToStoreSyncStrategy, writer),
                directory};
    }
    return {std::make_unique<CDataFrame>(false, numberColumns,
                                         readWriteToStoreSyncStrategy, writer),
            directory};
}
}
}
