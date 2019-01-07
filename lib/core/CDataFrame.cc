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

CRowRef::CRowRef(std::size_t index, TFloatVecItr beginColumns, TFloatVecItr endColumns, std::int32_t docId)
    : m_Index{index}, m_BeginColumns{beginColumns}, m_EndColumns{endColumns}, m_DocId{docId} {
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

std::int32_t CRowRef::docId() const {
    return m_DocId;
}

CRowIterator::CRowIterator(std::size_t numberColumns,
                           std::size_t rowCapacity,
                           std::size_t index,
                           TFloatVecItr rowItr,
                           TInt32VecCItr docIdItr)
    : m_NumberColumns{numberColumns},
      m_RowCapacity{rowCapacity}, m_Index{index}, m_RowItr{rowItr}, m_DocIdItr{docIdItr} {
}

bool CRowIterator::operator==(const CRowIterator& rhs) const {
    return m_RowItr == rhs.m_RowItr && m_DocIdItr == rhs.m_DocIdItr;
}

bool CRowIterator::operator!=(const CRowIterator& rhs) const {
    return m_RowItr != rhs.m_RowItr || m_DocIdItr != rhs.m_DocIdItr;
}

CRowRef CRowIterator::operator*() const {
    return CRowRef{m_Index, m_RowItr, m_RowItr + m_NumberColumns, *m_DocIdItr};
}

CRowPtr CRowIterator::operator->() const {
    return CRowPtr{m_Index, m_RowItr, m_RowItr + m_NumberColumns, *m_DocIdItr};
}

CRowIterator& CRowIterator::operator++() {
    ++m_Index;
    m_RowItr += m_RowCapacity;
    ++m_DocIdItr;
    return *this;
}

CRowIterator CRowIterator::operator++(int) {
    CRowIterator result{*this};
    ++m_Index;
    m_RowItr += m_RowCapacity;
    ++m_DocIdItr;
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

bool CDataFrame::reserve(std::size_t numberThreads, std::size_t rowCapacity) {
    if (m_RowCapacity >= rowCapacity) {
        return true;
    }

    m_RowCapacity = rowCapacity;

    std::atomic_bool successful{true};
    parallel_for_each(numberThreads, m_Slices.begin(), m_Slices.end(),
                      [this, &successful](TRowSlicePtr& slice) {
                          if (successful.load() == false) {
                              return;
                          }
                          if (slice->reserve(m_NumberColumns,
                                             m_RowCapacity - m_NumberColumns) == false) {
                              successful.store(false);
                          }
                      });
    return successful.load();
}

bool CDataFrame::resizeColumns(std::size_t numberThreads, std::size_t numberColumns) {
    if (this->reserve(numberThreads, numberColumns) == false) {
        return false;
    }
    m_NumberColumns = numberColumns;
    return true;
}

CDataFrame::TRowFuncVecBoolPr CDataFrame::readRows(std::size_t numberThreads,
                                                   TRowFunc reader) const {
    if (m_NumberRows == 0) {
        return {{std::move(reader)}, true};
    }
    return numberThreads > 1
               ? this->parallelApplyToAllRows(numberThreads, std::move(reader), false)
               : this->sequentialApplyToAllRows(std::move(reader), false);
}

bool CDataFrame::writeColumns(std::size_t numberThreads, TRowFunc writer) {
    if (m_NumberRows == 0) {
        return true;
    }
    bool successful;
    std::tie(std::ignore, successful) =
        numberThreads > 1
            ? this->parallelApplyToAllRows(numberThreads, std::move(writer), true)
            : this->sequentialApplyToAllRows(std::move(writer), true);
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
CDataFrame::parallelApplyToAllRows(std::size_t numberThreads, TRowFunc func, bool commitResult) const {

    // If we're reading in parallel then we don't want to interleave
    // reads from storage and applying the function because we're
    // already fully balancing our work across the slices.

    std::atomic_bool successful{true};

    CDataFrameRowSliceHandle readRow;
    auto results = parallel_for_each(
        numberThreads, m_Slices.begin(), m_Slices.end(),
        bindRetrievableState(
            [&successful, readRow, commitResult,
             this](TRowFunc& func_, const TRowSlicePtr& slice) mutable {
                if (successful.load() == false) {
                    return;
                }

                std::size_t firstRow;
                std::tie(firstRow, readRow) = slice->read();
                if (readRow.bad()) {
                    successful.store(false);
                    return;
                }

                this->applyToRowsOfOneSlice(func_, firstRow, readRow);

                if (commitResult) {
                    slice->write(readRow.rows(), readRow.docIds());
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

CDataFrame::TRowFuncVecBoolPr
CDataFrame::sequentialApplyToAllRows(TRowFunc func, bool commitResult) const {

    std::size_t firstRow;
    CDataFrameRowSliceHandle readSlice;

    switch (m_ReadAndWriteToStoreSyncStrategy) {
    case CDataFrame::EReadWriteToStorage::E_Async: {
        // The slices get read from storage on the thread executing this
        // function each slice is then concurrently read by the callback
        // on a worker thread.

        future<void> backgroundApply;

        // We need to wait and this isn't guaranteed by the future destructor.
        CWaitIfValidWhenExitingScope<void> waitFor(backgroundApply);

        for (auto& slice : m_Slices) {
            std::tie(firstRow, readSlice) = slice->read();
            if (readSlice.bad()) {
                return {{std::move(func)}, false};
            }

            // We wait here so at most one slice is copied into memory.
            wait_for_valid(backgroundApply);

            backgroundApply = async(defaultAsyncExecutor(), [
                &func, firstRow, readSlice_ = std::move(readSlice), &slice, commitResult, this
            ] {
                this->applyToRowsOfOneSlice(func, firstRow, readSlice_);
                if (commitResult) {
                    slice->write(readSlice_.rows(), readSlice_.docIds());
                }
            });
        }
        break;
    }
    case CDataFrame::EReadWriteToStorage::E_Sync:
        for (auto& slice : m_Slices) {
            std::tie(firstRow, readSlice) = slice->read();
            if (readSlice.bad()) {
                return {{std::move(func)}, false};
            }
            this->applyToRowsOfOneSlice(func, firstRow, readSlice);
            if (commitResult) {
                slice->write(readSlice.rows(), readSlice.docIds());
            }
        }
        break;
    }

    return {{std::move(func)}, true};
}

void CDataFrame::applyToRowsOfOneSlice(TRowFunc& func,
                                       std::size_t firstRow,
                                       const CDataFrameRowSliceHandle& slice) const {
    LOG_TRACE(<< "Applying function to slice starting at row " << firstRow);
    std::size_t rows{slice.size() / m_RowCapacity};
    std::size_t lastRow{firstRow + rows};
    func(CRowIterator{m_NumberColumns, m_RowCapacity, firstRow,
                      slice.beginRows(), slice.beginDocIds()},
         CRowIterator{m_NumberColumns, m_RowCapacity, lastRow, slice.endRows(),
                      slice.endDocIds()});
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
    m_DocIdsOfSliceBeingWritten.reserve(m_SliceCapacityInRows);
}

void CDataFrame::CDataFrameRowSliceWriter::operator()(const TWriteFunc& writeRow) {
    // Write the next row at the end of the current slice being written
    // and if the slice is full pass to the thread storing slices.

    std::size_t end{m_RowsOfSliceBeingWritten.size()};

    m_RowsOfSliceBeingWritten.resize(end + m_RowCapacity);
    m_DocIdsOfSliceBeingWritten.emplace_back();
    writeRow(m_RowsOfSliceBeingWritten.begin() + end,
             m_DocIdsOfSliceBeingWritten.back());
    ++m_NumberRows;

    if (m_DocIdsOfSliceBeingWritten.size() == m_SliceCapacityInRows) {
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
                      std::move(m_DocIdsOfSliceBeingWritten));
            break;
        }
        case EReadWriteToStorage::E_Sync:
            m_SlicesWrittenToStore.push_back(
                m_WriteSliceToStore(firstRow, std::move(m_RowsOfSliceBeingWritten),
                                    std::move(m_DocIdsOfSliceBeingWritten)));
            break;
        }
        m_RowsOfSliceBeingWritten.clear();
        m_DocIdsOfSliceBeingWritten.clear();
        m_RowsOfSliceBeingWritten.reserve(m_SliceCapacityInRows * m_RowCapacity);
        m_DocIdsOfSliceBeingWritten.reserve(m_SliceCapacityInRows);
    }
}

CDataFrame::TSizeDataFrameRowSlicePtrVecPr
CDataFrame::CDataFrameRowSliceWriter::finishWritingRows() {
    if (m_SliceWrittenAsyncToStore.valid()) {
        m_SlicesWrittenToStore.push_back(m_SliceWrittenAsyncToStore.get());
    }

    if (m_DocIdsOfSliceBeingWritten.size() > 0) {
        std::size_t firstRow{m_NumberRows - m_RowsOfSliceBeingWritten.size() / m_RowCapacity};
        LOG_TRACE(<< "Last slice [" << std::to_string(firstRow) << ","
                  << std::to_string(m_NumberRows) + ")");
        m_SlicesWrittenToStore.push_back(
            m_WriteSliceToStore(firstRow, std::move(m_RowsOfSliceBeingWritten),
                                std::move(m_DocIdsOfSliceBeingWritten)));
    }

    return {m_NumberRows, std::move(m_SlicesWrittenToStore)};
}

std::unique_ptr<CDataFrame>
makeMainStorageDataFrame(std::size_t numberColumns,
                         boost::optional<std::size_t> sliceCapacity,
                         CDataFrame::EReadWriteToStorage readWriteToStoreSyncStrategy) {
    auto writer = [](std::size_t firstRow, TFloatVec rows, TInt32Vec docIds) {
        return std::make_unique<CMainMemoryDataFrameRowSlice>(
            firstRow, std::move(rows), std::move(docIds));
    };

    if (sliceCapacity != boost::none) {
        return std::make_unique<CDataFrame>(true, numberColumns, *sliceCapacity,
                                            readWriteToStoreSyncStrategy, writer);
    }

    return std::make_unique<CDataFrame>(true, numberColumns,
                                        readWriteToStoreSyncStrategy, writer);
}

std::unique_ptr<CDataFrame>
makeDiskStorageDataFrame(const std::string& rootDirectory,
                         std::size_t numberColumns,
                         std::size_t numberRows,
                         boost::optional<std::size_t> sliceCapacity,
                         CDataFrame::EReadWriteToStorage readWriteToStoreSyncStrategy) {
    std::size_t minimumSpace{2 * numberRows * numberColumns * sizeof(CFloatStorage)};

    COnDiskDataFrameRowSlice::TTemporaryDirectoryPtr directory{
        std::make_shared<COnDiskDataFrameRowSlice::CTemporaryDirectory>(
            rootDirectory, minimumSpace)};

    // Note the writer lambda holding a reference to the directory shared
    // pointer is copied to the data frame. So this isn't destroyed, and
    // the folder cleaned up, until the data frame itself is destroyed.

    auto writer = [directory](std::size_t firstRow, TFloatVec rows, TInt32Vec docIds) {
        return std::make_unique<COnDiskDataFrameRowSlice>(
            directory, firstRow, std::move(rows), std::move(docIds));
    };

    if (sliceCapacity != boost::none) {
        return std::make_unique<CDataFrame>(false, numberColumns, *sliceCapacity,
                                            readWriteToStoreSyncStrategy, writer);
    }
    return std::make_unique<CDataFrame>(false, numberColumns,
                                        readWriteToStoreSyncStrategy, writer);
}
}
}
