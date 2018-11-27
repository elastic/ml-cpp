/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDataFrame.h>

#include <core/CConcurrentWrapper.h>
#include <core/CDataFrameRowSlice.h>
#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/Concurrency.h>

#include <boost/make_unique.hpp>

#include <algorithm>
#include <future>
#include <memory>

namespace ml {
namespace core {
namespace data_frame_detail {

CRowRef::CRowRef(std::size_t index, TFloatVecItr beginColumns, TFloatVecItr endColumns)
    : m_Index{index}, m_BeginColumns{beginColumns}, m_EndColumns{endColumns} {
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

CRowIterator::CRowIterator(std::size_t numberColumns,
                           std::size_t rowCapacity,
                           std::size_t index,
                           TFloatVecItr base)
    : m_NumberColumns{numberColumns}, m_RowCapacity{rowCapacity}, m_Index{index}, m_Base{base} {
}

bool CRowIterator::operator==(const CRowIterator& rhs) const {
    return m_Base == rhs.m_Base;
}

bool CRowIterator::operator!=(const CRowIterator& rhs) const {
    return m_Base != rhs.m_Base;
}

CRowRef CRowIterator::operator*() const {
    return CRowRef{m_Index, m_Base, m_Base + m_NumberColumns};
}

CRowPtr CRowIterator::operator->() const {
    return CRowPtr{m_Index, m_Base, m_Base + m_NumberColumns};
}

CRowIterator& CRowIterator::operator++() {
    ++m_Index;
    m_Base += m_RowCapacity;
    return *this;
}

CRowIterator CRowIterator::operator++(int) {
    CRowIterator result{*this};
    ++m_Index;
    m_Base += m_RowCapacity;
    return result;
}

TFloatVecItr CRowIterator::base() const {
    return m_Base;
}
}
using namespace data_frame_detail;

namespace {
using TFloatVec = CDataFrame::TFloatVec;
using TSizeFloatVecPrQueue = CDataFrame::TSizeFloatVecPrQueue;
using TRowSlicePtr = CDataFrame::TRowSlicePtr;
using TRowSlicePtrVec = CDataFrame::TRowSlicePtrVec;
using TRowSlicePtrVecCItr = TRowSlicePtrVec::const_iterator;

//! \brief A worker function object used to asynchronously write slices
//! to storage.
class CAsyncDataFrameStorageWriter final {
public:
    using TWriteSliceToStoreFunc = CDataFrame::TWriteSliceToStoreFunc;

public:
    CAsyncDataFrameStorageWriter(const TWriteSliceToStoreFunc& writeSliceToStore)
        : m_WriteSliceToStore{writeSliceToStore} {}

    TRowSlicePtrVec operator()(TSizeFloatVecPrQueue& sliceQueue) {
        // Loop until we're signaled to exit by an empty slice.

        TRowSlicePtrVec slices;

        std::size_t firstRow;
        TFloatVec slice;
        for (;;) {
            std::tie(firstRow, slice) = sliceQueue.pop();
            std::size_t size{slice.size()};
            if (size > 0) {
                auto storedSlice = m_WriteSliceToStore(firstRow, std::move(slice));
                slices.push_back(std::move(storedSlice));
            } else {
                break;
            }
        }

        return slices;
    }

private:
    TWriteSliceToStoreFunc m_WriteSliceToStore;
};

//! \brief Reserves extra columns in data frame.
class CDataFrameRowSliceReserver {
public:
    CDataFrameRowSliceReserver(std::size_t numberColumns, std::size_t extraColumns)
        : m_NumberColumns{numberColumns}, m_ExtraColumns{extraColumns} {}

    bool operator()(TRowSlicePtrVecCItr beginSlices, TRowSlicePtrVecCItr endSlices) {
        for (auto i = beginSlices; i != endSlices; ++i) {
            if ((*i)->reserve(m_NumberColumns, m_ExtraColumns) == false) {
                return false;
            }
        }
        return true;
    }

private:
    std::size_t m_NumberColumns;
    std::size_t m_ExtraColumns;
};

//! \brief Reads a collection of slices from a data frame.
class CDataFrameRowSliceApply final {
public:
    using TRowFunc = CDataFrame::TRowFunc;
    using TRowFuncRef = std::reference_wrapper<TRowFunc>;

public:
    CDataFrameRowSliceApply(TRowFunc& function,
                            bool commitFunctionAction,
                            std::size_t numberColumns,
                            std::size_t rowCapacity,
                            CDataFrame::EReadWriteToStorage asyncReadFromStore)
        : m_Function{function}, m_CommitFunctionAction{commitFunctionAction}, m_NumberColumns{numberColumns},
          m_RowCapacity{rowCapacity}, m_AsyncReadFromStore{asyncReadFromStore} {}

    //! Read all slices in [\p beginSlices, \p endSlices) passing to the
    //! callback supplied to the constructor.
    //!
    //! \return False if the rows couldn't all be read.
    bool operator()(TRowSlicePtrVecCItr beginSlices, TRowSlicePtrVecCItr endSlices) {
        std::size_t firstRow;
        CDataFrameRowSliceHandle sliceHandle;

        switch (m_AsyncReadFromStore) {
        case CDataFrame::EReadWriteToStorage::E_Async: {
            // The slices get read from storage on the thread executing this
            // function each slice is then concurrently read by the callback
            // on a worker thread.

            std::shared_ptr<task<void>> backgroundApply;
            for (auto i = beginSlices; i != endSlices; ++i) {
                std::tie(firstRow, sliceHandle) = (*i)->read();
                if (sliceHandle.bad()) {
                    return false;
                }
                // We wait here so at most one slice is copied into memory.
                await(backgroundApply);
                LOG_TRACE(<< "applying function to slice starting at row " << firstRow);
                backgroundApply = 
                    async(defaultAsyncExecutor(), 
                          [ firstRow, slice = std::move(sliceHandle), i, this ] {
                            std::size_t rows{slice.size() / m_RowCapacity};
                            std::size_t lastRow{firstRow + rows};
                            m_Function(CRowIterator{m_NumberColumns, m_RowCapacity,
                                                    firstRow, slice.begin()},
                                        CRowIterator{m_NumberColumns, m_RowCapacity,
                                                    lastRow, slice.end()});
                            if (m_CommitFunctionAction) {
                                (*i)->write(slice.values());
                            }
                        });
            }
            await(backgroundApply);
            break;
        }
        case CDataFrame::EReadWriteToStorage::E_Sync:
            for (auto i = beginSlices; i != endSlices; ++i) {
                std::tie(firstRow, sliceHandle) = (*i)->read();
                if (sliceHandle.bad()) {
                    return false;
                }
                LOG_TRACE(<< "applying function to slice starting at row " << firstRow);
                std::size_t rows{sliceHandle.size() / m_NumberColumns};
                std::size_t lastRow{firstRow + rows};
                m_Function(CRowIterator{m_NumberColumns, m_RowCapacity,
                                        firstRow, sliceHandle.begin()},
                           CRowIterator{m_NumberColumns, m_RowCapacity, lastRow,
                                        sliceHandle.end()});
                if (m_CommitFunctionAction) {
                    (*i)->write(sliceHandle.values());
                }
            }
            break;
        }

        return true;
    }

private:
    TRowFuncRef m_Function;
    bool m_CommitFunctionAction;
    std::size_t m_NumberColumns;
    std::size_t m_RowCapacity;
    CDataFrame::EReadWriteToStorage m_AsyncReadFromStore;
};

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
    if (m_NumberColumns >= rowCapacity) {
        return true;
    }

    m_RowCapacity = rowCapacity;

    // We pass a dummy function which is ignored since we'll use the reserver
    // to "apply" this function to slices and this simply ignores the supplied
    // value.

    bool successful;
    std::tie(std::ignore, successful) =
        this->applyFunctionToRows(numberThreads, TRowFunc{}, [this](TRowFunc) {
            return CDataFrameRowSliceReserver{m_NumberColumns, m_RowCapacity - m_NumberColumns};
        });
    return successful;
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
        return {{reader}, true};
    }
    return this->applyFunctionToRows(numberThreads, std::move(reader), [this](TRowFunc& read) {
        return CDataFrameRowSliceApply{read, false, m_NumberColumns, m_RowCapacity,
                                       m_ReadAndWriteToStoreSyncStrategy};
    });
}

bool CDataFrame::writeColumns(std::size_t numberThreads, TRowFunc writer) {
    bool successful;
    std::tie(std::ignore, successful) = this->applyFunctionToRows(
        numberThreads, std::move(writer), [this](TRowFunc& append) {
            return CDataFrameRowSliceApply{append, true, m_NumberColumns, m_RowCapacity,
                                           m_ReadAndWriteToStoreSyncStrategy};
        });
    return successful;
}

void CDataFrame::writeRow(const TWriteFunc& writeRow) {
    if (m_Writer == nullptr) {
        m_Writer = boost::make_unique<CDataFrameRowSliceWriter>(
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

CDataFrame::CDataFrameRowSliceWriter::CDataFrameRowSliceWriter(
    std::size_t numberRows,
    std::size_t rowCapacity,
    std::size_t sliceCapacityInRows,
    EReadWriteToStorage writeToStoreSyncStrategy,
    TWriteSliceToStoreFunc writeSliceToStore)
    : m_NumberRows{numberRows}, m_RowCapacity{rowCapacity}, m_SliceCapacityInRows{sliceCapacityInRows},
      m_WriteToStoreSyncStrategy{writeToStoreSyncStrategy}, m_WriteSliceToStore{writeSliceToStore} {
    m_SliceBeingWritten.reserve(m_SliceCapacityInRows * m_RowCapacity);
    if (m_WriteToStoreSyncStrategy == EReadWriteToStorage::E_Async) {
        m_AsyncWriteToStoreResult = std::async(
            std::launch::async, CAsyncDataFrameStorageWriter{writeSliceToStore},
            std::ref(m_SlicesToAsyncWriteToStore));
    }
}

CDataFrame::CDataFrameRowSliceWriter::~CDataFrameRowSliceWriter() {
    this->finishAsyncWriteToStore();
}

void CDataFrame::CDataFrameRowSliceWriter::operator()(const TWriteFunc& writeRow) {
    // Write the next row at the end of the current slice being written
    // and if the slice is full pass to the thread storing slices.

    std::size_t end{m_SliceBeingWritten.size()};

    m_SliceBeingWritten.resize(end + m_RowCapacity);
    writeRow(m_SliceBeingWritten.begin() + end);
    ++m_NumberRows;

    if (m_SliceBeingWritten.size() == m_SliceCapacityInRows * m_RowCapacity) {
        std::size_t firstRow{m_NumberRows - m_SliceCapacityInRows};
        LOG_TRACE(<< "Storing slice [" << firstRow << "," << m_NumberRows << ")");

        switch (m_WriteToStoreSyncStrategy) {
        case EReadWriteToStorage::E_Async: {
            TSizeFloatVecPr slice{firstRow, std::move(m_SliceBeingWritten)};
            m_SlicesToAsyncWriteToStore.push(std::move(slice));
            break;
        }
        case EReadWriteToStorage::E_Sync:
            m_SyncWrittenSlices.push_back(
                m_WriteSliceToStore(firstRow, std::move(m_SliceBeingWritten)));
            break;
        }
        m_SliceBeingWritten.clear();
        m_SliceBeingWritten.reserve(m_SliceCapacityInRows * m_RowCapacity);
    }
}

CDataFrame::TSizeDataFrameRowSlicePtrVecPr
CDataFrame::CDataFrameRowSliceWriter::finishWritingRows() {
    std::size_t firstRow{m_NumberRows - m_SliceBeingWritten.size() / m_RowCapacity};
    LOG_TRACE(<< "Last slice "
              << (firstRow == m_NumberRows ? "empty"
                                           : "[" + std::to_string(firstRow) + "," +
                                                 std::to_string(m_NumberRows) + ")"));

    switch (m_WriteToStoreSyncStrategy) {
    case EReadWriteToStorage::E_Async:
        if (m_SliceBeingWritten.size() > 0) {
            TSizeFloatVecPr slice{firstRow, std::move(m_SliceBeingWritten)};
            m_SlicesToAsyncWriteToStore.push(std::move(slice));
        }
        this->finishAsyncWriteToStore();
        return {m_NumberRows, m_AsyncWriteToStoreResult.get()};
    case EReadWriteToStorage::E_Sync:
        if (m_SliceBeingWritten.size() > 0) {
            m_SyncWrittenSlices.push_back(
                m_WriteSliceToStore(firstRow, std::move(m_SliceBeingWritten)));
        }
        break;
    }
    return {m_NumberRows, std::move(m_SyncWrittenSlices)};
}

CDataFrame::TSizeSizePr CDataFrame::numberOfThreadsAndStride(std::size_t target) const {

    if (m_Slices.empty()) {
        return {1, 0};
    }

    std::size_t numberSlices{m_Slices.size()};
    std::size_t numberThreads{std::min(numberSlices, target)};
    std::size_t strideLowerBound{numberSlices / numberThreads};
    std::size_t strideUpperBound{strideLowerBound + 1};

    if (numberSlices % numberThreads == 0) {
        return {numberThreads, strideLowerBound};
    }
    if (numberSlices % strideUpperBound == 0) {
        return {numberThreads - 1, strideUpperBound};
    }
    return {numberThreads, strideUpperBound};
}

template<typename SLICE_FUNCTION_FACTORY>
CDataFrame::TRowFuncVecBoolPr
CDataFrame::applyFunctionToRows(std::size_t numberThreads,
                                TRowFunc function,
                                SLICE_FUNCTION_FACTORY factory) const {
    if (numberThreads == 1) {
        // This all happens on the main thread to avoid a context switch.

        auto sliceFunction = factory(function);
        bool successful{sliceFunction(m_Slices.begin(), m_Slices.end())};
        return {{std::move(function)}, successful};
    }

    // We use a fixed schedule whereby each reader reads non-overlapping
    // slices. This means we can get no contention on reads from the slice
    // vector. This is naturally load balanced because we arrange for each
    // reader to read, as close as possible, the same number of rows.

    std::size_t stride;
    std::tie(numberThreads, stride) = this->numberOfThreadsAndStride(numberThreads);
    LOG_TRACE(<< "numberThreads = " << numberThreads << " stride = " << stride);

    TRowFuncVec functions{numberThreads, function};

    std::vector<std::shared_ptr<task<bool>>> tasks;
    tasks.reserve(numberThreads);
    std::size_t j{0};
    for (std::size_t i = 0; i + 1 < numberThreads; ++i, j += stride) {
        auto begin = m_Slices.begin() + j;
        auto end = m_Slices.begin() + j + stride;
        auto sliceFunction = factory(functions[i]);
        tasks.push_back(async(defaultAsyncExecutor(), sliceFunction, begin, end));
    }
    auto begin = m_Slices.begin() + j;
    auto end = m_Slices.end();
    if (begin != end) {
        auto sliceFunction = factory(functions.back());
        tasks.push_back(async(defaultAsyncExecutor(), sliceFunction, begin, end));
    }

    bool successful{true};
    for (auto& task : tasks) {
        successful &= task->get();
    }

    return {std::move(functions), successful};
}

void CDataFrame::CDataFrameRowSliceWriter::finishAsyncWriteToStore() {
    // Passing an empty slice signals to the thread writing slices to
    // storage that we're done.

    if (m_Writing) {
        if (m_WriteToStoreSyncStrategy == EReadWriteToStorage::E_Async) {
            m_SlicesToAsyncWriteToStore.push({0, TFloatVec{}});
        }
        m_Writing = false;
    }
}

CDataFrame makeMainStorageDataFrame(std::size_t numberColumns,
                                    boost::optional<std::size_t> sliceCapacity,
                                    CDataFrame::EReadWriteToStorage readWriteToStoreSyncStrategy) {
    // The return copy is elided so we never need to call the explicitly
    // deleted data frame copy constructor.

    auto writer = [](std::size_t firstRow, TFloatVec slice) {
        return boost::make_unique<CMainMemoryDataFrameRowSlice>(firstRow, std::move(slice));
    };

    if (sliceCapacity != boost::none) {
        return {true, numberColumns, *sliceCapacity, readWriteToStoreSyncStrategy, writer};
    }

    return {true, numberColumns, readWriteToStoreSyncStrategy, writer};
}

CDataFrame makeDiskStorageDataFrame(const std::string& rootDirectory,
                                    std::size_t numberColumns,
                                    std::size_t numberRows,
                                    boost::optional<std::size_t> sliceCapacity,
                                    CDataFrame::EReadWriteToStorage readWriteToStoreSyncStrategy) {
    // The return copy is elided so we never need to call the explicitly
    // deleted data frame copy constructor.

    std::size_t minimumSpace{2 * numberRows * numberColumns * sizeof(CFloatStorage)};

    COnDiskDataFrameRowSlice::TTemporaryDirectoryPtr directory{
        std::make_shared<COnDiskDataFrameRowSlice::CTemporaryDirectory>(
            rootDirectory, minimumSpace)};

    // Note the writer lambda holding a reference to the directory shared
    // pointer is copied to the data frame. So this isn't destroyed, and
    // the folder cleaned up, until the data frame itself is destroyed.

    auto writer = [directory](std::size_t firstRow, TFloatVec slice) {
        return boost::make_unique<COnDiskDataFrameRowSlice>(directory, firstRow,
                                                            std::move(slice));
    };

    if (sliceCapacity != boost::none) {
        return {false, numberColumns, *sliceCapacity, readWriteToStoreSyncStrategy, writer};
    }
    return {false, numberColumns, readWriteToStoreSyncStrategy, writer};
}
}
}
