/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_core_CDataFrame_h
#define INCLUDED_ml_core_CDataFrame_h

#include <core/CConcurrentQueue.h>
#include <core/CDataFrameRowSlice.h>
#include <core/CFloatStorage.h>
#include <core/ImportExport.h>

#include <boost/optional.hpp>

#include <algorithm>
#include <functional>
#include <future>
#include <iterator>
#include <memory>
#include <vector>

namespace ml {
namespace core {

namespace data_frame_detail {

using TFloatVec = std::vector<CFloatStorage>;
using TFloatVecItr = TFloatVec::iterator;

//! \brief A lightweight wrapper around a single row of the data frame.
//!
//! DESCRIPTION:\n
//! This is a helper class used to read rows of a CDataFrame object. It is
//! lightweight (24 bytes) and is expected to only be valid transiently
//! during a read. It should not be stored.
//!
//! If the row resides in main memory then its data can be referenced. If
//! it resides on disk then this is only valid whilst being read and it
//! should be copied if needed longer.
class CORE_EXPORT CRowRef {
public:
    //! \param[in] index The row index.
    //! \param[in] beginColumns The iterator for the columns of row \p index.
    //! \param[in] endColumns The iterator for the end of the columns of row
    //! \p index.
    CRowRef(std::size_t index, TFloatVecItr beginColumns, TFloatVecItr endColumns);

    //! Get column \p i value.
    CFloatStorage operator[](std::size_t i) const;

    //! Get the row's index.
    std::size_t index() const;

    //! Get the number of columns.
    std::size_t numberColumns() const;

    //! Write \p value to \p column of the row.
    void writeColumn(std::size_t index, double value) const;

    //! Get the data backing the row.
    CFloatStorage* data() const;

    //! Copy the range to \p output iterator.
    //!
    //! \warning The output iterator that must be able to receive number of
    //! columns values.
    template<typename ITR>
    void copyTo(ITR output) const {
        std::copy(m_BeginColumns, m_EndColumns, output);
    }

private:
    std::size_t m_Index;
    TFloatVecItr m_BeginColumns;
    TFloatVecItr m_EndColumns;
};

//! \brief Decorates CRowCRef to give it pointer semantics.
class CORE_EXPORT CRowPtr : public CRowRef {
public:
    template<typename... ARGS>
    CRowPtr(ARGS&&... args) : CRowRef{std::forward<ARGS>(args)...} {}

    CRowPtr* operator->() { return this; }
    const CRowPtr* operator->() const { return this; }
};

//! \brief A forward iterator over rows of the data frame.
//!
//! DESCRIPTION:\n
//! This is a helper class used to read rows of a CDataFrame object which
//! dereferences to CRowRef objects.
class CORE_EXPORT CRowIterator
    : public std::iterator<std::forward_iterator_tag, CRowRef, std::ptrdiff_t, CRowPtr, CRowRef> {
public:
    CRowIterator() = default;

    //! \param[in] numberColumns The number of columns in the data frame.
    //! \param[in] rowCapacity The capacity of each row in the data frame.
    //! \param[in] index The row index.
    //! \param[in] base The iterator for the columns of row \p index.
    CRowIterator(std::size_t numberColumns, std::size_t rowCapacity, std::size_t index, TFloatVecItr base);

    //! \name Forward Iterator Contract
    //@{
    bool operator==(const CRowIterator& rhs) const;
    bool operator!=(const CRowIterator& rhs) const;
    CRowRef operator*() const;
    CRowPtr operator->() const;
    CRowIterator& operator++();
    CRowIterator operator++(int);
    //@}

    TFloatVecItr base() const;

private:
    std::size_t m_NumberColumns = 0;
    std::size_t m_RowCapacity = 0;
    std::size_t m_Index = 0;
    TFloatVecItr m_Base;
};
}

//! \brief A data frame representation.
//!
//! DESECRIPTION:\n
//! A table data structure with "fixed" number of columns and a dynamic number
//! of rows.
//!
//! It can be read in row order and is append only. Reading rows can also be
//! parallelized in which case each reader reads a disjoint subset of the data
//! frame's rows.
//!
//! Space can be reserved at any point to hold one or more additional columns.
//! These are not visible until they are written.
//!
//! IMPLEMENTATION:\n
//! This is a fairly lightweight container which is essentially responsible
//! for managing the read and write process to some underlying store format.
//! The store format is determined by the user implementing functionality to
//! read and write state from the store. For example, these could copy to /
//! from main memory, "write to" / "read from" disk, etc. A factory function
//! must be provided to the constructor which effectively that determines the
//! type of storage used. It is assumed that copying this has no side effects.
//!
//! The data frame is divided into slices each of which represent a number of
//! contiguous rows. The idea is that they contain a reasonable amount of memory
//! so that, for example, they significantly reduce the number of "writes to" /
//! "reads from" disk (a whole slice being written or read in one go), mean we'll
//! get good locality of reference and mean there is minimal book keeping overhead
//! (such as state for vector sizes, pointers to starts of memory blocks, etc).
//! In addition, it is assumed that access to the individual slices is thread
//! safe. If they share state the implementation must ensure that access to this
//! is synchronized.
//!
//! Reads and writes of a single row are also done via call backs supplied to the
//! readRows and writeRow functions. This is to achieve maximum decoupling from
//! the calling code for how the underlying values are used or where they come
//! from. It also means certain operations can be done very efficiently. For example,
//! a stream can be attached to a row writer function to copy the values directly
//! into the data frame storage.
//!
//! Read and writes to storage can optionally happen in a separate thread to the
//! row reading and writing to deal with the case that these operations can by
//! time consuming.
class CORE_EXPORT CDataFrame final {
public:
    using TFloatVec = std::vector<CFloatStorage>;
    using TFloatVecItr = TFloatVec::iterator;
    using TSizeFloatVecPr = std::pair<std::size_t, TFloatVec>;
    using TRowItr = data_frame_detail::CRowIterator;
    using TRowFunc = std::function<void(TRowItr, TRowItr)>;
    using TRowFuncVec = std::vector<TRowFunc>;
    using TRowFuncVecBoolPr = std::pair<TRowFuncVec, bool>;
    using TWriteFunc = std::function<void(TFloatVecItr)>;
    using TRowSlicePtr = std::unique_ptr<CDataFrameRowSlice>;
    using TRowSlicePtrVec = std::vector<TRowSlicePtr>;
    using TSizeRowSliceHandlePr = std::pair<std::size_t, CDataFrameRowSliceHandle>;
    using TWriteSliceToStoreFunc = std::function<TRowSlicePtr(std::size_t, TFloatVec)>;
    using TSizeFloatVecPrQueue = CConcurrentQueue<TSizeFloatVecPr, 1>;

    //! Controls whether to read and write to storage asynchronously.
    enum class EReadWriteToStorage { E_Async, E_Sync };

public:
    //! \param[in] numberColumns The number of columns in the data frame.
    //! \param[in] sliceCapacityInRows The capacity of a slice of the data frame
    //! as a number of rows.
    //! \param[in] readAndWriteToStoreSyncStrategy Controls whether reads and
    //! writes from slice storage are synchronous or asynchronous.
    //! \param[in] writeSliceToStore The callback to write a slice to storage.
    //!
    //! \warning This requires that \p writeSliceToStore and \p readSliceFromStore
    //! can be copied and are thread safe. If they are not stateless then it is
    //! the implementers responsibility to ensure these conditions are satisfied.
    CDataFrame(bool inMainMemory,
               std::size_t numberColumns,
               std::size_t sliceCapacityInRows,
               EReadWriteToStorage readAndWriteToStoreSyncStrategy,
               const TWriteSliceToStoreFunc& writeSliceToStore);

    //! Overload which manages the setting of slice capacity to a sensible default.
    CDataFrame(bool inMainMemory,
               std::size_t numberColumns,
               EReadWriteToStorage readAndWriteToStoreSyncStrategy,
               const TWriteSliceToStoreFunc& writeSliceToStore);

    ~CDataFrame();

    CDataFrame(const CDataFrame&) = delete;
    CDataFrame& operator=(const CDataFrame&) = delete;
    CDataFrame(CDataFrame&&) = default;
    CDataFrame& operator=(CDataFrame&&) = default;

    //! Check if the data frame resides in main memory.
    bool inMainMemory() const;

    //! Get the number of rows in the data frame.
    std::size_t numberRows() const;

    //! Get the number of columns in the data frame.
    std::size_t numberColumns() const;

    //! Reserve space for up to \p numberColumns.
    //!
    //! This enables in-place updates of the data frame for analytics operations
    //! that append columns.
    //!
    //! \param[in] numberThreads The target number of threads to use.
    //! \param[in] rowCapacity The desired number of columns.
    bool reserve(std::size_t numberThreads, std::size_t rowCapacity);

    //! Resize to contain \p numberColumns columns.
    //!
    //! \param[in] numberThreads The target number of threads to use.
    //! \param[in] numberColumns The desired number of columns.
    bool resizeColumns(std::size_t numberThreads, std::size_t numberColumns);

    //! This reads rows using one or more readers.
    //!
    //! One reader is bound to one thread. Each thread reads a disjoint subset
    //! of slices of the data frame with the i'th reader reading slices
    //! \f$[ i \lfloor m / \min(m,n) \rfloor, (i+1) \lfloor m / \min(m,n) \rfloor)\f$
    //! for \f$m\f$ slices and \f$n\f$ readers to eliminate contention on the
    //! data frame state.
    //!
    //! \warning If there is more than one thread and the reader has shared
    //! state then the caller must ensure that access to this is thread safe.
    //!
    //! \param[in] numberThreads The target number of threads to use.
    //! \param[in] reader The callback to read rows.
    //! \return The readers used. This is intended to allow the reader to
    //! accumulate state in the reader which is passed back. RVO means any
    //! copy will be elided. Otherwise, the reader must hold the state by
    //! reference and must synchronize access to it.
    TRowFuncVecBoolPr readRows(std::size_t numberThreads, TRowFunc reader) const;

    //! Convenience overload for typed readers.
    //!
    //! The reason for this is to wrap up the code to extract the typed readers
    //! from the return type. A common case is that these will cache some state
    //! for which the calling code will need access to the underlying type to
    //! retrieve.
    //!
    //! \note READER must implement the TRowFunc contract.
    template<typename READER>
    std::pair<std::vector<READER>, bool>
    readRows(std::size_t numberThreads, READER reader) const {

        TRowFuncVecBoolPr result_{readRows(numberThreads, TRowFunc(std::move(reader)))};

        std::vector<READER> result;
        result.reserve(result_.first.size());
        for (auto& reader_ : result_.first) {
            result.push_back(std::move(*reader_.target<READER>()));
        }

        return {std::move(result), result_.second};
    }

    //! Overwrite a number of columns with \p writer.
    //!
    //! The caller must ensure that the columns overwritten are in range.
    //!
    //! \warning If there is more than one thread and the writer has shared
    //! state then the caller must ensure that access to this is thread safe.
    //!
    //! \param[in] numberThreads The target number of threads to use.
    //! \param[in] writer The callback to write the columns.
    bool writeColumns(std::size_t numberThreads, TRowFunc writer);

    //! This writes a single row of the data frame via a callback.
    //!
    //! If asynchronous read and write to store was selected in the constructor
    //! this creates a thread to store slices which is only joined when the
    //! finishWritingRows is called (or the data frame is destroyed).
    //!
    //! Until finishWritingRows is called the written row is not visible outside
    //! the data frame.
    //!
    //! \param[in] writeRow A function which writes a single row setting the
    //! values in supplied an output iterator to the column values of the row.
    //!
    //! \warning The caller MUST call finishWritingRows after they have finished
    //! writing rows.
    void writeRow(const TWriteFunc& writeRow);

    //! This retrieves the asynchronous work from writing the rows to the store
    //! and updates the stored rows.
    //!
    //! Until this is called the written rows are not visible outside the data
    //! frame.
    //!
    //! \warning This MUST be called after the last row is written to commit the
    //! work and to join the thread used to store the slices.
    void finishWritingRows();

    //! Get the memory used by the data frame.
    std::size_t memoryUsage() const;

    //! Get a checksum of all the data stored in the data frame.
    std::uint64_t checksum() const;

    // TODO Better error case diagnostics.

    // TODO We may want an architecture agnostic check pointing mechanism for long
    // running tasks.

private:
    using TSizeSizePr = std::pair<std::size_t, std::size_t>;
    using TSizeDataFrameRowSlicePtrVecPr = std::pair<std::size_t, TRowSlicePtrVec>;

    //! \brief Writes rows to the data frame.
    class CDataFrameRowSliceWriter final {
    public:
        CDataFrameRowSliceWriter(std::size_t numberRows,
                                 std::size_t rowCapacity,
                                 std::size_t sliceCapacityInRows,
                                 EReadWriteToStorage writeToStoreSyncStrategy,
                                 TWriteSliceToStoreFunc writeSliceToStore);
        ~CDataFrameRowSliceWriter();

        //! Write a single row using the callback \p writeRow.
        void operator()(const TWriteFunc& writeRow);

        //! Finish writing the rows and return the number of rows written and
        //! the slices.
        TSizeDataFrameRowSlicePtrVecPr finishWritingRows();

    private:
        //! This is called to flush the queue to write to the store.
        void finishAsyncWriteToStore();

    private:
        std::size_t m_NumberRows;
        std::size_t m_RowCapacity;
        std::size_t m_SliceCapacityInRows;
        EReadWriteToStorage m_WriteToStoreSyncStrategy;
        TWriteSliceToStoreFunc m_WriteSliceToStore;

        TFloatVec m_SliceBeingWritten;

        //! This is true while the rows are still being added.
        bool m_Writing = true;
        //! A queue of finished slices shared with the thread that writes
        //! them to storage if writes to storage are asynchronous.
        TSizeFloatVecPrQueue m_SlicesToAsyncWriteToStore;
        //! The result of the asynchronous work to write slices to storage
        //! if there is any.
        std::future<TRowSlicePtrVec> m_AsyncWriteToStoreResult;
        //! The synchronously written stored slices.
        TRowSlicePtrVec m_SyncWrittenSlices;
    };
    using TRowSliceWriterPtr = std::unique_ptr<CDataFrameRowSliceWriter>;

private:
    TRowFuncVecBoolPr parallelApplyToAllRows(std::size_t numberThreads,
                                             TRowFunc func,
                                             bool commitResult) const;

    TRowFuncVecBoolPr sequentialApplyToAllRows(TRowFunc func, bool commitResult) const;

    void applyToRowsOfOneSlice(TRowFunc& func,
                               std::size_t firstRow,
                               const CDataFrameRowSliceHandle& slice) const;

private:
    //! True if the data frame resides in main memory.
    bool m_InMainMemory;
    //! The number of rows in the data frame.
    std::size_t m_NumberRows = 0;
    //! The number of columns in the data frame.
    std::size_t m_NumberColumns;
    //! The number of columns a row could contain. This is greater than or
    //! equal to m_NumberColumns.
    std::size_t m_RowCapacity;
    //! The capacity of a slice of the data frame as a number of rows.
    std::size_t m_SliceCapacityInRows;

    //! If true read and write asynchronously to storage.
    EReadWriteToStorage m_ReadAndWriteToStoreSyncStrategy;
    //! The callback to write a slice to storage.
    TWriteSliceToStoreFunc m_WriteSliceToStore;

    //! The stored slices.
    TRowSlicePtrVec m_Slices;

    //! The slice writer which is currently active.
    TRowSliceWriterPtr m_Writer;
};

//! Make a data frame which uses main memory storage for its slices.
//!
//! \param[in] numberColumns The number of columns in the data frame created.
//! \param[in] sliceCapacity If none null this overrides the default slice
//! capacity in rows.
//! \param[in] readWriteToStoreSyncStrategy Controls whether reads and writes
//! from slice storage are synchronous or asynchronous.
CORE_EXPORT
std::unique_ptr<CDataFrame>
makeMainStorageDataFrame(std::size_t numberColumns,
                         boost::optional<std::size_t> sliceCapacity = boost::none,
                         CDataFrame::EReadWriteToStorage readWriteToStoreSyncStrategy =
                             CDataFrame::EReadWriteToStorage::E_Sync);

//! Make a data frame which uses disk storage for its slices.
//!
//! \param[in] rootDirectory The name of the directory to which write the
//! data frame slices.
//! \param[in] numberColumns The number of columns in the data frame created.
//! \param[in] numberRows The number of rows that will be added.
//! \param[in] sliceCapacity If none null this overrides the default slice
//! capacity in rows.
//! \param[in] readWriteToStoreSyncStrategy Controls whether reads and writes
//! from slice storage are synchronous or asynchronous.
CORE_EXPORT
std::unique_ptr<CDataFrame>
makeDiskStorageDataFrame(const std::string& rootDirectory,
                         std::size_t numberColumns,
                         std::size_t numberRows,
                         boost::optional<std::size_t> sliceCapacity = boost::none,
                         CDataFrame::EReadWriteToStorage readWriteToStoreSyncStrategy =
                             CDataFrame::EReadWriteToStorage::E_Async);
}
}

#endif // INCLUDED_ml_core_CDataFrame_h
