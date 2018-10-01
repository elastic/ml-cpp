/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_core_CDataFrameRowSlice_h
#define INCLUDED_ml_core_CDataFrameRowSlice_h

#include <core/CFloatStorage.h>
#include <core/CompressUtils.h>
#include <core/ImportExport.h>

#include <boost/filesystem.hpp>

#include <string>
#include <vector>

namespace ml {
namespace core {
class CDataFrame;

namespace data_frame_row_slice_detail {
//! \brief The implementation backing a data frame row slice handle.
class CORE_EXPORT CDataFrameRowSliceHandleImpl {
public:
    using TFloatVec = std::vector<CFloatStorage>;
    using TImplPtr = std::unique_ptr<CDataFrameRowSliceHandleImpl>;

public:
    virtual ~CDataFrameRowSliceHandleImpl() = default;
    virtual TImplPtr clone() const = 0;
    virtual const TFloatVec& values() const = 0;
    virtual bool bad() const = 0;
};
}

//! \brief A handle which can be used to read values from a slice of
//! CDataFrame storage.
class CORE_EXPORT CDataFrameRowSliceHandle {
public:
    using TFloatVec = std::vector<CFloatStorage>;
    using TFloatVecCItr = TFloatVec::const_iterator;
    using TImplPtr = std::unique_ptr<data_frame_row_slice_detail::CDataFrameRowSliceHandleImpl>;

public:
    CDataFrameRowSliceHandle() = default;
    CDataFrameRowSliceHandle(TImplPtr impl);
    CDataFrameRowSliceHandle(const CDataFrameRowSliceHandle& other);
    CDataFrameRowSliceHandle(CDataFrameRowSliceHandle&& other);

    CDataFrameRowSliceHandle& operator=(const CDataFrameRowSliceHandle& other);
    CDataFrameRowSliceHandle& operator=(CDataFrameRowSliceHandle&& other);

    std::size_t size() const;
    TFloatVecCItr begin() const;
    TFloatVecCItr end() const;
    bool bad() const;

private:
    TImplPtr m_Impl;
};

//! \brief CDataFrame slice storage interface.
class CORE_EXPORT CDataFrameRowSlice {
public:
    using TFloatVec = std::vector<CFloatStorage>;
    using TSizeHandlePr = std::pair<std::size_t, CDataFrameRowSliceHandle>;

public:
    virtual ~CDataFrameRowSlice() = default;
    virtual TSizeHandlePr read() const = 0;
    virtual std::size_t staticSize() const = 0;
    virtual std::size_t memoryUsage() const = 0;
};

//! \brief In main memory CDataFrame slice storage.
//!
//! DESCRIPTION:\n
//! This provides maximum speed at the expense of maximum main memory
//! usage. The intention is to provide a speed optimized implementation
//! suitable for small data frames which comfortably fit into memory.
//!
//! IMPLEMENTATION:\n
//! This is basically a wrapper around a single std::vector storing all
//! rows to adapt it for use by the data frame.
class CORE_EXPORT CMainMemoryDataFrameRowSlice final : public CDataFrameRowSlice {
public:
    CMainMemoryDataFrameRowSlice(std::size_t firstRow, TFloatVec state);
    virtual TSizeHandlePr read() const;
    virtual std::size_t staticSize() const;
    virtual std::size_t memoryUsage() const;

private:
    std::size_t m_FirstRow;
    TFloatVec m_State;
};

//! \brief On disk CDataFrame slice storage.
//!
//! DESCRIPTION:\n
//! This writes the data in binary format to a single file. As such, there is
//! essentially no main memory usage per slice. The intention is that the data
//! the corresponding frame can be linearly scanned efficiently and specific
//! rows can be copied to main memory during the running of some ML task over
//! the data frame. It is the task responsibility to decide how best to do this
//! whilst meeting some specified memory constraint.
//!
//! IMPLEMENTATION:\n
//! A temporary directory is created to hold the data frame files. It is the
//! calling code's responsibility to provide a suitable path for this.
//!
//! We have one directory for all slices and one file per slice. These are deleted
//! in one go when the data frame object is destroyed. The cleanup is performed
//! in the destructor of CTemporaryDirectory. There is one such object per data
//! frame object whose ownership is shared between all slices and the data frame
//! itself.
//!
//! The slices are stored in binary format to maximize the read and write speed.
//! We cache the number of bytes so we can read the file directory into a pre
//! allocated vector. Note that these files are intended to be short lived and
//! stay on the machine (or in the container) where the analysis action is being
//! performed. So we have no architecture related issues with interpreting the
//! stored bytes as floating point values.
class CORE_EXPORT COnDiskDataFrameRowSlice final : public CDataFrameRowSlice {
public:
    //! \brief Manages the resource associated with the temporary directory
    //! which contains all the slices of a single data frame.
    class CORE_EXPORT CTemporaryDirectory {
    public:
        CTemporaryDirectory(const std::string& name, std::size_t minimumSpace);
        ~CTemporaryDirectory();
        const std::string& name() const;
        bool bad() const;

    private:
        bool m_StateIsBad = false;
        boost::filesystem::path m_Name;
    };

    using TTemporaryDirectoryPtr = std::shared_ptr<CTemporaryDirectory>;

public:
    COnDiskDataFrameRowSlice(const TTemporaryDirectoryPtr& directory,
                             std::size_t firstRow,
                             TFloatVec state);
    virtual TSizeHandlePr read() const;
    virtual std::size_t staticSize() const;
    virtual std::size_t memoryUsage() const;

private:
    using TByteVec = CCompressUtil::TByteVec;

private:
    mutable bool m_StateIsBad = false;
    std::size_t m_FirstRow;
    std::size_t m_NumberRows;
    TTemporaryDirectoryPtr m_Directory;
    boost::filesystem::path m_FileName;
    uint64_t m_Checksum;
};
}
}

#endif // INCLUDED_ml_core_CDataFrameRowSlice_h
