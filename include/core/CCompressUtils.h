/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CCompressUtils_h
#define INCLUDED_ml_core_CCompressUtils_h

#include <core/CNonCopyable.h>
#include <core/ImportExport.h>

#include <zlib.h>

#include <string>
#include <vector>

namespace ml {
namespace core {

//! \brief
//! Shrink wrap zlib calls.
//!
//! DESCRIPTION:\n
//! Shrink wrap zlib calls.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Data can be added incrementally and then 'finished' to
//! complete deflation or inflation.
//!
//! This object retains in memory the entire compressed state
//! so it not good for file read/write.
//!
//! A single Z stream is used for the lifetime of the object,
//! so each object can only work on one task at a time.  In
//! a multi-threaded application it would be best to create
//! one object for each thread.
//!
class CORE_EXPORT CCompressUtils : private CNonCopyable {
public:
    using TByteVec = std::vector<Bytef>;
    enum EOperation { E_Deflate, E_Inflate };

public:
    CCompressUtils(EOperation operation, bool lengthOnly, int level = Z_DEFAULT_COMPRESSION);
    ~CCompressUtils();

    //! Add a string.
    //!
    //! \note Multiple calls to this function without finishing
    //! are equivalent to deflating or inflating the concatenation
    //! of the strings passed in the order they are passed.
    bool addString(const std::string& input);

    //! Add a vector of trivially copyable types.
    //!
    //! \note Multiple calls to this function without finishing
    //! are equivalent to deflating or inflating the concatenation
    //! of the vectors passed in the order they are passed.
    template<typename T>
    bool addVector(const std::vector<T>& input) {
        static_assert(std::is_trivially_copyable<T>::value, "Type must be trivially copyable");
        if (m_State == E_Finished) {
            // If the last round of data processing has finished
            // and we're adding a new vector then we need to reset
            // the stream so that a new round starts from scratch.
            this->reset();
        }
        return this->processInput(false, input);
    }

    //! Get transformed representation.
    //!
    //! \warning This will fail if the lengthOnly constructor argument
    //!  was set to true.
    //!
    //! \note The output representation is a byte array NOT a string,
    //! and hence not printable.
    //!
    //! If finish==false then retrieve partial state.
    bool data(bool finish, TByteVec& result);

    //! Get transformed data length.
    //!
    //! If finish==false then retrieve partial length.
    bool length(bool finish, size_t& length);

    //! Reset the underlying stream.  This will happen automatically
    //! when adding a new string after having finished the previous
    //! round, but sometimes, for example when recovering from an
    //! error, it may be desirable to explicitly reset the state.
    void reset();

private:
    enum EState { E_Unused, E_Active, E_Finished };

private:
    static const size_t CHUNK_SIZE{4096};

private:
    //! Get an unsigned character pointer to the address of the start
    //! of the vector data.
    template<typename T>
    static Bytef* bytes(const std::vector<T>& input) {
        return const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input.data()));
    }

    //! Get an unsigned character pointer to the address of the start
    //! of the string character array.
    static Bytef* bytes(const std::string& input) {
        return reinterpret_cast<Bytef*>(const_cast<char*>(input.data()));
    }

    //! Get the vector data size in bytes.
    template<typename T>
    static uInt size(const std::vector<T>& input) {
        return static_cast<uInt>(input.size() * sizeof(T));
    }

    //! Get the string size in bytes.
    static uInt size(const std::string& input) {
        return static_cast<uInt>(input.size());
    }

    //! Process a chunk of state (optionally flushing).
    bool processChunk(int flush);

    //! Process the input \p input in chunks.
    template<typename T>
    bool processInput(bool finish, const T& input) {
        if (input.empty() && m_State == E_Active && !finish) {
            return true;
        }

        m_State = E_Active;

        m_ZlibStrm.next_in = bytes(input);
        m_ZlibStrm.avail_in = size(input);

        int flush(finish ? Z_FINISH : Z_NO_FLUSH);
        do {
            if (this->processChunk(flush) == false) {
                return false;
            }
        } while (m_ZlibStrm.avail_out == 0);

        m_State = finish ? E_Finished : E_Active;

        return true;
    }

private:
    //! The current state of deflation or inflation.
    EState m_State;

    //! The mode of operation i.e. deflation or inflation.
    EOperation m_Operation;

    //! Is this object only fit for getting output lengths?
    bool m_LengthOnly;

    //! The buffer for a chunk of output from (de|in)flation.
    Bytef m_Chunk[CHUNK_SIZE];

    //! The output buffer when the compressed result is being
    //! stored.
    TByteVec m_FullResult;

    //! The zlib data structure.
    z_stream m_ZlibStrm;
};
}
}

#endif // INCLUDED_ml_core_CCompressUtils_h
