/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#ifndef INCLUDED_ml_core_CHexUtils_h
#define INCLUDED_ml_core_CHexUtils_h

#include <core/ImportExport.h>

#include <iosfwd>
#include <vector>

#include <stdint.h>

namespace ml {
namespace core {

//! \brief
//! Print out binary data in hex format.
//!
//! DESCRIPTION:\n
//! Binary data is printed out in hex format, optionally with a
//! header stating the data length and an ASCII representation
//! (for printable characters only) by the side.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Can be used with STL streams, or with a simple static dump()
//! function.
//!
class CORE_EXPORT CHexUtils {
public:
    typedef std::vector<uint8_t> TDataVec;

public:
    //! Construct an object of this class, which can then be output to a
    //! stream - only a shallow copy is done, so the data must exist for
    //! the lifetime of the object
    CHexUtils(const uint8_t* pkt, size_t pktLen, bool printHeader = true, bool printAscii = true);
    CHexUtils(const TDataVec& data, bool printHeader = true, bool printAscii = true);

    //! Dump a packet of given length to stdout
    static void dump(const uint8_t* pkt, size_t pktLen);

private:
    //! Pointer to raw data - we don't own this
    const uint8_t* m_Pkt;

    //! Packet length
    size_t m_PktLen;

    //! Should we print a header?
    bool m_PrintHeader;

    //! Should we the raw ASCII (where possible) next to the hex?
    bool m_PrintAscii;

    friend CORE_EXPORT std::ostream& operator<<(std::ostream&, const CHexUtils&);
};

CORE_EXPORT std::ostream& operator<<(std::ostream& strm, const CHexUtils& hex);
}
}

#endif // INCLUDED_ml_core_CHexUtils_h
