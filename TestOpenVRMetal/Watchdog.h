//
// MIT License
//
// Copyright (c) 2017 Chris Birkhold
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __WATCHDOG_H__
#define __WATCHDOG_H__

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stddef.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//------------------------------------------------------------------------------
// Watchdog is a utility to keep track of the timing of synchronous operations
// that manage asynchronous activity internally.
//
// A watchdog thread is created as needed internally for each thread from which
// markers are created.
//------------------------------------------------------------------------------

class Watchdog
{
public:

    enum MarkerResult_e {
        MARKER_RESULT_PREVIOUS_MARKER_OK,
        MARKER_RESULT_PREVIOUS_MARKER_EXPIRED,
    };

    //------------------------------------------------------------------------------
    // Use a mutex to protect output to cerr. Must be called before any other
    // watchdog function (other than set_cout_mutex).
    static void set_cerr_mutex(void (*lock)(), void (*unlock)());

    //------------------------------------------------------------------------------
    // Use a mutex to protect output to cout. Must be called before any other
    // watchdog function (other than set_cerr_mutex).
    static void set_cout_mutex(void (*lock)(), void (*unlock)());

    //------------------------------------------------------------------------------
    // Create a marker with the given name and timeout (ms). The previous marker is
    // implicitly reset. A timeout of zero creates a marker that never (in
    // reasonable time) expires.
    static MarkerResult_e marker(const char* const name, size_t timeout);

    //------------------------------------------------------------------------------
    // Explicitly reset the previous marker.
    static MarkerResult_e reset_marker() { return marker("<reset>", 0); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // __WATCHDOG_H__

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
