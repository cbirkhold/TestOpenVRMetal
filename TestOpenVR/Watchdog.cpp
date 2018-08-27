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

#include "Watchdog.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <future>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

    //------------------------------------------------------------------------------
    // Utility to use std::unique_lock with an external mutex that meets the
    // Lockable requirements.
    //------------------------------------------------------------------------------

    class MutexProxy
    {
    public:

        void (*m_lock)() = []() {};
        void (*m_unlock)() = []() {};

        void lock() { m_lock(); }
        void unlock() { m_unlock(); };
    };

    MutexProxy cerr_mutex_proxy;
    MutexProxy cout_mutex_proxy;

    //------------------------------------------------------------------------------
    // The watchdog thread keeps track of watchdog marker expiration. A marker is
    // created by calling marker() and reset either explicitly by calling
    // marker_reset() or when the next marker is created. Markers that
    // expire before being rest are logged.
    //------------------------------------------------------------------------------

    class Thread
    {
    public:

        typedef std::chrono::milliseconds milliseconds;
        typedef std::chrono::steady_clock steady_clock;
        typedef std::chrono::time_point<steady_clock> time_point;

    public:

        Thread()
        {
            start_thread();
        }

        ~Thread()
        {
            terminate_thread();
        }

    public:

        Watchdog::MarkerResult_e marker(const char* const name, size_t timeout)
        {
            Watchdog::MarkerResult_e result = Watchdog::MARKER_RESULT_PREVIOUS_MARKER_OK;

            //------------------------------------------------------------------------------
            // Get the timestamp used for checking the previous marker as early as possible
            // to reduce the influence of the cost of this function.
            const time_point now = steady_clock::now();

            {
                std::unique_lock<std::mutex> lock(m_mutex);

                //------------------------------------------------------------------------------
                // Check if previous marker has expired.
                if (m_marker_timeout < now) {
                    double overrun = std::chrono::duration<double, std::milli>(now - m_marker_timeout).count();
                    const char* overrun_unit = "ms";

                    if (overrun >= 1000.0) {
                        overrun /= 1000.0;
                        overrun_unit = "s";
                    }
                    else if (overrun < 1.0) {
                        overrun *= 1000.0;
                        overrun_unit = "ys";
                    }

                    {
                        std::unique_lock<MutexProxy> lock(cerr_mutex_proxy);
                        std::cerr << "Warning: " << m_marker_name << " (" << m_marker_sequence << "): expired " << overrun << " [" << overrun_unit << "] ago!" << std::endl;
                    }

                    result = Watchdog::MARKER_RESULT_PREVIOUS_MARKER_EXPIRED;
                }

                //------------------------------------------------------------------------------
                // Set new marker with updated 'now' to reduce the influence of the cost of
                // this function as much as possible.
                ++m_marker_sequence;
                m_marker_name = (name ? name : "<nullptr>");
                m_marker_timeout = (timeout == 0 ? time_point::max() : (steady_clock::now() + milliseconds(timeout)));
            }

            //------------------------------------------------------------------------------
            // Wake the thread.
            m_event.notify_one();

            //------------------------------------------------------------------------------
            // ...
            return result;
        }

    private:

        //------------------------------------------------------------------------------
        // Start the watchdog thread and wait until it is running.
        void start_thread()
        {
            std::promise<void> thread_started_event;
            std::future<void> thread_started_result = thread_started_event.get_future();

            //------------------------------------------------------------------------------
            // Name the thread.
            uint64_t thread_id = 0;
            pthread_threadid_np(pthread_self(), &thread_id);

            const char thread_name_format[] = "Watchdog (%ju)";
            char thread_name[sizeof(thread_name_format) + std::numeric_limits<uintmax_t>::digits10];
            const int thread_name_length = sprintf(thread_name, thread_name_format, uintmax_t(thread_id));
            assert(thread_name_length < sizeof(thread_name));

            //------------------------------------------------------------------------------
            // Start the thread.
            m_thread = std::async(std::launch::async, [this, &thread_started_event, &thread_name]() {
                pthread_setname_np(thread_name);

                {
                    std::unique_lock<MutexProxy> lock(cout_mutex_proxy);
                    std::cout << "Info: Watchdog thread is running" << '\n';
                }

                thread_started_event.set_value();

                std::unique_lock<std::mutex> lock(m_mutex);      // Released during each wait below.

                assert(m_marker_sequence == 0);
                assert(m_marker_timeout == time_point::max());

                decltype(m_marker_timeout) timeout = time_point::max();

                while (not m_terminate_thread) {
                    const decltype(m_marker_sequence) sequence = m_marker_sequence;

                    //------------------------------------------------------------------------------
                    // Wait for the current marker to expire or an event triggered by a new marker
                    // (sequence changed) or an exit request.
                    std::cv_status status = m_event.wait_until(lock, timeout);

                    //------------------------------------------------------------------------------
                    // In case of marker expiration, if the sequence has changed the expiration of
                    // the marker has already been reported in marker().
                    if ((status == std::cv_status::timeout) && (m_marker_sequence == sequence)) {
                        {
                            std::unique_lock<MutexProxy> lock(cerr_mutex_proxy);
                            std::cerr << "Warning: " << m_marker_name << " (" << sequence << "): expired!" << std::endl;
                        }

                        timeout = time_point::max();       // Wait indefinitely for a new marker
                    }
                    else {
                        timeout = m_marker_timeout;
                    }
                }

                {
                    std::unique_lock<MutexProxy> lock(cout_mutex_proxy);
                    std::cout << "Info: Watchdog thread has terminated" << '\n';
                }
            });

            thread_started_result.get();
        }

        //------------------------------------------------------------------------------
        // Request the watchdog thread to exit and wait until it is terminated.
        void terminate_thread()
        {
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_terminate_thread = true;
            }

            m_event.notify_one();
            m_thread.get();
        }

    private:

        std::future<void>           m_thread;
        bool                        m_terminate_thread = false;

        std::mutex                  m_mutex;
        std::condition_variable     m_event;

        size_t                      m_marker_sequence = 0;
        const char*                 m_marker_name = nullptr;
        time_point                  m_marker_timeout = time_point::max();
    };

    thread_local std::unique_ptr<Thread> thread;

} // unnamed namespace;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
Watchdog::set_cerr_mutex(void (*lock)(), void (*unlock)())
{
    assert(lock && unlock);
    cerr_mutex_proxy.m_lock = lock;
    cerr_mutex_proxy.m_unlock = unlock;
}

void
Watchdog::set_cout_mutex(void (*lock)(), void (*unlock)())
{
    assert(lock && unlock);
    cout_mutex_proxy.m_lock = lock;
    cout_mutex_proxy.m_unlock = unlock;
}

Watchdog::MarkerResult_e
Watchdog::marker(const char* const name, size_t duration)
{
    if (not thread) {
        thread.reset(new Thread());
    }

    return thread->marker(name, duration);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
