#include "utils.hpp"



// --- --- --- --- --- ---
// Timing
// --- --- --- --- --- ---
namespace tempo::utils {

  /** Create a time point for "now" */
  time_point_t now() { return myclock_t::now(); }

  /** Print a duration in a human readable form (from nanoseconds to hours) in an output stream. */
  void printDuration(std::ostream& out, const duration_t& elapsed) {
    namespace c = std::chrono;
    auto execution_time_ns = c::duration_cast<c::nanoseconds>(elapsed).count();
    auto execution_time_us = c::duration_cast<c::microseconds>(elapsed).count();
    auto execution_time_ms = c::duration_cast<c::milliseconds>(elapsed).count();
    auto execution_time_sec = c::duration_cast<c::seconds>(elapsed).count();
    auto execution_time_min = c::duration_cast<c::minutes>(elapsed).count();
    auto execution_time_hour = c::duration_cast<c::hours>(elapsed).count();

    bool first = true;

    if (execution_time_hour>0) {
      first = false; // no need to test, if above condition is true, this is the first
      out << execution_time_hour << "h";
    }
    if (execution_time_min>0) {
      if (first) { first = false; } else { out << " "; }
      out << execution_time_min%60 << "m";
    }
    if (execution_time_sec>0) {
      if (first) { first = false; } else { out << " "; }
      out << "" << execution_time_sec%60 << "s";
    }
    if (execution_time_ms>0) {
      if (first) { first = false; } else { out << " "; }
      out << "" << execution_time_ms%long(1E+3) << "ms";
    }
    if (execution_time_us>0) {
      if (first) { first = false; } else { out << " "; }
      out << "" << execution_time_us%long(1E+3) << "us";
    }
    if (execution_time_ns>=0) {
      if (first) { first = false; } else { out << " "; }
      out << "" << execution_time_ns%long(1E+3) << "ns";
    }
  }

  /** Shortcut to print in a string */
  std::string as_string(const duration_t& elapsed) {
    std::stringstream ss;
    printDuration(ss, elapsed);
    return ss.str();
  }

  /** Shortcut for the above function, converting two time points into a duration. */
  void printExecutionTime(std::ostream& out, time_point_t start_time, time_point_t end_time) {
    const auto elapsed = end_time - start_time;
    printDuration(out, elapsed);
  }

  /** Shortcut to print in a string */
  std::string as_string(time_point_t start_time, time_point_t end_time) {
    std::stringstream ss;
    printExecutionTime(ss, start_time, end_time);
    return ss.str();
  }

}


// --- --- --- --- --- ---
// --- ParTasks
// --- --- --- --- --- ---
namespace tempo::utils {

  /// Non thread safe! Add all the task before calling "execute"
  void ParTasks::push_task(task_t func) { tasklist.push(std::move(func)); }

  /// Blocking call
  void ParTasks::execute(int nbthreads) {
    if (nbthreads<=1) {
      while (!tasklist.empty()) {
        auto task = std::move(tasklist.front());
        tasklist.pop();
        task();
      }
    } else {
      threads.reserve(nbthreads);
      for (int i = 0; i<nbthreads; ++i) { threads.emplace_back([this]() { run_thread(); }); }
      // Wait for all threads to stop
      for (auto& thread : threads) { thread.join(); }
      threads.clear();
    }
  }

  /// Blocking call
  void ParTasks::execute(int nbthreads, int nbtask) {
    if (nbthreads<=1) {
      while (!tasklist.empty()) {
        auto task = std::move(tasklist.front());
        tasklist.pop();
        task();
      }
    } else {
      threads.reserve(nbthreads);
      for (int i = 0; i<nbthreads; ++i) { threads.emplace_back([this, nbtask]() { run_thread(nbtask); }); }
      // Wait for all threads to stop
      for (auto& thread : threads) { thread.join(); }
      threads.clear();
    }
  }

  /// Blocking call using a task generator
  void ParTasks::execute(int nbthread, taskgen_t tgenerator) {
    // --- --- --- 1 thread
    if (nbthread<=1) {
      auto ntask = tgenerator();
      while (ntask.has_value()) {
        auto task = ntask.value();
        task();
        ntask = tgenerator();
      }
    }
      // --- --- --- Multi thread
    else {
      threads.reserve(nbthread);
      for (int i = 0; i<nbthread; ++i) {
        threads.emplace_back([this, &tgenerator]() { run_thread_generator(tgenerator); });
      }
      // Wait for all threads to stop
      for (auto& thread : threads) { thread.join(); }
      threads.clear();
    }
  }

  /// Blocking call using an incremental task with i [start, stop[
  void ParTasks::execute(int nbthread, itask_t itask, size_t start, size_t stop, size_t step) {
    // --- --- --- 1 thread
    if (nbthread<=1) {
      for (size_t i = start; i<stop; i += step) {
        itask(i);
      }
    }
      // --- --- --- Multi thread
    else {
      threads.reserve(nbthread);
      itask_idx = start;
      for (int i = 0; i<nbthread; ++i) {
        threads.emplace_back([this, &itask, stop, step]() { run_thread_itask(itask, stop, step); });
      }
      // Wait for all threads to stop
      for (auto& thread : threads) { thread.join(); }
      threads.clear();
    }
  }




  // --- --- ---  Private

  void ParTasks::run_thread() {
    mtx.lock();
    while (!tasklist.empty()) {
      auto task = std::move(tasklist.front());
      tasklist.pop();
      mtx.unlock();
      task();
      mtx.lock();
    }
    mtx.unlock();
  }

  void ParTasks::run_thread(size_t nbtask) {
    if (nbtask<=1) { run_thread(); }
    else {
      std::vector<task_t> tasks;
      tasks.reserve(nbtask);
      mtx.lock();
      while (!tasklist.empty()) {
        while (!tasklist.empty()&&tasks.size()<nbtask) {
          tasks.emplace_back(std::move(tasklist.front()));
          tasklist.pop();
        }
        mtx.unlock();
        for (auto& t : tasks) { t(); }
        tasks.clear();
        mtx.lock();
      }
      mtx.unlock();
    }
  }

  void ParTasks::run_thread_generator(taskgen_t& tgenerator) {
    mtx.lock();
    auto ntask = tgenerator();
    mtx.unlock();
    while (ntask.has_value()) {
      auto task = ntask.value();
      task();
      {
        std::lock_guard lg(mtx);
        ntask = tgenerator();
      }
    }
  }

  void ParTasks::run_thread_itask(itask_t& itask, size_t stop, size_t step) {
    // Lock block to compare/get/increment shared itask_idx
    mtx.lock();
    while (itask_idx<stop) {
      size_t myidx = itask_idx;
      itask_idx += step;
      mtx.unlock();
      // Execute the task
      itask(myidx);
      // Lock block re-start
      mtx.lock();
    }
    // Lock block re-end
    mtx.unlock();
  }

}



// --- --- --- --- --- ---
// --- Progress Monitor
// --- --- --- --- --- ---
namespace tempo::utils {

  ProgressMonitor::ProgressMonitor(size_t max) : total(max) {};

  // --- --- --- Print progress
  void ProgressMonitor::print_progress(std::ostream& out, size_t nbdone) {
    if (nbdone>0) {
      const size_t vprev = (nbdone - 1)*100/total;
      const size_t vnow = nbdone*100/total;
      const size_t vnow_tenth = vnow/10;
      const size_t vprev_tenth = vprev/10;
      if (vprev<vnow) {
        if (vprev_tenth<vnow_tenth) { out << vnow_tenth*10 << "% "; } else { out << "."; }
        std::flush(out);
      }
    }
  }

  void ProgressMonitor::print_progress(std::ostream* out, size_t nbdone) {
    if(out!=nullptr){ print_progress(*out, nbdone); }
  }

}