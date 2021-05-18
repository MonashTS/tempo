#pragma once

#include <cassert>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>

namespace tempo {

  /** Helper class to execute several tasks in parallel.
   *  Tasks must be prepared (with push_task) before being executed.
   *  The 'execute' method waits for all task to be completed.
   *  If the number of thread required is <= 1, the current thread is used.
   *  Else, the requested number of threads are spawned, and the current thread waits for their completion.
   */
  class ParTasks {
  public:

    using task_t = std::function<void()>;
    using taskgen_t = std::function<std::optional<task_t>()>;

    ParTasks() = default;

    /// Non thread safe! Add all the task before calling "execute"
    void push_task(task_t func) {
      tasklist.emplace(std::move(func));
    }

    /// Template version
    template< class F, class... Args >
    void push_task( F&& f, Args&&... args ){
      tasklist.emplace(std::move(std::bind(f, args...)));
    }

    /// Blocking call
    void execute(int nbthreads) {
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

    /// Blocking call using a task generator
    void execute(int nbthread, taskgen_t tgenerator){
      // --- --- --- 1 thread
      if(nbthread<=1){
        auto ntask = tgenerator();
        while(ntask.has_value()){
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

  private:

    std::mutex mtx;
    std::vector<std::thread> threads;
    std::queue<task_t> tasklist;

    void run_thread() {
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

    void run_thread_generator(taskgen_t& tgenerator) {
      mtx.lock();
      auto ntask = tgenerator();
      mtx.unlock();
      while(ntask.has_value()){
        auto task = ntask.value();
        task();
        { std::lock_guard lg(mtx); ntask = tgenerator(); }
      }
    }

  };

} // End of namespace tempo

