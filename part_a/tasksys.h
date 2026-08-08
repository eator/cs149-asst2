#ifndef _TASKSYS_H
#define _TASKSYS_H

#include "itasksys.h"
#include <thread>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <queue>
#include <future>
#include <type_traits>
#include <utility>
#include <functional>
#include <condition_variable>

/*
 * TaskSystemSerial: This class is the student's implementation of a
 * serial task execution engine.  See definition of ITaskSystem in
 * itasksys.h for documentation of the ITaskSystem interface.
 */
class TaskSystemSerial: public ITaskSystem {
    public:
        TaskSystemSerial(int num_threads);
        ~TaskSystemSerial();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

/*
 * TaskSystemParallelSpawn: This class is the student's implementation of a
 * parallel task execution engine that spawns threads in every run()
 * call.  See definition of ITaskSystem in itasksys.h for documentation
 * of the ITaskSystem interface.
 */
class TaskSystemParallelSpawn: public ITaskSystem {
    public:
        TaskSystemParallelSpawn(int num_threads);
        ~TaskSystemParallelSpawn();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
    private:
        int thread_num_;
};

/*
 * TaskSystemParallelThreadPoolSpinning: This class is the student's
 * implementation of a parallel task execution engine that uses a
 * thread pool. See definition of ITaskSystem in itasksys.h for
 * documentation of the ITaskSystem interface.
 */
class TaskSystemParallelThreadPoolSpinning: public ITaskSystem {
    public:
        TaskSystemParallelThreadPoolSpinning(int num_threads);
        ~TaskSystemParallelThreadPoolSpinning();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
        void worker_loop();
    private:
        std::vector<std::thread> workers_;
        std::mutex queue_mtx_;
        std::queue<std::tuple<int, int, IRunnable*>> tasks_;
        bool stop_{false};
        std::atomic<int> remain_tasks_{0};
};

/*
 * TaskSystemParallelThreadPoolSleeping: This class is the student's
 * optimized implementation of a parallel task execution engine that uses
 * a thread pool. See definition of ITaskSystem in
 * itasksys.h for documentation of the ITaskSystem interface.
 */
class TaskSystemParallelThreadPoolSleeping: public ITaskSystem {
    public:
        TaskSystemParallelThreadPoolSleeping(int num_threads);
        ~TaskSystemParallelThreadPoolSleeping();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
        
        template <typename F, typename ...Args>
        auto submit(F&& f, Args... args) -> std::future<typename std::result_of<F(Args...)>::type>
        {
            using RetType = typename std::result_of<F(Args...)>::type;

            auto task = std::make_shared<std::packaged_task<RetType()>>(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...)
            );
            auto res = task->get_future();

            {
                std::lock_guard<std::mutex> lock(queue_mtx_);

                if (stop_) {
                    throw std::runtime_error("submit on a stopped thread pool");
                }

                tasks_.emplace([task]() { (*task)(); });
                remain_tasks_++;
            }

            cv_.notify_one();

            return res;
        }

        void worker_loop() {
            while (true) {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(queue_mtx_);
                    cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

                    if (stop_ && tasks_.empty()) {
                        return;
                    }

                    task = std::move(tasks_.front());
                    tasks_.pop();
                }

                task();

                {
                    std::lock_guard<std::mutex>  lock(queue_mtx_);
                    remain_tasks_--;

                    if (remain_tasks_ == 0) {
                        done_cv_.notify_all();
                    }
                }
            }
        }
    private:
        std::vector<std::thread>            workers_;
        std::queue<std::function<void()>>   tasks_; 

        std::mutex                          queue_mtx_;
        std::mutex                          shutdown_mtx_;
        std::condition_variable             cv_;
        std::condition_variable             done_cv_;

        int remain_tasks_{0};
        bool stop_{false};
};

#endif
