#include "phy_graph/inference_queue.h"
#include <algorithm>

namespace phy_graph {

InferenceQueue::InferenceQueue(int num_workers, int max_queue_size)
    : num_workers_(num_workers)
    , max_queue_size_(max_queue_size)
    , running_(false)
    , processing_count_(0) {
}

InferenceQueue::~InferenceQueue() {
    stop();
}

void InferenceQueue::start(InferenceCallback inference_callback, ResultCallback result_callback) {
    if (running_) {
        ROS_WARN("InferenceQueue already running");
        return;
    }
    
    inference_callback_ = inference_callback;
    result_callback_ = result_callback;
    running_ = true;
    
    // 启动 worker 线程
    for (int i = 0; i < num_workers_; ++i) {
        workers_.emplace_back(&InferenceQueue::workerLoop, this);
    }
    
    ROS_INFO("InferenceQueue started with %d workers", num_workers_);
}

void InferenceQueue::stop() {
    if (!running_) return;
    
    running_ = false;
    queue_cv_.notify_all();
    
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
    
    ROS_INFO("InferenceQueue stopped");
}

bool InferenceQueue::enqueue(InferenceTask task) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    // 检查队列是否已满
    if (static_cast<int>(queue_.size()) >= max_queue_size_) {
        ROS_WARN_THROTTLE(5.0, "InferenceQueue full (%zu tasks), dropping task for %s",
                          queue_.size(), task.label.c_str());
        return false;
    }
    
    // 检查是否已在队列中
    if (queued_node_ids_.count(task.node_id)) {
        ROS_DEBUG("Node %lu already in queue, skipping", task.node_id);
        return false;
    }
    
    task.enqueue_time = ros::Time::now();
    queue_.push_back(std::move(task));
    std::push_heap(queue_.begin(), queue_.end());  // 维护最大堆
    queued_node_ids_.insert(task.node_id);
    
    queue_cv_.notify_one();
    return true;
}

bool InferenceQueue::updateTask(hydra::NodeId node_id, const cv::Mat& new_image, double new_score) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    for (auto& task : queue_) {
        if (task.node_id == node_id) {
            if (new_score > task.image_score) {
                task.image = new_image.clone();
                task.image_score = new_score;
                // 重新排序堆
                std::make_heap(queue_.begin(), queue_.end());
                ROS_DEBUG("Updated task for node %lu with better score %.1f -> %.1f",
                          node_id, task.image_score, new_score);
                return true;
            }
            return false;  // 新分数不够高
        }
    }
    return false;  // 未找到任务
}

bool InferenceQueue::contains(hydra::NodeId node_id) const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return queued_node_ids_.count(node_id) > 0;
}

size_t InferenceQueue::size() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return queue_.size();
}

size_t InferenceQueue::processingCount() const {
    return processing_count_.load();
}

void InferenceQueue::clear() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    queue_.clear();
    queued_node_ids_.clear();
}

void InferenceQueue::workerLoop() {
    while (running_) {
        InferenceTask task;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            // 等待任务或停止信号
            queue_cv_.wait(lock, [this] {
                return !running_ || !queue_.empty();
            });
            
            if (!running_ && queue_.empty()) {
                break;
            }
            
            if (queue_.empty()) {
                continue;
            }
            
            // 取出优先级最高的任务
            std::pop_heap(queue_.begin(), queue_.end());
            task = std::move(queue_.back());
            queue_.pop_back();
            queued_node_ids_.erase(task.node_id);
        }
        
        // 在锁外执行推理
        processing_count_++;
        
        try {
            InferenceResult result = inference_callback_(task);
            result_callback_(result);
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in inference callback: %s", e.what());
        }
        
        processing_count_--;
    }
}

} // namespace phy_graph

