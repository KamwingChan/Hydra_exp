#pragma once

#include <ros/ros.h>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <unordered_set>

#include <hydra/common/dsg_types.h>
#include <kimera_pgmo_msgs/KimeraPgmoMesh.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>

namespace phy_graph {

/**
 * @brief 推理任务结构体
 * 
 * 包含待推理物体的所有必要信息
 */
struct InferenceTask {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    hydra::NodeId node_id;
    std::string label;
    cv::Mat image;
    double image_score;
    uint64_t last_update_ns;
    ros::Time enqueue_time;
    int defer_count;
    
    // 优先级比较（分数越高优先级越高）
    bool operator<(const InferenceTask& other) const {
        return image_score < other.image_score;
    }
};

/**
 * @brief 推理结果结构体
 */
struct InferenceResult {
    hydra::NodeId node_id;
    std::string label;
    std::string description;
    int friction_level;
    bool pushable;
    int weight_level;
    std::string estimated_weight_kg;
    double image_score;
    double processing_time_ms;
    bool success;
};

/**
 * @brief 异步推理队列
 * 
 * 管理待推理物体的队列，支持：
 * - 优先级排序（高分图像优先）
 * - 异步处理（不阻塞主循环）
 * - 可配置的 worker 数量
 */
class InferenceQueue {
public:
    using InferenceCallback = std::function<InferenceResult(const InferenceTask&)>;
    using ResultCallback = std::function<void(const InferenceResult&)>;

    /**
     * @brief 构造函数
     * 
     * @param num_workers worker 数量（默认 1）
     * @param max_queue_size 队列最大长度
     */
    InferenceQueue(int num_workers = 1, int max_queue_size = 100);
    
    ~InferenceQueue();
    
    /**
     * @brief 启动 worker 线程
     * 
     * @param inference_callback 执行推理的回调函数
     * @param result_callback 结果回调函数
     */
    void start(InferenceCallback inference_callback, ResultCallback result_callback);
    
    /**
     * @brief 停止所有 worker 线程
     */
    void stop();
    
    /**
     * @brief 添加任务到队列
     * 
     * @param task 推理任务
     * @return true 如果成功入队
     */
    bool enqueue(InferenceTask task);
    
    /**
     * @brief 更新现有任务（如果找到更好的图像）
     * 
     * @param node_id 节点 ID
     * @param new_image 新图像
     * @param new_score 新分数
     * @return true 如果更新成功
     */
    bool updateTask(hydra::NodeId node_id, const cv::Mat& new_image, double new_score);
    
    /**
     * @brief 检查节点是否在队列中
     */
    bool contains(hydra::NodeId node_id) const;
    
    /**
     * @brief 获取队列大小
     */
    size_t size() const;
    
    /**
     * @brief 获取正在处理的任务数
     */
    size_t processingCount() const;
    
    /**
     * @brief 清空队列
     */
    void clear();

private:
    void workerLoop();
    
    int num_workers_;
    int max_queue_size_;
    
    // 优先级队列（使用 vector + 堆操作）
    std::vector<InferenceTask> queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // 跟踪队列中的节点 ID
    std::unordered_set<uint64_t> queued_node_ids_;
    
    // Worker 线程
    std::vector<std::thread> workers_;
    std::atomic<bool> running_;
    std::atomic<size_t> processing_count_;
    
    InferenceCallback inference_callback_;
    ResultCallback result_callback_;
};

} // namespace phy_graph

