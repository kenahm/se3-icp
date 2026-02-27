#pragma once
#include "open3d/Open3D.h"
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <fstream>

// some common computations for the experiments, e.g., rotation matrix generation, error metrics, etc.
namespace cc {

Eigen::Matrix3d rot_3d(double roll_, double pitch_, double yaw_);

double angularErrorSO3 (const Eigen::Matrix3d& rot_mat_1, const Eigen::Matrix3d& rot_mat_2);

double angularErrorSO3_alt (const Eigen::Matrix3d& rot_mat_1, const Eigen::Matrix3d& rot_mat_2);

double evaluate_LRF_quality(const std::vector<Eigen::Matrix4d>& source_SE3, const std::vector<Eigen::Matrix4d>& target_SE3 ,
                            const Eigen::Matrix4d& map_gt, const std::vector<std::pair<int,int>>& corr_pairs);

double evaluate_LRF_quality(const std::vector<Eigen::Matrix4d>& source_SE3, const std::vector<Eigen::Matrix4d>& target_SE3 ,
                            const Eigen::Matrix4d& map_gt, const std::vector<std::pair<int,int>>& corr_pairs, std::string path);

std::vector<std::pair<int,int>> compute_corrs_with_gt(const open3d::geometry::PointCloud& src, 
                                                      const open3d::geometry::PointCloud& tgt,
                                                      const Eigen::Matrix4d rigid_map_gt);

std::vector<Eigen::Matrix4d> read_trajectory(std::string traj_path);

double error_filterreg(const open3d::geometry::PointCloud& src, Eigen::Matrix4d T_gt, Eigen::Matrix4d T_est);

void evaluate_trajectory_quality(std::string gt_traj_path, std::string est_traj_path);

std::vector<std::pair<int,int>> compute_nearest_neighbor_correspondences(const open3d::geometry::PointCloud& source, 
                                                                         const open3d::geometry::PointCloud& target,
                                                                         const open3d::geometry::KDTreeFlann& kd_tree_target);

}; // cc namespace (common computations)
