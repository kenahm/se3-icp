#pragma once

#include <iostream>
#include <vector>

#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <pcl/registration/correspondence_rejection_trimmed.h>

///////// To test with PCL LRF /////// 
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
//////////////////////////////////////

#include "open3d/Open3D.h"

#include <chrono>
#include <thread>
#include <omp.h>

double largestDistanceFromGivenPoint(const Eigen::Vector3d& ref_point, const open3d::geometry::PointCloud& cloud);

struct CorrespondencesSet {
    std::vector<Eigen::Vector2i> correspondences_vec;
    std::vector<double> distances_vec;
};

class IterativeSE3Registration {
public:
	IterativeSE3Registration();

	void setSourceCloud(const std::string& filename);
	void setTargetCloud(const std::string& filename);
    void setSourceCloud(const open3d::geometry::PointCloud& cloud);
    void setTargetCloud(const open3d::geometry::PointCloud& cloud);

	void update_correspondences_kd_tree_XYZ(const open3d::geometry::KDTreeFlann& target_kd_tree);
	void update_correspondences_raw_flann_SE3(); // raw NanoFlann
	void update_correspondences_raw_flann_SE3(const open3d::geometry::KDTreeFlann& se3_tree, const std::vector<Eigen::Matrix4d>& cloud_vector);

	double estimate_current_mse(const pcl::Correspondences pcl_corrs);
	double estimate_current_mse_compute_euclidean(const open3d::geometry::PointCloud& cloud_src, 
												  const open3d::geometry::PointCloud& cloud_tgt, 
												  const pcl::Correspondences pcl_corrs);

	// main align methods
	void run_icp(const std::string &variant_name); 	   // vanilla variants: "pt2pt", "pt2pl", "gicp" 
	void run_se3_icp(const std::string &variant_name); // proposed SE(3) variants: "pt2pt", "pt2pl", "gicp"
	void run_se3_icp_with_cf(); // proposed SE(3) variant with sensor uncertainty used for stanford lounge dataset

	void run_se3_pure(const std::string &variant_name); // pure SE(3) registration, without reducing to alpha=0.0, i.e. without switching to standard ICP

	//////////////////////////////////////////////////////////
	// Open3D point clouds (all XYZ)
    open3d::geometry::PointCloud source_;
    open3d::geometry::PointCloud source_moving_;
    open3d::geometry::PointCloud target_;

	// SE(3) point clouds
	std::vector<Eigen::Matrix4d> source_se3_cloud_;
	std::vector<Eigen::Matrix4d> target_se3_cloud_;

	// Estimated history
	std::vector<Eigen::Matrix4d> estimated_history_;

	// varius kd trees
	open3d::geometry::KDTreeFlann kd_tree_target_XYZ;
	open3d::geometry::KDTreeFlann kd_tree_source_XYZ;
	open3d::geometry::KDTreeFlann raw_flann_kd_tree_target_SE3; // raw NanoFlann
	//////////////////////////////////////////////////////////

    // in eigen this is typdef-ed into 'CorrespondenceSet'
    // this vector is passed into transformation estimation functions
    // std::vector<Eigen::Vector2i> current_estimated_correspondences_;
    CorrespondencesSet current_correspondences_set;
	pcl::CorrespondencesPtr current_correspondences_set_pcl; // pretty ugly but useful
    open3d::pipelines::registration::TransformationEstimationPointToPoint o3d_estimator;
	open3d::pipelines::registration::TransformationEstimationPointToPlane o3d_estimator_po2pl;
	open3d::pipelines::registration::TransformationEstimationForGeneralizedICP o3d_estimator_generalized;

	int number_of_nn_for_LRF_;
	double mse_;
	double estimated_overlap_;
	double lrf_radius_;
	double mse_switch_error_;
	double time_before_pure_icp_;
	double time_se3_correspondence_search_;

	double alpha_rot;
	double beta_transl;
	double scale_preprocessing;

	int num_iterations_;
	int max_num_iterations_;
	int max_num_se3_iterations_;
	int num_pure_se3_iterations_;

	// RESULT of registration is stored here; after executing a "run_*" method
	Eigen::Matrix4d current_estimated_T_; // used in algos but also holds final result
};
