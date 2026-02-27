#include <iostream>
#include <fstream>

#include <cc.hpp>
#include <iterative_SE3_registration.hpp>

int main() {
    std::cout << "Registration usage example\n";

    // Load a point cloud 
    auto source = open3d::io::CreatePointCloudFromFile("../stanford_bunny.ply");

    std::cout << "source size = " << source->points_.size() << std::endl;
    
    // Random downsample
    open3d::utility::random::Seed(1);
    auto source_ds = source->RandomDownSample(0.02);
    // auto source_ds = source->VoxelDownSample(0.0042);

    std::cout << "source_ds size = " << source_ds->points_.size() << std::endl;

    // Create a ground truth rotation and translation
    Eigen::Vector3d translation_gt {2.5, 3.5, 1.2};
    Eigen::Matrix3d rotation_gt = cc::rot_3d(M_PI/5, M_PI/4, M_PI/6);
    Eigen::Matrix4d transformation_gt = Eigen::Matrix4d::Identity();
    transformation_gt.block<3, 3>(0, 0) = rotation_gt; 
    transformation_gt.block<3, 1>(0, 3) = translation_gt;

    // Construct target point cloud
    auto target = std::make_shared<open3d::geometry::PointCloud>(*source);
    target->Transform(transformation_gt);
    auto target_ds = target->RandomDownSample(0.02); // random downsample independently from source, which has a bit of effect as if we added noise

    // Setup the registration object
    IterativeSE3Registration reg_obj;
    reg_obj.setSourceCloud(*source_ds);
    reg_obj.setTargetCloud(*target_ds);         
    reg_obj.estimated_overlap_ = 1.0; 
    reg_obj.max_num_se3_iterations_ = 10; // max num of iterations using SE(3) distance, before switching to standard R3 distance
    reg_obj.mse_ = 0.00001;
    reg_obj.mse_switch_error_ = 5 * reg_obj.mse_; // another criteria to switch to standard R3 distance
    reg_obj.number_of_nn_for_LRF_ = 90; // 90 nearest neighbors will be used to compute TOLDI local reference frames

    // Run the main SE(3)-ICP registration algorithm
    reg_obj.run_se3_icp("pt2pl"); // options: pt2pt, pt2pl, gicp

    // Alternativelly use a vanilla ICP variant
    // reg_obj.run_icp("pt2pl"); // options: pt2pt, pt2pl, gicp

    // Print estimated 4x4 homogeneous matrix
    std::cout << "Estimated transformation = \n" << reg_obj.current_estimated_T_ << std::endl;

    // Print ground truth transformation
    std::cout << "Ground truth transformation = \n" << transformation_gt << std::endl;

    return 0;
}