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

    target_ds->EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(30));
    source_ds->EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(30));

    auto target_ds_fpfh = open3d::pipelines::registration::ComputeFPFHFeature(*target_ds, open3d::geometry::KDTreeSearchParamKNN(100));
    auto source_ds_fpfh = open3d::pipelines::registration::ComputeFPFHFeature(*source_ds, open3d::geometry::KDTreeSearchParamKNN(100));


    // Visualize the problem
    source_ds->PaintUniformColor(Eigen::Vector3d(1.0, 0.0, 0.0)); // red
    target_ds->PaintUniformColor(Eigen::Vector3d(0.0, 1.0, 0.0)); // green
    open3d::visualization::DrawGeometries({source_ds, target_ds}, "Two Point Clouds");  

    open3d::pipelines::registration::FastGlobalRegistrationOption option; // use default parameters
    // option.maximum_correspondence_distance_ = 0.025;
    // option.iteration_number_ = 64;

    // auto result = open3d::pipelines::registration::FastGlobalRegistration(*source_ds, *target_ds, *source_ds_fpfh, *target_ds_fpfh, option);

    auto result = open3d::pipelines::registration::FastGlobalRegistrationBasedOnFeatureMatching(*source_ds, *target_ds, *source_ds_fpfh, *target_ds_fpfh, option);

    std::cout << "Estimated transformation = \n" << result.transformation_ << std::endl;

    // Print ground truth transformation
    std::cout << "Ground truth transformation = \n" << transformation_gt << std::endl;

    source_ds->Transform(result.transformation_);
    open3d::visualization::DrawGeometries({source_ds, target_ds}, "After FGR Alignment");

    return 0;
}