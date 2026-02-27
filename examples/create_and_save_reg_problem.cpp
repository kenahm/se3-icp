#include <iostream>
#include <fstream>

#include <chrono>
#include <thread>
#include <random>
#include <filesystem>
#include <iomanip>

#include "open3d/Open3D.h"
#include "cc.hpp"


int main() {
    std::cout << "Creating and saving one registration proplem\n";

    // source is shared pointer to standard Open3D point cloud type
    auto source = open3d::io::CreatePointCloudFromFile("../stanford_bunny.ply");
    source->Scale(50.0, Eigen::Vector3d {0, 0, 0});

    // set seed in Open3D to have repeatable random down sample 
    open3d::utility::random::Seed(1);

    auto source_ds = source->RandomDownSample(0.02); // bunny    
    std::cout << "source->points_.size()    = \n" << source->points_.size() << std::endl;
    std::cout << "source_ds->points_.size() = \n" << source_ds->points_.size() << std::endl;
    auto target_ds = std::make_shared<open3d::geometry::PointCloud>(*source_ds);

    // auto target = std::make_shared<open3d::geometry::PointCloud>(*source);

    Eigen::Vector3d translation_gt {1.0, 2.0, 3.0};
    Eigen::Matrix3d rotation_gt = cc::rot_3d(M_PI/9, M_PI/8, -M_PI/7);
    Eigen::Matrix4d transformation_gt = Eigen::Matrix4d::Identity();
    transformation_gt.block<3, 3>(0, 0) = rotation_gt; 
    transformation_gt.block<3, 1>(0, 3) = translation_gt;    

    target_ds->Transform(transformation_gt);

    // draw source and target point clouds in Open3D visualizer
    open3d::visualization::DrawGeometries({source_ds, target_ds},
        "Source and Target Point Clouds", 800, 600);

    // make a directory to save the registration problem data
    std::filesystem::create_directory("../created_example_reg_problem");

    bool success1 = open3d::io::WritePointCloud("../created_example_reg_problem/source.ply", *source_ds);
    bool success2 = open3d::io::WritePointCloud("../created_example_reg_problem/target.ply", *target_ds);
    
    // write ground truth transformation to text file
    std::ofstream gt_file("../created_example_reg_problem/transformation_gt.txt");
    if (gt_file.is_open()) {
        gt_file << std::fixed << std::setprecision(6) << transformation_gt << std::endl;
        gt_file.close();
    } else {
        std::cerr << "Unable to open file to write ground truth transformation\n";
    }
    
    if (success1 && success2) { std::cout << "Registration problem data written successfully\n"; }    
    
    std::cout << "Gt transformation = \n" << transformation_gt << std::endl;

    return 0;
}