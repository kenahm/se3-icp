#include <iostream>
#include <fstream>

#include <iterative_SE3_registration.hpp>



int main(int argc, char* argv[]) {

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <AlgorithmName> <SourcePointCloudFilePath> <TargetPointCloudFilePath>" << std::endl;
        return 1;
    }

    std::string algorithmName = argv[1];
    std::string sourceCloudPath = argv[2];
    std::string targetCloudPath = argv[3];

    if (algorithmName!= "pt2pt" && algorithmName!= "pt2pl" && algorithmName!= "gicp" &&
        algorithmName!= "se3_pt2pt" && algorithmName!= "se3_pt2pl" && algorithmName!= "se3_gicp") {
            std::cerr << "Not a valid algorithm name\n" 
                      << "Available names are: pt2pt, pt2pl, gicp, se3_pt2pt, se3_pt2pl, and se3_gicp\n"; 
            return 1;
    }

    // Load a source and target point clouds 
    auto source = open3d::io::CreatePointCloudFromFile(sourceCloudPath);
    std::cout << "source point cloud size = " << source->points_.size() << std::endl;

    auto target = open3d::io::CreatePointCloudFromFile(targetCloudPath);
    std::cout << "target point cloud size = " << target->points_.size() << std::endl;
    

    // Setup the registration object
    IterativeSE3Registration reg_obj;
    reg_obj.setSourceCloud(*source);
    reg_obj.setTargetCloud(*target);         
    reg_obj.estimated_overlap_ = 1.0; 
    reg_obj.max_num_se3_iterations_ = 10; // max num of iterations using SE(3) distance, before switching to standard R3 distance
    reg_obj.mse_ = 0.00001;
    reg_obj.mse_switch_error_ = 5 * reg_obj.mse_; // another criteria to switch to standard R3 distance
    reg_obj.number_of_nn_for_LRF_ = 90; // 90 nearest neighbors will be used to compute TOLDI local reference frames


    if (algorithmName=="pt2pt" || algorithmName=="pt2pl" || algorithmName=="gicp") {
        std::cout << "Running standard ICP variant: " << algorithmName << std::endl;
        reg_obj.run_icp(algorithmName);
    }
    else if (algorithmName=="se3_pt2pt" || algorithmName=="se3_pt2pl" || algorithmName=="se3_gicp") {
        std::cout << "Running SE(3)-ICP variant: " << algorithmName.substr(4) << std::endl; // remove "se3_" prefix to get the variant name
        reg_obj.run_se3_icp(algorithmName.substr(4));
    }
    else {
        std::cerr << "Not a valid algorithm name\n" 
                  << "Available names are: pt2pt, pt2pl, gicp, se3_pt2pt, se3_pt2pl, se3_gicp\n"; 
        return 1;
    }  

    // Print the estimated 4x4 homogeneous matrix
    std::cout << "Estimated transformation = \n" << reg_obj.current_estimated_T_ << std::endl;

    return 0;
}