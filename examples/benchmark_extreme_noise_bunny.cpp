#include <iostream>
#include <fstream>
#include <iterative_SE3_registration.hpp>

#include <chrono>
#include <thread>
#include <random>
#include <filesystem>
#include <iomanip>

#include <cc.hpp>

double largestDistanceFromGivenPoint(const Eigen::Vector3d& ref_point, const open3d::geometry::PointCloud& cloud) {
    double current_largest {-1.0};
    for (const auto& point : cloud.points_) {
        double dist = (point - ref_point).norm();
        if (dist > current_largest) current_largest = dist;
    }
    return current_largest;
}

void add_noise_to_point_cloud(open3d::geometry::PointCloud& cloud, double noise=0.005) {

    struct normal_random_variable
    {
        normal_random_variable(Eigen::MatrixXd const& covar)
            : normal_random_variable(Eigen::VectorXd::Zero(covar.rows()), covar)
        {}

        normal_random_variable(Eigen::VectorXd const& mean, Eigen::MatrixXd const& covar)
            : mean(mean)
        {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
            transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
        }

        Eigen::VectorXd mean;
        Eigen::MatrixXd transform;

        Eigen::VectorXd operator()() const
        {
            // static std::mt19937 gen{ std::random_device{}() };
            static std::mt19937 gen{1};
            static std::normal_distribution<> dist;

            return mean + transform * Eigen::VectorXd{ mean.size() }.unaryExpr([&](auto x) { return dist(gen); });
        }
    };

    // we are adding sample from Gaussan MV distribution
    // with zero mean and diagonal covariance matrix
    Eigen::Matrix3d cov_mat;
    cov_mat.setIdentity();
    cov_mat *= noise;

    normal_random_variable norm_generator {cov_mat};

    // Add noise to the point cloud
    for (auto& one_point : cloud.points_) {
        auto rnd_sample = norm_generator();
        one_point[0] += rnd_sample(0);
        one_point[1] += rnd_sample(1);
        one_point[2] += rnd_sample(2);
    }
}


void adding_different_noise_levels() {

    auto source = open3d::io::CreatePointCloudFromFile("/home/kenan/3D_models/stanford_bunny.ply");
    source->Scale(50.0, Eigen::Vector3d {0, 0, 0});

    // open3d::utility::random::Seed(2);
    auto source_ds = source->RandomDownSample(0.02);
    
    auto target_ds = std::make_shared<open3d::geometry::PointCloud>(*source_ds);
    target_ds->Translate(Eigen::Vector3d {3.0, 5.0, 7.0});

    Eigen::Vector3d centorid_source = source_ds->GetCenter();
    double radius_source = largestDistanceFromGivenPoint(centorid_source, *source_ds);
    double diametar_approx = 2.0 * radius_source; // I just randomly did this, but its not so important that it is precise

    std::cout << "source_ds->points_.size() = " << source_ds->points_.size() << std::endl;
    std::cout << "radius_source = " << radius_source << std::endl;
    std::cout << "diametar_approx = " << diametar_approx << std::endl;

    // noise level 1
    double perc = 0.01;
    add_noise_to_point_cloud(*source_ds, (perc * diametar_approx) * (perc * diametar_approx));

    // open3d::visualization::DrawGeometries({source_ds, target_ds});


    // Eigen Random uses seed from standard library
    srand((unsigned int) time(0));
    Eigen::Vector3d rnd_axis = Eigen::Vector3d::Random().normalized(); 
    std::cout << "rnd_axis = " << rnd_axis.transpose() << std::endl; 
    std::cout << "rnd_axis.norm() = " << rnd_axis.norm() << std::endl; 

    double angle_rad = 50.0 * M_PI / 180.0;

    Eigen::Matrix3d rotation_matrix = Eigen::AngleAxisd(angle_rad, rnd_axis).toRotationMatrix();

    std::cout << "rotation_matrix = \n" << rotation_matrix << std::endl;

    // everything seems nice!
    std::cout << "rotation matrix checks: \n"
              << rotation_matrix * rotation_matrix.transpose() << std::endl
              << rotation_matrix.transpose() * rotation_matrix << std::endl
              << "det = " << rotation_matrix.determinant() << std::endl; 
}


double benchmark_algorithm_at_noise_level(const std::string& algorithm_name, Eigen::Vector3d rnd_axis, double rot_angle_rad, 
                                          double noise_level, int num_runs=30, bool write_data=false, std::string write_folder="") {
    // noise_level scales the bunny point cloud diametar, which is taken to be std
    // then MV gaussian with diagonal covariance matrix diag(std*std)
    // is added to each point of both source and target point clouds independently

    // auto source = open3d::io::CreatePointCloudFromFile("../bunny_synthetic_data/stanford_bunny.ply");
    
    // this is already downsampled                                
    auto source_ds = open3d::io::CreatePointCloudFromFile("/home/kenan/bunny_from_freg_repo.pcd");
    std::cout << "source_ds->points_.size() = " << source_ds->points_.size() << std::endl;



    open3d::utility::random::Seed(1);
    // auto source_ds = source->RandomDownSample(0.02);
    // auto source_ds = source->VoxelDownSample(0.0042);
    
    auto target_ds = std::make_shared<open3d::geometry::PointCloud>(*source_ds);
    Eigen::Matrix3d rot_gt = Eigen::AngleAxisd(rot_angle_rad, rnd_axis).toRotationMatrix();
    Eigen::Vector3d tra_gt = Eigen::Vector3d {0.0, 0.0, 0.0};
    Eigen::Matrix4d T_gt = Eigen::Matrix4d::Identity();
    T_gt.block<3,3>(0,0) = rot_gt;
    T_gt.block<3,1>(0,3) = tra_gt;
    target_ds->Transform(T_gt);

    std::ofstream outFile(write_folder + "/gt_transform");
    if (write_data=true) {
        outFile << std::fixed << std::setprecision(8);
        outFile << T_gt(0,0) << " " 
                << T_gt(0,1) << " " 
                << T_gt(0,2) << " " 
                << T_gt(0,3) << " "
                << T_gt(1,0) << " " 
                << T_gt(1,1) << " " 
                << T_gt(1,2) << " " 
                << T_gt(1,3) << " "
                << T_gt(2,0) << " " 
                << T_gt(2,1) << " " 
                << T_gt(2,2) << " " 
                << T_gt(2,3) << "\n";
    }

    // open3d::visualization::DrawGeometries({source_ds, target_ds});

    Eigen::Vector3d centorid_source = source_ds->GetCenter();
    double radius_source = largestDistanceFromGivenPoint(centorid_source, *source_ds);
    double diametar = 2.0 * radius_source;     

    std::cout << "radius_source = " << radius_source << std::endl;
    std::cout << "diametar = " << diametar << std::endl;

    /*
    // pretty close to previously computed diametar, just wanted to sanity check
    auto max_bounds_vec3d = source_ds->GetMaxBound();
    auto min_bounds_vec3d = source_ds->GetMinBound();
    std::cout << "min_bounds_vec3d = " << min_bounds_vec3d.transpose() << std::endl;
    std::cout << "max_bounds_vec3d = " << max_bounds_vec3d.transpose() << std::endl;
    std::cout << "dist betw beounds= " << (max_bounds_vec3d - min_bounds_vec3d).norm() << std::endl;
    */

    double avg_RE {0.0}, avg_TE {0.0}, point_error {0.0};

    for (int i=0; i<num_runs; i++) {
        auto source_ds_noisy = std::make_shared<open3d::geometry::PointCloud>(*source_ds);
        auto target_ds_noisy = std::make_shared<open3d::geometry::PointCloud>(*target_ds);

        //>>// add_noise_to_point_cloud(*source_ds_noisy, (noise_level * diametar) * (noise_level * diametar));
        //>>// add_noise_to_point_cloud(*target_ds_noisy, (noise_level * diametar) * (noise_level * diametar));

        add_noise_to_point_cloud(*source_ds_noisy, noise_level * noise_level * diametar * diametar);
        add_noise_to_point_cloud(*target_ds_noisy, noise_level * noise_level * diametar * diametar);


        if (write_data=true) {
            
            std::string name1 {"source" + std::to_string(i) + ".ply"};
            std::string name2 {"target" + std::to_string(i) + ".ply"};

            bool success1 = open3d::io::WritePointCloud(write_folder + "/" + name1, *source_ds_noisy);
            bool success2 = open3d::io::WritePointCloud(write_folder + "/" + name2, *target_ds_noisy);            
        }

        IterativeSE3Registration reg_obj_icp;
        reg_obj_icp.setSourceCloud(*source_ds_noisy);
        reg_obj_icp.setTargetCloud(*target_ds_noisy);
        reg_obj_icp.estimated_overlap_=1.0;
        
        reg_obj_icp.max_num_iterations_ = 100;
        reg_obj_icp.mse_ =  0.000000001; // default value 0.00001;

        reg_obj_icp.number_of_nn_for_LRF_ = 90;   // relevant only for proposed SE(3) variants
        reg_obj_icp.mse_switch_error_ = 0.00005;  // relevant only for proposed SE(3) variants
        reg_obj_icp.max_num_se3_iterations_ = 10; // relevant only for proposed SE(3) variants


        if (algorithm_name=="pt2pt" || algorithm_name=="pt2pl" || algorithm_name=="gicp") {
            reg_obj_icp.run_icp(algorithm_name);
        }
        else if (algorithm_name=="se3_pt2pt" || algorithm_name=="se3_pt2pl" || algorithm_name=="se3_gicp") {
            
            if (algorithm_name=="se3_pt2pt") reg_obj_icp.run_se3_icp("pt2pt"); 
            else if (algorithm_name=="se3_pt2pl") reg_obj_icp.run_se3_icp("pt2pl");
            else if (algorithm_name=="se3_gicp") reg_obj_icp.run_se3_icp("gicp");
            else {
                std::cerr << "Invalid algorithm name!\n";
            }
        }
        else {
            std::cerr << "Invalid algorithm name!\n";
        }

        Eigen::Matrix3d est_rot_icp_i = reg_obj_icp.current_estimated_T_.block<3,3>(0,0);
        Eigen::Vector3d est_tra_icp_i = reg_obj_icp.current_estimated_T_.block<3,1>(0,3);     

        double TE_error_i = (tra_gt - est_tra_icp_i).norm();
        double RE_error_i = cc::angularErrorSO3(est_rot_icp_i, rot_gt);        
        
        avg_TE += TE_error_i;
        avg_RE += RE_error_i;
        point_error += cc::error_filterreg(*source_ds_noisy, T_gt, reg_obj_icp.current_estimated_T_);

        // open3d::visualization::DrawGeometries({source_ds_noisy, target_ds_noisy});
    }

    // std::cout << "avg_TE = " << avg_TE / num_runs << std::endl;
    // std::cout << "avg_RE = " << avg_RE / num_runs << std::endl;
    // std::cout << "error  = " << 1000.0 * (point_error / num_runs) << std::endl;

    return 1000.0 * (point_error / num_runs);
}

Eigen::Matrix4d readMatrixFromFile(const std::string& filename) {
    std::ifstream inFile(filename);
    Eigen::Matrix4d T_gt;

    if (!inFile) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return T_gt;
    }

    for (int i = 0; i < 3; ++i) {  // Read only 3 rows (since last row is implicit for homogeneous transformation)
        for (int j = 0; j < 4; ++j) {
            inFile >> T_gt(i, j);
        }
    }

    // Set last row as [0 0 0 1] for a valid homogeneous transformation matrix
    T_gt.row(3) << 0, 0, 0, 1;

    inFile.close();
    return T_gt;
}

double benchmark_algorithm_on_saved_data(const std::string& algorithm_name, std::string data_folder="") {

    // std::cout << "Reading previously constructed noisy data";
    std::cout << "Solving problems from folder: " << data_folder << std::endl;

    double avg_RE {0.0}, avg_TE {0.0}, point_error {0.0};

    auto T_gt = readMatrixFromFile(data_folder + "/gt_transform");
    Eigen::Matrix3d rot_gt = T_gt.block<3,3>(0,0);
    Eigen::Vector3d tra_gt = T_gt.block<3,1>(0,3);

    // std::cout << T_gt << std::endl;

    for (int i=0; i<30; i++) {

        auto source = open3d::io::CreatePointCloudFromFile(data_folder + "/source" + std::to_string(i) + ".ply");
        auto target = open3d::io::CreatePointCloudFromFile(data_folder + "/target" + std::to_string(i) + ".ply");

        IterativeSE3Registration reg_obj_icp;
        reg_obj_icp.setSourceCloud(*source);
        reg_obj_icp.setTargetCloud(*target);
        reg_obj_icp.estimated_overlap_=1.0;
        
        reg_obj_icp.max_num_iterations_ = 100;
        reg_obj_icp.mse_ =  0.000000001; // default value 0.00001;

        reg_obj_icp.number_of_nn_for_LRF_ = 90;   // relevant only for proposed SE(3) variants
        reg_obj_icp.mse_switch_error_ = 0.00005;  // relevant only for proposed SE(3) variants
        reg_obj_icp.max_num_se3_iterations_ = 10; // relevant only for proposed SE(3) variants


        if (algorithm_name=="pt2pt" || algorithm_name=="pt2pl" || algorithm_name=="gicp") {
            reg_obj_icp.run_icp(algorithm_name);
        }
        else if (algorithm_name=="se3_pt2pt" || algorithm_name=="se3_pt2pl" || algorithm_name=="se3_gicp") {
            
            if (algorithm_name=="se3_pt2pt") reg_obj_icp.run_se3_icp("pt2pt"); 
            else if (algorithm_name=="se3_pt2pl") reg_obj_icp.run_se3_icp("pt2pl");
            else if (algorithm_name=="se3_gicp") reg_obj_icp.run_se3_icp("gicp");
            else {
                std::cerr << "Invalid algorithm name!\n";
            }
        }
        else {
            std::cerr << "Invalid algorithm name!\n";
        }

        Eigen::Matrix3d est_rot_icp_i = reg_obj_icp.current_estimated_T_.block<3,3>(0,0);
        Eigen::Vector3d est_tra_icp_i = reg_obj_icp.current_estimated_T_.block<3,1>(0,3);     

        double TE_error_i = (tra_gt - est_tra_icp_i).norm();
        double RE_error_i = cc::angularErrorSO3(est_rot_icp_i, rot_gt);        
        
        avg_TE += TE_error_i;
        avg_RE += RE_error_i;
        point_error += cc::error_filterreg(*source, T_gt, reg_obj_icp.current_estimated_T_);

        // open3d::visualization::DrawGeometries({source_ds_noisy, target_ds_noisy});
    }

    // std::cout << "avg_TE = " << avg_TE / num_runs << std::endl;
    // std::cout << "avg_RE = " << avg_RE / num_runs << std::endl;
    // std::cout << "error  = " << 1000.0 * (point_error / num_runs) << std::endl;

    return 1000.0 * (point_error / 30.0);
}


int main(int argc, char* argv[]) {

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <AlgorithmName> <FolderPath>" << std::endl;
        return 1;
    }

    std::string algorithmName = argv[1];
    std::string folderPath = argv[2];

    if (algorithmName!= "pt2pt" && algorithmName!= "pt2pl" && algorithmName!= "gicp" &&
        algorithmName!= "se3_pt2pt" && algorithmName!= "se3_pt2pl" && algorithmName!= "se3_gicp" && 
        algorithmName!= "se3_gicp_with_cf") {
            std::cerr << "Not a valid algorithm name\n" 
                      << "Available names are: pt2pt, pt2pl, gicp, se3_pt2pt, se3_pt2pl, se3_gicp and se3_gicp_with_cf \n"; 
            return 1;
    }

    std::cout << "Extreme noise experiment\n";

    std::vector<double> errors; 

    /*
    Eigen::Vector3d rot_axis;
    rot_axis[0] = 0.0;
    rot_axis[1] = 0.0;
    rot_axis[2] = 1.0;

    // generate the rotation axis randomly
    // Eigen Random uses seed from standard library
    // srand((unsigned int) time(0));
    // rot_axis = Eigen::Vector3d::Random().normalized(); 
    // std::cout << "rot_axis = " << rot_axis.transpose() << std::endl; 
    // std::cout << "rot_axis.norm() = " << rot_axis.norm() << std::endl;    

    // comment out some noise levels for faster benchmarking
    errors.push_back(benchmark_algorithm_at_noise_level("pt2pt", rot_axis, 50.0 * M_PI / 180.0, 0.0, 30, true, "../bunny_extreme_noise_voxel/noise_0_0"));
    errors.push_back(benchmark_algorithm_at_noise_level("pt2pt", rot_axis, 50.0 * M_PI / 180.0, 0.01, 30, true, "../bunny_extreme_noise_voxel/noise_0_01"));
    errors.push_back(benchmark_algorithm_at_noise_level("pt2pt", rot_axis, 50.0 * M_PI / 180.0, 0.02, 30, true, "../bunny_extreme_noise_voxel/noise_0_02"));
    errors.push_back(benchmark_algorithm_at_noise_level("pt2pt", rot_axis, 50.0 * M_PI / 180.0, 0.03, 30, true, "../bunny_extreme_noise_voxel/noise_0_03"));
    errors.push_back(benchmark_algorithm_at_noise_level("pt2pt", rot_axis, 50.0 * M_PI / 180.0, 0.04, 30, true, "../bunny_extreme_noise_voxel/noise_0_04"));
    errors.push_back(benchmark_algorithm_at_noise_level("pt2pt", rot_axis, 50.0 * M_PI / 180.0, 0.05, 30, true, "../bunny_extreme_noise_voxel/noise_0_05"));
    errors.push_back(benchmark_algorithm_at_noise_level("pt2pt", rot_axis, 50.0 * M_PI / 180.0, 0.06, 30, true, "../bunny_extreme_noise_voxel/noise_0_06"));
    errors.push_back(benchmark_algorithm_at_noise_level("pt2pt", rot_axis, 50.0 * M_PI / 180.0, 0.07, 30, true, "../bunny_extreme_noise_voxel/noise_0_07"));
    errors.push_back(benchmark_algorithm_at_noise_level("pt2pt", rot_axis, 50.0 * M_PI / 180.0, 0.08, 30, true, "../bunny_extreme_noise_voxel/noise_0_08"));
    errors.push_back(benchmark_algorithm_at_noise_level("pt2pt", rot_axis, 50.0 * M_PI / 180.0, 0.09, 30, true, "../bunny_extreme_noise_voxel/noise_0_09"));
    errors.push_back(benchmark_algorithm_at_noise_level("pt2pt", rot_axis, 50.0 * M_PI / 180.0, 0.1, 30, true, "../bunny_extreme_noise_voxel/noise_0_1"));

    std::cout << "Errors for different noise levels: \n";
    for (auto error: errors) {
        std::cout << "error = " << error << std::endl;
    }
    */
    
    // errors.push_back(benchmark_algorithm_on_saved_data(algorithmName, "../bunny_extreme_noise/noise_0_0"));
    errors.push_back(benchmark_algorithm_on_saved_data(algorithmName, folderPath + "/noise_0_0"));
    errors.push_back(benchmark_algorithm_on_saved_data(algorithmName, folderPath + "/noise_0_01"));
    errors.push_back(benchmark_algorithm_on_saved_data(algorithmName, folderPath + "/noise_0_02"));
    errors.push_back(benchmark_algorithm_on_saved_data(algorithmName, folderPath + "/noise_0_03"));
    errors.push_back(benchmark_algorithm_on_saved_data(algorithmName, folderPath + "/noise_0_04"));
    errors.push_back(benchmark_algorithm_on_saved_data(algorithmName, folderPath + "/noise_0_05"));
    errors.push_back(benchmark_algorithm_on_saved_data(algorithmName, folderPath + "/noise_0_06"));
    errors.push_back(benchmark_algorithm_on_saved_data(algorithmName, folderPath + "/noise_0_07"));
    errors.push_back(benchmark_algorithm_on_saved_data(algorithmName, folderPath + "/noise_0_08"));
    errors.push_back(benchmark_algorithm_on_saved_data(algorithmName, folderPath + "/noise_0_09"));
    errors.push_back(benchmark_algorithm_on_saved_data(algorithmName, folderPath + "/noise_0_1"));  

    std::cout << "Errors for different noise levels (saved data): \n";
    for (auto error: errors) {
        std::cout << "error = " << error << std::endl;
    }
    
    return 0;
}