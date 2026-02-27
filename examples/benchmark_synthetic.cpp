#include <iostream>
#include <fstream>
#include <iterative_SE3_registration.hpp>

#include <chrono>
#include <thread>
#include <random>
#include <filesystem>
#include <iomanip>

#include <cc.hpp>

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

// Construct values for rotation scale parameter alpha in a hybrid manner
std::vector<double> makeHybridLGrid() {
    std::vector<double> L;

    // Always include 0
    L.push_back(0.0);

    // Dense linear: 0.01 .. 0.10 step 0.01
    for (int i = 1; i <= 10; ++i) {
        L.push_back(i * 0.01);
    }

    // Medium linear: 0.2 .. 1.0 step 0.1
    for (int i = 2; i <= 10; ++i) {
        L.push_back(i * 0.1);
    }

    // Coarser linear: 1.0 .. 5.0 step 0.5
    for (int i = 0; i <= 8; ++i) {
        L.push_back(1.0 + i * 0.5);
    }

    // Geometric tail: {5, 7, 10, 15, 25, 50} + extra stuff to 100 and 10000
    std::vector<double> extra = {5, 7, 10, 15, 25, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
    L.insert(L.end(), extra.begin(), extra.end());

    // Remove duplicates and sort
    std::sort(L.begin(), L.end());
    L.erase(std::unique(L.begin(), L.end()), L.end());

    return L;
}

void syntetic_experiment_noisy_over_common_cases(int num_test_cases=3, double diag_cov_noise=0.005, bool write_data=false) {

    // source is shared pointer to standard Open3D point cloud type
    auto source = open3d::io::CreatePointCloudFromFile("../3D_models/stanford_bunny.ply");
    source->Scale(50.0, Eigen::Vector3d {0, 0, 0});

    // set seed in Open3D to have repeatable random down sample 
    open3d::utility::random::Seed(1);

    auto source_ds = source->RandomDownSample(0.02); // bunny

    // create stuff for random rigid transformation generation
    std::random_device rd;  // seed for the random number engine
    std::mt19937 gen(1); // standard mersenne_twister_engine seeded with rd()

    // Easy setup 
    // std::uniform_real_distribution<double> dist_T(-5.0,5.0);
    // std::uniform_real_distribution<double> dist_R(-M_PI/4.0,M_PI/4.0);        

    // Moderate setup 
    std::uniform_real_distribution<double> dist_T(-10.0,10.0);
    std::uniform_real_distribution<double> dist_R(-M_PI/2.0,M_PI/2.0); 

    // Difficult setup
    // std::uniform_real_distribution<double> dist_T(-15.0,15.0);
    // std::uniform_real_distribution<double> dist_R(-M_PI,M_PI);        

    // store ICP registration results
    double icp_avg_iterations{0.0}, icp_avg_total_time{0.0}, 
           icp_avg_rot_error{0.0}, icp_avg_tran_error{0.0},
           icp_avg_SO3_error{0.0};
    int icp_num_fails {0};

    // store ICP registration results
    double se3_avg_iterations{0.0}, se3_avg_total_time{0.0}, se3_pure_se3_iterations{0.0},
           se3_avg_rot_error{0.0}, se3_avg_tran_error{0.0}, 
           se3_avg_SO3_error{0.0},
           se3_avg_time_before_icp_finetune{0.0},
           se3_avg_corr_time{0.0};
           

    int se3_num_fails {0};    

    int num_common_cases {0};

    std::ofstream outFile("../simulation_synthetic_generated_data/gt_data");
    outFile << std::fixed << std::setprecision(8);

    for (int i=0; i<num_test_cases; i++) {

        Eigen::Vector3d translation_gt {dist_T(gen), dist_T(gen), dist_T(gen)};
        Eigen::Matrix3d rotation_gt = cc::rot_3d(dist_R(gen), dist_R(gen), dist_R(gen));
        Eigen::Matrix4d transformation_gt = Eigen::Matrix4d::Identity();
        transformation_gt.block<3, 3>(0, 0) = rotation_gt; 
        transformation_gt.block<3, 1>(0, 3) = translation_gt;

        auto source_final = std::make_shared<open3d::geometry::PointCloud>(*source_ds); // same
        auto target_final_notDS = std::make_shared<open3d::geometry::PointCloud>(*source); // copy original cloud
        target_final_notDS->Transform(transformation_gt); // and transform it
        auto target_final = target_final_notDS->RandomDownSample(0.02); // now downsample independently

        ////////////////////////
        /////// ADD NOISE //////
        ////////////////////////
        add_noise_to_point_cloud(*source_final, diag_cov_noise);
        add_noise_to_point_cloud(*target_final, diag_cov_noise);

        std::cout << "source/target ds size = " << source_final->points_.size() << "/" << target_final->points_.size() << std::endl;

        // At this point we have generated the data and gt, we can optionally store the data

        if (write_data) {
            std::string name1 {"source" + std::to_string(i) + ".ply"};
            std::string name2 {"target" + std::to_string(i) + ".ply"};

            bool success1 = open3d::io::WritePointCloud("../simulation_synthetic_generated_data/" + name1, *source_final);
            bool success2 = open3d::io::WritePointCloud("../simulation_synthetic_generated_data/" + name2, *target_final);
            if (success1 && success2) { std::cout << "Registration problem data written successfully\n"; }

            outFile << transformation_gt(0,0) << " " 
                    << transformation_gt(0,1) << " " 
                    << transformation_gt(0,2) << " " 
                    << transformation_gt(0,3) << " "
                    << transformation_gt(1,0) << " " 
                    << transformation_gt(1,1) << " " 
                    << transformation_gt(1,2) << " " 
                    << transformation_gt(1,3) << " "
                    << transformation_gt(2,0) << " " 
                    << transformation_gt(2,1) << " " 
                    << transformation_gt(2,2) << " " 
                    << transformation_gt(2,3) << "\n";
        }

        ////////////////////// VANILLA ICP ////////////////////////
        IterativeSE3Registration reg_obj_icp;
        reg_obj_icp.setSourceCloud(*source_final);
        reg_obj_icp.setTargetCloud(*target_final);
        reg_obj_icp.estimated_overlap_ = 1.0;

        auto tx2 = std::chrono::high_resolution_clock::now(); 
        reg_obj_icp.run_icp("gicp");
        auto ty2 = std::chrono::high_resolution_clock::now();
        std::cout << "ICP num iterations = " << reg_obj_icp.num_iterations_ << std::endl;
        double singlev2 = std::chrono::duration_cast<std::chrono::nanoseconds>(ty2 - tx2).count() / 1e6;
        std::cout << "ICP time for alignment [ms] = " << singlev2 << std::endl;

        Eigen::Matrix3d est_rot_icp_i = reg_obj_icp.current_estimated_T_.block<3,3>(0,0);
        Eigen::Vector3d est_tra_icp_i = reg_obj_icp.current_estimated_T_.block<3,1>(0,3);
        double icp_rot_error_i = (rotation_gt - est_rot_icp_i).norm();
        double icp_tra_error_i = (translation_gt - est_tra_icp_i).norm();
        double icp_SO3_error_i = cc::angularErrorSO3(est_rot_icp_i, rotation_gt);
        std::cout << "ICP rot fro norm error = " << icp_rot_error_i << std::endl;
        std::cout << "ICP translation  error = " << icp_tra_error_i << std::endl;
        std::cout << "ICP SO(3) error        = " << icp_SO3_error_i << std::endl;
        ///////////////////////////////////////////////////////////////////

        
        //////////////////////// SE(3) ICP ////////////////////////////////
        IterativeSE3Registration reg_obj_se3;
        reg_obj_se3.setSourceCloud(*source_final);
        reg_obj_se3.setTargetCloud(*target_final);
        reg_obj_se3.estimated_overlap_ = 1.0; // this synthetic dataset has 100% overlap
        reg_obj_se3.lrf_radius_ = 0.75;
        reg_obj_se3.mse_ = 0.00001;
        reg_obj_se3.max_num_se3_iterations_ = 10; // effectivelly not capping se3 iterations
        reg_obj_se3.mse_switch_error_ = 5.0 * reg_obj_se3.mse_;
        reg_obj_se3.number_of_nn_for_LRF_ = 90; // standard stuff

        auto tx1 = std::chrono::high_resolution_clock::now(); 
        reg_obj_se3.run_se3_icp("gicp");
        auto ty1 = std::chrono::high_resolution_clock::now();

        std::cout << "SE3 num iterations = " << reg_obj_se3.num_iterations_ << std::endl;
        double singlev1 = std::chrono::duration_cast<std::chrono::nanoseconds>(ty1 - tx1).count() / 1e6;
        std::cout << "SE3 time for alignment [ms] = " << singlev1 << std::endl;  
        
        Eigen::Matrix3d est_rot_se3_i = reg_obj_se3.current_estimated_T_.block<3,3>(0,0);
        Eigen::Vector3d est_tra_se3_i = reg_obj_se3.current_estimated_T_.block<3,1>(0,3);
        double se3_rot_error_i = (rotation_gt - est_rot_se3_i).norm();
        double se3_tra_error_i = (translation_gt - est_tra_se3_i).norm();  
        double se3_SO3_error_i = cc::angularErrorSO3(rotation_gt, est_rot_se3_i);
        std::cout << "SE3 rot fro norm error = " << se3_rot_error_i << std::endl;
        std::cout << "SE3 translation  error = " << se3_tra_error_i << std::endl;
        std::cout << "SE3 SO(3) error        = " << se3_SO3_error_i << std::endl;
        ///////////////////////////////////////////////////////////////////

        // store results
        if (icp_SO3_error_i > 2.0 or icp_tra_error_i > 0.25) {
            std::cout << "ICP did not converge\n";
            icp_num_fails++;
        }

        if (se3_SO3_error_i > 2.0 or se3_tra_error_i > 0.25) {
            std::cout << "SE3 did not converge\n";
            se3_num_fails++;
        }

        // common cases
        if ((icp_SO3_error_i <= 2.0) and (icp_tra_error_i <= 0.25) and 
            (se3_SO3_error_i <= 2.0) and (se3_tra_error_i <= 0.25)) {
                num_common_cases++;

                icp_avg_iterations += reg_obj_icp.num_iterations_;
                icp_avg_total_time += singlev2;
                icp_avg_rot_error  += icp_rot_error_i;
                icp_avg_tran_error += icp_tra_error_i;    
                icp_avg_SO3_error  += icp_SO3_error_i;

                se3_avg_iterations += reg_obj_se3.num_iterations_;
                se3_pure_se3_iterations += reg_obj_se3.num_pure_se3_iterations_;
                se3_avg_total_time += singlev1;
                se3_avg_rot_error  += se3_rot_error_i;
                se3_avg_tran_error += se3_tra_error_i;   
                se3_avg_SO3_error  += se3_SO3_error_i;
                se3_avg_time_before_icp_finetune += reg_obj_se3.time_before_pure_icp_;
                se3_avg_corr_time += reg_obj_se3.time_se3_correspondence_search_;                     
        }
    }

    std::cout << "=== Point cloud information ===\n";
    std::cout << "Original           number of points = " << source->points_.size() << std::endl;
    std::cout << "After downsampling number of points = " << source_ds->points_.size() << std::endl;
    std::cout << "Added noise diag covariance value   = " << diag_cov_noise << std::endl;
    std::cout << "=============================== RESULTS ==================================\n";
    std::cout << "Over common " << num_common_cases << "/" << num_test_cases << " cases\n";
    std::cout << "ICP total num of fails = " << icp_num_fails << std::endl; 
    std::cout << "ICP avg time           = " << icp_avg_total_time/num_common_cases << std::endl;
    std::cout << "ICP avg iterations     = " << icp_avg_iterations/num_common_cases << std::endl;
    std::cout << "ICP avg rot error      = " << icp_avg_rot_error/num_common_cases << std::endl;
    std::cout << "ICP avg tra error      = " << icp_avg_tran_error/num_common_cases << std::endl;
    std::cout << "ICP avg SO(3) error    = " << icp_avg_SO3_error/num_common_cases << std::endl;
    std::cout << "ICP success rate       = " << double(num_test_cases-icp_num_fails)/double(num_test_cases) << std::endl;
    std::cout << "==========================================================================\n";
    std::cout << "SE3 total num of fails = " << se3_num_fails << std::endl; 
    std::cout << "SE3 avg time           = " << se3_avg_total_time/num_common_cases << std::endl;
    std::cout << "SE3 avg time bef icp   = " << se3_avg_time_before_icp_finetune/num_common_cases << std::endl;
    std::cout << "SE3 avg time se3 corr  = " << se3_avg_corr_time/num_common_cases << std::endl;
    std::cout << "SE3 avg iterations     = " << se3_avg_iterations/num_common_cases << std::endl;
    std::cout << "pure SE3 iterations    = " << se3_pure_se3_iterations/num_common_cases << std::endl;
    std::cout << "SE3 avg rot error      = " << se3_avg_rot_error/num_common_cases << std::endl;
    std::cout << "SE3 avg tra error      = " << se3_avg_tran_error/num_common_cases << std::endl;
    std::cout << "SE3 avg SO(3) error    = " << se3_avg_SO3_error/num_common_cases << std::endl;
    std::cout << "SE3 success rate       = " << double(num_test_cases-se3_num_fails)/double(num_test_cases) << std::endl;
    std::cout << "==========================================================================\n";
    std::cout << "Noisy simulation over " << num_test_cases << " test cases finished!\n";

    outFile.close();
}

void benchmark_algorithm_on_saved_data(const std::string& algorithm_name, const std::string& data_folder_path, double rot_alpha=3.0, bool write_results=false) {

    double avg_iterations{0.0}, avg_time{0.0}, avg_rot_error{0.0}, avg_tran_error{0.0}, avg_SO3_error{0.0};
    int num_fails {0};    

    // read ground truth transformation/matrices
    std::vector<Eigen::Matrix4d> gt44_matrices;
    std::ifstream file(data_folder_path + "/gt_data");
    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
    }   
    std::string line;
    bool readLine {true};
    while (std::getline(file, line)) {
        std::istringstream stream(line);
        Eigen::VectorXd rowVector(12);

        for (int j = 0; j < 12; ++j) {
            stream >> rowVector(j);
        }

        Eigen::Matrix4d mat44;
        mat44 <<  rowVector(0), rowVector(1), rowVector(2), rowVector(3),
                rowVector(4), rowVector(5), rowVector(6), rowVector(7),
                rowVector(8), rowVector(9), rowVector(10), rowVector(11),
                0, 0, 0, 1;

        gt44_matrices.push_back(mat44);
    }
    file.close();        
    std::cout << "gt44_matrices.size() = " << gt44_matrices.size() << std::endl;    

    int index {0};
    while (true) {

        std::string source_file = data_folder_path + "/source" + std::to_string(index) + ".ply";
        std::string target_file = data_folder_path + "/target" + std::to_string(index) + ".ply";

        // Read point clouds and perform registration
        if (std::filesystem::exists(source_file) && std::filesystem::exists(target_file) && index < gt44_matrices.size()) {

            // needs to be visible at this scope
            open3d::pipelines::registration::RegistrationResult fgr_result;
            
            // Get ground truth values
            const Eigen::Matrix3d rotation_gt = gt44_matrices[index].block<3,3>(0,0);
            const Eigen::Vector3d translation_gt = gt44_matrices[index].block<3,1>(0,3);

            // Read source and target point clouds
            auto source_final = open3d::io::CreatePointCloudFromFile(source_file);
            auto target_final = open3d::io::CreatePointCloudFromFile(target_file);            

            // Point Cloud Registration
            IterativeSE3Registration reg_obj;
            reg_obj.setSourceCloud(*source_final);
            reg_obj.setTargetCloud(*target_final);         
            reg_obj.estimated_overlap_ = 1.0; 
            reg_obj.max_num_se3_iterations_ = 10;
            reg_obj.mse_ = 0.00001;
            reg_obj.mse_switch_error_ = 5 * reg_obj.mse_;
            reg_obj.number_of_nn_for_LRF_ = 90;

            // rotation scale parameter
            reg_obj.alpha_rot = rot_alpha;

            auto tx2 = std::chrono::high_resolution_clock::now();

            if (algorithm_name=="pt2pt" || algorithm_name=="pt2pl" || algorithm_name=="gicp") {
                reg_obj.run_icp(algorithm_name);
            }
            else if (algorithm_name=="se3_pt2pt" || algorithm_name=="se3_pt2pl" || algorithm_name=="se3_gicp") {
                
                if (algorithm_name=="se3_pt2pt") 
                    reg_obj.run_se3_icp("pt2pt"); 
                else if (algorithm_name=="se3_pt2pl") 
                    reg_obj.run_se3_icp("pt2pl");
                else if (algorithm_name=="se3_gicp") 
                    reg_obj.run_se3_icp("gicp");
                else {
                    std::cerr << "Invalid algorithm name!\n";
                }
            }
            else if (algorithm_name=="fgr") {
                std::cout << "Running FGR registration \n";
                source_final->EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(30));
                target_final->EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(30));
                
                auto source_fpfh = open3d::pipelines::registration::ComputeFPFHFeature(*source_final, open3d::geometry::KDTreeSearchParamKNN(100));
                auto target_fpfh = open3d::pipelines::registration::ComputeFPFHFeature(*target_final, open3d::geometry::KDTreeSearchParamKNN(100));

                open3d::pipelines::registration::FastGlobalRegistrationOption option; // use default parameters
                fgr_result = open3d::pipelines::registration::FastGlobalRegistrationBasedOnFeatureMatching(*source_final, *target_final, *source_fpfh, *target_fpfh, option);
            }
            else {
                std::cerr << "Invalid algorithm name! Options: pt2pt, pt2pl, gicp, se3_pt2pt, se3_pt2pl, se3_gicp, fgr\n";
            }

            auto ty2 = std::chrono::high_resolution_clock::now();
            
            if (algorithm_name!="fgr") { // this must be one of icp or se3_icp variants
                std::cout << "num iterations = " << reg_obj.num_iterations_ << std::endl;
                double singlev2 = std::chrono::duration_cast<std::chrono::nanoseconds>(ty2 - tx2).count() / 1e6;
                avg_time += singlev2;

                Eigen::Matrix3d est_rot_i = reg_obj.current_estimated_T_.block<3,3>(0,0);
                Eigen::Vector3d est_tra_i = reg_obj.current_estimated_T_.block<3,1>(0,3);
                double rot_error_i = (rotation_gt - est_rot_i).norm();
                double tra_error_i = (translation_gt - est_tra_i).norm();
                double SO3_error_i = cc::angularErrorSO3(est_rot_i, rotation_gt);

                if (SO3_error_i > 2.0 or tra_error_i > 0.25) {
                    std::cout << "Algorithm did not converge correctly\n";
                    num_fails++;
                }                 
            }
            else if (algorithm_name=="fgr") {
                double singlev2 = std::chrono::duration_cast<std::chrono::nanoseconds>(ty2 - tx2).count() / 1e6;
                avg_time += singlev2;

                Eigen::Matrix3d est_rot_i = fgr_result.transformation_.block<3,3>(0,0);
                Eigen::Vector3d est_tra_i = fgr_result.transformation_.block<3,1>(0,3);
                double rot_error_i = (rotation_gt - est_rot_i).norm();
                double tra_error_i = (translation_gt - est_tra_i).norm();
                double SO3_error_i = cc::angularErrorSO3(est_rot_i, rotation_gt);

                if (SO3_error_i > 2.0 or tra_error_i > 0.25) {
                    std::cout << "FGR Algorithm did not converge correctly\n";
                    num_fails++;
                }   
                else {
                    std::cout << "FGR Algorithm did successfully converge\n";
                }
            }
            else {
                std::cout << "This else statement print should be impossible! \n";
            }
        }

        // no more registration problem to read & solve
        else {
            std::cout << "===== Synthetic data results of algorithm: " << algorithm_name << " =====\n";
            std::cout << "Data folder: " << data_folder_path << std::endl;
            int num_test_cases = index;
            int num_succ_cases = num_test_cases-num_fails;

            std::cout << "Num of fails over " << num_test_cases << " problems is: " << num_fails << std::endl; 
            std::cout << "ICP success rate     = " << double(num_succ_cases)/double(num_test_cases) << std::endl;
            std::cout << "ICP avg time overall = " << avg_time/num_test_cases << std::endl;            
            std::cout << "==========================================================================\n";

            if (write_results) {
                std::string filename = "../experiments_rot_scale_synthetic.txt";

                // Open in append mode: creates the file if it doesn't exist
                std::ofstream file(filename, std::ios::app);
                if (!file) {
                    std::cerr << "Error opening file: " << filename << std::endl;
                }

                // Write value followed by newline
                file << algorithm_name << " | " << "rot_alpha = " << rot_alpha << " | success_rate = " << double(num_succ_cases)/double(num_test_cases) << "\n";
                file.close(); 
                }


            break;
        }

        ++index;
    }
}

void benchmark_different_rot_scales(const std::string& algorithm_name, const std::string& lounge_folder_path) {
    auto rot_alphas = makeHybridLGrid();
    for (const auto& alpha : rot_alphas) {
        benchmark_algorithm_on_saved_data(algorithm_name, lounge_folder_path, alpha, true);
    }
}

int main(int argc, char* argv[]) {

    std::cout << "Testing o3d-based registration impl on synthetic data \n";

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <AlgorithmName> <FolderPath>" << std::endl;
        return 1;
    }

    std::string algorithmName = argv[1];
    std::string folderPath = argv[2];

    if (algorithmName!= "pt2pt" && algorithmName!= "pt2pl" && algorithmName!= "gicp" &&
        algorithmName!= "se3_pt2pt" && algorithmName!= "se3_pt2pl" && algorithmName!= "se3_gicp" && algorithmName!= "fgr") {
            std::cerr << "Not a valid algorithm name\n" 
                      << "Available names are: pt2pt, pt2pl, gicp, se3_pt2pt, se3_pt2pl, se3_gicp , fgr\n"; 
            return 1;
    }

    // benchmark_algorithm_on_saved_data(algorithmName, folderPath, 3.0, true);
    benchmark_algorithm_on_saved_data(algorithmName, folderPath);

    // test for different rotation scales accross the dataset
    // benchmark_different_rot_scales(algorithmName, folderPath);    

    return 0;
}