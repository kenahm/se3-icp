#include "cc.hpp"

// Error measure used in FilterReg 6.1, and LSG-CPD 4.1
double cc::error_filterreg(const open3d::geometry::PointCloud& src, Eigen::Matrix4d T_gt, Eigen::Matrix4d T_est) {

    auto src_transf_gt  = std::make_shared<open3d::geometry::PointCloud>(src);
    src_transf_gt->Transform(T_gt);

    auto src_transf_est = std::make_shared<open3d::geometry::PointCloud>(src);
    src_transf_est->Transform(T_est);


    double accumulated_error = 0.0;
    for (int i=0; i<src.points_.size(); i++) {
        accumulated_error += (src_transf_gt->points_[i] - src_transf_est->points_[i]).norm();  
    }

    return accumulated_error / (src.points_.size());
}

// Implementing some common point cloud related computations
Eigen::Matrix3d cc::rot_3d(double roll_, double pitch_, double yaw_) {
    // generate 3d rotation matrix from yaw, pitch, roll angles
    Eigen::AngleAxisd rollAngle(roll_, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch_, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw_, Eigen::Vector3d::UnitZ());  
    Eigen::Quaternion<double> q =  yawAngle * pitchAngle * rollAngle;
    Eigen::Matrix3d rotMat = q.matrix(); 
    return rotMat;
}

double cc::angularErrorSO3 (const Eigen::Matrix3d& rot_mat_1, const Eigen::Matrix3d& rot_mat_2) {
    Eigen::Matrix3d logR = (rot_mat_1.transpose() * rot_mat_2).log();
    Eigen::Vector3d logR_vee {-logR(1, 2), logR(0, 2), -logR(0, 1)};
    double error = logR_vee.norm() * (180.0/M_PI);
    return error;
}

double safe_acos(double x) {
    if (x<=-1.0) {
        return M_PI;
    } else if (x>=1.0) {
        return 0;
    } else {
        return acos(x);
    }    
}

double cc::angularErrorSO3_alt (const Eigen::Matrix3d& rot_mat_1, const Eigen::Matrix3d& rot_mat_2) {

    double acos_argument = ((rot_mat_1.transpose() * rot_mat_2).trace() - 1.0)/2.0;

    /*
    if (acos_argument > 1.0 or acos_argument < -1.0) {
        std::cout << "BAAAAAAAAAAD acos_argument = " << acos_argument << std::endl;
    }
    */

    double error = std::abs(safe_acos(acos_argument)) * (180.0/M_PI);
    return error;
}

double cc::evaluate_LRF_quality(const std::vector<Eigen::Matrix4d>& source_SE3, const std::vector<Eigen::Matrix4d>& target_SE3 ,
                          const Eigen::Matrix4d& map_gt, const std::vector<std::pair<int,int>>& corr_pairs) {

    // std::cout << "==Evaluating LRF quality==\n";

    double rot_SO3_error{0.0};
    
    std::ofstream outFile("../LRF_SO3_ERRORS.txt");
    // outFile << std::fixed << std::setprecision(10);

    for (auto pair : corr_pairs) {
        auto source_LRF_transformed {map_gt * source_SE3[pair.first]};
        // double err = cc::angularErrorSO3(source_LRF_transformed.block<3,3>(0,0), target_SE3[pair.second].block<3,3>(0,0)); // standard way
        double err = cc::angularErrorSO3_alt(source_LRF_transformed.block<3,3>(0,0), target_SE3[pair.second].block<3,3>(0,0));
        rot_SO3_error += err;
        // std::cout << "SO3 error = " << err << std::endl; 
        outFile << err << std::endl;
    }
    outFile.close();

    rot_SO3_error = rot_SO3_error/((double)corr_pairs.size());
    // std::cout << "Average SO3 error on LRF = " << rot_SO3_error << std::endl;  
    return rot_SO3_error;
}




double cc::evaluate_LRF_quality(const std::vector<Eigen::Matrix4d>& source_SE3, const std::vector<Eigen::Matrix4d>& target_SE3 ,
                            const Eigen::Matrix4d& map_gt, const std::vector<std::pair<int,int>>& corr_pairs, std::string path) {

    // std::cout << "==Evaluating LRF quality==\n";

    double rot_SO3_error{0.0};
    
    std::ofstream outFile(path);
    // outFile << std::fixed << std::setprecision(10);

    for (auto pair : corr_pairs) {
        auto source_LRF_transformed {map_gt * source_SE3[pair.first]};
        // double err = cc::angularErrorSO3(source_LRF_transformed.block<3,3>(0,0), target_SE3[pair.second].block<3,3>(0,0));
        double err = cc::angularErrorSO3_alt(source_LRF_transformed.block<3,3>(0,0), target_SE3[pair.second].block<3,3>(0,0));
        rot_SO3_error += err;
        // std::cout << "SO3 error = " << err << std::endl; 
        outFile << err << std::endl;
    }
    outFile.close();

    rot_SO3_error = rot_SO3_error/((double)corr_pairs.size());
    // std::cout << "Average SO3 error on LRF = " << rot_SO3_error << std::endl;  
    return rot_SO3_error;
}

std::vector<std::pair<int,int>> cc::compute_corrs_with_gt(const open3d::geometry::PointCloud& src, 
                                                   const open3d::geometry::PointCloud& tgt,
                                                   const Eigen::Matrix4d rigid_map_gt) {
    
    // transform create copy and transform src point cloud
    auto src_transformed = std::make_shared<open3d::geometry::PointCloud>(src);
    src_transformed->Transform(rigid_map_gt);

    // create kd-tree on tgt point cloud
    open3d::geometry::KDTreeFlann tgt_tree;
    tgt_tree.SetGeometry(tgt);

    // create vector in which the result will be stored
    std::vector<std::pair<int,int>> corr_pairs;
    corr_pairs.resize(src_transformed->points_.size());

    // calculate nearest neighbors / correspondences
	#pragma omp parallel for
		for (int i=0; i < (int)src_transformed->points_.size(); i++) {
            std::vector<int> indices_kd(1);
            std::vector<double> distances2_kd(1);
            tgt_tree.SearchKNN(src_transformed->points_[i], 1, indices_kd, distances2_kd);
            corr_pairs[i].first  = i;
            corr_pairs[i].second = indices_kd[0];            
		}    
    
    return corr_pairs;
}

std::vector<Eigen::Matrix4d> cc::read_trajectory(std::string traj_path) {

    std::vector<Eigen::Matrix4d> gt44_matrices;

    std::ifstream file(traj_path);
    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
    }   

    std::string line;
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
    std::cout << "Number of estimated roto-translations in file = " << gt44_matrices.size() << std::endl;

    return gt44_matrices;
}


void cc::evaluate_trajectory_quality(std::string gt_traj_path, std::string est_traj_path) {
    // gt_traj and est_traj are paths to txt files with ground truth and estimated trajectories
    // each row of the txt file consists of 12 entries of the 4x4 homogeneous matrix (omitting 
    // the last [0 0 0 1] row).

    std::vector<Eigen::Matrix4d> gt_traj = cc::read_trajectory(gt_traj_path);
    std::vector<Eigen::Matrix4d> est_traj = cc::read_trajectory(est_traj_path);

    if (gt_traj.size() != est_traj.size()) {
        std::cout << "Error: trajectories have different size!\n";
    }

    std::cout << "Evaluating the provided estimated trajectory" << std::endl;

    double rel_rot_error{0.0}, rel_tra_error{0.0};

    int num_fails {0};

    for (int i=0; i<gt_traj.size(); i++) {
        
        double rel_rot_error_i = cc::angularErrorSO3(gt_traj[i].block<3,3>(0,0), est_traj[i].block<3,3>(0,0));
        double rel_tra_error_i = (gt_traj[i].block<3,1>(0,3) - est_traj[i].block<3,1>(0,3)).norm();

        rel_rot_error += rel_rot_error_i;
        rel_tra_error += rel_tra_error_i;

        if (rel_rot_error_i > 2.0 or rel_tra_error_i > 0.25) {
            num_fails++;
        }

    }

    rel_rot_error = rel_rot_error/((double)gt_traj.size()); 
    rel_tra_error = rel_tra_error/((double)gt_traj.size()); 

    int num_test_cases = gt_traj.size();
    std::cout << "Average translation error = " << rel_tra_error << std::endl;
    std::cout << "Average rotation    error = " << rel_rot_error << std::endl;
    std::cout << "Success rate              = " << double(num_test_cases-num_fails)/double(num_test_cases) << std::endl;
}


std::vector<std::pair<int,int>> cc::compute_nearest_neighbor_correspondences(const open3d::geometry::PointCloud& source, 
                                                                         const open3d::geometry::PointCloud& target,
                                                                         const open3d::geometry::KDTreeFlann& kd_tree_target) {

    // same data as CorrespondencesSet, convinient for visualizing
    std::vector<std::pair<int, int>> correspondences_result;  
    correspondences_result.resize(source.points_.size());
    
    // Compute correspondences in parallel 
	#pragma omp parallel for
		for (int i=0; i < (int)source.points_.size(); i++) {
            std::vector<int> indices_kd(1);
            std::vector<double> distances2_kd(1); // squared distances
            kd_tree_target.SearchKNN(source.points_[i], 1, indices_kd, distances2_kd); // need to find two I think
            correspondences_result[i] = std::make_pair(i, indices_kd[0]);
		}
    return correspondences_result;
}
