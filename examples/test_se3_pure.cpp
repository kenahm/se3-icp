#include <iostream>
#include <iterative_SE3_registration.hpp>
#include <chrono>
#include <thread>
#include <random>

#include <cc.hpp>

// Adapted from: https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/src/Utils.cpp#L194
// Converts a Rotation Matrix to Euler angles
// Convention used is Y-Z-X Tait-Bryan angles
// Reference code implementation:
// https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToEuler/index.htm
Eigen::Vector3d rot2euler(const Eigen::Matrix3d& rotationMatrix)
{
    Eigen::Vector3d euler;

    double m00 = rotationMatrix(0,0);
    double m02 = rotationMatrix(0,2);
    double m10 = rotationMatrix(1,0);
    double m11 = rotationMatrix(1,1);
    double m12 = rotationMatrix(1,2);
    double m20 = rotationMatrix(2,0);
    double m22 = rotationMatrix(2,2);


    double bank, attitude, heading;

    // Assuming the angles are in radians.
    if (m10 > 0.998) { // singularity at north pole
        bank = 0;
        attitude = M_PI/2;
        heading = std::atan2(m02,m22);
    }
    else if (m10 < -0.998) { // singularity at south pole
        bank = 0;
        attitude = -M_PI/2;
        heading = std::atan2(m02,m22);
    }
    else
    {
        bank = std::atan2(-m12,m11);
        attitude = std::asin(m10);
        heading = std::atan2(-m20,m00);
    }

    euler(0) = bank;
    euler(1) = attitude;
    euler(2) = heading;

    return euler;
}

// Error metric that many papers use on lounge dataset (avg euler angle deviation)
double angleDifference(double angle1, double angle2) {
    double diff = fmod(angle1 - angle2, 360.0);
    if (diff > 180.0) diff = 360.0 - diff;
    return std::abs(diff);
}

double avgEulError (const Eigen::Matrix3d& rot_mat_1, const Eigen::Matrix3d& rot_mat_2) {
    Eigen::Vector3d E = rot2euler(rot_mat_1);
    Eigen::Vector3d K = rot2euler(rot_mat_2);

    // Turn to degrees
    E[0] *= (180.0 / M_PI); E[1] *= (180.0 / M_PI); E[2] *= (180.0 / M_PI);
    K[0] *= (180.0 / M_PI); K[1] *= (180.0 / M_PI); K[2] *= (180.0 / M_PI);
    
    // Make sure it is in (0, 360) range
    E[0] = fmod(E[0], 360.0); E[1] = fmod(E[1], 360.0); E[2] = fmod(E[2], 360.0);
    K[0] = fmod(K[0], 360.0); K[1] = fmod(K[1], 360.0); K[2] = fmod(K[2], 360.0);

    double diff1 = angleDifference(E(0), K(0));
    double diff2 = angleDifference(E(1), K(1));
    double diff3 = angleDifference(E(2), K(2));

    // Compute average error
    double avgError = (diff1 + diff2 + diff3) / 3.0;    
    
    return avgError;
}

// from http://redwood-data.org/indoor/fileformat.html
// struct to hold ground truth transformation
struct FramedTransformation {
    int id1_;
    int id2_;
    int frame_;
    Eigen::Matrix4d transformation_;
    FramedTransformation( int id1, int id2, int f, Eigen::Matrix4d t )
        : id1_( id1 ), id2_( id2 ), frame_( f ), transformation_( t )
    {}
};

// from http://redwood-data.org/indoor/fileformat.html
// struct to load ground truth transformation file
struct RGBDTrajectory {
    std::vector< FramedTransformation > data_;
    int index_;

    void LoadFromFile( std::string filename ) {
        data_.clear();
        index_ = 0;
        int id1, id2, frame;
        Eigen::Matrix4d trans;
        FILE * f = fopen( filename.c_str(), "r" );
        if ( f != NULL ) {
            char buffer[1024];
            while ( fgets( buffer, 1024, f ) != NULL ) {
                if ( strlen( buffer ) > 0 && buffer[ 0 ] != '#' ) {
                    sscanf( buffer, "%d %d %d", &id1, &id2, &frame);
                    fgets( buffer, 1024, f );
                    sscanf( buffer, "%lf %lf %lf %lf", &trans(0,0), &trans(0,1), &trans(0,2), &trans(0,3) );
                    fgets( buffer, 1024, f );
                    sscanf( buffer, "%lf %lf %lf %lf", &trans(1,0), &trans(1,1), &trans(1,2), &trans(1,3) );
                    fgets( buffer, 1024, f );
                    sscanf( buffer, "%lf %lf %lf %lf", &trans(2,0), &trans(2,1), &trans(2,2), &trans(2,3) );
                    fgets( buffer, 1024, f );
                    sscanf( buffer, "%lf %lf %lf %lf", &trans(3,0), &trans(3,1), &trans(3,2), &trans(3,3) );
                    data_.push_back( FramedTransformation( id1, id2, frame, trans ) );
                }
            }
            fclose( f );
        }
    }
    void SaveToFile( std::string filename ) {
        FILE * f = fopen( filename.c_str(), "w" );
        for ( int i = 0; i < ( int )data_.size(); i++ ) {
            Eigen::Matrix4d & trans = data_[ i ].transformation_;
            fprintf( f, "%d\t%d\t%d\n", data_[ i ].id1_, data_[ i ].id2_, data_[ i ].frame_ );
            fprintf( f, "%.8f %.8f %.8f %.8f\n", trans(0,0), trans(0,1), trans(0,2), trans(0,3) );
            fprintf( f, "%.8f %.8f %.8f %.8f\n", trans(1,0), trans(1,1), trans(1,2), trans(1,3) );
            fprintf( f, "%.8f %.8f %.8f %.8f\n", trans(2,0), trans(2,1), trans(2,2), trans(2,3) );
            fprintf( f, "%.8f %.8f %.8f %.8f\n", trans(3,0), trans(3,1), trans(3,2), trans(3,3) );
        }
        fclose( f );
    }
};

void benchmark_algorithm_lounge(const std::string& algorithm_name, const std::string& lounge_folder_path, double rot_alpha=3.0, bool write_results=false) {
    std::string traj_path {lounge_folder_path + "/lounge_data/lounge_trajectory.log"};
    RGBDTrajectory trajectory_gt;
    trajectory_gt.LoadFromFile(traj_path);

    double avg_iterations{0.0}, avg_total_time{0.0}, 
           avg_rot_frob_error{0.0}, avg_tra_error{0.0}, 
           avg_angular_SO3_error {0.0}, avg_eul_angle_error {0.0};  

    open3d::utility::random::Seed(42);

    int total_num_cases {0};
    for (int i=1; i < 395; i+=5) {
        total_num_cases++;

        int num_length = std::to_string(i).length();
        std::string source_cloud_name = lounge_folder_path + "/lounge_data/" + std::string(6 - num_length, '0') + std::to_string(i) + ".ply";
        num_length = std::to_string(i+5).length();
        std::string target_cloud_name = lounge_folder_path + "/lounge_data/" + std::string(6 - num_length, '0') + std::to_string(i+5) + ".ply";

        // Using LSG-CPD already downsampled clouds
        std::cout << "Registering point clouds: \n" <<
        source_cloud_name << std::endl <<
        target_cloud_name << std::endl << "===\n";

        auto source_ds_ptr = open3d::io::CreatePointCloudFromFile(source_cloud_name); 
        auto target_ds_ptr = open3d::io::CreatePointCloudFromFile(target_cloud_name); 

        // Extract ground truth transformation matrix
        Eigen::Matrix4d T1 = trajectory_gt.data_[(i-1)].transformation_; 
        Eigen::Matrix4d T2 = trajectory_gt.data_[(i-1)+5].transformation_; 
        Eigen::Matrix4d T12 = (T2.inverse()) * T1;
        Eigen::Matrix3d rot_gt = T12.block<3,3>(0,0);
        Eigen::Vector3d tra_gt = T12.block<3,1>(0,3);    

        // Setup and execute ICP algorithm
        IterativeSE3Registration reg_obj_icp;
        reg_obj_icp.setSourceCloud(*source_ds_ptr);
        reg_obj_icp.setTargetCloud(*target_ds_ptr);

        // set up parameters
        reg_obj_icp.estimated_overlap_ = 0.75;
        reg_obj_icp.number_of_nn_for_LRF_ = 90;   
        reg_obj_icp.mse_switch_error_ = 0.00005;  
        reg_obj_icp.max_num_se3_iterations_ = 30; // tried 50, 100, results are similarish as 30, but slower

        // reg_obj_icp.mse_ = 0.00001; // default value, used in benchmark_lounge.cpp
        // reg_obj_icp.mse_ = 0.0000001; // basically almost the same precision, but slower

        // rotation scale parameter
        reg_obj_icp.alpha_rot = rot_alpha;

        // perform alignment
        auto tx2 = std::chrono::high_resolution_clock::now(); 

        if (algorithm_name=="se3_pure_pt2pt" || algorithm_name=="se3_pure_pt2pl" || algorithm_name=="se3_pure_gicp") {
            
            if (algorithm_name=="se3_pure_pt2pt") reg_obj_icp.run_se3_pure("pt2pt"); 
            else if (algorithm_name=="se3_pure_pt2pl") reg_obj_icp.run_se3_pure("pt2pl");
            else if (algorithm_name=="se3_pure_gicp") reg_obj_icp.run_se3_pure("gicp");
            else { // should not happen
                std::cerr << "Invalid algorithm name [should not happen]\n";
                break;
            }
        }
        else {
            std::cerr << "Invalid algorithm name!\n";
            break;
        }

        auto ty2 = std::chrono::high_resolution_clock::now();
        double singlev2 = std::chrono::duration_cast<std::chrono::nanoseconds>(ty2 - tx2).count() / 1e6;

        // compute and print results, errors
        std::cout << "num iterations = " << reg_obj_icp.num_iterations_ << std::endl;
        std::cout << "time for alignment [ms] = " << singlev2 << std::endl;    
        Eigen::Matrix3d est_rot_icp_i = reg_obj_icp.current_estimated_T_.block<3,3>(0,0);
        Eigen::Vector3d est_tra_icp_i = reg_obj_icp.current_estimated_T_.block<3,1>(0,3);
        double icp_rot_fro_error_i = (rot_gt - est_rot_icp_i).norm();
        double icp_tra_error_i = (tra_gt - est_tra_icp_i).norm();
        double icp_so3_rot_error_i = cc::angularErrorSO3(est_rot_icp_i, rot_gt);
        double icp_avg_eul_angle_error_i = avgEulError(est_rot_icp_i, rot_gt);
        std::cout << "rot fro norm error = " << icp_rot_fro_error_i << std::endl;
        std::cout << "translation  error = " << icp_tra_error_i << std::endl;
        std::cout << "SO3 rotation error = " << icp_so3_rot_error_i << std::endl;        
        std::cout << "euler rot error    = " << icp_avg_eul_angle_error_i << std::endl;        

        // store results
        avg_iterations += reg_obj_icp.num_iterations_;
        avg_total_time += singlev2;
        avg_rot_frob_error += icp_rot_fro_error_i;
        avg_angular_SO3_error += icp_so3_rot_error_i;
        avg_tra_error += icp_tra_error_i;
        avg_eul_angle_error += icp_avg_eul_angle_error_i;
    }

    // Average out and print final results
    avg_iterations /= (double)total_num_cases ;
    avg_total_time /= (double)total_num_cases ;
    avg_rot_frob_error /= (double)total_num_cases ;
    avg_angular_SO3_error /= (double)total_num_cases ;
    avg_tra_error /= (double)total_num_cases ; 
    avg_eul_angle_error /= (double)total_num_cases ;   

    std::cout << "=== Final results of algorithm: " << algorithm_name << " ===" << std::endl;
    std::cout << "avg_total_time = " << avg_total_time << std::endl;
    std::cout << "avg_iterations = " << avg_iterations << std::endl;
    std::cout << "avg_tra_error  = " << avg_tra_error << std::endl;
    std::cout << "avg_rot_frob_error  = " << avg_rot_frob_error << std::endl;
    std::cout << "avg_angular_SO3_error  = " << avg_angular_SO3_error << std::endl;    
    std::cout << "avg_eul_angle_error    = " << avg_eul_angle_error << std::endl;

    if (write_results) {
        std::string filename = "../experiments_rot_scale_lounge.txt";

        // Open in append mode: creates the file if it doesn't exist
        std::ofstream file(filename, std::ios::app);
        if (!file) {
            std::cerr << "Error opening file: " << filename << std::endl;
        }

        // Write value followed by newline
        file << algorithm_name << " PURE | " << "rot_alpha = " << rot_alpha << " | avg_eul_angle_error = " << avg_eul_angle_error << "\n";
        file.close(); 
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


void benchmark_different_rot_scales(const std::string& algorithm_name, const std::string& lounge_folder_path) {
    
    auto rot_alphas = makeHybridLGrid();
    for (const auto& alpha : rot_alphas) {
        benchmark_algorithm_lounge(algorithm_name, lounge_folder_path, alpha, true);
    }
}

int main(int argc, char* argv[]) {

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <AlgorithmName> <FolderPath>" << std::endl;
        return 1;
    }

    std::string algorithmName = argv[1];
    std::string folderPath = argv[2];

    // this is standard single run test
    // benchmark_algorithm_lounge(algorithmName, folderPath);

    // test for different rotation scales accross the dataset
    benchmark_different_rot_scales(algorithmName, folderPath);

    return 0;
}