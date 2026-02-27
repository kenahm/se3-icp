#include <iostream>
#include <iterative_SE3_registration.hpp>
#include <chrono>
#include <thread>
#include <random>

#include <cc.hpp>

// stuff to load the dataset
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

// stuff to load the dataset
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

void benchmark_algorithm_kitti(const std::string& algorithm_name, const std::string& kitti_folder_path, double rot_alpha=3.0, bool write_results=false) {

    //////////////////// Read all point clouds and all GT matrices //////////////////////
    std::vector<Eigen::Matrix4d> gt44_matrices;
    std::ifstream file(kitti_folder_path + "/Sequence_07/07.txt");
    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
    }    
    
    std::string line;
    bool readLine {true};
    while (std::getline(file, line)) {

        if (readLine) {
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
        readLine = !readLine;
    }
    file.close();

    std::vector<std::shared_ptr<open3d::geometry::PointCloud>> all_clouds; 
    for (int i=0; i<=1100; i+=2) {
        int num_length = std::to_string(i).length();
        std::string cloud_file_name = kitti_folder_path + "/Sequence_07/Downsampled/" + std::string(6 - num_length, '0') + std::to_string(i) + ".ply";
        auto cloud = open3d::io::CreatePointCloudFromFile(cloud_file_name);
        all_clouds.push_back(cloud);
    }
    /////////////////////////////////////////////////////////////////////////////////////////       

    // Initialize errors and various metrics
    double rel_rot_error{0.0}, rel_tra_error{0.0}; 
    double max_rel_rot_error{0.0}, max_rel_tra_error{0.0};
    double abs_rot_error{0.0}, abs_tra_error{0.0};
    double max_abs_rot_error{0.0}, max_abs_tra_error{0.0};
    double last_frame_rot_error {-1.0}, last_frame_tra_error{-1.0};
    double time {0.0};
    Eigen::Matrix4d current_estimated_transf {Eigen::Matrix4d::Identity()}; // will be used to compute absolute errors    

    int counter=0;
    for (int i=0; i < all_clouds.size()-1; i++) {

        std::cout << "Solving kitti seq 07 registration problem #" << i << " using " << algorithm_name << " algorithm\n";

        auto gt_matrix = (gt44_matrices[i].inverse()) * gt44_matrices[i+1];
        Eigen::Matrix3d gt_rot = gt_matrix.block<3,3>(0,0);
        Eigen::Vector3d gt_tra = gt_matrix.block<3,1>(0,3);         

        IterativeSE3Registration reg_obj;

        reg_obj.setSourceCloud(*all_clouds[i+1]);
        reg_obj.setTargetCloud(*all_clouds[i]);

        reg_obj.max_num_se3_iterations_ = 10;
        reg_obj.number_of_nn_for_LRF_ = 90;  

        // rotation scale parameter
        reg_obj.alpha_rot = rot_alpha;

        auto t_start = std::chrono::high_resolution_clock::now();
        // Empirically verified that se3-icp and vanilla-icp algorithms prefer slightly different rejection rates for best results
        if (algorithm_name=="pt2pt" || algorithm_name=="pt2pl" || algorithm_name=="gicp") {
            reg_obj.estimated_overlap_ = 0.8;
            reg_obj.run_icp(algorithm_name);
        }
        else if (algorithm_name=="se3_pt2pt" || algorithm_name=="se3_pt2pl" || algorithm_name=="se3_gicp") {
            reg_obj.estimated_overlap_ = 0.7;
            reg_obj.mse_ = 0.0000001;
            reg_obj.mse_switch_error_ = 5 * reg_obj.mse_;
            if (algorithm_name=="se3_pt2pt") 
                reg_obj.run_se3_icp("pt2pt");
            else if (algorithm_name=="se3_pt2pl") 
                reg_obj.run_se3_icp("pt2pl");
            else if (algorithm_name=="se3_gicp") 
                reg_obj.run_se3_icp("gicp");
            else {
                std::cout << "Invalid algorithm name!\n";
            }
        }
        else { 
            std::cerr << "Invalid algorithm name. Possible names: pt2pt, pt2pl, gicp, se3_pt2pt, se3_pt2pl, se3_gicp \n"; 
        }        

        auto t_finish = std::chrono::high_resolution_clock::now();
        double t_total = std::chrono::duration_cast<std::chrono::nanoseconds>(t_finish - t_start).count() / 1e6;
        time+= t_total;

        // Collect estimated results
        Eigen::Matrix3d estimated_R = reg_obj.current_estimated_T_.block<3,3>(0,0);
        Eigen::Vector3d estimated_t = reg_obj.current_estimated_T_.block<3,1>(0,3); 
        current_estimated_transf = current_estimated_transf *  reg_obj.current_estimated_T_;
        Eigen::Matrix3d accumulated_R = current_estimated_transf.block<3,3>(0,0);
        Eigen::Vector3d accumulated_t = current_estimated_transf.block<3,1>(0,3);         

        // Calculate and update errors
        double rel_rot_error_i = cc::angularErrorSO3(estimated_R, gt_rot);
        double rel_tra_error_i = (estimated_t - gt_tra).norm();
        double abs_rot_error_i = cc::angularErrorSO3(accumulated_R, gt44_matrices[i+1].block<3,3>(0,0));
        double abs_tra_error_i = (accumulated_t - gt44_matrices[i+1].block<3,1>(0,3)).norm();      

        if (rel_rot_error_i > max_rel_rot_error) max_rel_rot_error = rel_rot_error_i;
        if (rel_tra_error_i > max_rel_tra_error) max_rel_tra_error = rel_tra_error_i;
        if (abs_rot_error_i > max_abs_rot_error) max_abs_rot_error = abs_rot_error_i;
        if (abs_tra_error_i > max_abs_tra_error) max_abs_tra_error = abs_tra_error_i;


        rel_rot_error += rel_rot_error_i;
        rel_tra_error += rel_tra_error_i;
        abs_rot_error += abs_rot_error_i;
        abs_tra_error += abs_tra_error_i;

        if (i==all_clouds.size()-2) {
            last_frame_rot_error = abs_rot_error_i;
            last_frame_tra_error = abs_tra_error_i;
        }         

        counter++; 
    }

    std::cout << "===== Kitti sequence 07 results of algorithm: " << algorithm_name << " =====\n";
    std::cout << "Mean(max) REL rot error = " << rel_rot_error/((double)counter) << " (" << max_rel_rot_error << ")" << std::endl;
    std::cout << "Mean(max) REL tra error = " << rel_tra_error/((double)counter) << " (" << max_rel_tra_error << ")" << std::endl;
    std::cout << "Mean(max) ABS rot error = " << abs_rot_error/((double)counter) << " (" << max_abs_rot_error << ")" << std::endl;
    std::cout << "Mean(max) ABS tra error = " << abs_tra_error/((double)counter) << " (" << max_abs_tra_error << ")" << std::endl;
    std::cout << "Last frame rot error    = " << last_frame_rot_error << std::endl;
    std::cout << "Last frame tra error    = " << last_frame_tra_error << std::endl;
    std::cout << "Avg time = " << time/((double)counter) << std::endl;

    if (write_results) {
        std::string filename = "../experiments_rot_scale_kitti.txt";

        // Open in append mode: creates the file if it doesn't exist
        std::ofstream file(filename, std::ios::app);
        if (!file) {
            std::cerr << "Error opening file: " << filename << std::endl;
        }

        // Write value followed by newline
        file << algorithm_name << " | " << "rot_alpha = " << rot_alpha << " | last_frame_rot_error = " << last_frame_rot_error << " | last_frame_tra_error = " << last_frame_tra_error << "\n";
        file.close(); 
        }

}


void benchmark_algorithm_kitti_fgr(const std::string& kitti_folder_path) {

    //////////////////// Read all point clouds and all GT matrices //////////////////////
    std::vector<Eigen::Matrix4d> gt44_matrices;
    std::ifstream file(kitti_folder_path + "/Sequence_07/07.txt");
    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
    }    
    
    std::string line;
    bool readLine {true};
    while (std::getline(file, line)) {

        if (readLine) {
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
        readLine = !readLine;
    }
    file.close();

    std::vector<std::shared_ptr<open3d::geometry::PointCloud>> all_clouds; 
    for (int i=0; i<=1100; i+=2) {
        int num_length = std::to_string(i).length();
        std::string cloud_file_name = kitti_folder_path + "/Sequence_07/Downsampled/" + std::string(6 - num_length, '0') + std::to_string(i) + ".ply";
        auto cloud = open3d::io::CreatePointCloudFromFile(cloud_file_name);
        all_clouds.push_back(cloud);
    }
    /////////////////////////////////////////////////////////////////////////////////////////       

    // Initialize errors and various metrics
    double rel_rot_error{0.0}, rel_tra_error{0.0}; 
    double max_rel_rot_error{0.0}, max_rel_tra_error{0.0};
    double abs_rot_error{0.0}, abs_tra_error{0.0};
    double max_abs_rot_error{0.0}, max_abs_tra_error{0.0};
    double last_frame_rot_error {-1.0}, last_frame_tra_error{-1.0};
    double time {0.0};
    Eigen::Matrix4d current_estimated_transf {Eigen::Matrix4d::Identity()}; // will be used to compute absolute errors    

    int counter=0;
    for (int i=0; i < all_clouds.size()-1; i++) {

        std::cout << "Solving kitti seq 07 registration problem #" << i << " using <<FGR>> algorithm\n";

        auto gt_matrix = (gt44_matrices[i].inverse()) * gt44_matrices[i+1];
        Eigen::Matrix3d gt_rot = gt_matrix.block<3,3>(0,0);
        Eigen::Vector3d gt_tra = gt_matrix.block<3,1>(0,3);         

        // reg_obj.setSourceCloud(*all_clouds[i+1]);
        // reg_obj.setTargetCloud(*all_clouds[i]);

        auto source_ds_ptr = std::make_shared<open3d::geometry::PointCloud>(*all_clouds[i+1]);
        auto target_ds_ptr = std::make_shared<open3d::geometry::PointCloud>(*all_clouds[i]);

        // Setup FGR algorithm
        open3d::pipelines::registration::FastGlobalRegistrationOption option; // use default parameters
        open3d::pipelines::registration::RegistrationResult fgr_result;

        auto t_start = std::chrono::high_resolution_clock::now();

        source_ds_ptr->EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(30));
        target_ds_ptr->EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(30));

        auto source_fpfh = open3d::pipelines::registration::ComputeFPFHFeature(*source_ds_ptr, open3d::geometry::KDTreeSearchParamKNN(100));
        auto target_fpfh = open3d::pipelines::registration::ComputeFPFHFeature(*target_ds_ptr, open3d::geometry::KDTreeSearchParamKNN(100));

        fgr_result = open3d::pipelines::registration::FastGlobalRegistrationBasedOnFeatureMatching(*source_ds_ptr, *target_ds_ptr, *source_fpfh, *target_fpfh, option);
        
        auto t_finish = std::chrono::high_resolution_clock::now();
        double t_total = std::chrono::duration_cast<std::chrono::nanoseconds>(t_finish - t_start).count() / 1e6;
        time+= t_total;

        // Collect estimated results
        Eigen::Matrix3d estimated_R = fgr_result.transformation_.block<3,3>(0,0);
        Eigen::Vector3d estimated_t = fgr_result.transformation_.block<3,1>(0,3); 
        current_estimated_transf = current_estimated_transf *  fgr_result.transformation_;
        Eigen::Matrix3d accumulated_R = current_estimated_transf.block<3,3>(0,0);
        Eigen::Vector3d accumulated_t = current_estimated_transf.block<3,1>(0,3);         

        // Calculate and update errors
        double rel_rot_error_i = cc::angularErrorSO3(estimated_R, gt_rot);
        double rel_tra_error_i = (estimated_t - gt_tra).norm();
        double abs_rot_error_i = cc::angularErrorSO3(accumulated_R, gt44_matrices[i+1].block<3,3>(0,0));
        double abs_tra_error_i = (accumulated_t - gt44_matrices[i+1].block<3,1>(0,3)).norm();      

        if (rel_rot_error_i > max_rel_rot_error) max_rel_rot_error = rel_rot_error_i;
        if (rel_tra_error_i > max_rel_tra_error) max_rel_tra_error = rel_tra_error_i;
        if (abs_rot_error_i > max_abs_rot_error) max_abs_rot_error = abs_rot_error_i;
        if (abs_tra_error_i > max_abs_tra_error) max_abs_tra_error = abs_tra_error_i;

        rel_rot_error += rel_rot_error_i;
        rel_tra_error += rel_tra_error_i;
        abs_rot_error += abs_rot_error_i;
        abs_tra_error += abs_tra_error_i;

        if (i==all_clouds.size()-2) {
            last_frame_rot_error = abs_rot_error_i;
            last_frame_tra_error = abs_tra_error_i;
        }         

        counter++; 
    }

    std::cout << "===== Kitti sequence 07 results of <<FGR>> algorithm =====\n";
    std::cout << "Mean(max) REL rot error = " << rel_rot_error/((double)counter) << " (" << max_rel_rot_error << ")" << std::endl;
    std::cout << "Mean(max) REL tra error = " << rel_tra_error/((double)counter) << " (" << max_rel_tra_error << ")" << std::endl;
    std::cout << "Mean(max) ABS rot error = " << abs_rot_error/((double)counter) << " (" << max_abs_rot_error << ")" << std::endl;
    std::cout << "Mean(max) ABS tra error = " << abs_tra_error/((double)counter) << " (" << max_abs_tra_error << ")" << std::endl;
    std::cout << "Last frame rot error    = " << last_frame_rot_error << std::endl;
    std::cout << "Last frame tra error    = " << last_frame_tra_error << std::endl;
    std::cout << "Avg time = " << time/((double)counter) << std::endl;

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


void benchmark_different_rot_scales(const std::string& algorithm_name, const std::string& kitti_folder_path) {
    std::vector<double> rot_alphas = makeHybridLGrid();

    for (const auto& alpha : rot_alphas) {
        benchmark_algorithm_kitti(algorithm_name, kitti_folder_path, alpha, true);
    }
}

void evaluate_kitti_registration_difficulty(const std::string& kitti_folder_path) {
    // Will be used to evaluate kitti seq 07 registration difficulty w.r.t. some 
    // criteria, e.g., how noisy registration pairs are, etc.

    std::cout << "Evaluating kitti dataset w.r.t. some measures\n";

    //////////////////// Read all point clouds and all GT matrices //////////////////////
    std::vector<Eigen::Matrix4d> gt44_matrices;
    std::ifstream file(kitti_folder_path + "/Sequence_07/07.txt");
    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
    }    
    
    std::string line;
    bool readLine {true};
    while (std::getline(file, line)) {

        if (readLine) {
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
        readLine = !readLine;
    }
    file.close();

    std::vector<std::shared_ptr<open3d::geometry::PointCloud>> all_clouds; 
    for (int i=0; i<=1100; i+=2) {
        int num_length = std::to_string(i).length();
        std::string cloud_file_name = kitti_folder_path + "/Sequence_07/Downsampled/" + std::string(6 - num_length, '0') + std::to_string(i) + ".ply";
        auto cloud = open3d::io::CreatePointCloudFromFile(cloud_file_name);
        all_clouds.push_back(cloud);
    }
    /////////////////////////////////////////////////////////////////////////////////////////          

    double average_point_cloud_distance = 0.0;
    double average_point_cloud_distance_top_k = 0.0;
    for (int i=0; i < all_clouds.size()-1; i++) {

        auto gt_matrix = (gt44_matrices[i].inverse()) * gt44_matrices[i+1];
        Eigen::Matrix3d gt_rot = gt_matrix.block<3,3>(0,0);
        Eigen::Vector3d gt_tra = gt_matrix.block<3,1>(0,3);   

        std::shared_ptr<open3d::geometry::PointCloud> source_cloud_ptr {new open3d::geometry::PointCloud};
        std::shared_ptr<open3d::geometry::PointCloud> target_cloud_ptr {new open3d::geometry::PointCloud};

        for (const auto& point : all_clouds[i+1]->points_) {
            source_cloud_ptr->points_.push_back(point);
        }

        for (const auto& point : all_clouds[i]->points_) {
            target_cloud_ptr->points_.push_back(point);     
        }    

        auto vec_mean_value = [](const std::vector<double>& v) {
            double sum = 0.0;
            for (const auto& val : v) {
                sum += val;
            }
            return sum / static_cast<double>(v.size());
        };


        // Now evaluate some measures on source_cloud_ptr and target_cloud_ptr
        auto starting_distance = source_cloud_ptr->ComputePointCloudDistance(*target_cloud_ptr);
        std::cout << "starting_distance mean =  " << vec_mean_value(starting_distance) << std::endl;

        source_cloud_ptr->Transform(gt_matrix);
        auto after_transform_distance = source_cloud_ptr->ComputePointCloudDistance(*target_cloud_ptr);

        // Take first the top 70% smallest distances to avoid outliers
        std::sort(after_transform_distance.begin(), after_transform_distance.end());
        size_t top_k = static_cast<size_t>(0.7 * after_transform_distance.size());
        std::vector<double> top_k_distances(after_transform_distance.begin(), after_transform_distance.begin() + top_k);
        average_point_cloud_distance_top_k += vec_mean_value(top_k_distances);

        average_point_cloud_distance += vec_mean_value(after_transform_distance);


        std::cout << "after_transform_distance mean =  " << vec_mean_value(after_transform_distance) << std::endl;
        std::cout << "after_transform_distance mean (top 70%) =  " << vec_mean_value(top_k_distances) << std::endl;
        std::cout << "---------------------------------------------\n";
    }

    average_point_cloud_distance /= static_cast<double>(all_clouds.size()-1);
    std::cout << "=============================================\n";
    std::cout << "Overall average point cloud distance = " << average_point_cloud_distance << std::endl;  
    std::cout << "Overall average point cloud distance (top 70%) = " << average_point_cloud_distance_top_k / static_cast<double>(all_clouds.size()-1) << std::endl;
}


void evaluate_lounge_avg_chamfer_distance(const std::string& kitti_folder_path) {

    //////////////////// Read all point clouds and all GT matrices //////////////////////
    std::vector<Eigen::Matrix4d> gt44_matrices;
    std::ifstream file(kitti_folder_path + "/Sequence_07/07.txt");
    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
    }    
    
    std::string line;
    bool readLine {true};
    while (std::getline(file, line)) {

        if (readLine) {
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
        readLine = !readLine;
    }
    file.close();

    std::vector<std::shared_ptr<open3d::geometry::PointCloud>> all_clouds; 
    for (int i=0; i<=1100; i+=2) {
        int num_length = std::to_string(i).length();
        std::string cloud_file_name = kitti_folder_path + "/Sequence_07/Downsampled/" + std::string(6 - num_length, '0') + std::to_string(i) + ".ply";
        auto cloud = open3d::io::CreatePointCloudFromFile(cloud_file_name);
        all_clouds.push_back(cloud);
    }
    /////////////////////////////////////////////////////////////////////////////////////////          

    double average_chamfer_distance = 0.0;
    for (int i=0; i < all_clouds.size()-1; i++) {

        auto gt_matrix = (gt44_matrices[i].inverse()) * gt44_matrices[i+1];
        Eigen::Matrix3d gt_rot = gt_matrix.block<3,3>(0,0);
        Eigen::Vector3d gt_tra = gt_matrix.block<3,1>(0,3);   

        std::shared_ptr<open3d::geometry::PointCloud> source_cloud_ptr {new open3d::geometry::PointCloud};
        std::shared_ptr<open3d::geometry::PointCloud> target_cloud_ptr {new open3d::geometry::PointCloud};

        for (const auto& point : all_clouds[i+1]->points_) {
            source_cloud_ptr->points_.push_back(point);
        }

        for (const auto& point : all_clouds[i]->points_) {
            target_cloud_ptr->points_.push_back(point);     
        }    

        auto vec_mean_value_of_squares = [](const std::vector<double>& v) {
            double sum = 0.0;
            for (const auto& val : v) {
                sum += (val*val);
            }
            return sum / static_cast<double>(v.size());
        };

        source_cloud_ptr->Transform(gt_matrix);
        auto L_distances = source_cloud_ptr->ComputePointCloudDistance(*target_cloud_ptr); // computes for each point in source the distance to nearest point in target
        auto R_distances = target_cloud_ptr->ComputePointCloudDistance(*source_cloud_ptr);  

        // As defined in "A Point Set Generation Network for 3D Object Reconstruction from a Single Image", Fan et al., CVPR 2017
        double chamfer_dist = vec_mean_value_of_squares(L_distances) + vec_mean_value_of_squares(R_distances);
        
        average_chamfer_distance += chamfer_dist;

        // std::cout << "L dist = " << vec_mean_value_of_squares(L_distances) << std::endl;
        // std::cout << "R dist = " << vec_mean_value_of_squares(R_distances) << std::endl;
        // std::cout << "Chamfer distance (kitti) =  " << chamfer_dist << std::endl;
        // std::cout << "--------------------------------------\n";        
    }
    
    average_chamfer_distance /= static_cast<double>(all_clouds.size()-1);
    std::cout << "=============================================\n";
    std::cout << "Overall average chamfer distance (kitti) = " << average_chamfer_distance << std::endl;
}



int main(int argc, char* argv[]) {

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <AlgorithmName> <FolderPath>" << std::endl;
        return 1;
    }

    std::string algorithmName = argv[1];
    std::string folderPath = argv[2];

    
    ///// Run standard benchmark /////
    if (algorithmName=="pt2pt" || algorithmName=="pt2pl" || algorithmName=="gicp" ||
        algorithmName=="se3_pt2pt" || algorithmName=="se3_pt2pl" || algorithmName=="se3_gicp") {
        benchmark_algorithm_kitti(algorithmName, folderPath);
    }
    else if (algorithmName=="fgr") {
        benchmark_algorithm_kitti_fgr(folderPath);
    }
    else {
        std::cerr << "Not a valid algorithm name\n" 
                    << "Available names are: pt2pt, pt2pl, gicp, se3_pt2pt, se3_pt2pl, se3_gicp, fgr \n"; 
        return 1;
    }
    ////////////////////////////////////
    

    // Evaluate Kitti Difficulty
    // evaluate_kitti_registration_difficulty(folderPath);    
    // evaluate_lounge_avg_chamfer_distance(folderPath);

    // benchmark_different_rot_scales(algorithmName, folderPath);

    /*
    std::vector<double> rot_alphas = makeHybridLGrid();
    for (const auto& alpha : rot_alphas) {
        std::cout << "alpha = " << alpha << std::endl;
    }
    */

    return 0;
}