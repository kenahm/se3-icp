#include "iterative_SE3_registration.hpp"

// Copy-pasted from Open3D because it is a private class member
inline Eigen::Matrix3d GetRotationFromE1ToX(const Eigen::Vector3d &x) {
    const Eigen::Vector3d e1{1, 0, 0};
    const Eigen::Vector3d v = e1.cross(x);
    const double c = e1.dot(x);
    if (c < -0.99) {
        return Eigen::Matrix3d::Identity();
    }
    const Eigen::Matrix3d sv = open3d::utility::SkewMatrix(v);
    const double factor = 1 / (1 + c);
    return Eigen::Matrix3d::Identity() + sv + (sv * sv) * factor;
}

double lounge_point_confidence(const Eigen::Vector3d& v) {
    /*
    Compute confidence of a point v, which only depends on the depth of v.
    Function adapted from the matlab code of LSG-CPD paper, 
    https://github.com/ChirikjianLab/LSG-CPD
    */

    double depth = v(2);

    double p1{0.002203}, p2{-0.001028}, p3{0.0005351}, min_depth{0.4};
    double error = p1 * depth * depth + p2 * depth + p3;
    double confidence = (p1 * min_depth + p2 * min_depth + p3) / error;

    return confidence;
}

// Modified open3d function to directly change the input point cloud, rather then return shared ptr output for a resulting pc
void InitializePointCloudForGeneralizedICP_modified(open3d::geometry::PointCloud &pcd, double epsilon) {
    if (pcd.HasCovariances()) {
        open3d::utility::LogDebug("GeneralizedICP: Using pre-computed covariances.");
        return;
    }
    if (pcd.HasNormals()) {
        open3d::utility::LogDebug("GeneralizedICP: Computing covariances from normals");
    } else {
        // Compute covariances the same way is done in the original GICP paper.
        open3d::utility::LogDebug("GeneralizedICP: Computing covariances from points.");
        pcd.EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(20));
    }
    pcd.covariances_.resize(pcd.points_.size());
    const Eigen::Matrix3d C = Eigen::Vector3d(epsilon, 1, 1).asDiagonal();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)pcd.normals_.size(); i++) {
        const auto Rx = GetRotationFromE1ToX(pcd.normals_[i]);
        pcd.covariances_[i] = Rx * C * Rx.transpose();
    }
}

// Generalized ICP optimization solver where we can include weights easily
// This is a modification of Open3D source code "TransformationEstimationForGeneralizedICP::ComputeTransformation"
// where we add the weights input argument
Eigen::Matrix4d optimize_generalizedICP_manual(
        const open3d::geometry::PointCloud &source,
        const open3d::geometry::PointCloud &target,
        const open3d::pipelines::registration::CorrespondenceSet &corres, 
        std::vector<double> &weights) {
    
    if (corres.empty() || !target.HasCovariances() ||
        !source.HasCovariances()) {
        return Eigen::Matrix4d::Identity();
    }

    auto compute_jacobian_and_residual =
            [&](int i,
                std::vector<Eigen::Vector6d, open3d::utility::Vector6d_allocator> &J_r,
                std::vector<double> &r, std::vector<double> &w) {
                const Eigen::Vector3d &vs = source.points_[corres[i][0]];
                const Eigen::Matrix3d &Cs = source.covariances_[corres[i][0]];
                const Eigen::Vector3d &vt = target.points_[corres[i][1]];
                const Eigen::Matrix3d &Ct = target.covariances_[corres[i][1]];
                const Eigen::Vector3d d = vs - vt;
                const Eigen::Matrix3d M = Ct + Cs;
                const Eigen::Matrix3d W = weights[i] * M.inverse().sqrt(); // K added weights here

                Eigen::Matrix<double, 3, 6> J;
                J.block<3, 3>(0, 0) = -open3d::utility::SkewMatrix(vs);
                J.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
                J = W * J;

                constexpr int n_rows = 3;
                J_r.resize(n_rows);
                r.resize(n_rows);
                w.resize(n_rows);
                for (size_t i = 0; i < n_rows; ++i) {
                    r[i] = W.row(i).dot(d);
                    // w[i] = kernel_->Weight(r[i]);
                    w[i] = 1.0; // again same as point2plane, under L2Loss these are just ones
                    J_r[i] = J.row(i);
                }
            };

    Eigen::Matrix6d JTJ;
    Eigen::Vector6d JTr;
    double r2 = -1.0;
    std::tie(JTJ, JTr, r2) =
            open3d::utility::ComputeJTJandJTr<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>(
                    compute_jacobian_and_residual, (int)corres.size());

    bool is_success = false;
    Eigen::Matrix4d extrinsic;
    std::tie(is_success, extrinsic) =
            open3d::utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, JTr);

    return is_success ? extrinsic : Eigen::Matrix4d::Identity();
}

double largestDistanceFromGivenPoint(const Eigen::Vector3d& ref_point, const open3d::geometry::PointCloud& cloud) {
    double current_largest {-1.0};
    for (const auto& point : cloud.points_) {
        double dist = (point - ref_point).norm();
        if (dist > current_largest) current_largest = dist;
    }
    return current_largest;
}

Eigen::Matrix4d computeSingleSHOTSE3Frame(const open3d::geometry::PointCloud& cloud, 
                                          const open3d::geometry::KDTreeFlann& kdtree_for_LRF, 
                                          int index_center_point, 
                                          double radius) {
    // Implements SHOTS paper local reference frame
    // Unique Signatures of Histograms for Local Surface Description, Tombari etal.
    // It is similar to PCL's implementation

    Eigen::Matrix4d frame_matrix;
    frame_matrix.setIdentity();

    Eigen::Vector3d central_point (cloud.points_[index_center_point]);
    std::vector<int> indices;
    std::vector<double> distances2; // these are squared distance
    
    kdtree_for_LRF.SearchRadius(central_point, radius, indices, distances2); // find neighborhood within radius ('point support')

    // Now we need to compute weighted covariance matrix
    Eigen::Matrix3d cov_mat_weighted;
    cov_mat_weighted.setZero();
    double total_sum = 0.0;

    std::vector<Eigen::Vector3d> diff_vectors;
    diff_vectors.reserve(indices.size()); // in the end it should have size indices.size()-1

    // central covariance point, will be closest in kd tree
    // hence at index i=0 (its sorted), which we simply skip
    for (int i = 1; i < indices.size(); i++) { 
        double temp_diff = (radius - std::sqrt(distances2[i]));
        diff_vectors.emplace_back(cloud.points_[indices[i]] - central_point);
        cov_mat_weighted += temp_diff * diff_vectors[i-1] * (diff_vectors[i-1].transpose());
        total_sum += temp_diff;
    }
    cov_mat_weighted /= total_sum;

    // Now compute eigendecomposition
    // Eigen::SelfAdjointEigenSolver <Eigen::Matrix3d> solver_adj(cov_mat_weighted); // old way
    Eigen::SelfAdjointEigenSolver <Eigen::Matrix3d> solver_adj;
    solver_adj.compute(cov_mat_weighted);

    // https://eigen.tuxfamily.org/dox-devel/classEigen_1_1SelfAdjointEigenSolver.html#af456f15c1ed7e03a2bca1d3f32075e14
    // k-th COLUMN of solver_adj.eigenvectors() 3x3 matrix is k-th eigenvector AND normalized already
    Eigen::Vector3d eigvals = solver_adj.eigenvalues();
    Eigen::Matrix3d eigvecs = solver_adj.eigenvectors(); 

    Eigen::Vector3d x_plus = eigvecs.col(2); // eigenvector corr to largest  eigvalue as described in the paper 
    Eigen::Vector3d z_plus = eigvecs.col(0); // eigenvector corr to smallest eigvalue as described in the paper

    int Sx_plus {0}, Sz_plus {0}; // counters of positive dot products
    for (const auto& vec : diff_vectors) {
        double dotX = vec.dot(x_plus);
        double dotZ = vec.dot(z_plus);
        if (dotX >= 0) { Sx_plus++; } 
        if (dotZ >= 0) { Sz_plus++; } 
    }

    int num_considered_points {int(indices.size()) - 1};
    
    if (num_considered_points < 5) {
        std::cout << "NOT ENOUGH VALID POINTS\n";
    }

    Sx_plus = 2 * Sx_plus - num_considered_points;
    if (Sx_plus == 0) {
        int points {5};
        int medianIndex {num_considered_points/2};
        for (int i = -points/2; i <= points/2; i++) {
            double dotX = diff_vectors[medianIndex - i].dot(x_plus);
            if (dotX >= 0) { Sx_plus++; } 
        }
        if (Sx_plus < points/2 + 1) {
            x_plus = -x_plus;
        }
    }
    else if (Sx_plus < 0) {
        x_plus = -x_plus;
    }

    Sz_plus = 2 * Sz_plus - num_considered_points;
    if (Sz_plus == 0) {
        int points {5};
        int medianIndex {num_considered_points/2};
        for (int i = -points/2; i <= points/2; i++) {
            double dotZ = diff_vectors[medianIndex - i].dot(z_plus);
            if (dotZ >= 0) { Sz_plus++; } 
        }
        if (Sz_plus < points/2 + 1) {
            z_plus = -z_plus;
        }
    }
    else if (Sz_plus < 0) {
        z_plus = -z_plus;
    }

    Eigen::Vector3d y_plus = z_plus.cross(x_plus);

    frame_matrix.block<3,1>(0,0) = x_plus; // rot axis 1
    frame_matrix.block<3,1>(0,1) = y_plus; // rot axis 2
    frame_matrix.block<3,1>(0,2) = z_plus; // rot axis 3
    frame_matrix.block<3,1>(0,3) = central_point;  // translation part which is equal to original point

    return frame_matrix;
}


void computeAllSHOTSE3FramesOMP(const open3d::geometry::PointCloud& cloud, 
							    const open3d::geometry::KDTreeFlann& kdtree_for_LRF, 
								double radius, 
								std::vector<Eigen::Matrix4d>& result_frames) {
    /*
        Compute SHOT-based SE(3) frames for all points in the input cloud, using OpenMP for parallelization.
        Result is store in the passed result_frames vector.
    */
    result_frames.resize(cloud.points_.size());
    #pragma omp parallel for
        for (int i=0; i < cloud.points_.size(); i++) {
            result_frames[i] = computeSingleSHOTSE3Frame(cloud, kdtree_for_LRF, i, radius);
        }    
}

Eigen::Matrix4d computeSingleTOLDISE3Frame(const open3d::geometry::PointCloud& cloud, 
                                           const open3d::geometry::KDTreeFlann& kdtree_for_LRF, 
                                           Eigen::Vector3d central_point, 
                                           int knn_pts) {

    /* 
        Implemetntation of TOLDI Local Reference Frame, kNN variant.               
    */

    // 1. Get the k nearest neighbors
    std::vector<int> indices;
    std::vector<double> distances2; // SQUARED DISTANCES
    kdtree_for_LRF.SearchKNN(central_point, knn_pts, indices, distances2);

    // they are ordered from closest to less-close in kNN search
    double computed_radius = (central_point - cloud.points_[indices.back()]).norm();

    // 2. Compute centroid withing radius/3
    Eigen::Vector3d centroid_point {0.0, 0.0, 0.0};
    int rz_size {0}; // num of pts within this radius
    for (int i = 1; i < (int)(indices.size()/3); i++) {
        centroid_point += cloud.points_[indices[i]];
    }
    rz_size = (int)(indices.size()/3);
    centroid_point = centroid_point/(double)rz_size;

    // 3. Compute the covariance matrix 
    Eigen::Matrix3d cov_mat {Eigen::Matrix3d::Zero()};
    for (int i=1; i < rz_size+1; i++) { // we exclude the first point
        auto point = cloud.points_[indices[i]] - centroid_point;
        cov_mat += point * (point.transpose()); 
    }    

    // 3. Compute normal as eigenvector corresponding to smallest eigenvalue of cov matrix 
    Eigen::SelfAdjointEigenSolver <Eigen::Matrix3d> solver_adj;
    solver_adj.compute(cov_mat);
    // https://eigen.tuxfamily.org/dox-devel/classEigen_1_1SelfAdjointEigenSolver.html#af456f15c1ed7e03a2bca1d3f32075e14
    // k-th COLUMN of solver_adj.eigenvectors() 3x3 matrix is k-th eigenvector AND normalized already
    Eigen::Vector3d eigvals = solver_adj.eigenvalues();
    Eigen::Matrix3d eigvecs = solver_adj.eigenvectors(); 
    Eigen::Vector3d point_normal = eigvecs.col(0); // eigenvector corr to smallest eigvalue

    // 4. Correct for the sign of the normal  
    // 5. Compute x-axis, in the same for loop 
    // This slightly faster computation is based on my equivalent Gram-Schmidt interpretation of TOLDI LRF
    Eigen::Vector3d accumulated_arrow_vector {0., 0., 0.}; 
    Eigen::Vector3d accumulated_arrow_vector_scaled {0., 0., 0.};
    for (int i=1; i < indices.size(); i++) { // goes over all points in R (not just R/3) radius
        Eigen::Vector3d single_arrow_vector = (cloud.points_[indices[i]] - central_point);
        accumulated_arrow_vector += single_arrow_vector; // central point, not centroid, as in the paper!

        double arrow_normal_dot = point_normal.dot(single_arrow_vector);

        double wi1 = (computed_radius - single_arrow_vector.norm()) * (computed_radius - single_arrow_vector.norm());
        double wi2 = arrow_normal_dot * arrow_normal_dot;
        accumulated_arrow_vector_scaled += (wi1 * wi2) * single_arrow_vector;
    }
    if (point_normal.transpose() * accumulated_arrow_vector < 0.0) point_normal = -point_normal;

    Eigen::Vector3d z_axis = point_normal;
    // Based on Gram-Schmidt (i.e. projection of second vector onto orthogonal subspace of the z_axis)
    Eigen::Vector3d x_axis = accumulated_arrow_vector_scaled - (accumulated_arrow_vector_scaled.dot(z_axis)) * z_axis;
    x_axis = (1/x_axis.norm()) * x_axis;

    // 6. Compute y-axis: simply cross-product of x and y to get right-handed coordinate system, i.e., proper SE(3)
    Eigen::Vector3d y_axis = z_axis.cross(x_axis);

    // 7. Construct the final attached SE(3) frame based on TOLDI LRF
    Eigen::Matrix4d result_lrf {Eigen::Matrix4d::Identity()};
    result_lrf.block<3,1>(0,0) = x_axis; // rot axis 1
    result_lrf.block<3,1>(0,1) = y_axis; // rot axis 2
    result_lrf.block<3,1>(0,2) = z_axis; // rot axis 3
    result_lrf.block<3,1>(0,3) = central_point;  // translation part which is equal to original point

    return result_lrf;
}

void computeAllTOLDISE3FramesOMP(const open3d::geometry::PointCloud& cloud, 
							     const open3d::geometry::KDTreeFlann& kdtree_for_LRF, 
								 int knn_pts, 
								 std::vector<Eigen::Matrix4d>& result_frames) {
    /*
        Compute TOLDI-based SE(3) frames for all points in the input cloud, using OpenMP for parallelization.
        Result is store in the passed result_frames vector.
    */
    result_frames.resize(cloud.points_.size());
    #pragma omp parallel for
        for (int i=0; i < cloud.points_.size(); i++) {
            result_frames[i] = computeSingleTOLDISE3Frame(cloud, kdtree_for_LRF, cloud.points_[i], knn_pts);
        }                                        
}


IterativeSE3Registration::IterativeSE3Registration() : 
	max_num_iterations_(150),
    max_num_se3_iterations_(20),
	num_iterations_(0),
    num_pure_se3_iterations_(-1),
	mse_(0.00001),
    lrf_radius_(0.8), // relevant only if SHOT LRF is used, which is currently commented out in the code 
    mse_switch_error_(0.001),
	number_of_nn_for_LRF_(30),
    current_correspondences_set_pcl (new pcl::Correspondences),
    estimated_overlap_(1.0),
    alpha_rot(3.0),
	beta_transl(1.0),
	scale_preprocessing(3.0)
	{ }

void IterativeSE3Registration::setSourceCloud(const std::string& filename) {	
    open3d::io::ReadPointCloud(filename, source_);
    open3d::io::ReadPointCloud(filename, source_moving_);
    current_correspondences_set.correspondences_vec.resize(source_.points_.size()); // custom corrs
    current_correspondences_set.distances_vec.resize(source_.points_.size()); // custom corrs
    current_correspondences_set_pcl->resize(source_.points_.size()); // PCL corrs
}

void IterativeSE3Registration::setSourceCloud(const open3d::geometry::PointCloud& cloud) {	
    for (auto point : cloud.points_) {
        source_.points_.push_back(point);
        source_moving_.points_.push_back(point);
    }
    current_correspondences_set.correspondences_vec.resize(cloud.points_.size()); // // custom corrs
    current_correspondences_set.distances_vec.resize(cloud.points_.size()); // // custom corrs
    current_correspondences_set_pcl->resize(source_.points_.size()); // PCL corrs
}

void IterativeSE3Registration::setTargetCloud(const std::string& filename) {
	open3d::io::ReadPointCloud(filename, target_);
}

void IterativeSE3Registration::setTargetCloud(const open3d::geometry::PointCloud& cloud) {
    for (auto point : cloud.points_) {
        target_.points_.push_back(point);
    }
}

// using distances in pcl::Correspondences, which are squared euclidean by default computation
double IterativeSE3Registration::estimate_current_mse(const pcl::Correspondences pcl_corrs) {
    double current_est_mse {0.0};
    int N=0;
    for (const auto& one_corr: pcl_corrs) {
        current_est_mse += one_corr.distance;
        N++;
    }
    return current_est_mse/N;  
}

// using euclidean distances, which are computed additionally
double IterativeSE3Registration::estimate_current_mse_compute_euclidean(const open3d::geometry::PointCloud& cloud_src, 
                                                              const open3d::geometry::PointCloud& cloud_tgt, 
                                                              const pcl::Correspondences pcl_corrs) {
    double current_est_mse {0.0};
    int N=0;
    for (const auto& one_corr: pcl_corrs) {
        current_est_mse += (cloud_src.points_[one_corr.index_query] - cloud_tgt.points_[one_corr.index_match]).norm();
        N++;
    }
    return current_est_mse/N;  
}

void IterativeSE3Registration::update_correspondences_kd_tree_XYZ(const open3d::geometry::KDTreeFlann& target_kd_tree) {
	#pragma omp parallel for
		for (int i=0; i < (int)source_moving_.points_.size(); i++) {
            std::vector<int> indices_kd(1);
            std::vector<double> distances2_kd(1);
            target_kd_tree.SearchKNN(source_moving_.points_[i], 1, indices_kd, distances2_kd);
            
            (current_correspondences_set.correspondences_vec[i])(0) = i;
            (current_correspondences_set.correspondences_vec[i])(1) = indices_kd[0];
            current_correspondences_set.distances_vec[i] = std::sqrt(distances2_kd[0]);

			pcl::Correspondence cc(i, indices_kd[0], float(std::sqrt(distances2_kd[0])));
			(*current_correspondences_set_pcl)[i] = cc;            
		}
}

void IterativeSE3Registration::update_correspondences_raw_flann_SE3() {
	#pragma omp parallel for
		for (int i=0; i < (int)source_se3_cloud_.size(); i++) {
            Eigen::VectorXd query_point(12);

            query_point(0) = source_se3_cloud_[i](0,0);  query_point(1) = source_se3_cloud_[i](1,0);   query_point(2) = source_se3_cloud_[i](2,0);
            query_point(3) = source_se3_cloud_[i](0,1);  query_point(4) = source_se3_cloud_[i](1,1);   query_point(5) = source_se3_cloud_[i](2,1);
            query_point(6) = source_se3_cloud_[i](0,2);  query_point(7) = source_se3_cloud_[i](1,2);   query_point(8) = source_se3_cloud_[i](2,2);
            query_point(9) = source_se3_cloud_[i](0,3); query_point(10) = source_se3_cloud_[i](1,3);  query_point(11) = source_se3_cloud_[i](2,3);        

            std::vector<int> indices_kd(1);
			std::vector<double> distances2_kd(1);

            raw_flann_kd_tree_target_SE3.SearchKNN(query_point, 1, indices_kd, distances2_kd);

            // Use R3 (euclidean) distances for trimming
            (current_correspondences_set.correspondences_vec[i])(0) = i;
            (current_correspondences_set.correspondences_vec[i])(1) = indices_kd[0];
            double dist_r3 = (source_se3_cloud_[i].col(3) - target_se3_cloud_[indices_kd[0]].col(3)).norm();
            current_correspondences_set.distances_vec[i] = dist_r3;
			pcl::Correspondence cc(i, indices_kd[0], float(dist_r3));
			(*current_correspondences_set_pcl)[i] = cc;
            
        }
}

void IterativeSE3Registration::update_correspondences_raw_flann_SE3(const open3d::geometry::KDTreeFlann& se3_tree, 
                                                                    const std::vector<Eigen::Matrix4d>& cloud_vector) {
	#pragma omp parallel for
		for (int i=0; i < (int)cloud_vector.size(); i++) {
            Eigen::VectorXd query_point(12);

            query_point(0) = cloud_vector[i](0,0);  query_point(1) = cloud_vector[i](1,0);   query_point(2) = cloud_vector[i](2,0);
            query_point(3) = cloud_vector[i](0,1);  query_point(4) = cloud_vector[i](1,1);   query_point(5) = cloud_vector[i](2,1);
            query_point(6) = cloud_vector[i](0,2);  query_point(7) = cloud_vector[i](1,2);   query_point(8) = cloud_vector[i](2,2);
            query_point(9) = cloud_vector[i](0,3); query_point(10) = cloud_vector[i](1,3);  query_point(11) = cloud_vector[i](2,3);        

            std::vector<int> indices_kd(1);
			std::vector<double> distances2_kd(1);

            se3_tree.SearchKNN(query_point, 1, indices_kd, distances2_kd);

            // Use R3 (euclidean) distances for trimming
            (current_correspondences_set.correspondences_vec[i])(0) = i;
            (current_correspondences_set.correspondences_vec[i])(1) = indices_kd[0];

            // standard way, using euclidean distances for rejection
            double dist_r3 = (cloud_vector[i].col(3) - target_se3_cloud_[indices_kd[0]].col(3)).norm();
            current_correspondences_set.distances_vec[i] = dist_r3;
			pcl::Correspondence cc(i, indices_kd[0], float(dist_r3));
			(*current_correspondences_set_pcl)[i] = cc;
        }
}


void IterativeSE3Registration::run_icp(const std::string &variant_name) {
    /*
    Implementation of standard ICP methods: point-to-point, point-to-plane, and generalized ICP
    Valid "variant_names" values are: "pt2pt", "pt2pl" and "gicp"
    */
    if (variant_name!= "pt2pt" && variant_name!= "pt2pl" && variant_name!= "gicp") {
        std::cerr << "Invalid ICP variant name. Valid names are pt2pt, pt2pl and gicp.\n";
    }

    kd_tree_target_XYZ.SetGeometry(target_);

    current_estimated_T_.setIdentity();
	double mse_previous {10000000.0}, mse_current {10000000.0}, mse_relative {10000000.0};

    pcl::registration::CorrespondenceRejectorTrimmed trimmed_rejector;
	trimmed_rejector.setOverlapRatio(estimated_overlap_);
    
    // estimated history for later visualization and analysis if needed
    estimated_history_.push_back(Eigen::Matrix4d::Identity());

    if (variant_name=="pt2pl") {
        target_.EstimateNormals();
    }

    if (variant_name=="gicp") {
        InitializePointCloudForGeneralizedICP_modified(source_moving_, 1e-3);
        InitializePointCloudForGeneralizedICP_modified(target_, 1e-3);
    }

    num_iterations_ = 0;
    while (true) {
        // find correspondences based on nearest neighbor, using euclidean distance
        update_correspondences_kd_tree_XYZ(kd_tree_target_XYZ);

		// apply trimmed corr rejection based on estimated/guessed overlap ratio
		trimmed_rejector.setInputCorrespondences(current_correspondences_set_pcl);
		pcl::Correspondences corrs_filtered;
		trimmed_rejector.getCorrespondences(corrs_filtered);

        std::vector<Eigen::Vector2i> corrs_filtered_o3d;
        for (auto corr: corrs_filtered) {
            Eigen::Vector2i corr_temp {corr.index_query, corr.index_match};
            corrs_filtered_o3d.emplace_back(corr_temp);
        }

        // compute mean erorr between corresponding points (distances)
		mse_previous = mse_current;
        mse_current = estimate_current_mse(corrs_filtered);
		mse_relative = std::abs(mse_current - mse_previous);

        Eigen::Matrix4d estimated_transformation_i;
        if (variant_name=="pt2pt") {
            estimated_transformation_i = o3d_estimator.ComputeTransformation(source_moving_, target_, corrs_filtered_o3d);
        }
        else if (variant_name=="pt2pl") {
            estimated_transformation_i = o3d_estimator_po2pl.ComputeTransformation(source_moving_, target_, corrs_filtered_o3d);
        }
        else if (variant_name=="gicp") {
            estimated_transformation_i = o3d_estimator_generalized.ComputeTransformation(source_moving_, target_, corrs_filtered_o3d);
        }
        else {
            std::cerr << "Invalid ICP variant name!\n";
        }

        // Update estimated history
        estimated_history_.push_back(estimated_transformation_i);

        // transform current point cloud 
        source_moving_.Transform(estimated_transformation_i);

        // update final estimated transformation
        current_estimated_T_ = estimated_transformation_i * current_estimated_T_;                

        // check convergence 
        num_iterations_++;
        if (num_iterations_ == max_num_iterations_ || mse_relative < mse_) {			
            break;
		}     
    }
}


void IterativeSE3Registration::run_se3_icp(const std::string &variant_name) {
    /*
    Implementation of the proposed SE(3)-ICP methods
    Valid "variant_names" values are: "pt2pt", "pt2pl" and "gicp"
    */
    
    if (variant_name!= "pt2pt" && variant_name!= "pt2pl" && variant_name!= "gicp") {
        std::cerr << "Invalid variant name. Choose one of: pt2pt, pt2pl, gicp \n";
    }

    //////////////////////////////////////////////////////////////////
    // Preprocessing source and target point clouds with the aim of roto-transl balancing in the SE(3) metric
    // Scaling with the inverse of the larger radius of points clouds & subtracting point cloud centroids
    Eigen::Vector3d centorid_source = source_.GetCenter();
    Eigen::Vector3d centorid_target = target_.GetCenter();

    double radius_source = largestDistanceFromGivenPoint(centorid_source, source_);
    double radius_target = largestDistanceFromGivenPoint(centorid_target, target_);
    double radius_max = std::max(radius_source, radius_target);
    double scaling_factor = scale_preprocessing * (1.0/radius_max);

    source_.Translate(-centorid_source);
    source_moving_.Translate(-centorid_source);
    target_.Translate(-centorid_target);

    source_.Scale(scaling_factor, Eigen::Vector3d {0,0,0}); // second argument is related to centering, so not relevant here 
    source_moving_.Scale(scaling_factor, Eigen::Vector3d {0,0,0});
    target_.Scale(scaling_factor, Eigen::Vector3d {0,0,0});
    //////////////////////////////////////////////////////////////////

    // Calculate SE(3) frames
    kd_tree_source_XYZ.SetGeometry(source_);
    kd_tree_target_XYZ.SetGeometry(target_);

    // TOLDI call with our kNN search (original formulation is directly with radius)
    computeAllTOLDISE3FramesOMP(source_, kd_tree_source_XYZ, number_of_nn_for_LRF_, source_se3_cloud_); // 90 nearest neighbors is default
    computeAllTOLDISE3FramesOMP(target_, kd_tree_target_XYZ, number_of_nn_for_LRF_, target_se3_cloud_); // 90 nearest neighbors is default

    // computeAllSHOTSE3FramesOMP(source_, kd_tree_source_XYZ, lrf_radius_, source_se3_cloud_);
    // computeAllSHOTSE3FramesOMP(target_, kd_tree_target_XYZ, lrf_radius_, target_se3_cloud_);
    ////////////////////////////////////

    #pragma omp parallel for
    for (int i = 0; i<source_se3_cloud_.size(); i++) {
        source_se3_cloud_[i].block<3,3>(0,0) *= alpha_rot; // rotation weight
        source_se3_cloud_[i].block<3,1>(0,3) *= beta_transl; // translation weight
    }

    #pragma omp parallel for
    for (int i = 0; i<target_se3_cloud_.size(); i++) {
        target_se3_cloud_[i].block<3,3>(0,0) *= alpha_rot; // rotation weight
        target_se3_cloud_[i].block<3,1>(0,3) *= beta_transl; // translation weight
    } 

    // Construct SE(3) data matrix needed for raw flann data structure
    Eigen::MatrixXd target_SE3_matrix (12, target_.points_.size());
    #pragma omp parallel for
    for (int i=0; i<target_se3_cloud_.size(); i++) {
        target_SE3_matrix.col(i)(0) = target_se3_cloud_[i](0,0);  
        target_SE3_matrix.col(i)(1) = target_se3_cloud_[i](1,0);   
        target_SE3_matrix.col(i)(2) = target_se3_cloud_[i](2,0);
        target_SE3_matrix.col(i)(3) = target_se3_cloud_[i](0,1);  
        target_SE3_matrix.col(i)(4) = target_se3_cloud_[i](1,1);   
        target_SE3_matrix.col(i)(5) = target_se3_cloud_[i](2,1);
        target_SE3_matrix.col(i)(6) = target_se3_cloud_[i](0,2);  
        target_SE3_matrix.col(i)(7) = target_se3_cloud_[i](1,2);   
        target_SE3_matrix.col(i)(8) = target_se3_cloud_[i](2,2);
        target_SE3_matrix.col(i)(9)  = target_se3_cloud_[i](0,3);
        target_SE3_matrix.col(i)(10) = target_se3_cloud_[i](1,3); 
        target_SE3_matrix.col(i)(11) = target_se3_cloud_[i](2,3); 
    }
    raw_flann_kd_tree_target_SE3.SetMatrixData(target_SE3_matrix);    


    current_estimated_T_.setIdentity();
    Eigen::Matrix4d previous_estimated_T {Eigen::Matrix4d::Identity()};
	double mse_previous {10000000.0}, mse_current {10000000.0}, mse_relative {10000000.0};
    double current_T_change {10000000.0};

	pcl::registration::CorrespondenceRejectorTrimmed trimmed_rejector;
	trimmed_rejector.setOverlapRatio(estimated_overlap_);

    time_se3_correspondence_search_ = 0.0;

    num_iterations_ = 0;
    num_pure_se3_iterations_ = 0;
    
    if (variant_name=="pt2pl"){ // point2plane ICP cost
        target_.EstimateNormals(); 
    } 
    else if (variant_name=="gicp") { // generalized ICP cost 
        InitializePointCloudForGeneralizedICP_modified(source_moving_, 1e-3);
        InitializePointCloudForGeneralizedICP_modified(target_, 1e-3);   
    }

    // flag used to switch to an ICP variant for extra precision at the end, theoretically corresponding to alpha=0
    bool switch_icp = false;

    std::vector<Eigen::Vector2i> corrs_filtered_o3d; // making this outside of loop for optional visualizations
    while (true) { 

        num_iterations_++;
        
        if (!switch_icp) {
            // SE(3) distance-based correspondence search
            num_pure_se3_iterations_++;
            update_correspondences_raw_flann_SE3(raw_flann_kd_tree_target_SE3, source_se3_cloud_);
        }
        else if (switch_icp) {
            // euclidean distance-based correspondence search (as used in standard ICP variant)
            update_correspondences_kd_tree_XYZ(kd_tree_target_XYZ);
        }
        
		// apply trimmed corr rejection based on estimated/guessed overlap ratio
		trimmed_rejector.setInputCorrespondences(current_correspondences_set_pcl);
		pcl::Correspondences corrs_filtered;
		trimmed_rejector.getCorrespondences(corrs_filtered);

        corrs_filtered_o3d.clear();
        corrs_filtered_o3d.reserve(corrs_filtered.size());
        std::vector<std::pair<int,int>> corrs_filtered_vis;
        for (auto corr : corrs_filtered) {
            Eigen::Vector2i corr_temp {corr.index_query, corr.index_match};
            corrs_filtered_o3d.emplace_back(corr_temp);

            auto one_pair = std::make_pair(corr.index_query, corr.index_match);
            corrs_filtered_vis.emplace_back(one_pair);
        }

		mse_previous = mse_current;
        mse_current = estimate_current_mse(corrs_filtered); // this one will use calculated distances in corrs_filtered passed
		mse_relative = std::abs(mse_current - mse_previous);	        

        // estimate unknown transformation based on computed correspondences
        Eigen::Matrix4d estimated_transformation_i;

        if (variant_name=="pt2pt") { // point2point
            estimated_transformation_i = o3d_estimator.ComputeTransformation(source_moving_, target_, corrs_filtered_o3d);
        }
        else if (variant_name=="pt2pl") { // point2plane
            estimated_transformation_i = o3d_estimator_po2pl.ComputeTransformation(source_moving_, target_, corrs_filtered_o3d);
        }
        else if (variant_name=="gicp") { // generalized
            estimated_transformation_i = o3d_estimator_generalized.ComputeTransformation(source_moving_, target_, corrs_filtered_o3d);
        }
        else {
            std::cout << "Unknown optimization strategy for SE(3) \n";
            break;
        }

        // transform current point cloud 
        source_moving_.Transform(estimated_transformation_i);

        // update final estimated transformation
        previous_estimated_T = current_estimated_T_;
        current_estimated_T_ = estimated_transformation_i * current_estimated_T_;
        current_T_change = (previous_estimated_T - current_estimated_T_).norm();

        #pragma omp parallel for 
            for (int k = 0; k<source_se3_cloud_.size(); k++) {
                source_se3_cloud_[k] = estimated_transformation_i * source_se3_cloud_[k];
            }

        if (!switch_icp) {
            // We are still with the SE(3) correspondence matching variant
            if (num_iterations_ == max_num_se3_iterations_ || current_T_change < mse_switch_error_) {
                switch_icp = true;
            }
        }
        else {
            // Now we are at pure ICP
            if (num_iterations_ == max_num_iterations_ || mse_relative < scaling_factor * mse_) {
                break;
            }
        }
          
        
    }

    // Convert current estimated transformation into the original coordinates & scale
    Eigen::Matrix3d rot_prime = current_estimated_T_.block<3,3>(0,0);
    Eigen::Vector3d tra_prime = current_estimated_T_.block<3,1>(0,3);
    Eigen::Vector3d tra_og_value = (1.0/scaling_factor) * tra_prime - rot_prime * centorid_source + centorid_target;
    current_estimated_T_.block<3,1>(0,3) = tra_og_value;
}


void IterativeSE3Registration::run_se3_icp_with_cf() {
    /*
    SE(3)-ICP variant that includes uncertainties about depth measurements 
    from RGBD camera and it is used in the paper for the Stanford Lounge RGBD
    dataset. It uses generalized icp cost function, but other variants could 
    be implemented as well.

    In principle, if via point cloud capturing sensor, or algorithmically, we have 
    a way to assign uncertainties to points of a point cloud, they could be included
    in this way.  
    */

    auto start_before_icp = std::chrono::high_resolution_clock::now();    

    std::vector<double> confidences_source, confidences_target;
    confidences_source.resize(source_.points_.size());
    confidences_target.resize(target_.points_.size());

    double confidence_source_total{0.0}, confidence_target_total{0.0};
    for (int i=0; i<source_.points_.size(); i++) {
        confidences_source[i] = lounge_point_confidence(source_.points_[i]);
        confidence_source_total += confidences_source[i]; 
    }

    for (int i=0; i<target_.points_.size(); i++) {
        confidences_target[i] = lounge_point_confidence(target_.points_[i]);
        confidence_target_total += confidences_target[i];
    }

    std::vector<double> confidences_source_normalized, confidences_target_normalized;
    confidences_source_normalized.resize(source_.points_.size());
    confidences_target_normalized.resize(target_.points_.size());    

    for (int i=0; i<source_.points_.size(); i++) {
        confidences_source_normalized[i] = confidences_source[i]/confidence_source_total;

    }

    for (int i=0; i<target_.points_.size(); i++) {
        confidences_target_normalized[i] = confidences_target[i]/confidence_target_total;
    }

    //////////////////////////////////////////////////////////////////
    // Point cloud precomputation
    Eigen::Vector3d centorid_source = source_.GetCenter();
    Eigen::Vector3d centorid_target = target_.GetCenter();

    double radius_source = largestDistanceFromGivenPoint(centorid_source, source_);
    double radius_target = largestDistanceFromGivenPoint(centorid_target, target_);
    double radius_max = std::max(radius_source, radius_target);
    double scaling_factor = scale_preprocessing * (1.0/radius_max);

    std::cout << "### scaling factor = " << scaling_factor << "\n";

    source_.Translate(-centorid_source);
    source_moving_.Translate(-centorid_source);
    target_.Translate(-centorid_target);

    source_.Scale(scaling_factor, Eigen::Vector3d {0,0,0}); 
    source_moving_.Scale(scaling_factor, Eigen::Vector3d {0,0,0});
    target_.Scale(scaling_factor, Eigen::Vector3d {0,0,0});

    // Calculate SE(3) frames
    kd_tree_source_XYZ.SetGeometry(source_);
    kd_tree_target_XYZ.SetGeometry(target_);

    // NEW TOLDI CALL WITH kNN SEARCH
    computeAllTOLDISE3FramesOMP(source_, kd_tree_source_XYZ, number_of_nn_for_LRF_, source_se3_cloud_); // 90 nearest neighbors is default
    computeAllTOLDISE3FramesOMP(target_, kd_tree_target_XYZ, number_of_nn_for_LRF_, target_se3_cloud_); // 90 nearest neighbors is default

    // computeAllSHOTSE3FramesOMP(source_, kd_tree_source_XYZ, lrf_radius_, source_se3_cloud_);
    // computeAllSHOTSE3FramesOMP(target_, kd_tree_target_XYZ, lrf_radius_, target_se3_cloud_);

    #pragma omp parallel for
    for (int i = 0; i<source_se3_cloud_.size(); i++) {
        source_se3_cloud_[i].block<3,3>(0,0) *= alpha_rot; // rotation weight
        source_se3_cloud_[i].block<3,1>(0,3) *= beta_transl; // translation weight
    }

    #pragma omp parallel for
    for (int i = 0; i<target_se3_cloud_.size(); i++) {
        target_se3_cloud_[i].block<3,3>(0,0) *= alpha_rot; // rotation weight
        target_se3_cloud_[i].block<3,1>(0,3) *= beta_transl; // translation weight
    } 

    // Construct SE(3) data matrix needed for raw flann data structure
    Eigen::MatrixXd target_SE3_matrix (12, target_.points_.size());
    #pragma omp parallel for
    for (int i=0; i<target_se3_cloud_.size(); i++) {
        target_SE3_matrix.col(i)(0) = target_se3_cloud_[i](0,0);  target_SE3_matrix.col(i)(1) = target_se3_cloud_[i](1,0);   target_SE3_matrix.col(i)(2) = target_se3_cloud_[i](2,0);
        target_SE3_matrix.col(i)(3) = target_se3_cloud_[i](0,1);  target_SE3_matrix.col(i)(4) = target_se3_cloud_[i](1,1);   target_SE3_matrix.col(i)(5) = target_se3_cloud_[i](2,1);
        target_SE3_matrix.col(i)(6) = target_se3_cloud_[i](0,2);  target_SE3_matrix.col(i)(7) = target_se3_cloud_[i](1,2);   target_SE3_matrix.col(i)(8) = target_se3_cloud_[i](2,2);
        target_SE3_matrix.col(i)(9)  = target_.points_[i](0); 
        target_SE3_matrix.col(i)(10) = target_.points_[i](1); 
        target_SE3_matrix.col(i)(11) = target_.points_[i](2); 
    }
    raw_flann_kd_tree_target_SE3.SetMatrixData(target_SE3_matrix);    
    //////////////////////////////////////////////////////////////////////

    current_estimated_T_.setIdentity();
    Eigen::Matrix4d previous_estimated_T {Eigen::Matrix4d::Identity()};
	double mse_previous {10000000.0}, mse_current {10000000.0}, mse_relative {10000000.0};
    double current_T_change {10000000.0};

	pcl::registration::CorrespondenceRejectorTrimmed trimmed_rejector;
	trimmed_rejector.setOverlapRatio(estimated_overlap_);

    time_se3_correspondence_search_ = 0.0;

    num_iterations_ = 0;
    num_pure_se3_iterations_ = 0;
    
    // needed for generalized ICP cost 
    InitializePointCloudForGeneralizedICP_modified(source_moving_, 1e-3);
    InitializePointCloudForGeneralizedICP_modified(target_, 1e-3);   

    // when to switch to ICP's R3 correspondences search to gain some extra precision 
    bool switch_icp = false;

    std::vector<Eigen::Vector2i> corrs_filtered_o3d; // making this outside of loop for visualizations
    while (true) { 

        num_iterations_++;
        
        auto start_se3_corr = std::chrono::high_resolution_clock::now();

        // NEW
        if (!switch_icp) {
            num_pure_se3_iterations_++;
            update_correspondences_raw_flann_SE3(raw_flann_kd_tree_target_SE3, source_se3_cloud_);
        }
        else if (switch_icp) {
            update_correspondences_kd_tree_XYZ(kd_tree_target_XYZ);
        }
        
        auto end_se3_corr = std::chrono::high_resolution_clock::now();
        time_se3_correspondence_search_ += std::chrono::duration_cast<std::chrono::nanoseconds>(end_se3_corr - start_se3_corr).count() / 1e6;

		// apply trimmed corr rejection based on estimated/guessed overlap ratio
		trimmed_rejector.setInputCorrespondences(current_correspondences_set_pcl);
		pcl::Correspondences corrs_filtered;
		trimmed_rejector.getCorrespondences(corrs_filtered);

        corrs_filtered_o3d.clear();
        corrs_filtered_o3d.reserve(corrs_filtered.size());
        std::vector<std::pair<int,int>> corrs_filtered_vis;
        for (auto corr : corrs_filtered) {
            Eigen::Vector2i corr_temp {corr.index_query, corr.index_match};
            corrs_filtered_o3d.emplace_back(corr_temp);

            auto one_pair = std::make_pair(corr.index_query, corr.index_match);
            corrs_filtered_vis.emplace_back(one_pair);
        }

		mse_previous = mse_current;
        mse_current = estimate_current_mse_compute_euclidean(source_moving_, target_, corrs_filtered);
		mse_relative = std::abs(mse_current - mse_previous);	        

        // estimate unknown transformation based on computed correspondences
        Eigen::Matrix4d estimated_transformation_i;

        // WITH CF (confidence filtering)
        std::vector<double> weights(corrs_filtered_o3d.size());
        std::vector<double> kept_weights;
        kept_weights.reserve(corrs_filtered_o3d.size());

        open3d::pipelines::registration::CorrespondenceSet kept_corrs;
        kept_corrs.reserve(corrs_filtered_o3d.size());

        for (int ci=0; ci<corrs_filtered_o3d.size(); ci++) {

            weights[ci] = (confidences_source[corrs_filtered_o3d[ci][0]] + confidences_target[corrs_filtered_o3d[ci][1]])/2.0;
            
            if ((confidences_source[corrs_filtered_o3d[ci][0]]<0.15 && confidences_target[corrs_filtered_o3d[ci][1]])<0.15) {
                kept_weights.emplace_back(weights[ci]); 
                kept_corrs.emplace_back(corrs_filtered_o3d[ci]);
            }
        }

        estimated_transformation_i = optimize_generalizedICP_manual(source_moving_, target_, corrs_filtered_o3d, weights);
      
        // transform current point cloud 
        source_moving_.Transform(estimated_transformation_i);

        // update final estimated transformation
        previous_estimated_T = current_estimated_T_;
        current_estimated_T_ = estimated_transformation_i * current_estimated_T_;
        current_T_change = (previous_estimated_T - current_estimated_T_).norm();

        #pragma omp parallel for 
            for (int k = 0; k<source_se3_cloud_.size(); k++) {
                source_se3_cloud_[k] = estimated_transformation_i * source_se3_cloud_[k];
            }
        
        if (!switch_icp) {
            // We are still with the SE(3) correspondence matching variant
            if (num_iterations_ == max_num_se3_iterations_ || current_T_change < mse_switch_error_) {
                switch_icp = true;
            }
        }
        else {
            // Now we are at pure ICP
            if (num_iterations_ == max_num_iterations_ || mse_relative < scaling_factor * mse_) {
                break;
            }
        } 
        
    }

    // Convert current estimated transformation into the original coordinates & scale
    Eigen::Matrix3d rot_prime = current_estimated_T_.block<3,3>(0,0);
    Eigen::Vector3d tra_prime = current_estimated_T_.block<3,1>(0,3);
    Eigen::Vector3d tra_og_value = (1.0/scaling_factor) * tra_prime - rot_prime * centorid_source + centorid_target;
    current_estimated_T_.block<3,1>(0,3) = tra_og_value;

    auto end_before_icp = std::chrono::high_resolution_clock::now(); 
    time_before_pure_icp_ = std::chrono::duration_cast<std::chrono::nanoseconds>(end_before_icp - start_before_icp).count() / 1e6;
}


void IterativeSE3Registration::run_se3_pure(const std::string &variant_name) {
    /*
        For experimentation.
        Implementation of the proposed SE(3)-ICP methods without standard ICP for final precision.
        Can get pretty decent precision on its own while retaining some of robustness benefits, but
        (alpha,beta) parametars should not be standard (3,1) values. Rotational part (alpha) should 
        be much smaller, something like 0.1, if I remember correctly.

        Valid "variant_names" values are: "pt2pt", "pt2pl" and "gicp"
    */
    
    if (variant_name!= "pt2pt" && variant_name!= "pt2pl" && variant_name!= "gicp") {
        std::cerr << "Invalid variant name. Choose one of: pt2pt, pt2pl, gicp \n";
    }

    //////////////////////////////////////////////////////////////////
    // Preprocessing source and target point clouds with the aim of roto-transl balancing in the SE(3) metric
    // Scaling with the inverse of the larger radius of points clouds & subtracting point cloud centroids
    Eigen::Vector3d centorid_source = source_.GetCenter();
    Eigen::Vector3d centorid_target = target_.GetCenter();

    double radius_source = largestDistanceFromGivenPoint(centorid_source, source_);
    double radius_target = largestDistanceFromGivenPoint(centorid_target, target_);
    double radius_max = std::max(radius_source, radius_target);
    double scaling_factor = scale_preprocessing * (1.0/radius_max);

    source_.Translate(-centorid_source);
    source_moving_.Translate(-centorid_source);
    target_.Translate(-centorid_target);

    source_.Scale(scaling_factor, Eigen::Vector3d {0,0,0}); // second argument is some centering thing 
    source_moving_.Scale(scaling_factor, Eigen::Vector3d {0,0,0});
    target_.Scale(scaling_factor, Eigen::Vector3d {0,0,0});
    //////////////////////////////////////////////////////////////////

    // Calculate SE(3) frames
    kd_tree_source_XYZ.SetGeometry(source_);
    kd_tree_target_XYZ.SetGeometry(target_);

    // TOLDI call with kNN search (original formulation is directly with radius)
    computeAllTOLDISE3FramesOMP(source_, kd_tree_source_XYZ, number_of_nn_for_LRF_, source_se3_cloud_); // 90 nearest neighbors is default
    computeAllTOLDISE3FramesOMP(target_, kd_tree_target_XYZ, number_of_nn_for_LRF_, target_se3_cloud_); // 90 nearest neighbors is default

    ////////////////////////////////////

    #pragma omp parallel for
    for (int i = 0; i<source_se3_cloud_.size(); i++) {
        source_se3_cloud_[i].block<3,3>(0,0) *= alpha_rot; // rotation weight
        source_se3_cloud_[i].block<3,1>(0,3) *= beta_transl; // translation weight
    }

    #pragma omp parallel for
    for (int i = 0; i<target_se3_cloud_.size(); i++) {
        target_se3_cloud_[i].block<3,3>(0,0) *= alpha_rot; // rotation weight
        target_se3_cloud_[i].block<3,1>(0,3) *= beta_transl; // translation weight
    } 

    // Construct needed flann SE(3) here manually!
    Eigen::MatrixXd target_SE3_matrix (12, target_.points_.size());
    #pragma omp parallel for
    for (int i=0; i<target_se3_cloud_.size(); i++) {
        target_SE3_matrix.col(i)(0) = target_se3_cloud_[i](0,0);  
        target_SE3_matrix.col(i)(1) = target_se3_cloud_[i](1,0);   
        target_SE3_matrix.col(i)(2) = target_se3_cloud_[i](2,0);
        target_SE3_matrix.col(i)(3) = target_se3_cloud_[i](0,1);  
        target_SE3_matrix.col(i)(4) = target_se3_cloud_[i](1,1);   
        target_SE3_matrix.col(i)(5) = target_se3_cloud_[i](2,1);
        target_SE3_matrix.col(i)(6) = target_se3_cloud_[i](0,2);  
        target_SE3_matrix.col(i)(7) = target_se3_cloud_[i](1,2);   
        target_SE3_matrix.col(i)(8) = target_se3_cloud_[i](2,2);
        target_SE3_matrix.col(i)(9)  = target_se3_cloud_[i](0,3);
        target_SE3_matrix.col(i)(10) = target_se3_cloud_[i](1,3); 
        target_SE3_matrix.col(i)(11) = target_se3_cloud_[i](2,3); 
    }
    raw_flann_kd_tree_target_SE3.SetMatrixData(target_SE3_matrix);    
    //////////////////////////////////////////////////////////////////////

    current_estimated_T_.setIdentity();
    Eigen::Matrix4d previous_estimated_T {Eigen::Matrix4d::Identity()};
	double mse_previous {10000000.0}, mse_current {10000000.0}, mse_relative {10000000.0};
    double current_T_change {10000000.0};

	pcl::registration::CorrespondenceRejectorTrimmed trimmed_rejector;
	trimmed_rejector.setOverlapRatio(estimated_overlap_);

    time_se3_correspondence_search_ = 0.0;

    num_iterations_ = 0;
    num_pure_se3_iterations_ = 0;
    
    if (variant_name=="pt2pl"){ // point2plane ICP cost
        target_.EstimateNormals(); 
    } 
    else if (variant_name=="gicp") { // generalized ICP cost 
        InitializePointCloudForGeneralizedICP_modified(source_moving_, 1e-3);
        InitializePointCloudForGeneralizedICP_modified(target_, 1e-3);   
    }

    std::vector<Eigen::Vector2i> corrs_filtered_o3d; // making this outside of loop for visualizations
    while (true) { 

        num_iterations_++;

        num_pure_se3_iterations_++;
        update_correspondences_raw_flann_SE3(raw_flann_kd_tree_target_SE3, source_se3_cloud_);
        
		// apply trimmed corr rejection based on estimated/guessed overlap ratio
		trimmed_rejector.setInputCorrespondences(current_correspondences_set_pcl);
		pcl::Correspondences corrs_filtered;
		trimmed_rejector.getCorrespondences(corrs_filtered);

        corrs_filtered_o3d.clear();
        corrs_filtered_o3d.reserve(corrs_filtered.size());
        std::vector<std::pair<int,int>> corrs_filtered_vis;
        for (auto corr : corrs_filtered) {
            Eigen::Vector2i corr_temp {corr.index_query, corr.index_match};
            corrs_filtered_o3d.emplace_back(corr_temp);

            auto one_pair = std::make_pair(corr.index_query, corr.index_match);
            corrs_filtered_vis.emplace_back(one_pair);
        }

		mse_previous = mse_current;
        mse_current = estimate_current_mse(corrs_filtered); // this one will use calculated distances in corrs_filtered passed
		mse_relative = std::abs(mse_current - mse_previous);	        

        // estimate unknown transformation based on computed correspondences
        Eigen::Matrix4d estimated_transformation_i;

        if (variant_name=="pt2pt") { // point2point
            estimated_transformation_i = o3d_estimator.ComputeTransformation(source_moving_, target_, corrs_filtered_o3d);
        }
        else if (variant_name=="pt2pl") { // point2plane
            estimated_transformation_i = o3d_estimator_po2pl.ComputeTransformation(source_moving_, target_, corrs_filtered_o3d);
        }
        else if (variant_name=="gicp") { // generalized
            estimated_transformation_i = o3d_estimator_generalized.ComputeTransformation(source_moving_, target_, corrs_filtered_o3d);
        }
        else {
            std::cout << "Unknown optimization strategy for SE(3) \n";
            break;
        }

        // transform current point cloud 
        source_moving_.Transform(estimated_transformation_i);

        // update final estimated transformation
        previous_estimated_T = current_estimated_T_;
        current_estimated_T_ = estimated_transformation_i * current_estimated_T_;
        current_T_change = (previous_estimated_T - current_estimated_T_).norm();

        #pragma omp parallel for 
            for (int k = 0; k<source_se3_cloud_.size(); k++) {
                source_se3_cloud_[k] = estimated_transformation_i * source_se3_cloud_[k];
            }

        if (num_iterations_ == max_num_se3_iterations_ || mse_relative < scaling_factor * mse_) 
            break;
    }

    // Convert current estimated transformation into the original coordinates & scale
    Eigen::Matrix3d rot_prime = current_estimated_T_.block<3,3>(0,0);
    Eigen::Vector3d tra_prime = current_estimated_T_.block<3,1>(0,3);
    Eigen::Vector3d tra_og_value = (1.0/scaling_factor) * tra_prime - rot_prime * centorid_source + centorid_target;
    current_estimated_T_.block<3,1>(0,3) = tra_og_value;
    std::cout << "pure se3 finished" << std::endl;
}