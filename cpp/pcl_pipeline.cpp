#include <pcl/common/io.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <filesystem>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Cloud = pcl::PointCloud<pcl::PointXYZ>;

Cloud::Ptr passAxis(const Cloud::Ptr& input, const char* field, float low, float high) {
  auto output = pcl::make_shared<Cloud>();
  pcl::PassThrough<pcl::PointXYZ> filter;
  filter.setInputCloud(input);
  filter.setFilterFieldName(field);
  filter.setFilterLimits(low, high);
  filter.filter(*output);
  return output;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: pcl_pipeline input.pcd output_dir [target_x target_y]\n";
    return 2;
  }
  const float targetX = argc > 3 ? std::stof(argv[3]) : 0.42f;
  const float targetY = argc > 4 ? std::stof(argv[4]) : 0.0f;
  const fs::path outputDir = argv[2];
  fs::create_directories(outputDir);

  auto input = pcl::make_shared<Cloud>();
  if (pcl::io::loadPCDFile(argv[1], *input) < 0) {
    std::cerr << "Cannot read " << argv[1] << '\n';
    return 1;
  }

  // Units are metres, matching the bundled sample. Change all constants together.
  auto roi = passAxis(passAxis(passAxis(input, "x", -2.0f, 2.0f), "y", -2.0f, 2.0f),
                      "z", -0.1f, 1.0f);
  auto down = pcl::make_shared<Cloud>();
  pcl::VoxelGrid<pcl::PointXYZ> voxel;
  voxel.setInputCloud(roi);
  voxel.setLeafSize(0.012f, 0.012f, 0.012f);
  voxel.filter(*down);

  auto clean = pcl::make_shared<Cloud>();
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud(down);
  sor.setMeanK(20);
  sor.setStddevMulThresh(2.5);
  sor.filter(*clean);

  pcl::ModelCoefficients::Ptr coefficients = pcl::make_shared<pcl::ModelCoefficients>();
  pcl::PointIndices::Ptr planeIndices = pcl::make_shared<pcl::PointIndices>();
  pcl::SACSegmentation<pcl::PointXYZ> segmentation;
  segmentation.setOptimizeCoefficients(true);
  segmentation.setModelType(pcl::SACMODEL_PLANE);
  segmentation.setMethodType(pcl::SAC_RANSAC);
  segmentation.setDistanceThreshold(0.006);
  segmentation.setInputCloud(clean);
  segmentation.segment(*planeIndices, *coefficients);
  if (planeIndices->indices.empty()) throw std::runtime_error("No plane found");

  auto parts = pcl::make_shared<Cloud>();
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(clean);
  extract.setIndices(planeIndices);
  extract.setNegative(true);
  extract.filter(*parts);

  auto tree = pcl::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
  std::vector<pcl::PointIndices> clusters;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> clustering;
  clustering.setClusterTolerance(0.035);
  clustering.setMinClusterSize(40);
  clustering.setMaxClusterSize(10000000);
  clustering.setSearchMethod(tree);
  clustering.setInputCloud(parts);
  clustering.extract(clusters);
  if (clusters.empty()) throw std::runtime_error("No workpiece cluster found");

  auto nearestCluster = clusters.begin();
  float bestDistance = std::numeric_limits<float>::max();
  for (auto cluster = clusters.begin(); cluster != clusters.end(); ++cluster) {
    float x = 0.0f, y = 0.0f;
    for (int index : cluster->indices) {
      x += parts->points[index].x;
      y += parts->points[index].y;
    }
    x /= cluster->indices.size();
    y /= cluster->indices.size();
    const float distance = std::hypot(x - targetX, y - targetY);
    if (distance < bestDistance) {
      bestDistance = distance;
      nearestCluster = cluster;
    }
  }
  auto target = pcl::make_shared<Cloud>();
  pcl::copyPointCloud(*parts, nearestCluster->indices, *target);

  auto normals = pcl::make_shared<pcl::PointCloud<pcl::Normal>>();
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
  normalEstimation.setInputCloud(target);
  normalEstimation.setSearchMethod(tree);
  normalEstimation.setKSearch(24);
  normalEstimation.compute(*normals);
  auto targetWithNormals = pcl::make_shared<pcl::PointCloud<pcl::PointNormal>>();
  pcl::concatenateFields(*target, *normals, *targetWithNormals);

  pcl::MomentOfInertiaEstimation<pcl::PointXYZ> inertia;
  inertia.setInputCloud(target);
  inertia.compute();
  pcl::PointXYZ minObb, maxObb, obbCenter;
  Eigen::Matrix3f obbRotation;
  inertia.getOBB(minObb, maxObb, obbCenter, obbRotation);

  pcl::io::savePCDFileBinary((outputDir / "parts_without_plane.pcd").string(), *parts);
  pcl::io::savePCDFileBinary((outputDir / "target.pcd").string(), *target);
  pcl::io::savePCDFileBinary((outputDir / "target_with_normals.pcd").string(), *targetWithNormals);

  std::cout << "raw=" << input->size() << " plane=" << planeIndices->indices.size()
            << " parts=" << parts->size() << " clusters=" << clusters.size()
            << " target=" << target->size() << '\n';
  std::cout << "plane=[";
  for (float value : coefficients->values) std::cout << value << ' ';
  std::cout << "] obb_center=[" << obbCenter.x << ' ' << obbCenter.y << ' '
            << obbCenter.z << "]\n";
  return 0;
}

