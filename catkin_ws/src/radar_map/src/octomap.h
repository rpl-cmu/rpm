// std includes
#include <filesystem>
#include <string>
#include <vector>
#include <csignal>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

// Eigen includes
#include <Eigen/Dense>

// ROS includes
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/MapMetaData.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_broadcaster.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <dynamic_reconfigure/server.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// PCL includes
#define PCL_NO_PRECOMPILE
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

// Octomap
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/GetOctomap.h>
#include <octomap_msgs/BoundingBoxQuery.h>
#include <octomap_msgs/conversions.h>
#include <octomap_ros/conversions.h>
#include <octomap/octomap.h>
#include <octomap/OcTreeKey.h>

using namespace pcl;
using namespace std;

typedef message_filters::sync_policies::ExactTime<sensor_msgs::PointCloud2, nav_msgs::Path> MySyncPolicy;
typedef octomap::OcTree OcTreeT;

class OctoMapping
{
private:
    bool use_voxel = false;
    float process_voxel_size = 0.2;
    float map_voxel_size = 0;
    float snr_thresh = 50;

    PassThrough<PointXYZ> z_filter;

    OcTreeT *m_octree;
    octomap::KeyRay m_keyRay; // temp storage for ray casting
    octomap::OcTreeKey m_updateBBXMin;
    octomap::OcTreeKey m_updateBBXMax;
    double m_res;
    unsigned m_treeDepth;
    unsigned m_maxTreeDepth;
    double m_colorFactor;
    double m_minSizeX;
    double m_minSizeY;
    double m_occupancyMinZ;
    double m_occupancyMaxZ;

    // downprojected 2D map:
    bool m_incrementalUpdate = false;
    nav_msgs::OccupancyGrid m_gridmap;
    octomap::OcTreeKey m_paddedMinKey;
    unsigned m_multires2DScale;
    bool m_projectCompleteMap;

    ros::Publisher m_mapPub, m_markerPub;

public:
    void insert_pc_pose(sensor_msgs::PointCloud2ConstPtr pcd, Eigen::Matrix4f &pose);
    void insert_scan(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr map_part,
                     const octomath::Vector3 sensorOrigin);
    void reset_octree_map();
    void handlePreNodeTraversal(const ros::Time &rostime);
    void handlePostNodeTraversal(const ros::Time &rostime);
    void handleOccupiedNode(const OcTreeT::iterator &it);
    void handleFreeNode(const OcTreeT::iterator &it);
    void handleOccupiedNodeInBBX(const OcTreeT::iterator &it);
    void handleFreeNodeInBBX(const OcTreeT::iterator &it);
    void update2DMap(const OcTreeT::iterator &it, bool occupied);
    void publishProjected2DMap(const ros::Time &rostime);
    void adjustMapData(nav_msgs::OccupancyGrid &map, const nav_msgs::MapMetaData &oldMapInfo) const;

    visualization_msgs::MarkerArray getROSMarkerArray(ros::Time stamp);
    std_msgs::ColorRGBA heightMapColor(double h);
    OctoMapping(ros::NodeHandle *nh);
    ~OctoMapping();

protected:
    inline static void updateMinKey(const octomap::OcTreeKey &in, octomap::OcTreeKey &min)
    {
        for (unsigned i = 0; i < 3; ++i)
            min[i] = std::min(in[i], min[i]);
    };

    inline static void updateMaxKey(const octomap::OcTreeKey &in, octomap::OcTreeKey &max)
    {
        for (unsigned i = 0; i < 3; ++i)
            max[i] = std::max(in[i], max[i]);
    };

    /// Test if key is within update area of map (2D, ignores height)
    inline bool isInUpdateBBX(const OcTreeT::iterator &it) const
    {
        // 2^(tree_depth-depth) voxels wide:
        unsigned voxelWidth = (1 << (m_maxTreeDepth - it.getDepth()));
        octomap::OcTreeKey key = it.getIndexKey(); // lower corner of voxel
        return (key[0] + voxelWidth >= m_updateBBXMin[0] && key[1] + voxelWidth >= m_updateBBXMin[1] && key[0] <= m_updateBBXMax[0] && key[1] <= m_updateBBXMax[1]);
    };

    inline unsigned mapIdx(int i, int j) const
    {
        return m_gridmap.info.width * j + i;
    };

    inline unsigned mapIdx(const octomap::OcTreeKey &key) const
    {
        return mapIdx((key[0] - m_paddedMinKey[0]) / m_multires2DScale,
                      (key[1] - m_paddedMinKey[1]) / m_multires2DScale);
    };

    inline bool mapChanged(const nav_msgs::MapMetaData &oldMapInfo, const nav_msgs::MapMetaData &newMapInfo)
    {
        return (oldMapInfo.height != newMapInfo.height || oldMapInfo.width != newMapInfo.width || oldMapInfo.origin.position.x != newMapInfo.origin.position.x || oldMapInfo.origin.position.y != newMapInfo.origin.position.y);
    };

};