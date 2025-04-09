#include "octomap.h"

OctoMapping::OctoMapping(ros::NodeHandle *nh)
{
    nh->param("zmax", m_occupancyMaxZ, 1.0);
    nh->param("zmin", m_occupancyMinZ, -1.0);

    z_filter.setFilterFieldName("z");
    z_filter.setFilterLimits(m_occupancyMinZ, m_occupancyMaxZ);

    m_colorFactor = 0.8;
    m_res = 0.1;

    m_octree = new OcTreeT(m_res);
    m_octree->setProbHit(0.7);
    m_octree->setProbMiss(0.4);
    m_octree->setClampingThresMin(0.12);
    m_octree->setClampingThresMax(0.97);
    m_treeDepth = m_octree->getTreeDepth();
    m_maxTreeDepth = m_treeDepth;
    m_gridmap.info.resolution = m_res;
    m_minSizeX = 0;
    m_minSizeY = 0;

    m_markerPub = nh->advertise<visualization_msgs::MarkerArray>("octomap", 1);
    m_mapPub = nh->advertise<nav_msgs::OccupancyGrid>("projected_map", 5, true);

    printf("radar mapping init\n");
}

void OctoMapping::insert_pc_pose(sensor_msgs::PointCloud2ConstPtr msg, Eigen::Matrix4f &pose)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcd(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*msg, *pcd);

    z_filter.setInputCloud(pcd);
    z_filter.filter(*pcd);

    pcl::transformPointCloud(*pcd, *pcd, pose);
    octomath::Vector3 sensorOrigin(pose(0, 3), pose(1, 3), pose(2, 3));
    insert_scan(pcd, sensorOrigin);

    getROSMarkerArray(msg->header.stamp);
}

void OctoMapping::reset_octree_map()
{
    visualization_msgs::MarkerArray occupiedNodesVis;
    occupiedNodesVis.markers.resize(m_treeDepth + 1);
    ros::Time rostime = ros::Time::now();
    m_octree->clear();

    for (std::size_t i = 0; i < occupiedNodesVis.markers.size(); ++i)
    {
        occupiedNodesVis.markers[i].header.frame_id = "map";
        occupiedNodesVis.markers[i].header.stamp = rostime;
        occupiedNodesVis.markers[i].ns = "map";
        occupiedNodesVis.markers[i].id = i;
        occupiedNodesVis.markers[i].type = visualization_msgs::Marker::CUBE_LIST;
        occupiedNodesVis.markers[i].action = visualization_msgs::Marker::DELETE;
    }
    // m_markerPub.publish(occupiedNodesVis);
}

void OctoMapping::insert_scan(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr map_part,
                              const octomath::Vector3 sensorOrigin)
{
    octomap::KeySet free_cells, occupied_cells;

    for (auto p : map_part->points)
    {

        octomath::Vector3 point(p.x, p.y, p.z);
        float range = (point - sensorOrigin).norm();
        if (p.z > 1.5 || p.z < -2 || range > 5)
            continue;

        // free cells
        if (m_octree->computeRayKeys(sensorOrigin, point, m_keyRay))
        {
            free_cells.insert(m_keyRay.begin(), m_keyRay.end());
        }
        // occupied endpoint
        octomap::OcTreeKey key;
        if (m_octree->coordToKeyChecked(point, key))
        {
            occupied_cells.insert(key);

            updateMinKey(key, m_updateBBXMin);
            updateMaxKey(key, m_updateBBXMax);
        }
    }

    // mark free cells only if not seen occupied in this cloud
    for (octomap::KeySet::iterator it = free_cells.begin(), end = free_cells.end(); it != end; ++it)
    {
        if (occupied_cells.find(*it) == occupied_cells.end())
        {
            m_octree->updateNode(*it, false);
        }
    }
    // now mark all occupied cells:
    for (octomap::KeySet::iterator it = occupied_cells.begin(), end = occupied_cells.end(); it != end; it++)
    {
        m_octree->updateNode(*it, true);
    }
}

void OctoMapping::handlePreNodeTraversal(const ros::Time &rostime)
{
    // init projected 2D map:
    m_gridmap.header.frame_id = "map";
    m_gridmap.header.stamp = rostime;
    nav_msgs::MapMetaData oldMapInfo = m_gridmap.info;

    // TODO: move most of this stuff into c'tor and init map only once (adjust if size changes)
    double minX, minY, minZ, maxX, maxY, maxZ;
    m_octree->getMetricMin(minX, minY, minZ);
    m_octree->getMetricMax(maxX, maxY, maxZ);

    octomap::point3d minPt(minX, minY, minZ);
    octomap::point3d maxPt(maxX, maxY, maxZ);
    octomap::OcTreeKey minKey = m_octree->coordToKey(minPt, m_maxTreeDepth);
    octomap::OcTreeKey maxKey = m_octree->coordToKey(maxPt, m_maxTreeDepth);

    // add padding if requested (= new min/maxPts in x&y):
    double halfPaddedX = 0.5 * m_minSizeX;
    double halfPaddedY = 0.5 * m_minSizeY;
    minX = std::min(minX, -halfPaddedX);
    maxX = std::max(maxX, halfPaddedX);
    minY = std::min(minY, -halfPaddedY);
    maxY = std::max(maxY, halfPaddedY);
    minPt = octomap::point3d(minX, minY, minZ);
    maxPt = octomap::point3d(maxX, maxY, maxZ);

    octomap::OcTreeKey paddedMaxKey;
    if (!m_octree->coordToKeyChecked(minPt, m_maxTreeDepth, m_paddedMinKey))
    {
        ROS_ERROR("Could not create padded min OcTree key at %f %f %f", minPt.x(), minPt.y(), minPt.z());
        return;
    }
    if (!m_octree->coordToKeyChecked(maxPt, m_maxTreeDepth, paddedMaxKey))
    {
        ROS_ERROR("Could not create padded max OcTree key at %f %f %f", maxPt.x(), maxPt.y(), maxPt.z());
        return;
    }

    assert(paddedMaxKey[0] >= maxKey[0] && paddedMaxKey[1] >= maxKey[1]);

    m_multires2DScale = 1 << (m_treeDepth - m_maxTreeDepth);
    m_gridmap.info.width = (paddedMaxKey[0] - m_paddedMinKey[0]) / m_multires2DScale + 1;
    m_gridmap.info.height = (paddedMaxKey[1] - m_paddedMinKey[1]) / m_multires2DScale + 1;

    int mapOriginX = minKey[0] - m_paddedMinKey[0];
    int mapOriginY = minKey[1] - m_paddedMinKey[1];
    assert(mapOriginX >= 0 && mapOriginY >= 0);

    // might not exactly be min / max of octree:
    octomap::point3d origin = m_octree->keyToCoord(m_paddedMinKey, m_treeDepth);
    double gridRes = m_octree->getNodeSize(m_maxTreeDepth);
    m_projectCompleteMap = (!m_incrementalUpdate || (std::abs(gridRes - m_gridmap.info.resolution) > 1e-6));
    m_gridmap.info.resolution = gridRes;
    m_gridmap.info.origin.position.x = origin.x() - gridRes * 0.5;
    m_gridmap.info.origin.position.y = origin.y() - gridRes * 0.5;
    if (m_maxTreeDepth != m_treeDepth)
    {
        m_gridmap.info.origin.position.x -= m_res / 2.0;
        m_gridmap.info.origin.position.y -= m_res / 2.0;
    }

    // workaround for  multires. projection not working properly for inner nodes:
    // force re-building complete map
    if (m_maxTreeDepth < m_treeDepth)
        m_projectCompleteMap = true;

    if (m_projectCompleteMap)
    {
        ROS_DEBUG("Rebuilding complete 2D map");
        m_gridmap.data.clear();
        // init to unknown:
        m_gridmap.data.resize(m_gridmap.info.width * m_gridmap.info.height, -1);
    }
    else
    {

        if (mapChanged(oldMapInfo, m_gridmap.info))
        {
            ROS_DEBUG("2D grid map size changed to %dx%d", m_gridmap.info.width, m_gridmap.info.height);
            adjustMapData(m_gridmap, oldMapInfo);
        }
        nav_msgs::OccupancyGrid::_data_type::iterator startIt;
        size_t mapUpdateBBXMinX = std::max(0, (int(m_updateBBXMin[0]) - int(m_paddedMinKey[0])) / int(m_multires2DScale));
        size_t mapUpdateBBXMinY = std::max(0, (int(m_updateBBXMin[1]) - int(m_paddedMinKey[1])) / int(m_multires2DScale));
        size_t mapUpdateBBXMaxX = std::min(int(m_gridmap.info.width - 1), (int(m_updateBBXMax[0]) - int(m_paddedMinKey[0])) / int(m_multires2DScale));
        size_t mapUpdateBBXMaxY = std::min(int(m_gridmap.info.height - 1), (int(m_updateBBXMax[1]) - int(m_paddedMinKey[1])) / int(m_multires2DScale));

        assert(mapUpdateBBXMaxX > mapUpdateBBXMinX);
        assert(mapUpdateBBXMaxY > mapUpdateBBXMinY);

        size_t numCols = mapUpdateBBXMaxX - mapUpdateBBXMinX + 1;

        // test for max idx:
        uint max_idx = m_gridmap.info.width * mapUpdateBBXMaxY + mapUpdateBBXMaxX;
        if (max_idx >= m_gridmap.data.size())
            ROS_ERROR("BBX index not valid: %d (max index %zu for size %d x %d) update-BBX is: [%zu %zu]-[%zu %zu]", max_idx, m_gridmap.data.size(), m_gridmap.info.width, m_gridmap.info.height, mapUpdateBBXMinX, mapUpdateBBXMinY, mapUpdateBBXMaxX, mapUpdateBBXMaxY);

        // reset proj. 2D map in bounding box:
        for (unsigned int j = mapUpdateBBXMinY; j <= mapUpdateBBXMaxY; ++j)
        {
            std::fill_n(m_gridmap.data.begin() + m_gridmap.info.width * j + mapUpdateBBXMinX,
                        numCols, -1);
        }
    }
}

void OctoMapping::handlePostNodeTraversal(const ros::Time &rostime)
{
    publishProjected2DMap(rostime);
}

void OctoMapping::handleOccupiedNode(const OcTreeT::iterator &it)
{
    if (m_projectCompleteMap)
    {
        update2DMap(it, true);
    }
}

void OctoMapping::handleFreeNode(const OcTreeT::iterator &it)
{
    if (m_projectCompleteMap)
    {
        update2DMap(it, false);
    }
}

void OctoMapping::handleOccupiedNodeInBBX(const OcTreeT::iterator &it)
{
    if (!m_projectCompleteMap)
    {
        update2DMap(it, true);
    }
}

void OctoMapping::handleFreeNodeInBBX(const OcTreeT::iterator &it)
{
    if (!m_projectCompleteMap)
    {
        update2DMap(it, false);
    }
}

void OctoMapping::update2DMap(const OcTreeT::iterator &it, bool occupied)
{
    // update 2D map (occupied always overrides):

    if (it.getDepth() == m_maxTreeDepth)
    {
        unsigned idx = mapIdx(it.getKey());
        if (occupied)
            m_gridmap.data[mapIdx(it.getKey())] = 100;
        else if (m_gridmap.data[idx] == -1)
        {
            m_gridmap.data[idx] = 0;
        }
    }
    else
    {
        int intSize = 1 << (m_maxTreeDepth - it.getDepth());
        octomap::OcTreeKey minKey = it.getIndexKey();
        for (int dx = 0; dx < intSize; dx++)
        {
            int i = (minKey[0] + dx - m_paddedMinKey[0]) / m_multires2DScale;
            for (int dy = 0; dy < intSize; dy++)
            {
                unsigned idx = mapIdx(i, (minKey[1] + dy - m_paddedMinKey[1]) / m_multires2DScale);
                if (occupied)
                    m_gridmap.data[idx] = 100;
                else if (m_gridmap.data[idx] == -1)
                {
                    m_gridmap.data[idx] = 0;
                }
            }
        }
    }
}

void OctoMapping::publishProjected2DMap(const ros::Time &rostime)
{
    if (m_mapPub.getNumSubscribers() > 0)
    {
        m_gridmap.header.stamp = rostime;
        m_mapPub.publish(m_gridmap);
    }
}

void OctoMapping::adjustMapData(nav_msgs::OccupancyGrid &map, const nav_msgs::MapMetaData &oldMapInfo) const
{
    if (map.info.resolution != oldMapInfo.resolution)
    {
        ROS_ERROR("Resolution of map changed, cannot be adjusted");
        return;
    }

    int i_off = int((oldMapInfo.origin.position.x - map.info.origin.position.x) / map.info.resolution + 0.5);
    int j_off = int((oldMapInfo.origin.position.y - map.info.origin.position.y) / map.info.resolution + 0.5);

    if (i_off < 0 || j_off < 0 || oldMapInfo.width + i_off > map.info.width || oldMapInfo.height + j_off > map.info.height)
    {
        ROS_ERROR("New 2D map does not contain old map area, this case is not implemented");
        return;
    }

    nav_msgs::OccupancyGrid::_data_type oldMapData = map.data;

    map.data.clear();
    // init to unknown:
    map.data.resize(map.info.width * map.info.height, -1);

    nav_msgs::OccupancyGrid::_data_type::iterator fromStart, fromEnd, toStart;

    for (int j = 0; j < int(oldMapInfo.height); ++j)
    {
        // copy chunks, row by row:
        fromStart = oldMapData.begin() + j * oldMapInfo.width;
        fromEnd = fromStart + oldMapInfo.width;
        toStart = map.data.begin() + ((j + j_off) * m_gridmap.info.width + i_off);
        copy(fromStart, fromEnd, toStart);
    }
}

visualization_msgs::MarkerArray OctoMapping::getROSMarkerArray(ros::Time stamp)
{
    visualization_msgs::MarkerArray occupiedNodesVis;
    // each array stores all cubes of a different size, one for each depth level:
    occupiedNodesVis.markers.resize(m_treeDepth + 1);

    // Height
    double minX, minY, minZ, maxX, maxY, maxZ;
    m_octree->getMetricMin(minX, minY, minZ);
    m_octree->getMetricMax(maxX, maxY, maxZ);

    handlePreNodeTraversal(stamp);

    // traverse all leafs in the tree:
    for (OcTreeT::iterator it = m_octree->begin(m_maxTreeDepth), end = m_octree->end(); it != end; ++it)
    {
        bool inUpdateBBX = isInUpdateBBX(it);

        if (m_octree->isNodeOccupied(*it))
        {
            double z = it.getZ();
            double half_size = it.getSize() / 2.0;
            double size = it.getSize();
            double x = it.getX();
            double y = it.getY();

            handleOccupiedNode(it);
            if (inUpdateBBX)
                handleOccupiedNodeInBBX(it);

            unsigned idx = it.getDepth();
            assert(idx < occupiedNodesVis.markers.size());

            geometry_msgs::Point cubeCenter;
            cubeCenter.x = x;
            cubeCenter.y = y;
            cubeCenter.z = z;

            occupiedNodesVis.markers[idx].points.push_back(cubeCenter);
            occupiedNodesVis.markers[idx].pose.orientation.w = 1;

            double h = (1.0 - std::min(std::max((cubeCenter.z - minZ) / (maxZ - minZ), 0.0), 1.0)) * m_colorFactor;
            occupiedNodesVis.markers[idx].colors.push_back(heightMapColor(h));
        }
        else
        { // node not occupied => mark as free in 2D map if unknown so far
            double z = it.getZ();
            double half_size = it.getSize() / 2.0;
            if (z + half_size > m_occupancyMinZ && z - half_size < m_occupancyMaxZ)
            {
                handleFreeNode(it);
                if (inUpdateBBX)
                    handleFreeNodeInBBX(it);
            }
        }
    }

    // publish 2D occupancy
    handlePostNodeTraversal(stamp);

    for (unsigned i = 0; i < occupiedNodesVis.markers.size(); ++i)
    {
        double size = m_octree->getNodeSize(i);

        occupiedNodesVis.markers[i].header.frame_id = "map";
        occupiedNodesVis.markers[i].header.stamp = stamp;
        occupiedNodesVis.markers[i].ns = "radar_map";
        occupiedNodesVis.markers[i].id = i;
        occupiedNodesVis.markers[i].type = visualization_msgs::Marker::CUBE_LIST;
        occupiedNodesVis.markers[i].scale.x = size;
        occupiedNodesVis.markers[i].scale.y = size;
        occupiedNodesVis.markers[i].scale.z = size;
        if (occupiedNodesVis.markers[i].points.size() > 0)
            occupiedNodesVis.markers[i].action = visualization_msgs::Marker::ADD;
        else
            occupiedNodesVis.markers[i].action = visualization_msgs::Marker::DELETE;
    }

    return occupiedNodesVis;
}

std_msgs::ColorRGBA OctoMapping::heightMapColor(double h)
{

    std_msgs::ColorRGBA color;
    color.a = 1.0;
    // blend over HSV-values (more colors)

    double s = 1.0;
    double v = 1.0;

    h -= floor(h);
    h *= 6;
    int i;
    double m, n, f;

    i = floor(h);
    f = h - i;
    if (!(i & 1))
        f = 1 - f; // if i is even
    m = v * (1 - s);
    n = v * (1 - s * f);

    switch (i)
    {
    case 6:
    case 0:
        color.r = v;
        color.g = n;
        color.b = m;
        break;
    case 1:
        color.r = n;
        color.g = v;
        color.b = m;
        break;
    case 2:
        color.r = m;
        color.g = v;
        color.b = n;
        break;
    case 3:
        color.r = m;
        color.g = n;
        color.b = v;
        break;
    case 4:
        color.r = n;
        color.g = m;
        color.b = v;
        break;
    case 5:
        color.r = v;
        color.g = m;
        color.b = n;
        break;
    default:
        color.r = 1;
        color.g = 0.5;
        color.b = 0.5;
        break;
    }

    return color;
}

OctoMapping::~OctoMapping()
{
    if (m_octree)
    {
        delete m_octree;
        m_octree = NULL;
    }
}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "octomap_node");
    ros::NodeHandle nh;
    OctoMapping mapper(&nh);

    std::string pose_path = "/default/pose/path";
    nh.getParam("pose_path", pose_path);
    ROS_INFO("Pose path: %s", pose_path.c_str());
    if (!std::filesystem::exists(pose_path))
    {
        ROS_ERROR("Pose path does not exist: %s", pose_path.c_str());
        return -1;
    }
    std::ifstream pose_file(pose_path);

    // stamp, seq, tx, ty, tz, qx, qy, qz, qw
    std::vector<std::pair<double, Eigen::Matrix4f>> poses;
    std::string line;
    std::getline(pose_file, line);

    while (std::getline(pose_file, line))
    {
        std::stringstream ss(line);
        std::string token;
        std::vector<double> data;
        while (std::getline(ss, token, ','))
        {
            try
            {
                data.push_back(std::stod(token));
            }
            catch (const std::exception &e)
            {
            }
        }

        Eigen::Matrix4f p = Eigen::Matrix4f::Identity();
        p(0, 3) = data[2];
        p(1, 3) = data[3];
        p(2, 3) = data[4];
        Eigen::Quaternionf q(data[7], data[4], data[5], data[6]);
        p.block<3, 3>(0, 0) = q.toRotationMatrix();
        poses.emplace_back(data[0], p);
    }
    pose_file.close();

    // open bags
    std::string bag_path = "/default/bag/path";
    nh.getParam("bag_folder", bag_path);
    ROS_INFO("Bag path: %s", bag_path.c_str());
    std::vector<std::string> bags;
    for (const auto &entry : std::filesystem::directory_iterator(bag_path))
        bags.emplace_back(entry.path().string());
    std::sort(bags.begin(), bags.end());

    std::vector<std::string> topics;
    std::string sensor = "/velodyne_points";
    nh.getParam("topic", sensor);
    topics.push_back(sensor);
    double msg_time = 0;

    for (auto bf : bags)
    {
        rosbag::Bag bag;
        bag.open(bf, rosbag::bagmode::Read);
        std::cout << bf << std::endl;

        rosbag::View view(bag, rosbag::TopicQuery(topics));
        foreach (rosbag::MessageInstance const m, view)
        {
            sensor_msgs::PointCloud2::ConstPtr msg = m.instantiate<sensor_msgs::PointCloud2>();
            if (msg != NULL)
            {
                msg_time = msg->header.stamp.toSec();
                while (!poses.empty() && poses.front().first < msg_time)
                {
                    poses.erase(poses.begin());
                }

                if (poses.empty())
                    break;

                Eigen::Matrix4f pose = poses.front().second;
                double time = poses.front().first;
                printf("time: %lf, %lf\n", time, msg_time);

                mapper.insert_pc_pose(msg, pose);
            }

            if (!ros::ok())
                break;
        }
        bag.close();

        if (!ros::ok())
            break;
    }

    // mapper.getROSMarkerArray(ros::Time(msg_time));

    return 0;
}