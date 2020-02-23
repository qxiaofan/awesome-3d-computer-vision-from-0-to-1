## 四 答疑

**4.1 星主，您好，我已经用双目摄像头生成一个目标物体的五六个方向的点云，怎么把这些融合一起，生成一个完整点的点云，用什么方法**
答：这个需要将点云进行配准，可以参考之前我在我们公众号总结的三篇文章：
1)[一分钟详解PCL中点云配准技术](https://mp.weixin.qq.com/s?__biz=MzU1MjY4MTA1MQ==&mid=2247484425&idx=1&sn=fffa30c88cbd0c51d159fa1ea7d738c2&chksm=fbff2f3dcc88a62b953a95552a3db01e554e341978e1b0165ea342b9765b907396802baed7e7&token=906868167&lang=zh_CN#rd)
2)[[点云配准(一 两两配准)](https://mp.weixin.qq.com/s?__biz=MzU1MjY4MTA1MQ==&mid=2247484414&idx=1&sn=1636d47b38cc0b47e1f3ce60d3670d65&chksm=fbff28cacc88a1dcceb15d1819f9f7c2a138886c8dc39f1bcbc26ff40acda64a170dd6cff0a7&token=906868167&lang=zh_CN#rd)
3、[3Ｄ点云配准（二多幅点云配准）](https://mp.weixin.qq.com/s?__biz=MzU1MjY4MTA1MQ==&mid=2247484422&idx=1&sn=55b7497b5262a184fc183283cd34e27b&chksm=fbff2f32cc88a62402eb37957d8d496e0b34cdcb13db973b8b27d6cbb93ee3a45dfc8324f8dc&token=906868167&lang=zh_CN#rd)
你这个双目摄像机是搭载在机械臂上还是什么物体上呢，如果是机械臂上，需要将相机与机械臂进行手眼标定，手眼标定的问题可以查看我们的快速导航，里面汇总了手眼标定的一些经验，可以参考下。

**4.2 群主，你好，请问下，在深度学习tensorflow中，调用多个GPU，怎么调用用了。现在公司给了我两块GPU，(是一个机器多个gpu)。请问如何调用，有这方面资料参考吗？**
答：tensorflow实现单机下的多GPU调用，相对于多机分布式的集群操作还是比较简单的，其实tensorflow官网上有多GPUs使用教程，主要通过设定显卡设备号，给你三个参考链接：
1、https://cloud.tencent.com/developer/article/1155836
2、http://www.tensorfly.cn/tfdoc/how_tos/using_gpu.html
3、https://www.baidu.com/link?url=CEifTwWRJudgJt1zoZq...https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/6_MultiGPU/multigpu_basics.ipynb

**4.3 最近在论文中看到这种，可以模拟散乱场景，可以选择相机位姿，生成深度图的，所谓的 virtual camera，星主有接触过吗**
答：可以试试如下方法：1、有很多三维软件，比如meshlab或者PolyWorks、GeoMetric等，直接导入CAD等三维模型。 2、可以试试利用V-Rep进行仿真：[VREP探索（一）——认识VREP_嵌入式](https://blog.csdn.net/hyhop150/article/details/54562411)或者ROS系统下的gazebo模块进行仿真。

**4.4 哈咯，我想咨询一下，最近让调研一下移动机器人的室内定位问题。老板让考虑Wi-Fi，这个您有了解吗？他和现在大家说的激光slam，视觉slam有什么关系吗？**
答：这个移动机器人，比如以扫地机为例，有很多是以Wi-Fi来进行数据通讯的。也就是移动机器人和手机端共用一个网关，即可进行数据交互。
至于这个与视觉SLAM和激光SLAM的关系，那仅仅是数据交互的一环，视觉SLAM是以视觉传感器为主的即时定位与构建系统，而激光SLAM，是以激光雷达传感器为主的即时定位与构建系统。

嘉宾补充回答：身边有朋友做室内定位，主要用wifi和uwb比较多。但室内的难点是，wifi和uwb信号多次反弹造成的信息叠加形成复杂网络，对信号处理，复杂网络定位，优化滤波等要求比较高。这是学术上，具体工程上的不太了解。

**4.5 星主您好，请问是否可以推荐一些双目视觉的数据集？**
答：[超全的3D视觉数据集汇总](https://mp.weixin.qq.com/s?__biz=MzU1MjY4MTA1MQ==&mid=2247484787&idx=1&sn=d781d80abb06a1b79e80a04248892b49&chksm=fbff2e47cc88a7517c8c381cd6e55f04f8b5433053e8d896fc18cd09878fb58a85efabb1670b&token=498280384&lang=zh_CN#rd) 

**4.6 请问一下，点云匹配后怎么计算匹配误差，看到知乎上您的回答是如图所示，能不能进行一个解释？**
答：不过ICP的匹配误差思想还是很简单的，主要目的就是找到旋转和平移参数，将两个不同坐标系下的点云，以其中一个点云坐标系为全局坐标系，另一个点云经过旋转和平移后两组点云重合部分完全重叠。针对采集的点云和待匹配点云，首先初始化变换矩阵R和T，然后计算出两幅点云最近点对经过变换之后的误差（最小二乘），通过误差不断的迭代更新R和T，直至最终误差小于某一阈值停止迭代。

**4.7 你好vio还有哪些发展方向了？是不是可以将感知和定位结合比如语义slam？**
答：VIO，其实也就是VO+IMU，这里的VO也即视觉SLAM的前端，VIO即视觉惯性里程计，有时也叫视觉惯性系统，是融合相机和IMU数据实现SLAM的算法。根据融合框架的区别，又分为紧耦合和松耦合，松耦合中视觉运动估计和惯导运动估计系统是两个独立的模块，将每个模块的输出结果进行融合，而紧耦合则是使用两个传感器的原始数据共同估计一组变量，传感器噪声也是相互影响的，紧耦合算法上比较复杂，但充分利用了传感器数据，可以实现更好的效果，是目前的研究重点。

其实SLAM本身也是具有感知和定位的，而语义SLAM则是目前很多SLAM公司的产品研究热点，比如扫地机上，希望可以通过语义信息给用户在APP更直观的展示，AR的产品上也同样如此。

**4.8 小凡你好，我是刚入门3d视觉的小白，本身是从2d视觉背景入行的，熟悉opencv，只对2d算法熟悉，但没有学习过有关3d的相关知识，目前在自学pcl，发现3d有关的中文教学相对2d很少，本人英文很差，查很多网上课程大多是国外大学的视频英语授课，有名的教材课本也都是英语原文。请问3d 视觉处理的相关应用和开发有那些中文的视频和教材呢？网路上的资源很多，但是太过零散，对小白来说很需要系统化的教程循序渐进的走完教程，谢谢。**
答：对于3D视觉的入门与学习，其实这块说起来很大。就我的学习经验来说，还是要以项目为依托，在实战中学习效率最高。
如果现在只是想单纯地学习的话，我的建议可以动手实操如下几个方面：
1、单目、双目的标定;
2、三维重建算法的研究;
3、点云后处理;
4、手眼标定;
5、界面编程QT的学习，主要是三维可视化地显示。
如果你想学习点云后处理的话，可以参考《点云库PCL从入门到精通》，书中附有代码示例，可以学习，或者PCL库本身的Samples也可以的。
如果是标定或者三维重建，建议可以先看一些论文，代码的话在网上也有很多，可以先跑起来。

**4.9 星主你好！pcl中双边滤波（binary filter）针对的点云类型是XYZI，带有强度信息I，但是双目扫出来的点云只有XYZ，请问I咋弄**
答：这里点云的强度图，又叫点云的亮度图，与深度图不同，点云的强度图一般在点云采集时，相机同步给出，反应的是光的亮度情况。

**4.10 lk光流法和orb特征匹配跟踪哪个环境适应性好？**
答：ORB特征点，主要通过特征点匹配来跟踪点，计算几何关系得到R,t,BA来优化R,t。

光流法主要是基于灰度不变假设，把特征点法中的描述子和匹配换成了光流跟踪，之后求解R,t。

其实对于每种方法，都会有其优缺点。
比如对于特征点法，优点是：运动过大时，只要匹配点还在像素内，则不太会引起无匹配，相对于直接法或者光流法来说，有更好的鲁棒性。缺点是：特征点过多过少都无法正常工作，且只能用来构建稀疏地图，计算效率不高。

而对于光流法，优点是不需要计算描述子，不需要匹配特征点，节省了很多计算量，关键点提取多少都能工作，从稀疏到稠密重构基本上都可以用。缺点呢，基于灰度不变假设，容易受外界光照影响；相机发生大尺度移动或者旋转时无法很好的追踪，非凸优化，容易局部极值。用尺度金字塔改善局部极值，组合光流法（增加旋转描述）改善旋转。

如果综合来看，个人还是觉得ORB特征点跟踪对环境适应性更好点。

**4.11 想问下，有哪些适合看的关于三维方面的开源项目呢，能学习下的，国内的或是国外的都可以。或是有哪些点云处理相关的也可以的，想平时学习下，最好附上git库的链接了**
答：对于3D方面的开源项目，我倒是没有特刻意整理Git上的，不过我们星球里之前倒是有相关的，比如三维重建的：1. http://mesh.brown.edu/byo3d/source.html  2. https://www.3dunderworld.org/ 此外，如果想找git上的开源代码，可以参考https://mp.weixin.qq.com/s/mR6LClodXNFrp_uJuYUx7g
你关注一下对应的3D模块就可以啦

**4.12 如何对不规则物体点云进行体力计算呢**
答：这里是指点云的体积计算吗，这个问题倒是没有遇见过，这个我的建议是1、自己实现各部分体积计算方法，然后累加；2、使用下PolyWorks那边的技术支持，可以问一下那边是否提供这样的功能，PolyWorks在逆向工程上还是很强大的。这个问题我晚点和伟哥也一起讨论研究下，后面有好建议告诉你啊。
嘉宾补充回答：三角化之后，放到UG或者solidworks里，可以输出体积，还可以根据不同密度出质量等。polyworks有c++API，导入也可以三角化，但是没试过计算体积。还有开源的meshlab可能有你想要的功能。如果点云是凸包的，可以类似积分的思想：https://www.cnblogs.com/21207-iHome/p/7461458.html


**4.13 星主有没有研究过Drost的PPF方法（三维位姿估计），opencv自带的surfacematching 无法在windows下跑，有没有啥解决方法**
答：不好意思啊，问题我居然刚看到。这个问题我记得我们星球里之前有位作者倒是回答过这个问题，你可以查看下[这个帖子](https://t.zsxq.com/AyfEUzF)
同时他的[个人博客地址](https://littlebearsama.github.io/categories/ObjectDetection-and-PoseEstimation/0-tutorial/) 可以参考下。
如果需要联系他的，可以微信上联系我，我给你引荐下。

**4.14 问您一个问题,我想利用双目相机(相机固定)获得图像中某个像素点(u,v)在世界坐标系坐标(二维码所在坐标系)中的坐标(x,y,z),有哪些方法以及步骤?我现在思维有点乱,想让您帮我梳理一下.O(∩_∩)O谢谢**
答：为了得到三维点，简单总结要经过以下几个大致步骤：
1、左右相机的单目标定，标定出相机的内参
2、两个相机的双目标定，标定出外参R,T矩阵。
3、左右图像的特征点检测，检测得到左右图的二维点坐标;
4、代入内外参，进行三角测量，计算出三维点。
每个环节都会有精度误差，尽量保证每个环节的精度都要高点。

**4.15 博主你好，论文中说得有序点云（organized pointcloud）如何获取呢**
答：有序点云一般来讲，需要自己将无序点云转化成有序点云，也就是自己设定x、y、z方向的密度，根据任意方向的最大最小值，将点云均分，类似于二维图像那样，给出x,y的列数，即可推算出其三维坐标。建议自己实现，如果后续实现过程中有疑问的，可以加我微信QYong2014,我晚点可发你参考下点云有序化的实现。（暂时代码还在网盘里，家里网速太渣了，晚点有需要我下载分享给你，不过是我们自己的点云库，非PCL实现）

**4.16 博主，请问有没有推荐的深度相机的深度图和彩色图对齐的博客或者资源，看星球里的资源没有专门针对这个的～**
答：可以参考如下代码demo：

// C++ 标准库
#include <iostream>
#include <string>

// OpenCV 库
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// PCL 库
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// 定义点云类型
typedef pcl::PointXYZ PointT;
// typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud; 

using namespace std;
using namespace cv;
// 相机内参
const double camera_factor = 1000;
//RGB内参
// const double camera_cx = 327.411224;
// const double camera_cy = 243.387131;
// const double camera_fx = 454.778259;
// const double camera_fy = 454.778259;
//IR内参
const double camera_cx = 311.829987;
const double camera_cy = 274.658997;
const double camera_fx = 579.234009;
const double camera_fy = 579.234009;

//RGB外参
//Mat tvc_rgb = (Mat_<float>(4,4)<<-0.001208959960532363,0.01149703337966534,0.9999331760869226,0.19,
				 // -0.9995679131951007,0.02935296841445894,-0.001546012953174262,0.020,
				//  -0.02936878149677709,-0.9995029872235134,0.01145657909971065,0.05400000095367432,
				 // 0.0,0.0,0.0,1.0);
//RGB IR外参
//Mat rgb_ir = (Mat_<float>(4,4)<< 0.999875,  -0.002629,  -0.015589, -0.030002598,
				 // 0.002623,  0.999996,  -0.000453 , -0.000049383,
				  //0.015590,  0.000412,  0.999878,  -0.001055870,
				 // 0.0,  0.0,  0.0 ,1.0);

// 主函数 
int main( int argc, char** argv )
{
    // 读取./data/rgb.png和./data/depth.png，并转化为点云

    cv::Mat rgb, depth;
    //没有用到RGB
//     rgb = cv::imread( "../data/Color_46.png" );

    depth = cv::imread( "../depthImg/Depth_00056366.png", -1 );

    std::cout<<"read cloud data from local file success.."<<endl;

    // 点云变量
    // 使用智能指针，创建一个空点云。这种指针用完会自动释放。
    PointCloud::Ptr cloud ( new PointCloud );
    
    
        //点云滤波，直通滤波
//     pcl::PassThrough<pcl::PointXYZ> pass;
// 
//     pass.setInputCloud(cloud);
//     pass.setFilterFieldName("z");
//     pass.setFilterLimits(0.0,1.0);
    
    
//     PointCloud::Ptr cloud_filtered ( new PointCloud );
    
//     pass.filter(*cloud_filtered);
    
    //转换到地宝坐标系
    float x,y,z;
    Mat w_point = Mat_<float>(4,1);  //世界坐标系下的位姿
    Mat ir_point;		//IR坐标系		
    // 遍历深度图
    for (int m = 0; m < depth.rows; m++)
    {
        for (int n=0; n < depth.cols; n++)
        {
            // 获取深度图中(m,n)处的值
            ushort d = depth.ptr<ushort>(m)[n];
            // d 可能没有值，若如此，跳过此点
            if (d == 0)
                continue;
            // d 存在值，则向点云增加一个点
            PointT p;


	    
	    // 计算这个点的空间坐标
	    p.z = double(d) / camera_factor;
	    p.x = (n - camera_cx) * p.z / camera_fx;
	    p.y = (m - camera_cy) * p.z / camera_fy;
	    
	    //ir_point = (Mat_<float>(4,1)<< x ,y, z ,1);
	    
	   // w_point = tvc_rgb*rgb_ir*ir_point;
	    
	    //p.x = w_point.at<float>(0,0);
	   // p.y = w_point.at<float>(1,0);
	   // p.z = w_point.at<float>(2,0);
	    

            
	      // 从rgb图像中获取它的颜色
//            p.b = rgb.ptr<uchar>(m)[n*3];
//            p.g = rgb.ptr<uchar>(m)[n*3+1];
//            p.r = rgb.ptr<uchar>(m)[n*3+2];
	    
            // 把p加入到点云中
            cloud->points.push_back( p );
        }
    }
    
    // 设置并保存点云
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cout<<"point cloud size = "<<cloud->points.size()<<endl;
    cloud->is_dense = false;
    
   
    pcl::io::savePCDFile( "../Depth_00056366.pcd", *cloud );

    // 清除数据并退出
    cloud->points.clear();
//     cloud_filtered->points.clear();
    cout<<"Point cloud saved."<<endl;
    return 0;
}

**4.17 博主好，最近在做一个双目重建项目，需要很高的的标定精度，所以定制了一个圆环标定板，然后用halcon来进行双目标定，博主有没有这方面的代码 或者相关资料呀？ 能否指导一下，谢谢博主了**
答：如果是halcon的算子的话，我之前项目不用的，主要是用的OpenCV实现的，对于双目标定，OpenCV里面的stereoCalibrate函数你可以参考下，到OpenCV的documentation里查看其相关用法就好啦，这个函数不难的。
嘉宾补充回答：halcon双目标定我实操过，精度稍微差一些。如果追求精度可以使用激光三角进行标定，自己造一个就OK了

**4.18 星主您好，最近我需要做一个高精度结构光相机，对于高精度的要求，我没有头绪，请问该从哪些方面着手呢？**
答：问题中你想做高精度的结构光相机，那么以下几个环节你应该都要绕不开：
1、相机标定；
2、投影仪标定;
3、手眼标定;
4、三维重建;
5、点云后处理.
以上的每个环节都会影响到你最终的测量精度，把每个环节都能做到极致，你的测量精度才能达到最优。
嘉宾补充回答：说下单目散斑结构光精度问题。以下都是以单点角度说明（较低分辨率图像）标定reference图的单点精度（提取散点中心点的精度（小数点有效位数）匹配成功后单点像素的最优视差（亚像素拟合）

其他的问题集锦都更新在星球里，列表如下：
[01~60](https://wx.zsxq.com/dweb2/index/group/825412441552?from=mweb&type=detail)
[61-120](https://wx.zsxq.com/dweb2/index/group/825412441552?from=mweb&type=detail)
[181-240](https://wx.zsxq.com/dweb2/index/group/825412441552?from=mweb&type=detail)

**更新于 Date：2020-02-23**
