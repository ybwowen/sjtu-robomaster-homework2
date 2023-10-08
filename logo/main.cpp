
#include <iostream>
#include <fstream>
#include <random>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>  
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace std;

const int randmax = 32768;
const int min_depth = 18;
const int max_depth = 22; 
const int base_depth = 20;

cv::Size img_size;
vector <cv::Mat> photos;

namespace Random{
    double randNumber(int low, int up){
        return low + 1.0 * 1ll * (up - low) * (rand() % randmax) / randmax;
    }
}

namespace Contour{
    vector <Eigen::Vector3d> points; //世界坐标 右手系 相机向z轴正方向拍摄 x轴向下
    vector < vector <cv::Point> > contours;
    vector < cv::Vec4i > hierachy;
    void work(){
        cv::Mat img_src = cv::imread("../logo.png");
        img_size = cv::Size(img_src.cols, img_src.rows);
        cv::Mat img_gray;
        cv::cvtColor(img_src, img_gray, cv::COLOR_BGR2GRAY);
        cv::threshold(img_gray, img_gray, 100, 255, cv::THRESH_OTSU);
        // cv::imshow("img_gray", img_gray);
        // cv::waitKey(0);
        cv::findContours(img_gray, contours, hierachy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
        cv::Mat drawer = cv::Mat::zeros(img_size, CV_8UC3);
        // cout << img_size << endl;
        // for(int i = 0; i < contours.size(); i++)
        //     cv::drawContours(drawer, contours, i, cv::Scalar(255, 255, 255));
        // cv::imshow("img_contour", drawer);
        // cv::waitKey(0);
        // cout << img_size.height << ' ' << img_size.width << endl;
        cv::Point center = (cv::Point){img_size.width / 2, img_size.height / 2};
        for(int i = 0; i < contours.size(); i++){
            for(auto point_cv : contours[i]){
                Eigen::Vector3d point_eigen;
                // cout << point_cv << endl;
                point_cv -= center;
                point_eigen[0] = point_cv.y;
                point_eigen[1] = point_cv.x; //一定要改成右手系
                point_eigen[2] = Random::randNumber(min_depth, max_depth);
                // point_eigen[2] = 1.0;
                // cout << point_eigen << endl;
                points.push_back(point_eigen);
            }
        }
    }
}

class Camera{
    private:
        Eigen::Matrix <double, 3, 4> K; //内参矩阵
        Eigen::Quaterniond q; //位姿四元数
        Eigen::Matrix3d R; //旋转矩阵
        Eigen::Vector3d T; //位移向量，即相机位置
        Eigen::Matrix4d W; //外参矩阵
        double f = 100.0;
        double alpha = 1.0;
        double beta = 1.0;
        double fx = alpha * f;
        double fy = beta * f;

    public:
        vector <Eigen::Vector3d> pointsWorld;
        vector <Eigen::Vector4d> pointsWorldHomo;
        vector <Eigen::Vector2d> pointsPixel;
        vector <Eigen::Vector3d> pointsPixelHomo;

        Camera(){
            K << fx, 0., img_size.height / 2.0, 0.,
                 0., fy, img_size.width / 2.0, 0.,
                 0., 0., 1., 0.;
            R = Eigen::Matrix3d::Identity();
            T << 0., 0. ,0.;
            W = Eigen::Matrix4d::Zero();
            W.block(0, 0, 3, 3) = R;
            W.block(0, 3, 3, 1) = -R * T;
            W(3, 3) = 1; 
        }

        void setPosture(Eigen::Vector3d pos, Eigen::Matrix3d rotation_matrix, double focus){
            f = focus;
            fx = alpha * f;
            fy = beta * f;
            K << fx, 0., img_size.height / 2.0, 0.,
                 0., fy, img_size.width / 2.0, 0.,
                 0., 0., 1., 0.;
            T = pos;
            R = rotation_matrix;
            W = Eigen::Matrix4d::Zero();
            W.block(0, 0, 3, 3) = R;
            W.block(0, 3, 3, 1) = -R * T;
            W(3, 3) = 1; 
        }

        void takePhoto(){
            pointsWorldHomo.resize(pointsWorld.size());
            assert(pointsWorld.size() == pointsWorldHomo.size());
            for(int i = 0; i < pointsWorld.size(); i++){
                pointsWorldHomo[i] = pointsWorld[i].homogeneous();
            }
            pointsPixelHomo.resize(pointsWorldHomo.size());
            assert(pointsPixelHomo.size() == pointsWorldHomo.size()); 
            for(int i = 0; i < pointsWorld.size(); i++){
                pointsPixelHomo[i] = K * W * pointsWorldHomo[i];
            }
            pointsPixel.resize(pointsPixelHomo.size());
            assert(pointsPixel.size() == pointsPixelHomo.size());
            for(int i = 0; i < pointsWorld.size(); i++){
                pointsPixel[i] = pointsPixelHomo[i].head(2) / pointsPixelHomo[i][2];
            }
        }

        void generatePhoto(){
            cv::Mat result_img = cv::Mat::zeros(img_size, CV_8UC3);
            for(int i = 0; i < pointsPixel.size(); i++){
                cv::Point2f p;
                p.x = pointsPixel[i][1], p.y = pointsPixel[i][0]; //改回像素坐标系
                if(p.x < 0 || p.x > img_size.width || p.y < 0 || p.y > img_size.height) continue;
                double opt = Random::randNumber(1, 10);
                if(opt < 8) continue;
                if(pointsWorld[i][2] > base_depth){
                    double tmp = Random::randNumber(1, 10);
                    if(tmp > 9)
                        cv::circle(result_img, p, Random::randNumber(1, 4), cv::Scalar(255, 255, 255), -1);
                    else
                        cv::circle(result_img, p, Random::randNumber(1, 3), cv::Scalar(255, 255, 255), -1);
                }
                    
                else if(0.75 * (base_depth - min_depth) + min_depth < pointsWorld[i][2] \
                && pointsWorld[i][2] < base_depth)
                    cv::circle(result_img, p, Random::randNumber(1, 2), cv::Scalar(255, 255, 255), -1);
                else
                    cv::circle(result_img, p, Random::randNumber(1, 2), cv::Scalar(255, 255, 255), -1);
            }
            // cv::imshow("result_img", result_img);
            // cv::waitKey(0);
            photos.push_back(result_img);
            img_size = result_img.size();
        }
};

Camera* camera;

namespace Trajectory{
    const double pi = acos(-1);
    const double linestart = 5.0;
    const double lineend = 10.0;
    const double coef_log = 0.3;

    double degree2Rad(double degree){
        return 1.0 * degree / 180 * pi;
    }

    //把相机的运动路径写成参数曲线的形式，计算相机位置和姿态

    void curveLine(const double t, Eigen::Matrix3d &rotation_matrix, Eigen::Vector3d &pos){ 
        double pos_z = t + linestart;
        pos << 0., 0., pos_z;
        // double angle = atan(pos[1] / pos[2]);

        Eigen::AngleAxisd rotation_vector(0, Eigen::Vector3d(-1, 0, 0));
        rotation_matrix = rotation_vector.matrix();

        // Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0);
        // cout << "yaw pitch roll = " << euler_angles.transpose() << endl;
        // cout << "position: " << pos << endl;

    }

    void moveLine(){
        const double start = 0.0;
        const double delta = 0.05;
        const double focus = 10.0;
        const double end = lineend - linestart;

        for(double t = start ; t <= end; t += delta){
            Eigen::Vector3d pos; 
            Eigen::Matrix3d rotation_matrix;
            curveLine(t, rotation_matrix, pos);
            // rotation_matrix = Eigen::Matrix3d::Identity();
            camera -> setPosture(pos, rotation_matrix, focus);
            camera -> takePhoto();
            camera -> generatePhoto();
        }

    }

    void curveCircle(const double t, Eigen::Matrix3d &rotation_matrix, Eigen::Vector3d &pos){ 
        const double radius_a = 800.0;
        const double radius_b = 20.0;

        pos[0] = 0.0;
        pos[1] = radius_a * sin(degree2Rad(t));
        pos[2] = linestart - radius_b + radius_b * cos(degree2Rad(t));

        Eigen::Vector3d center;
        center << 0., 0., 20;
        Eigen::Vector3d direction_vector = (center - pos).normalized();
        

        double angle = atan(direction_vector[1] / direction_vector[2]);

        Eigen::AngleAxisd rotation_vector(angle, Eigen::Vector3d(-1, 0, 0));
        // rotation_matrix = rotation_vector.matrix();
        rotation_matrix = Eigen::Matrix3d::Identity();

        // cout << "position: " << pos << endl;

    }

    void moveCircle(){
        double start = 270;
        double end = 360;
        double delta = 1.0;
        double focus = 10.0;

        for(double theta = start; theta <= end; theta += delta){
            Eigen::Vector3d pos; 
            Eigen::Matrix3d rotation_matrix;
            curveCircle(theta, rotation_matrix, pos);
            camera -> setPosture(pos, rotation_matrix, focus);
            camera -> takePhoto();
            camera -> generatePhoto();
        }
    }

    void move(){
        moveCircle();
        moveLine();
    }
}

int main(){
    srand(time(0));
    Contour::work();
    camera = new Camera();
    (*camera).pointsWorld = Contour::points;
    Trajectory::move();
    cv::VideoWriter videowriter("../logo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25, img_size, true);
    for(auto photo : photos) videowriter << photo;
    videowriter.release();
    return 0;
}