#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>  
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <cassert>

using namespace std;

int n;
const double eps = 1e-7;
const int centerx = 190;
const int centery = 160;
constexpr int rows = 720;
constexpr int cols = 1280;

class Converter{
    public:
        Eigen::Matrix <double, 3, 4> K; //内参矩阵
        vector <Eigen::Vector3d> pointsWorld; //世界坐标
        vector <Eigen::Vector4d> pointsWorldHomo;
        vector <Eigen::Vector2d> pointsPixel;
        vector <Eigen::Vector3d> pointsPixelHomo;
        Eigen::Quaterniond q; //位姿四元数
        Eigen::Matrix3d R; //旋转矩阵
        Eigen::Vector3d T; //位移向量
        Eigen::Matrix4d W; //外参矩阵
    Converter(){
        K << 400., 0., 190., 0.,
             0., 400., 160., 0.,
             0., 0., 1., 0.;
        T << 2., 2., 2.;
        q = Eigen::Quaterniond(-0.5, 0.5, 0.5, -0.5);
        R = q.toRotationMatrix();
        // W << R, T, 
        //      Eigen::Matrix <double, 1, 3>::Zero(), 1;
        W = Eigen::Matrix4d::Zero();
        W.block(0, 0, 3, 3) = R.transpose();
        W.block(0, 3, 3, 1) = -R.transpose() * T;
        W(3, 3) = 1;
    }
    void World2Pixel(){
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
};  

void init(Converter *converter){
    ifstream fin("../points.txt");
    fin >> n;
    for(int idx = 0; idx < n; idx++){
        double data[3];
        for(auto &tmp: data) fin >> tmp;
        Eigen::Vector3d point(data);
        converter -> pointsWorld.push_back(point);
    }
}

void output(Converter *converter){
    int cnt = 0;
    cv::Mat result_img = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC3);
    for(auto point : converter -> pointsPixel){
        cv::Point2f p;
        p.x = point[0], p.y = point[1];
        if(p.x < 0 || p.x > cols || p.y < 0 || p.y > rows) continue;
        cv::circle(result_img, p, 2, cv::Scalar(255, 255, 255), -1);
        cnt ++;
    }
    cv::imwrite("../result.jpg", result_img);
    cout << "Total number of points that could be captured by camera:" << cnt << endl;
}

int main(){
    Converter *converter = new Converter();
    init(converter);
    converter -> World2Pixel();
    output(converter);
    delete converter;
    return 0;
}