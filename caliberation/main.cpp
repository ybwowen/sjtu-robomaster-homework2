
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>  
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

const int board_width = 9;
const int board_height = 6;
constexpr int board_num = board_width * board_height;
const int block_width = 10;
const int block_height = 10;

class Caliberation{
    public:
        int cnt = 0;
        vector <cv::Mat> images;
        vector <vector <cv::Point2f> > imagescorners;  
        vector <vector <cv::Point3f> > imagescorners_world;
        cv::Mat K; //内参矩阵
        cv::Mat dist_coeffs; //畸变系数
        vector <cv::Mat> rvecs;
        vector <cv::Mat> tvecs;
        cv::Size img_size;

        void validation(cv::Mat img_src){
            vector <cv::Point2f> corners; corners.clear();
            if(cv::findChessboardCorners(img_src, cv::Size(board_width, board_height), corners) \
                && corners.size() == board_num){
                cv::Mat img_gray;
                cv::cvtColor(img_src, img_gray, cv::COLOR_BGR2GRAY);
                cv::find4QuadCornerSubpix(img_gray, corners, cv::Size(5, 5));
                cv::Mat img_corner = img_src.clone();
                // cv::drawChessboardCorners(img_corner, cv::Size(board_width, board_height), corners, true);
                // cv::imshow("img_corner", img_corner);
                // cv::waitKey(0);
                imagescorners.push_back(corners);
                images.push_back(img_src);
                img_size = cv::Size(img_src.rows, img_src.cols);
                cnt ++;
            }
        }

        void caliberate(){
            for(int i = 0; i < cnt; i++){ // 生成角点的世界坐标
                vector <cv::Point3f> corners_world; corners_world.clear();
                for(int h = 0; h < board_height; h++)
                    for(int w = 0; w < board_width; w++){
                        cv::Point3f point;
                        point.x = w * block_width;
                        point.y = h * block_height;
                        point.z = 0;
                        corners_world.push_back(point);
                    }
                imagescorners_world.push_back(corners_world);
            }
            cout << "Reprojection Error: " << cv::calibrateCamera(imagescorners_world, imagescorners, img_size, K, dist_coeffs, rvecs, tvecs) << endl;
            cout << K << endl;
            cout << dist_coeffs << endl;
            
            ofstream fout("../parameters.txt");
            fout << "Camera Matrix:" << endl;
            for(int i = 0; i < K.rows; i++){
                for(int j = 0; j < K.cols; j++){
                    fout << K.at<double>(i, j) << " ";
                }
                fout << endl;
            }  
            fout << endl;
            fout << "Distorted Coefficients:" << endl;
            for(int i = 0; i < dist_coeffs.cols; i++)
                fout << dist_coeffs.at<double>(0, i) << " ";
            fout << endl;
        }
        void undistorted(){
            for(int i = 0; i < images.size(); i++){
                cv::Mat img_undist;
                cv::undistort(images[i], img_undist, K, dist_coeffs);
                // cv::imshow("img_undist", img_undist);
                // cv::waitKey(0);
                cv::imwrite("../chess_undist/" + to_string(i) + ".jpg", img_undist);
            }
        }
};

int main(){
    Caliberation *caliberation = new Caliberation();
    for(int i = 0; i <= 40; i++){
        string filename = "../chess/" + to_string(i) + ".jpg";
        cv::Mat img_src = cv::imread(filename);    
        caliberation -> validation(img_src);
    }
    cout << "Valid num: " << (caliberation -> cnt) << endl;
    caliberation -> caliberate();
    caliberation -> undistorted();
    delete caliberation;
    return 0;
}