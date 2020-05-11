// 
// This code is the combination and modification of:
// 		1/ Apache 2.0 : Intel RealSense SDK Examples
//		2/ Nghia Ho   : Recorded video stabilization using Point-Features Matching
//		3/ Chen Jia   : Code Structure for stabilizing Real-time UAV video recording
//
//

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <time.h>

using namespace std;
using namespace cv;

// ============================================================ //
// USER-DEFINED CLASSES & SUB-FUNCTIONS
// ============================================================ //

//const int SMOOTHING_RADIUS = 0; // In frames. The larger the more stable the video, but less reactive to sudden panning
// In pixels. Crops the border to reduce the black borders from stabilisation being too noticeable.
const int HORIZONTAL_BORDER_CROP = 0;

// ============================================================ //
struct TransformParam
{
    TransformParam() {}
    TransformParam(double _dx, double _dy, double _da) {
        dx = _dx;
        dy = _dy;
        da = _da;
    }

    double dx;
    double dy;
    double da; // angle
};

// ============================================================ //
struct Trajectory
{
    Trajectory() {}
    Trajectory(double _x, double _y, double _a) {
        x = _x;
        y = _y;
        a = _a;
    }
    // "+"
    friend Trajectory operator+(const Trajectory& c1, const Trajectory& c2) {
        return Trajectory(c1.x + c2.x, c1.y + c2.y, c1.a + c2.a);
    }
    //"-"
    friend Trajectory operator-(const Trajectory& c1, const Trajectory& c2) {
        return Trajectory(c1.x - c2.x, c1.y - c2.y, c1.a - c2.a);
    }
    //"*"
    friend Trajectory operator*(const Trajectory& c1, const Trajectory& c2) {
        return Trajectory(c1.x * c2.x, c1.y * c2.y, c1.a * c2.a);
    }
    //"/"
    friend Trajectory operator/(const Trajectory& c1, const Trajectory& c2) {
        return Trajectory(c1.x / c2.x, c1.y / c2.y, c1.a / c2.a);
    }
    //"="
    Trajectory operator =(const Trajectory& rx) {
        x = rx.x;
        y = rx.y;
        a = rx.a;
        return Trajectory(x, y, a);
    }

    double x;
    double y;
    double a; // angle
};



// ============================================================ //
// MAIN FUNCTION
// ============================================================ //

int main(int argc, char* argv[]) try
{
    // ============================================================ //
    // STEP 01 : SETUP REALSENSE CAMERA
    // ============================================================ //    
    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Start streaming with default recommended configuration
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    pipe.start(cfg);
    const int w = 640;
    const int h = 480;

    // Prepared displayed window to show stabilized video stream
    const auto window_name = "Original Video (left) & Stabilized Video (right)";
    namedWindow(window_name, WINDOW_AUTOSIZE);

    // Set up output video
    VideoWriter out("video_out.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(2 * w, h));

    
    
    // ============================================================ //
    // STEP 02 : INPUT FIRST FRAME TO OPENCV & CONVERT TO GRAYSCALE
    // ============================================================ // 
    // Get the 1st frame from RS camera
    rs2::frameset data = pipe.wait_for_frames();
    rs2::frame prev_rs = data.get_color_frame();
    // Input RS 1st frame to OpenCV matrix & Convert to grayscale
    Mat prev(Size(w, h), CV_8UC3, (void*)prev_rs.get_data(), Mat::AUTO_STEP);
    Mat prev_gray;
    cvtColor(prev, prev_gray, COLOR_BGR2GRAY);

    // ============================================================ //
    // STEP 03 : INITIALIZE VECTORS TO STORE THE OPTICAL FLOW & TRANSFORM ESTIMATION
    // ============================================================ //
    // Vector : Optical Flow (dx, dy, da)
    vector <TransformParam> prev_to_cur_transform;
    double a = 0;
    double x = 0;
    double y = 0;

    // Vector : Trajectory of Camera Motion
    vector <Trajectory> trajectory;

    // Prepare average window to smooth the transformation
    vector <Trajectory> smoothed_trajectory;
    Trajectory X;        // posteriori state estimate
    Trajectory	X_;      // priori estimate
    Trajectory P;        // posteriori estimate error covariance
    Trajectory P_;       // priori estimate error covariance
    Trajectory K;        // gain
    Trajectory	z;       // actual measurement
    double pstd = 4e-3;  // can be changed
    double cstd = 0.25;  // can be changed
    Trajectory Q(pstd, pstd, pstd);  // process noise covariance
    Trajectory R(cstd, cstd, cstd);  // measurement noise covariance 

    // Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
    vector <TransformParam> new_prev_to_cur_transform;

    // Apply new transformation to the video
    int k = 1;
    int vert_border = HORIZONTAL_BORDER_CROP * prev.rows / prev.cols; // get the aspect ratio correct
    Mat T(2, 3, CV_64F);
    Mat last_T;


    // ============================================================ //
    // STEP 04 : INITIALIZE VIDEO STREAM  >>  CAPTURE 2ND FRAME
    // ============================================================ //
    while (waitKey(1) < 0 && getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
    {
        // Load the current frame from RealSense camera
        rs2::frameset data1 = pipe.wait_for_frames(); // Wait for next set of frames from the camera
        rs2::frame curr_rs = data1.get_color_frame();
        
        // Add watch-timer 1
        auto start_1 = chrono::steady_clock::now();
        // Create OpenCV matrix of current frame
        Mat curr(Size(w, h), CV_8UC3, (void*)curr_rs.get_data(), Mat::AUTO_STEP);
        Mat curr_gray;
        cvtColor(curr, curr_gray, COLOR_BGR2GRAY);


    // ============================================================ //
    // STEP 05 : COLLECT CORNER POINTS & CALCULATE OPTICAL FLOW BETWEEN 2 FRAMES
    // ============================================================ //
        // vector from prev to cur
        vector <Point2f> prev_corner, curr_corner;
        vector <Point2f> prev_corner2, curr_corner2;
        vector <uchar> status;
        vector <float> err;

        goodFeaturesToTrack(prev_gray, prev_corner, 200, 0.01, 30);
        calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_corner, curr_corner, status, err);

        // weed out bad matches
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i]) {
                prev_corner2.push_back(prev_corner[i]);
                curr_corner2.push_back(curr_corner[i]);
            }
        }


    // ============================================================ //
    // STEP 06 : CALCULATE TRANSFORMATION MATRIX OF MOTION BETWEEN 2 FRAMES
    // ============================================================ //
        // translation + rotation only
        Mat T = estimateRigidTransform(prev_corner2, curr_corner2, false); // false = rigid transform, no scaling/shearing

        // in rare cases no transform is found. We'll just use the last known good transform.
        if (T.data == NULL) {
            last_T.copyTo(T);
        }

        T.copyTo(last_T);

        // decompose T
        double dx = T.at<double>(0, 2);
        double dy = T.at<double>(1, 2);
        double da = atan2(T.at<double>(1, 0), T.at<double>(0, 0));
        //
        prev_to_cur_transform.push_back(TransformParam(dx, dy, da));


    // ============================================================ //
    // STEP 07 : ADDING TRANSFORMATION (dx, dy, da) TO FIND CAMERA TRAJECTORY  >>  SMOOTH THE TRAJECTORY BEFORE APPLYING TO FRAMES
    // ============================================================ //
        // Accumulated frame to frame transform
        x += dx;
        y += dy;
        a += da;
        trajectory.push_back(Trajectory(x,y,a));

        //
        z = Trajectory(x, y, a);

        if (k == 1) {
            // intial guesses
            X = Trajectory(0, 0, 0);             //Initial estimate,  set 0
            P = Trajectory(1, 1, 1);             //set error variance,set 1
        }
        else
        {
            // time update（prediction）
            X_ = X;                              //X_(k) = X(k-1);
            P_ = P + Q;                          //P_(k) = P(k-1)+Q;
            
            // measurement update（correction）
            K = P_ / (P_ + R);                   //gain; K(k) = P_(k)/( P_(k)+R );
            X = X_ + K * (z - X_);               //z-X_ is residual,X(k) = X_(k)+K(k)*(z(k)-X_(k)); 
            P = (Trajectory(1, 1, 1) - K) * P_;  //P(k) = (1-K(k))*P_(k);
        }
        smoothed_trajectory.push_back(X);


    // ============================================================ //
    // STEP 08 : APPLY THE SMOOTHED CAMERA TRAJECTORY TO STABILIZE THE DISPLAYED FRAMES
    // ============================================================ //
        // target - current
        double diff_x = X.x - x;//
        double diff_y = X.y - y;
        double diff_a = X.a - a;

        dx = dx + diff_x;
        dy = dy + diff_y;
        da = da + diff_a;

        new_prev_to_cur_transform.push_back(TransformParam(dx, dy, da));

        //
        T.at<double>(0, 0) = cos(da);
        T.at<double>(0, 1) = -sin(da);
        T.at<double>(1, 0) = sin(da);
        T.at<double>(1, 1) = cos(da);

        T.at<double>(0, 2) = dx;
        T.at<double>(1, 2) = dy;

        Mat curr2;

        warpAffine(prev, curr2, T, curr.size());

        curr2 = curr2(Range(vert_border, curr2.rows - vert_border), Range(HORIZONTAL_BORDER_CROP, curr2.cols - HORIZONTAL_BORDER_CROP));
        // Resize cur2 back to cur size, for better side by side comparison
        resize(curr2, curr2, curr.size());

        // Now draw the original and stablised side by side for coolness
        Mat canvas = Mat::zeros(curr.rows, curr.cols * 2, curr.type());

        prev.copyTo(canvas(Range::all(), Range(0, curr2.cols)));
        curr2.copyTo(canvas(Range::all(), Range(curr2.cols, curr2.cols * 2)));

        // If too big to fit on the screen, then scale it down by 2, hopefully it'll fit :)
        //if (canvas.cols > 1920) {
            //resize(canvas, canvas, Size(canvas.cols / 2, canvas.rows / 2));
        //}

        // Update the window with new data
        imshow(window_name, canvas);

        // Save the displayed window into video file
        out.write(canvas);
        
        // Stop stop-watch timer
        auto end_1 = chrono::steady_clock::now();
        auto timing = end_1 - start_1;
        cout << chrono::duration <double, milli>(timing).count() << endl;
        
        //
        prev = curr.clone();
        curr_gray.copyTo(prev_gray);

        //
        k++;
                
    }
    out.release();
    destroyAllWindows();

    return EXIT_SUCCESS;
}
catch (const rs2::error& e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}