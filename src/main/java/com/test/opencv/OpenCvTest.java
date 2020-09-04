package com.test.opencv;

import com.test.opencv.recognition.CvMatch;
import org.opencv.core.*;
import org.opencv.features2d.AKAZE;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class OpenCvTest {
    public static final String opencvDll = "opencv_java440.dll";
    public static final String opencvdylib = "libopencv_java440.dylib";

    public static void main(String[] args) {
        setEnv();
        //Mat mat = Mat.eye(3, 3, CvType.CV_8UC1);
        //Mat mat1 = Imgcodecs.imread("1.jpg`");
        //System.out.println("mat = " + mat1.dump());
        //knn();
        //knn();
        //tempMatch2();
        knn();
    }

    public static void akaMatch() {
        Mat src = Imgcodecs.imread("box_in_scene.png");//to match
        Mat template = Imgcodecs.imread("box.png");//template
        Point point = CvMatch.akaMatch(template, src);
        if (point != null) {
            Imgproc.rectangle(src, point, new Point(point.x + 5, point.y + 5),
                    new Scalar(0, 0, 255), 2, Imgproc.LINE_AA);
            HighGui.imshow("模板匹配", src);
            HighGui.waitKey();
        }
    }

    public static void tempMatch2() {
        Mat src = Imgcodecs.imread("box_in_scene.png");//to match
        Mat template = Imgcodecs.imread("box.png");//template
        Point point = CvMatch.templateMatch(template, src);
        Imgproc.rectangle(src, point, new Point(point.x + 5, point.y + 5),
                new Scalar(0, 0, 255), 2, Imgproc.LINE_AA);
        HighGui.imshow("模板匹配", src);
        HighGui.waitKey();
    }

    public static void knn() {
        Mat template = Imgcodecs.imread("aat.png");
        Mat img2 = Imgcodecs.imread("swipe-start-screenshot.png");
        if (template.empty() || img2.empty()) {
            System.err.println("Cannot read images!");
            System.exit(0);
        }

        //-- Step 1: Detect the keypoints using AKAZE Detector, compute the descriptors
        AKAZE detector = AKAZE.create();
        MatOfKeyPoint keypoints1 = new MatOfKeyPoint(), keypoints2 = new MatOfKeyPoint();
        Mat descriptors1 = new Mat(), descriptors2 = new Mat();
        detector.detectAndCompute(template, new Mat(), keypoints1, descriptors1);
        detector.detectAndCompute(img2, new Mat(), keypoints2, descriptors2);

        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
        List<MatOfDMatch> knnMatches = new ArrayList<>();
        matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

        //-- Filter matches using the Lowe's ratio test
        float ratioThresh = 0.8f;
        List<DMatch> listOfGoodMatches = new ArrayList<>();
        for (int i = 0; i < knnMatches.size(); i++) {
            if (knnMatches.get(i).rows() > 1) {
                DMatch[] matches = knnMatches.get(i).toArray();
                if (matches[0].distance < ratioThresh * matches[1].distance) {
                    listOfGoodMatches.add(matches[0]);
                }
            }
        }
        MatOfDMatch goodMatches = new MatOfDMatch();
        goodMatches.fromList(listOfGoodMatches);


        //-- Draw matches
        Mat imgMatches = new Mat();
        Features2d.drawMatches(template, keypoints1, img2, keypoints2, goodMatches, imgMatches, Scalar.all(-1),
                Scalar.all(-1), new MatOfByte(), Features2d.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS);

        Mat dst = Mat.zeros(800, 600, CvType.CV_8UC3);
        Imgproc.resize(imgMatches, dst, dst.size());

        //-- Show detected matches
        HighGui.imshow("Good Matches", dst);
        HighGui.waitKey(0);
    }

    public static void aKase() {
        try {
            Mat src1 = Imgcodecs.imread("s.png");
            Mat src2 = Imgcodecs.imread("test1.png");
            if (src1.empty() || src2.empty()) {
                throw new Exception("no file");
            }
            MatOfKeyPoint keyPoint1 = new MatOfKeyPoint();
            MatOfKeyPoint keyPoint2 = new MatOfKeyPoint();
            /*
            FeatureDetector sifDetecotr = FeatureDetector.create(FeatureDetector.AKAZE);//用它的名字创建一个功能检测器。
            sifDetecotr.detect(src1, keyPoint1);//在图像（第一个变体）或图像集（第二种变体）中检测关键点。
            sifDetecotr.detect(src2, keyPoint2);
            DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.AKAZE);//根据名称创建一个描述符提取器。
            Mat descriptor1 = new Mat(src1.rows(), src1.cols(), src1.type());
            extractor.compute(src1, keyPoint1, descriptor1);//计算在一个图像（第一个变体）或图像集（第二个变种）中检测到的一组关键点的描述符。
            Mat descriptor2 = new Mat(src2.rows(), src2.cols(), src2.type());
            extractor.compute(src2, keyPoint2, descriptor2);
            MatOfDMatch matches = new MatOfDMatch();
            DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);//用默认参数（使用默认的构造函数）创建给定类型的描述符matcher。

            matcher.match(descriptor1, descriptor2, matches);//从查询集中找到每个描述符的最佳匹配。

            Mat dst = new Mat();
            Features2d.drawMatches(src1, keyPoint1, src2, keyPoint2, matches, dst);//从两个图像中提取出的关键点匹配。
            HighGui.imshow("aKase", dst);
            HighGui.waitKey();*/
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void tempMatch() {
        Mat src = Imgcodecs.imread("s.png");//to match
        Mat template = Imgcodecs.imread("shape.png");//template
        HighGui.imshow("原图", src);
        HighGui.waitKey();
        /**
         * TM_SQDIFF = 0, 平方差匹配法，最好的匹配为0，值越大匹配越差
         * TM_SQDIFF_NORMED = 1,归一化平方差匹配法
         * TM_CCORR = 2,相关匹配法，采用乘法操作，数值越大表明匹配越好
         * TM_CCORR_NORMED = 3,归一化相关匹配法
         * TM_CCOEFF = 4,相关系数匹配法，最好的匹配为1，-1表示最差的匹配
         * TM_CCOEFF_NORMED = 5;归一化相关系数匹配法
         */
        int method = Imgproc.TM_CCORR_NORMED;

        int width = src.cols() - template.cols() + 1;
        int height = src.rows() - template.rows() + 1;
        // 创建32位模板匹配结果Mat
        Mat result = new Mat(width, height, CvType.CV_32FC1);

        /*
         * 将模板与重叠的图像区域进行比较。
         * @param image运行搜索的图像。 它必须是8位或32位浮点。
         * @param templ搜索的模板。 它必须不大于源图像并且具有相同的数据类型。
         * @param result比较结果图。 它必须是单通道32位浮点。 如果image是（W * H）并且templ是（w * h），则结果是（（W-w + 1）*（H-h + 1））。
         * @param方法用于指定比较方法的参数，请参阅默认情况下未设置的#TemplateMatchModes。
         * 当前，仅支持#TM_SQDIFF和#TM_CCORR_NORMED方法。
         */
        Imgproc.matchTemplate(src, template, result, method);

        // 归一化 详见https://blog.csdn.net/ren365880/article/details/103923813
        Core.normalize(result, result, 0, 1, Core.NORM_MINMAX, -1, new Mat());

        // 获取模板匹配结果 minMaxLoc寻找矩阵(一维数组当作向量,用Mat定义) 中最小值和最大值的位置.
        Core.MinMaxLocResult mmr = Core.minMaxLoc(result);

        // 绘制匹配到的结果 不同的参数对结果的定义不同
        double x, y;
        if (method == Imgproc.TM_SQDIFF_NORMED || method == Imgproc.TM_SQDIFF) {
            x = mmr.minLoc.x;
            y = mmr.minLoc.y;
        } else {
            x = mmr.maxLoc.x;
            y = mmr.maxLoc.y;
        }

        /*
         * 函数rectangle绘制一个矩形轮廓或一个填充的矩形，其两个相对角为pt1和pt2。
         * @param img图片。
         * @param pt1矩形的顶点。
         * @param pt2与pt1相反的矩形的顶点。
         * @param color矩形的颜色或亮度（灰度图像）。
         * @param thickness组成矩形的线的粗细。 负值（如#FILLED）表示该函数必须绘制一个填充的矩形。
         * @param lineType线的类型。 请参阅https://blog.csdn.net/ren365880/article/details/103952856
         */
        Imgproc.rectangle(src, new Point(x, y), new Point(x + template.cols(), y + template.rows()),
                new Scalar(0, 0, 255), 2, Imgproc.LINE_AA);
        Mat dst = Mat.zeros(800, 600, CvType.CV_8UC3);
        Imgproc.resize(src, dst, dst.size());
        HighGui.imshow("模板匹配", dst);
        HighGui.waitKey();
    }

    public static void setEnv() {
        try {
            String path = System.getProperty("opencv.lib");
            System.out.println(path);
            String arch = System.getProperty("os.arch");
            if (arch.contains("64")) {
                arch = "x64";
            } else {
                arch = "x86";
            }
            String osName = System.getProperties().getProperty("os.name");
            if (osName.startsWith("Win")) {
                path = path.replace("/", File.separator);
                String opencvLib = path + File.separator + arch + File.separator + opencvDll;
                File file = new File(opencvLib);
                System.out.println(file.getAbsolutePath());
                if (file.exists()) {
                    System.load(file.getAbsolutePath());
                }
            } else if (osName.startsWith("Mac")) {
                String opencvLib = path + File.separator + opencvdylib;
                File file = new File(opencvLib);
                System.out.println(file.getAbsolutePath());
                if (file.exists()) {
                    System.load(file.getAbsolutePath());
                }
            }
        } catch (Exception ex) {
            System.out.println("Failed to set Java Library Path: " + ex.getMessage());
        }
    }
}
