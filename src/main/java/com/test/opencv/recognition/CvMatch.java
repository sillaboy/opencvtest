package com.test.opencv.recognition;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.AKAZE;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class CvMatch {

    public static final int MAX_FEATURE_POINT = 10;

    public static Point templateMatch(Mat template, Mat scene) {
        Mat grayTemplate = new Mat(template.rows(), template.cols(), CvType.CV_8UC1);
        Mat grayScene = new Mat(scene.rows(), scene.cols(), CvType.CV_8UC1);
        Imgproc.cvtColor(template, grayTemplate, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(scene, grayScene, Imgproc.COLOR_BGR2GRAY);
        int width = grayScene.cols() - grayTemplate.cols() + 1;
        int height = grayScene.rows() - grayTemplate.rows() + 1;
        int method = Imgproc.TM_CCOEFF_NORMED;
        Mat result = new Mat(width, height, CvType.CV_32FC1);
        Imgproc.matchTemplate(grayScene, grayTemplate, result, Imgproc.TM_CCOEFF_NORMED);
        Core.normalize(result, result, 0, 1, Core.NORM_MINMAX, -1, new Mat());
        Core.MinMaxLocResult mmr = Core.minMaxLoc(result);
        double x, y;
        if (method == Imgproc.TM_SQDIFF_NORMED || method == Imgproc.TM_SQDIFF) {
            x = mmr.minLoc.x;
            y = mmr.minLoc.y;
        } else {
            x = mmr.maxLoc.x;
            y = mmr.maxLoc.y;
        }
        // return middle point
        double midx, midy;
        midx = x + grayTemplate.cols() /2;
        midy = y + grayTemplate.rows() /2;
        return new Point(midx, midy);
    }

    public static Point akaMatch(Mat template, Mat scene) {
        Point point = null;
        AKAZE detector = AKAZE.create();
        MatOfKeyPoint tempLatePoint = new MatOfKeyPoint(), scenePoint = new MatOfKeyPoint();
        Mat descriptors1 = new Mat(), descriptors2 = new Mat();
        detector.detectAndCompute(template, new Mat(), tempLatePoint, descriptors1);
        detector.detectAndCompute(scene, new Mat(), scenePoint, descriptors2);

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
        if (listOfGoodMatches.size() > MAX_FEATURE_POINT) {
            List<KeyPoint> templateKeyPointList = tempLatePoint.toList();
            List<KeyPoint> originalKeyPointList = scenePoint.toList();
            LinkedList<Point> objectPoints = new LinkedList();
            LinkedList<Point> scenePoints = new LinkedList();
            listOfGoodMatches.forEach(goodMatch -> {
                objectPoints.addLast(templateKeyPointList.get(goodMatch.queryIdx).pt);
                scenePoints.addLast(originalKeyPointList.get(goodMatch.trainIdx).pt);
            });

            MatOfPoint2f objMatOfPoint2f = new MatOfPoint2f();
            objMatOfPoint2f.fromList(objectPoints);
            MatOfPoint2f scnMatOfPoint2f = new MatOfPoint2f();
            scnMatOfPoint2f.fromList(scenePoints);
            //使用 findHomography 寻找匹配上的关键点的变换
            Mat homography = Calib3d.findHomography(objMatOfPoint2f, scnMatOfPoint2f,
                    Calib3d.RANSAC, 3);

            /**
             * 透视变换(Perspective Transformation)是将图片投影到一个新的视平面(Viewing Plane)，也称作投影映射(Projective Mapping)。
             */
            Mat templateCorners = new Mat(4, 1, CvType.CV_32FC2);
            Mat templateTransformResult = new Mat(4, 1, CvType.CV_32FC2);
            templateCorners.put(0, 0, new double[]{0, 0});
            templateCorners.put(1, 0, new double[]{template.cols(), 0});
            templateCorners.put(2, 0, new double[]{template.cols(), template.rows()});
            templateCorners.put(3, 0, new double[]{0, template.rows()});
            //使用 perspectiveTransform 将模板图进行透视变以矫正图象得到标准图片
            Core.perspectiveTransform(templateCorners, templateTransformResult, homography);

            //矩形四个顶点
            double[] pointA = templateTransformResult.get(0, 0);
            double[] pointB = templateTransformResult.get(1, 0);
            double[] pointC = templateTransformResult.get(2, 0);
            double[] pointD = templateTransformResult.get(3, 0);

            //指定取得数组子集的范围
            int rowStart = (int) pointA[1];
            int rowEnd = (int) pointC[1];
            int colStart = (int) pointD[0];
            int colEnd = (int) pointB[0];
            double pointX = colStart + template.cols() / 2;
            double pointY = rowStart + template.rows() / 2;
            point = new Point(pointX, pointY);
        } else {
            // not exist
        }
        return point;
    }
}
