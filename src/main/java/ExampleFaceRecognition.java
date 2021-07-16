import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

public class ExampleFaceRecognition {
    public static void main(String[] args) {
        OpenCV.loadShared();
        Mat mat = loadImage("MyFOTO.jpeg");
        MatOfRect facesDetected = new MatOfRect();

        CascadeClassifier cascadeClassifier = new CascadeClassifier();
        int minFaceSize = Math.round(mat.rows() * 0.1f);
        cascadeClassifier.load("./src/main/resources/haarcascade_frontalcatface_extended.xml");
        cascadeClassifier.detectMultiScale(mat,
                facesDetected,
                1.1,
                3,
                Objdetect.CASCADE_SCALE_IMAGE,
                new Size(minFaceSize, minFaceSize),
                new Size()
        );

        Rect[] facesArray = facesDetected.toArray();
        for(Rect face : facesArray) {
            Imgproc.rectangle(mat, face.tl(), face.br(), new Scalar(0, 0, 255), 3);
        }
        saveImage(mat, "targetImage.jpeg");

    }

    public static Mat loadImage(String imagePath) {
        Imgcodecs imageCodecs = new Imgcodecs();
        return imageCodecs.imread(imagePath);
    }

    public static void saveImage(Mat imageMatrix, String targetPath) {
        Imgcodecs imgcodecs = new Imgcodecs();
        imgcodecs.imwrite(targetPath, imageMatrix);
    }
}
