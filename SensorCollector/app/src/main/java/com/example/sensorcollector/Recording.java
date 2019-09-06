package com.example.sensorcollector;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.widget.ArrayAdapter;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Locale;

/**
 * Created by Fabian on 28.03.2018.
 */

public class Recording extends Thread implements SensorEventListener {

    private SensorManager mSensorManager;

    TextView counterView;

    private float x = 0;
    private float y = 0;
    private float z = 0;

    private final TextView xData;
    private final TextView yData;
    private final TextView zData;

    private static final int FEATURE_NUMBER = 4;
    private static final int DATASET_LENGTH = 256;
    private static final int TIMEOUT = 25;

    private ArrayList<float[][]> data;
    private float[][] dataSet;

    private Context context;
    private Activity activity;
    private int classNr;

    private boolean running = true;

    private boolean analysing = false;
    private boolean recording = false;

    private TensorFlowInferenceInterface inferenceInterface;
    private static final int NUMBER_OF_DATASETS_TO_BUNDEL = 4;
    private int[] analysedClasses;
    private int analysingCounter = 0;
    private float averageAccuracy = 0;

    private ArrayList<String> listViewContent;
    private ArrayAdapter adapter;
    private ArrayAdapter<CharSequence> classLabels;

    public Recording(Context context, Activity activity, TextView counterView, ArrayList<String> listViewContent, ArrayAdapter adapter, ArrayAdapter<CharSequence> classLabels, TextView xData, TextView yData, TextView zData) {
        data = new ArrayList<>();
        dataSet = new float[DATASET_LENGTH][FEATURE_NUMBER];
        this.counterView = counterView;
        this.context = context;
        this.activity = activity;
        this.listViewContent = listViewContent;
        this.adapter = adapter;
        this.classLabels = classLabels;
        this.xData = xData;
        this.yData = yData;
        this.zData = zData;

        // initialize sensors
        mSensorManager = (SensorManager) this.activity.getSystemService(Context.SENSOR_SERVICE);
        Sensor gyroSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        mSensorManager.registerListener(this, gyroSensor, SensorManager.SENSOR_DELAY_NORMAL);
        Sensor accSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        mSensorManager.registerListener(this, accSensor, SensorManager.SENSOR_DELAY_NORMAL);

        // initialize components for inference
        inferenceInterface = new TensorFlowInferenceInterface(this.activity.getAssets(), "frozen_graph.pb");
        analysedClasses = new int[NUMBER_OF_DATASETS_TO_BUNDEL];
    }

    public void run() {
        int overallCounter = 0;
        int counter = 0;
        long time = System.currentTimeMillis();
        while (running) {
            dataSet[counter][0] = (System.currentTimeMillis() - time) / 1000.0f;
            dataSet[counter][1] = x;
            dataSet[counter][2] = y;
            dataSet[counter][3] = z;

            counter++;

            if (counter == DATASET_LENGTH) {
                if (analysing) {
                    classifyDataSet(dataSet);
                }

                if (recording) {
                    data.add(dataSet);
                }

                if (recording || analysing) {
                    overallCounter++;

                    //print the number of data sets recorded on the UI
                    final int tmp = overallCounter;
                    new Handler(Looper.getMainLooper()).post(new Runnable() {
                        @Override
                        public void run() {
                            counterView.setText(String.valueOf(tmp));
                        }
                    });
                }

                // reset values
                time = System.currentTimeMillis();
                counter = 0;
                dataSet = new float[DATASET_LENGTH][FEATURE_NUMBER];
            }
            try {
                this.sleep(TIMEOUT);
            } catch (InterruptedException e) {
            }
        }
    }

    protected void stopThread() {
        running = false;
    }

    protected void startRecording(int classNr) {
        this.classNr = classNr;
        recording = true;
    }

    protected void stopRecording() {
        recording = false;
        saveData();
    }

    protected void startAnalysing() {
        analysing = true;
    }

    protected void stopAnalysing() {
        analysing = false;
    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {
        x = sensorEvent.values[0];
        y = sensorEvent.values[1];
        z = sensorEvent.values[2];

        new Handler(Looper.getMainLooper()).post(new Runnable() {
            @Override
            public void run() {
                xData.setText("X: " + x);
                yData.setText("Y: " + y);
                zData.setText("Z: " + z);
            }
        });
    }

    private void saveData() {
        File root = Environment.getExternalStorageDirectory();
        PrintWriter os = null;

        probablyCreateFolder(root.getPath());

        String fileName = classNr + "_class_time_gravity_" + getFileNumber(classNr) + ".txt";
        final String filePath = root.getPath() + MainActivity.APP_DIRECTORY + fileName;

        System.out.println(filePath);

        try {
            os = new PrintWriter(filePath);
            JSONArray json = new JSONArray(data);
            os.println(json.toString());
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            os.close();
        }
        //delete data from array
        data.clear();

        new Handler(Looper.getMainLooper()).post(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(context, "Saved data to: " + filePath, Toast.LENGTH_LONG).show();
            }
        });
    }

    private String getFileNumber(int itemClass) {
        SharedPreferences sharedPref = activity.getPreferences(Context.MODE_PRIVATE);
        int number = sharedPref.getInt(String.valueOf(itemClass), -1);
        if (number == -1) {
            number = 100;
            sharedPref.edit().putInt(String.valueOf(itemClass), number).apply();
        } else {
            number++;
            sharedPref.edit().putInt(String.valueOf(itemClass), number).apply();
        }
        return String.format(Locale.GERMANY, "%06d", number);
    }

    private void probablyCreateFolder(String sdCartPath) {
        File dir = new File(sdCartPath + MainActivity.APP_DIRECTORY);
        if (!dir.exists()) {
            dir.mkdir();
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {
    }

    private void classifyDataSet(float[][] data) {
        //x value
        inferenceInterface.feed("x", flatten(data), DATASET_LENGTH, FEATURE_NUMBER);
        //keep prob
        inferenceInterface.feed("keep_prob", new float[]{1.0f}, 1);
        String[] out = new String[]{"softmax"};
        //classify data set
        inferenceInterface.run(out);
        //get output
        float[] outputs = new float[FEATURE_NUMBER];
        inferenceInterface.fetch("softmax", outputs);

        int predictedClass = argMax(outputs);

        averageAccuracy += outputs[predictedClass];

        analysingCounter++;

        if (analysingCounter == NUMBER_OF_DATASETS_TO_BUNDEL) {
            final int classToDisplay = mostPredictedClass(analysedClasses);
            final float tmp = averageAccuracy;
            new Handler(Looper.getMainLooper()).post(new Runnable() {
                @Override
                public void run() {
                    //display result in user interface
                    listViewContent.add(String.valueOf(classLabels.getItem(classToDisplay)) + " | Accuracy: " + (tmp / NUMBER_OF_DATASETS_TO_BUNDEL));
                    adapter.notifyDataSetChanged();
                }
            });

            //reset detected classes
            analysingCounter = 0;
            averageAccuracy = 0;
            analysedClasses = new int[NUMBER_OF_DATASETS_TO_BUNDEL];
        } else {
            analysedClasses[analysingCounter] = predictedClass;
        }
    }

    private int mostPredictedClass(int[] data) {
        float[] tmp = new float[NUMBER_OF_DATASETS_TO_BUNDEL];
        for (int i = 0; i < NUMBER_OF_DATASETS_TO_BUNDEL; i++) {
            tmp[data[i]]++;
        }
        return argMax(tmp);
    }

    private float[] flatten(float[][] data) {
        float[] result = new float[DATASET_LENGTH * FEATURE_NUMBER];
        int index = 0;
        for (int i = 0; i < DATASET_LENGTH; i++) {
            for (int j = 0; j < FEATURE_NUMBER; j++) {
                result[index] = data[i][j];
                index++;
            }
        }
        return result;
    }

    private int argMax(float[] values) {
        int result = 0;
        float oldVal = 0;
        for (int i = 0; i < values.length; i++) {
            if (oldVal < values[i]) {
                result = i;
                oldVal = values[i];
            }
        }
        return result;
    }
}
