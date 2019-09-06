package com.example.sensorcollector;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Environment;
import android.os.PowerManager;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.BaseAdapter;
import android.widget.Button;
import android.widget.ListView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Locale;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends Activity {

    private PowerManager.WakeLock wakeLock;

    private TextView xData;
    private TextView yData;
    private TextView zData;
    private TextView counterView;
    private Button startButton;
    private Button analysingButton;
    private ListView listView;
    private Spinner classes;
    private ArrayAdapter<CharSequence> classLabels;
    private ArrayList<String> listViewContent;
    private ArrayAdapter adapter;

    private boolean recording = false;
    protected static final String APP_DIRECTORY = "/SensorCollector/";
    private static final int NUMBER_OF_CLASSES = 4;

    private ArrayList data;

    private boolean analysing = false;

    private Recording rec;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //request permission
        if (this.checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            this.requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 0);
        }

        //initialize text views
        xData = (TextView) findViewById(R.id.x);
        yData = (TextView) findViewById(R.id.y);
        zData = (TextView) findViewById(R.id.z);
        counterView = (TextView) findViewById(R.id.counter);
        startButton = (Button) findViewById(R.id.button);
        analysingButton = (Button) findViewById(R.id.analysing);
        listView = (ListView) findViewById(R.id.listview);

        //initialise list view
        listViewContent = new ArrayList<String>();
        adapter = new ArrayAdapter(this, android.R.layout.simple_list_item_1, listViewContent);
        listView.setAdapter(adapter);

        //initialize data array
        data = new ArrayList<>();

        //initialize dropdown with classes
        classes = (Spinner) findViewById(R.id.classes);
        classLabels = ArrayAdapter.createFromResource(this,
                R.array.classes, android.R.layout.simple_spinner_item);
        classLabels.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        classes.setAdapter(classLabels);

        //start wake lock
        PowerManager powerManager = (PowerManager) getSystemService(POWER_SERVICE);
        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK,
                "SensorCollector:RecordingSensorData");

        // start thread to record and analyse data
        rec = new Recording(this, this, counterView, listViewContent, adapter, classLabels, xData, yData, zData);
        rec.start();
    }

    public void startRecording(View view) {
        if(!recording) {
            wakeLock.acquire();
            rec.startRecording((int) classes.getSelectedItemId());
            startButton.setText("Stop");
        } else {
            startButton.setText("Start");
            rec.stopRecording();
            if(!analysing) {
                wakeLock.release();
                //reset counter view value
                counterView.setText("0");
            }
        }
        recording = !recording;
    }

    public void startAnalysing(View view) {
        if(!analysing) {
            wakeLock.acquire();
            rec.startAnalysing();
            analysingButton.setText("Stop Analysing");

        } else {
            analysingButton.setText("Start Analysing");
            rec.stopAnalysing();
            if(!recording) {
                wakeLock.release();
                //reset counter view value
                counterView.setText("0");
            }
        }
        analysing = !analysing;
    }

    protected void onDestroy() {
        super.onDestroy();

        // stop recording thread
        rec.stopThread();
        try {
            rec.join();
        } catch(InterruptedException e) {}
    }
}
