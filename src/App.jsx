import './App.css';
import { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Holistic, POSE_CONNECTIONS, FACEMESH_TESSELATION, HAND_CONNECTIONS } from '@mediapipe/holistic';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import { Camera } from '@mediapipe/camera_utils';

const actions = [
  'call', 'class', 'doctor', 'email',
  'family', 'form', 'learn', 'like', 
  'meet', 'No hands in frame', 'please', 'room',
  'teacher', 'thank you', 'today', 'water', 'you'
];

const SEQUENCE_LENGTH = 20;
const THRESHOLD = 0.75; 

const LANDMARK_IDX = [
  0,9,11,13,14,17,117,118,119,199,346,347,348,
  468,469,470,471,472,473,474,475,476,477,478,479,
  480,481,482,483,484,485,486,487,488,489,490,491,
  492,493,494,495,496,497,498,499,500,501,502,503,
  504,505,506,507,508,509,510,511,512,513,514,515,
  516,517,518,519,520,521,522,523,524,525,526,527,
  528,529,530,531,532,533,534,535,536,537,538,539,
  540,541,542
];

const NUM_FEATURES = 264;

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [sequence, setSequence] = useState([]);
  const [currentPrediction, setCurrentPrediction] = useState(null);
  const [probabilities, setProbabilities] = useState([]);

  useEffect(() => {
    const loadModel = async () => {
      try {
        const loadedModel = await tf.loadGraphModel('/submission/model.json');
        setModel(loadedModel);
        console.log('Model loaded successfully!');
      } catch (error) {
        console.error('Error loading the model:', error);
      }
    };
    loadModel();
  }, []);

  useEffect(() => {
    const startHolistic = async () => {
      if (!videoRef.current || !canvasRef.current) return;

      const holistic = new Holistic({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
      });

      holistic.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        enableSegmentation: false,
        refineFaceLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      holistic.onResults(handleResults);

      const video = videoRef.current;
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
      video.onloadedmetadata = () => {
        video.play();
        const camera = new Camera(video, {
          onFrame: async () => {
            await holistic.send({ image: video });
          },
          width: 640,
          height: 480,
        });
        camera.start();
      };
    };

    startHolistic();
  }, []);

  const handleResults = (results) => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.faceLandmarks) {
      drawConnectors(ctx, results.faceLandmarks, FACEMESH_TESSELATION, {color: '#C0C0C0', lineWidth: 1});
    }
    if (results.poseLandmarks) {
      drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, {color: '#FF0000', lineWidth: 2});
      drawLandmarks(ctx, results.poseLandmarks, {color: '#0000FF', lineWidth: 1});
    }
    if (results.leftHandLandmarks) {
      drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, {color: '#CC0000', lineWidth: 2});
      drawLandmarks(ctx, results.leftHandLandmarks, {color: '#00FF00', lineWidth: 1});
    }
    if (results.rightHandLandmarks) {
      drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, {color: '#00CC00', lineWidth: 2});
      drawLandmarks(ctx, results.rightHandLandmarks, {color: '#FF0000', lineWidth: 1});
    }

    const keypoints = extractKeypoints(results);
    setSequence(prev => {
      const newSeq = [...prev, keypoints];
      if (newSeq.length > SEQUENCE_LENGTH) {
        newSeq.shift();
      }
      return newSeq;
    });
  };

  useEffect(() => {
    if (sequence.length === SEQUENCE_LENGTH && model) {
      console.log("Running prediction...");
      const input = tf.tensor([sequence], [1, SEQUENCE_LENGTH, NUM_FEATURES]);
      console.log("Input shape:", input.shape);

      const prediction = model.predict(input);
      const data = prediction.dataSync();
      tf.dispose(prediction);

      console.log("Predicted probabilities:", data);
      setProbabilities(Array.from(data));

      let maxProb = -Infinity;
      let maxIndex = -1;
      data.forEach((p, i) => {
        if (p > maxProb) {
          maxProb = p;
          maxIndex = i;
        }
      });

      if (maxProb > THRESHOLD) {
        setCurrentPrediction(actions[maxIndex]);
      } else {
        setCurrentPrediction("Uncertain");
      }

      tf.dispose(input);
    }
  }, [sequence, model]);

  const extractKeypoints = (results) => {
    const face = results.faceLandmarks
      ? results.faceLandmarks.map(lm => [lm.x, lm.y, lm.z])
      : Array(468).fill([0,0,0]);

    const pose = results.poseLandmarks
      ? results.poseLandmarks.map(lm => [lm.x, lm.y, lm.z])
      : Array(33).fill([0,0,0]);

    const leftHand = results.leftHandLandmarks
      ? results.leftHandLandmarks.map(lm => [lm.x, lm.y, lm.z])
      : Array(21).fill([0,0,0]);

    const rightHand = results.rightHandLandmarks
      ? results.rightHandLandmarks.map(lm => [lm.x, lm.y, lm.z])
      : Array(21).fill([0,0,0]);

    const allLandmarks = [
      ...face,
      ...pose,
      ...leftHand,
      ...rightHand
    ]; 

    const selected = LANDMARK_IDX.map(i => allLandmarks[i]);
    const flattened = selected.flat();

    return flattened;
  };

  return (
    <div className="App">
      <header className="header-banner">
        <h1 className="header-title">Sign Language Gesture Recognition</h1>
      </header>

      <div className="content-container">
        <div className="instructions-card">
          <p>Welcome to our Sign Language Gesture Recognition tool! This application uses a pre-trained machine learning model combined with Mediapipe's Holistic solution to identify certain sign language gestures from your webcam.</p>
          <p><strong>How to Use:</strong></p>
          <ol>
            <li>Allow access to your webcam when prompted.</li>
            <li>Position yourself so your hands are clearly visible in the camera frame.</li>
            <li>Perform one of the known gestures (listed below) and wait a moment for the prediction to appear.</li>
          </ol>
        </div>

        <div className="video-section">
          <div className="video-container">
            <video
              ref={videoRef}
              style={{ width: '640px', height: '480px' }}
              autoPlay
              muted
            ></video>
            <canvas
              ref={canvasRef}
              className="overlay-canvas"
            ></canvas>
          </div>

          {currentPrediction && (
            <div className="prediction-box">
              <p className="prediction-text">Current Prediction: <span className="highlight">{currentPrediction}</span></p>
            </div>
          )}

          {probabilities.length > 0 && (
            <div className="card probabilities-card">
              <h2>Model Probabilities:</h2>
              <p>The following shows how confident the model is about each gesture. Higher means more confident.</p>
              <ul className="probability-list">
                {probabilities.map((prob, i) => (
                  <li key={i}>{actions[i]}: {prob.toFixed(4)}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
      
      <footer className="footer">
        <p>© {new Date().getFullYear()} Sign Gesture Recognition. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;
