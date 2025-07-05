import React, { useRef, useState, useEffect } from "react";
import Webcam from "react-webcam";

function App() {
  const webcamRef = useRef(null);
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState("");
  const [isDetecting, setIsDetecting] = useState(false);
  const [error, setError] = useState("");

  // Function to convert base64 to Blob
  function dataURLtoBlob(dataurl) {
    const arr = dataurl.split(",");
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new Blob([u8arr], { type: mime });
  }

  // Function to send frame to backend
  const sendFrameToBackend = async (imageSrc) => {
    try {
      const blob = dataURLtoBlob(imageSrc);
      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setPrediction(data.predicted_sign);
      setError("");
    } catch (error) {
      console.error("Error sending frame:", error);
      setError("Error: " + error.message);
    }
  };

  // Real-time detection loop
  useEffect(() => {
    let intervalId;

    if (isDetecting && webcamRef.current) {
      intervalId = setInterval(() => {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
          sendFrameToBackend(imageSrc);
        }
      }, 500); // Send frame every 1 second
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [isDetecting]);

  const startDetection = () => {
    setIsDetecting(true);
    setPrediction("");
    setError("");
  };

  const stopDetection = () => {
    setIsDetecting(false);
    setPrediction("");
    setError("");
  };

  return (
    <div style={{ 
      maxWidth: '800px', 
      margin: '0 auto', 
      padding: '20px',
      fontFamily: 'Arial, sans-serif'
    }}>
      <h1 style={{ textAlign: 'center', color: '#333' }}>
        🤟 Real-Time Sign Language Detection
      </h1>
      
      <div style={{ textAlign: 'center', marginBottom: '20px' }}>
        <button 
          onClick={startDetection}
          disabled={isDetecting}
          style={{
            backgroundColor: '#4CAF50',
            color: 'white',
            padding: '10px 20px',
            border: 'none',
            borderRadius: '5px',
            marginRight: '10px',
            cursor: isDetecting ? 'not-allowed' : 'pointer'
          }}
        >
          ▶️ Start Detection
        </button>
        
        <button 
          onClick={stopDetection}
          disabled={!isDetecting}
          style={{
            backgroundColor: '#f44336',
            color: 'white',
            padding: '10px 20px',
            border: 'none',
            borderRadius: '5px',
            cursor: !isDetecting ? 'not-allowed' : 'pointer'
          }}
        >
          ⏹️ Stop Detection
        </button>
      </div>

      <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-start' }}>
        {/* Camera Feed */}
        <div style={{ flex: 1 }}>
          <h3>📹 Camera Feed</h3>
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            width="100%"
            height="auto"
            style={{ 
              border: '3px solid #ddd',
              borderRadius: '10px',
              maxWidth: '400px'
            }}
          />
        </div>

        {/* Prediction Display */}
        <div style={{ flex: 1, padding: '20px' }}>
          <h3>🔍 Live Detection</h3>
          
          {isDetecting ? (
            <div style={{ 
              padding: '20px', 
              backgroundColor: '#f0f8ff', 
              borderRadius: '10px',
              border: '2px solid #4CAF50'
            }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '10px' }}>
                Detected Sign: <span style={{ color: '#4CAF50' }}>{prediction || '...'}</span>
              </div>
              <div style={{ fontSize: '16px', color: '#666' }}>
                Show your hand sign to the camera
              </div>
            </div>
          ) : (
            <div style={{ 
              padding: '20px', 
              backgroundColor: '#f5f5f5', 
              borderRadius: '10px',
              border: '2px solid #ddd'
            }}>
              <div style={{ fontSize: '18px', color: '#666' }}>
                Click "Start Detection" to begin real-time sign recognition
              </div>
            </div>
          )}

          {error && (
            <div style={{ 
              padding: '10px', 
              backgroundColor: '#ffebee', 
              color: '#c62828',
              borderRadius: '5px',
              marginTop: '10px'
            }}>
              {error}
            </div>
          )}
        </div>
      </div>

      {/* Instructions */}
      <div style={{ 
        marginTop: '30px', 
        padding: '20px', 
        backgroundColor: '#e8f5e8', 
        borderRadius: '10px' 
      }}>
        <h3>📋 Instructions:</h3>
        <ul style={{ textAlign: 'left' }}>
          <li>Click "Start Detection" to begin real-time sign recognition</li>
          <li>Position your hand clearly in front of the camera</li>
          <li>The system will automatically detect ASL alphabet signs (A-Z)</li>
          <li>Results update every second</li>
          <li>Click "Stop Detection" to pause</li>
        </ul>
      </div>
    </div>
  );
}

export default App;