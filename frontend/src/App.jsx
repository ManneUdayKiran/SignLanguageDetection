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
      }, 1000); // Send frame every 1 second
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
        🤟 Sign Language Detection with Boundary Guide
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
        {/* Camera Feed with Boundary */}
        <div style={{ flex: 1 }}>
          <h3>📹 Camera Feed</h3>
          <div style={{ position: 'relative', display: 'inline-block' }}>
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              width="100%"
              height="100%"
              style={{ 
                border: '3px solid #ddd',
                borderRadius: '10px',
                maxWidth: '500px'
              }}
            />
            
            {/* Detection Boundary Overlay */}
            {isDetecting && (
              <div style={{
                position: 'absolute',
                top: '20%',
                left: '20%',
                width: '60%',
                height: '60%',
                border: '3px solid #4CAF50',
                borderRadius: '10px',
                backgroundColor: 'rgba(76, 175, 80, 0.1)',
                pointerEvents: 'none',
                zIndex: 10
              }}>
                {/* Corner indicators */}
                <div style={{
                  position: 'absolute',
                  top: '-5px',
                  left: '-5px',
                  width: '10px',
                  height: '10px',
                  backgroundColor: '#4CAF50',
                  borderRadius: '50%'
                }}></div>
                <div style={{
                  position: 'absolute',
                  top: '-5px',
                  right: '-5px',
                  width: '10px',
                  height: '10px',
                  backgroundColor: '#4CAF50',
                  borderRadius: '50%'
                }}></div>
                <div style={{
                  position: 'absolute',
                  bottom: '-5px',
                  left: '-5px',
                  width: '10px',
                  height: '10px',
                  backgroundColor: '#4CAF50',
                  borderRadius: '50%'
                }}></div>
                <div style={{
                  position: 'absolute',
                  bottom: '-5px',
                  right: '-5px',
                  width: '10px',
                  height: '10px',
                  backgroundColor: '#4CAF50',
                  borderRadius: '50%'
                }}></div>
                
                {/* Center text */}
                <div style={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  color: '#4CAF50',
                  fontWeight: 'bold',
                  fontSize: '14px',
                  textAlign: 'center',
                  textShadow: '1px 1px 2px rgba(0,0,0,0.8)'
                }}>
                  Position your hand<br/>inside this box
                </div>
              </div>
            )}
          </div>
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
                Keep your hand inside the green boundary for best results
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
                Click "Start Detection" to begin sign recognition with boundary guide
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
          <li><strong>Position your hand inside the green boundary box</strong></li>
          <li>Ensure your hand is clearly visible and well-lit</li>
          <li>Keep your hand steady for better recognition</li>
          <li>The boundary helps the system focus on the correct area</li>
          <li>Results update every second</li>
          <li>Click "Stop Detection" to pause</li>
        </ul>
      </div>

      {/* Tips for Better Recognition */}
      <div style={{ 
        marginTop: '20px', 
        padding: '20px', 
        backgroundColor: '#fff3e0', 
        borderRadius: '10px' 
      }}>
        <h3>💡 Tips for Better Recognition:</h3>
        <ul style={{ textAlign: 'left' }}>
          <li><strong>Good Lighting:</strong> Ensure your hand is well-lit</li>
          <li><strong>Clear Background:</strong> Avoid cluttered backgrounds</li>
          <li><strong>Hand Position:</strong> Keep your hand centered in the boundary</li>
          <li><strong>Steady Hand:</strong> Hold your sign steady for 1-2 seconds</li>
          <li><strong>Distance:</strong> Keep your hand about 20-30cm from the camera</li>
          <li><strong>Clean Hands:</strong> Ensure your hands are clean and visible</li>
        </ul>
      </div>
    </div>
  );
}

export default App;