import { useState } from "react";
import { detectObjects, askAssistant, getNavigation } from "./api";
import "./App.css";

export default function App() {
  const [detections, setDetections] = useState([]);
  const [response, setResponse] = useState("");
  const [nav, setNav] = useState(null);
  const [loading, setLoading] = useState({
    detect: false,
    assistant: false,
    navigation: false
  });
  const [error, setError] = useState("");

  const handleDetect = async () => {
    setLoading(prev => ({ ...prev, detect: true }));
    setError("");
    try {
      const res = await detectObjects();
      setDetections(res.data.detections);
    } catch (err) {
      setError("Failed to detect objects. Make sure the backend is running.");
      console.error(err);
    } finally {
      setLoading(prev => ({ ...prev, detect: false }));
    }
  };

  const handleAsk = async () => {
    setLoading(prev => ({ ...prev, assistant: true }));
    setError("");
    try {
      const res = await askAssistant("What's around me?");
      setResponse(res.data.response);
    } catch (err) {
      setError("Failed to get assistant response. Make sure the backend is running.");
      console.error(err);
    } finally {
      setLoading(prev => ({ ...prev, assistant: false }));
    }
  };

  const handleNavigate = async () => {
    setLoading(prev => ({ ...prev, navigation: true }));
    setError("");
    try {
      // Example coordinates (New York City coordinates)
      const start = [40.7128, -74.0060]; // NYC
      const end = [40.7589, -73.9851];   // Times Square
      const res = await getNavigation(start, end);
      setNav(res.data);
    } catch (err) {
      setError("Failed to get navigation data. Make sure the backend is running.");
      console.error(err);
    } finally {
      setLoading(prev => ({ ...prev, navigation: false }));
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🚀 Vision AI Assistant</h1>
        <p>Detect objects, ask questions, and navigate with AI</p>
      </header>

      <main className="main-content">
        <div className="controls">
          <button
            onClick={handleDetect}
            disabled={loading.detect}
            className="control-btn detect-btn"
          >
            {loading.detect ? "🔍 Detecting..." : "🎯 Detect Objects"}
          </button>

          <button
            onClick={handleAsk}
            disabled={loading.assistant}
            className="control-btn assistant-btn"
          >
            {loading.assistant ? "🤖 Thinking..." : "🤖 Ask Assistant"}
          </button>

          <button
            onClick={handleNavigate}
            disabled={loading.navigation}
            className="control-btn navigation-btn"
          >
            {loading.navigation ? "🧭 Calculating..." : "🧭 Navigation"}
          </button>
        </div>

        {error && (
          <div className="error-message">
            <p>❌ {error}</p>
          </div>
        )}

        <div className="results">
          {detections.length > 0 && (
            <div className="result-card">
              <h3>📊 Detected Objects</h3>
              <ul className="detection-list">
                {detections.map((detection, index) => (
                  <li key={index} className="detection-item">
                    <div className="detection-info">
                      <strong>{detection.label}</strong>
                      <span className="confidence">
                        {(detection.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="detection-details">
                      <span className="distance">📏 {detection.distance}</span>
                      <span className="direction">🧭 {detection.direction}</span>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {response && (
            <div className="result-card">
              <h3>🤖 AI Assistant Response</h3>
              <div className="assistant-response">
                <p>{response}</p>
              </div>
            </div>
          )}

          {nav && (
            <div className="result-card">
              <h3>🧭 Navigation Information</h3>
              <div className="navigation-info">
                <div className="nav-item">
                  <strong>Distance:</strong>
                  <span>{(nav.distance_m / 1000).toFixed(2)} km</span>
                </div>
                <div className="nav-item">
                  <strong>ETA:</strong>
                  <span>{nav.duration_min} minutes</span>
                </div>
                <div className="nav-item">
                  <strong>Steps:</strong>
                  <div className="steps">
                    {nav.steps.split('\n').map((step, index) => (
                      <p key={index}>{step}</p>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      <footer className="footer">
        <p>🔗 Backend: http://localhost:8000 | 🎨 Frontend: React + FastAPI</p>
      </footer>
    </div>
  );
}
