import React, { useState, useRef } from 'react';
import { Upload, X } from 'lucide-react';
import './index.css';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add('active');
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('active');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('active');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  };

  const handleFileSelection = (selectedFile) => {
    if (!selectedFile.type.startsWith('image/')) {
      setError('Please select an image file (jpg, jpeg, png).');
      return;
    }
    
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResults(null);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // In development, the proxy or CORS will handle this.
      // Assuming FastAPI runs on 8000.
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error(err);
      setError('Failed to get prediction. Please ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResults(null);
    setError(null);
  };

  return (
    <div className="dashboard-container">
      <div className="header">
        <h1>Plant Disease Detection</h1>
        <p>AI-powered diagnosis for healthier crops.</p>
      </div>

      <div className="glass-panel main-content">
        <div className="upload-section">
          {!preview ? (
            <div 
              className="dropzone"
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload className="upload-icon" />
              <h3>Drag & Drop your leaf image here</h3>
              <p>or click to browse files</p>
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                accept="image/jpeg, image/png, image/jpg" 
                style={{ display: 'none' }} 
              />
            </div>
          ) : (
            <div className="image-preview">
              <img src={preview} alt="Selected leaf" />
              <button className="reset-button" onClick={reset}>
                <X size={20} />
              </button>
            </div>
          )}

          <button 
            className="upload-button" 
            onClick={handleUpload} 
            disabled={!file || loading}
          >
            {loading ? 'Analyzing...' : 'Analyze Image'}
          </button>
          
          {error && <div className="error-message">{error}</div>}
        </div>

        <div className="results-section">
          {loading && (
            <div className="loader">
              <div className="spinner"></div>
              <p>Analyzing plant tissue...</p>
            </div>
          )}

          {results && !loading && (
            <div className="result-card">
              <div className="plant-name">Plant: {results.plant}</div>
              <h2>{results.disease}</h2>
              <div className="confidence">{results.confidence.toFixed(2)}% Confidence</div>
              
              <div className="top-predictions">
                <h3>Top 5 Predictions</h3>
                {results.top5.map((item, index) => (
                  <div className="prediction-item" key={index}>
                    <div className="prediction-header">
                      <span>{item.class}</span>
                      <span>{item.confidence.toFixed(2)}%</span>
                    </div>
                    <div className="progress-bar-bg">
                      <div 
                        className="progress-bar-fill" 
                        style={{ width: `${item.confidence}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {!results && !loading && !error && (
            <div style={{ textAlign: 'center', color: 'var(--text-muted)', paddingTop: '3rem' }}>
              Upload an image to see the analysis results here.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
