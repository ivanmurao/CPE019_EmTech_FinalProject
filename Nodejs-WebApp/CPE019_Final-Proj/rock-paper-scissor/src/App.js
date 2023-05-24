import React, { useState, useEffect } from 'react';
import './App.css';
import * as tf from '@tensorflow/tfjs';
import { loadLayersModel, browserFromLocalStorage } from '@tensorflow/tfjs';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedImage, setUploadedImage] = useState(null);

  const handleFileInputChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setUploadedImage(URL.createObjectURL(event.target.files[0]));
  };

  const handlePredictClick = async () => {
    if (selectedFile) {
      setIsLoading(true);

      const model = await loadModel();
      if (model) {
        const image = await preprocessImage(selectedFile);
        const predictions = await predict(model, image);
        setPrediction(predictions);
      } else {
        setPrediction('The model failed to load due to an error.');
      }

      setIsLoading(false);
    } else {
      setPrediction('Select an image.');
    }
  };

  const loadModel = async () => {
    try {
      const model = await loadLayersModel('best-model.h5');
      return model;
    } catch (error) {
      console.error('The model failed to load due to an error:', error);
      return null;
    }
  };

  const preprocessImage = async (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const image = new Image();
        image.onload = () => {
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');

          // Calculate the new dimensions to maintain 3:4 aspect ratio
          const maxWidth = 400;
          const maxHeight = 533;
          let width = image.width;
          let height = image.height;

          if (width > maxWidth) {
            height *= maxWidth / width;
            width = maxWidth;
          }

          if (height > maxHeight) {
            width *= maxHeight / height;
            height = maxHeight;
          }

          // Set the canvas dimensions and draw the resized image
          canvas.width = width;
          canvas.height = height;
          ctx.drawImage(image, 0, 0, width, height);

          // Convert the canvas image to a tensor
          const tensor = tf.browser.fromPixels(canvas).expandDims();
          resolve(tensor);
        };
        image.onerror = (error) => {
          reject(error);
        };
        image.src = reader.result;
      };
      reader.onerror = (error) => {
        reject(error);
      };
      reader.readAsDataURL(file);
    });
  };

  const predict = async (model, image) => {
    try {
      const predictions = await model.predict(image).data();
      return predictions;
    } catch (error) {
      console.error('Prediction Error:', error);
      return 'Prediction Error.';
    }
  };

  useEffect(() => {
    return () => {
      // Clean up the uploaded image URL when the component is unmounted
      if (uploadedImage) {
        URL.revokeObjectURL(uploadedImage);
      }
    };
  }, [uploadedImage]);

  return (
    <div className="container">
      <h1 className="title">Rock Paper Scissor Prediction</h1>
        <input
          type="file"
          className="file-input"
          accept="image/*"       
          onChange={handleFileInputChange}
        />
    {selectedFile && (
      <div className="image-container">
        <img src={uploadedImage} alt="Uploaded" className="uploaded-image" />
      </div>
    )}
    <button
      className="predict-btn"
      onClick={handlePredictClick}
      disabled={!selectedFile || isLoading}
    >
      {isLoading ? 'Predicting...' : 'Predict'}
    </button>
    {prediction && <p className="result">Result: {prediction}</p>}
  </div>
  );
}

export default App;