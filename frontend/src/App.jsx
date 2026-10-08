import { useState } from 'react';
import ImageUploader from './components/ImageUploader';
import Metrics from './components/Metrics';
import ModelSelector from './components/ModelSelector';
import Results from './components/Results';
import { api } from './services/api';

const modelOptions = [{ id: 'fruit-segmentation', label: 'Fruit Segmentation' }];

export default function App() {
  const [selectedModel, setSelectedModel] = useState(modelOptions[0].id);
  const [originalFile, setOriginalFile] = useState(null);
  const [groundTruthFile, setGroundTruthFile] = useState(null);
  const [originalPreview, setOriginalPreview] = useState('');
  const [groundTruthPreview, setGroundTruthPreview] = useState('');
  const [result, setResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleOriginalFileChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setOriginalFile(file);
    setOriginalPreview(URL.createObjectURL(file));
    setErrorMessage('');
  };

  const handleGroundTruthChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setGroundTruthFile(file);
    setGroundTruthPreview(URL.createObjectURL(file));
    setErrorMessage('');
  };

  const handleSegment = async () => {
    if (!originalFile) {
      setErrorMessage('Please upload a fruit image before running segmentation.');
      return;
    }

    setIsLoading(true);
    setErrorMessage('');

    try {
      const response = await api.predict(originalFile, groundTruthFile);
      setResult(response);
    } catch (error) {
      setErrorMessage(error.message || 'Prediction failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="page-header">
        <h1>Fruit Segmentation</h1>
        <p>ResNet50-U-Net Image Segmentation</p>
      </header>

      <main className="container">
        <section className="panel">
          <ModelSelector
            options={modelOptions}
            value={selectedModel}
            onChange={(event) => setSelectedModel(event.target.value)}
          />

          <div className="upload-grid">
            <ImageUploader
              label="Upload Fruit Image"
              accept="image/*"
              preview={originalPreview}
              altText="Selected fruit image"
              onChange={handleOriginalFileChange}
            />

            <ImageUploader
              label="Upload Ground Truth Mask (Optional)"
              helperText="Upload the actual segmentation mask for this image if you want to calculate IoU, Dice, Precision, and Recall."
              accept="image/*"
              preview={groundTruthPreview}
              altText="Ground truth mask preview"
              onChange={handleGroundTruthChange}
            />
          </div>

          <button
            type="button"
            className="predict-button"
            disabled={isLoading}
            onClick={handleSegment}
          >
            {isLoading ? 'Segmenting Image...' : 'Segment Image'}
          </button>

          {errorMessage ? <div className="error-banner">{errorMessage}</div> : null}
        </section>

        {result ? (
          <section className="result-section">
            <Results result={result} />

            {result.metrics_available ? (
              <Metrics metrics={result.metrics} />
            ) : (
              <div className="info-banner">
                Ground-truth mask not provided. Metrics are unavailable.
              </div>
            )}
          </section>
        ) : null}
      </main>
    </div>
  );
}
