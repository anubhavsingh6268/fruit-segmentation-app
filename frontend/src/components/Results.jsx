export default function Results({ result }) {
  if (!result) {
    return null;
  }

  const buildImageSrc = (value) => `data:image/png;base64,${value}`;

  return (
    <section className="results-panel">
      <h2>Prediction Results</h2>
      <div className="result-grid">
        <div className="result-card">
          <h3>Original Image</h3>
          <img src={buildImageSrc(result.original_image)} alt="Original fruit input" />
        </div>
        <div className="result-card">
          <h3>Predicted Segmentation Mask</h3>
          <img src={buildImageSrc(result.mask_image)} alt="Predicted fruit mask" />
        </div>
        <div className="result-card">
          <h3>Segmentation Overlay</h3>
          <img src={buildImageSrc(result.overlay_image)} alt="Fruit mask overlay" />
        </div>
      </div>
    </section>
  );
}
