const metricLabels = {
  iou: 'IoU',
  dice: 'Dice Score',
  precision: 'Precision',
  recall: 'Recall',
};

export default function Metrics({ metrics }) {
  return (
    <section className="metrics-panel">
      <h2>Segmentation Metrics</h2>
      <div className="metrics-grid">
        {Object.entries(metrics).map(([key, value]) => (
          <div key={key} className="metric-card">
            <span>{metricLabels[key]}</span>
            <strong>{(value * 100).toFixed(2)}%</strong>
          </div>
        ))}
      </div>
    </section>
  );
}
