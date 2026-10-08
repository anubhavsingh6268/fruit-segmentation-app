export default function ModelSelector({ options, value, onChange }) {
  return (
    <div className="selector-box">
      <label className="field-label" htmlFor="model-selector">
        Model
      </label>
      <select id="model-selector" value={value} onChange={onChange}>
        {options.map((option) => (
          <option key={option.id} value={option.id}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}
