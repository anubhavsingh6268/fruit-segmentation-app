import ImagePreview from './ImagePreview';

export default function ImageUploader({
  label,
  helperText,
  accept,
  onChange,
  preview,
  altText,
  disabled = false,
}) {
  return (
    <div className="uploader-box">
      <div className="uploader-heading">
        <label className="field-label">{label}</label>
        {helperText ? <p className="field-help">{helperText}</p> : null}
      </div>
      <input type="file" accept={accept} onChange={onChange} disabled={disabled} />
      {preview ? (
        <ImagePreview src={preview} altText={altText || label} />
      ) : (
        <div className="empty-preview">No image selected</div>
      )}
    </div>
  );
}
