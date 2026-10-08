export default function ImagePreview({ src, altText }) {
  if (!src) {
    return null;
  }

  return (
    <div className="image-preview">
      <img src={src} alt={altText} />
    </div>
  );
}
