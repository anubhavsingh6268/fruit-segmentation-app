const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function parseResponse(response) {
  const contentType = response.headers.get('content-type') || '';
  const data = contentType.includes('application/json')
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const message = typeof data === 'object' && data !== null ? data.detail || data.error || 'Request failed.' : data || 'Request failed.';
    throw new Error(message);
  }

  return data;
}

export const api = {
  async health() {
    const response = await fetch(`${API_BASE_URL}/health`);
    return parseResponse(response);
  },

  async predict(originalFile, groundTruthFile) {
    const formData = new FormData();
    formData.append('image', originalFile);

    if (groundTruthFile) {
      formData.append('ground_truth_mask', groundTruthFile);
    }

    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
    });

    return parseResponse(response);
  },
};

export { API_BASE_URL };
