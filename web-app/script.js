document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('classifier-form');
    const input = document.getElementById('text-input');
    const submitBtn = document.getElementById('submit-btn');
    const resultContainer = document.getElementById('result-container');
    const resultText = document.getElementById('result-text');
    const errorContainer = document.getElementById('error-container');
    const errorText = document.getElementById('error-text');
 form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const textToClassify = input.value.trim();
        if (!textToClassify) return;

        setLoadingState(true);
        hideResult();
        hideError();
try {
            const response = await fetch("https://wrought-mold-confider.ngrok-free.dev/predict", {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'ngrok-skip-browser-warning': "true"
                },
                body: JSON.stringify({ text: textToClassify })
            });

            if (!response.ok) {
                throw new Error(`API returned status: ${response.status}`);
            }
    const data = await response.json();

            let categoryToDisplay = '';

            if (data && typeof data === 'object') {
                // If it returns { category: "..." } or { label: "..." } or { prediction: "..." }
                categoryToDisplay = data.category || data.label || data.prediction || data.result || Object.values(data)[0] || JSON.stringify(data);
            } else {
                // If it returns a plain string
                categoryToDisplay = String(data);
            }

            showResult(categoryToDisplay);
