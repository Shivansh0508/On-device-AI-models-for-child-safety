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

        // Reset UI state
        setLoadingState(true);
        hideResult();
        hideError();
try {
            // ngrok might block CORS or have a browser warning page. 
            // We set headers to request JSON explicitly.
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
