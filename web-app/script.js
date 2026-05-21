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
