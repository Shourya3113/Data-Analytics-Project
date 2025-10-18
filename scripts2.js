const fileInput = document.getElementById('file-upload');
const previewImg = document.getElementById('preview-img');
const thumbsUp = document.getElementById('thumbs-up');
const analyzeBtn = document.getElementById('analyze-btn');
const progressBar = document.getElementById('progress-bar');
const resultSection = document.getElementById('result-section');
const emotionResult = document.getElementById('emotion-result');
const confidence = document.getElementById('confidence');
const emotionBreakdown = document.getElementById('emotion-breakdown');
const saveBtn = document.getElementById('save-btn');
const historySection = document.getElementById('history-section');
const languageSelect = document.getElementById('language-select');

// Theme switching functionality
document.getElementById('light-theme').addEventListener('click', () => {
  document.body.classList.remove('dark-mode', 'classic-mode');
});
document.getElementById('dark-theme').addEventListener('click', () => {
  document.body.classList.add('dark-mode');
  document.body.classList.remove('classic-mode');
});
document.getElementById('classic-theme').addEventListener('click', () => {
  document.body.classList.add('classic-mode');
  document.body.classList.remove('dark-mode');
});

// Language selector
languageSelect.addEventListener('change', (e) => {
  alert(`Language switched to: ${e.target.value}`);
});

// File upload functionality
fileInput.addEventListener('change', handleFileSelect);
function handleFileSelect(e) {
  const file = e.target.files[0];
  const reader = new FileReader();
  
  reader.onloadend = function () {
    previewImg.src = reader.result;
    previewImg.style.display = 'block';
    thumbsUp.style.display = 'block';
  };
  if (file) {
    reader.readAsDataURL(file);
  }
}

// Analyzing the image (mock behavior)
function analyzeImage() {
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing...';
  document.getElementById('progress-section').style.display = 'block';

  // Simulating a delay in analysis
  let progress = 0;
  let interval = setInterval(() => {
    progress += 10;
    progressBar.style.width = `${progress}%`;
    if (progress === 100) {
      clearInterval(interval);
      showResults();
    }
  }, 500);
}

// Displaying results after analysis
function showResults() {
  emotionResult.textContent = 'Emotion: Happy';
  confidence.textContent = 'Confidence: 92%';
  emotionBreakdown.innerHTML = `
    <strong>Emotion Breakdown:</strong><br>
    Happiness: 80%<br>
    Sadness: 15%<br>
    Surprise: 5%
  `;
  resultSection.style.display = 'block';
  analyzeBtn.disabled = false;
  analyzeBtn.textContent = 'Analyze Again';
  document.getElementById('progress-section').style.display = 'none';
}

// Save & Share functionality
saveBtn.addEventListener('click', () => {
  alert('Results saved! Share on social media or email.');
});
