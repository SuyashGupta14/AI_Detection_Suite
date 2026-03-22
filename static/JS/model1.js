/* model1.js — Real vs AI Video */

const uploadBox = document.getElementById('uploadBox');
const videoInput = document.getElementById('videoInput');
const resultBox = document.getElementById('resultBox');
const loading = document.getElementById('loading');

uploadBox.addEventListener('click', () => videoInput.click());

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#6c63ff';
});
uploadBox.addEventListener('dragleave', () => {
    uploadBox.style.borderColor = '#3a3a4a';
});
uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files[0]) analyzeVideo(e.dataTransfer.files[0]);
});
videoInput.addEventListener('change', () => {
    if (videoInput.files[0]) analyzeVideo(videoInput.files[0]);
});

function analyzeVideo(file) {
    uploadBox.style.display = 'none';
    loading.style.display = 'block';
    resultBox.style.display = 'none';

    const fd = new FormData();
    fd.append('video', file);

    fetch('/predict', { method: 'POST', body: fd })
        .then(r => r.json())
        .then(data => {
            loading.style.display = 'none';
            resultBox.style.display = 'block';

            if (data.error) { alert('Error: ' + data.error); resetUI(); return; }

            const lbl = document.getElementById('resultLabel');
            lbl.textContent = data.label + ' — ' + data.confidence + '%';
            lbl.className = 'result-label ' + (data.label === 'Real Video' ? 'real' : 'ai');

            document.getElementById('realBar').style.width = data.real_prob + '%';
            document.getElementById('aiBar').style.width = data.ai_prob + '%';
            document.getElementById('realPct').textContent = data.real_prob + '%';
            document.getElementById('aiPct').textContent = data.ai_prob + '%';
            document.getElementById('metaInfo').textContent =
                'Frames: ' + data.frames + '  |  ' + data.conf_level +
                ' confidence  |  XGBoost + EfficientNet-B0';
        })
        .catch(() => {
            loading.style.display = 'none';
            alert('Server error — check terminal');
            resetUI();
        });
}

function resetUI() {
    resultBox.style.display = 'none';
    uploadBox.style.display = 'block';
    videoInput.value = '';
}