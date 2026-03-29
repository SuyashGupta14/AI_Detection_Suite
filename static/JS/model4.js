/* model4.js - Deepfake Audio Detection */

const uploadBox = document.getElementById('uploadBox');
const audioInput = document.getElementById('audioInput');
const loading = document.getElementById('loading');
const resultBox = document.getElementById('resultBox');

if (uploadBox) {
    uploadBox.addEventListener('click', () => audioInput.click());

    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = '#6c63ff';
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.style.borderColor = '#3a3a4a';
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        const f = e.dataTransfer.files[0];
        if (f) analyzeAudio(f);
    });

    audioInput.addEventListener('change', () => {
        if (audioInput.files[0]) analyzeAudio(audioInput.files[0]);
    });
}

function analyzeAudio(file) {
    uploadBox.style.display = 'none';
    loading.style.display = 'block';
    resultBox.style.display = 'none';

    const fd = new FormData();
    fd.append('audio', file);

    fetch('/predict_audio_deepfake', {
        method: 'POST',
        body: fd,
    })
        .then(r => r.json())
        .then(data => {
            loading.style.display = 'none';

            if (data.error) {
                alert('Error: ' + data.error);
                resetAudioUI();
                return;
            }

            resultBox.style.display = 'block';

            const lbl = document.getElementById('resultLabel');
            lbl.textContent = `${data.emoji} ${data.label} - ${data.confidence}%`;
            lbl.className = 'result-label ' + (data.is_fake ? 'ai' : 'real');
            if (data.prediction === 'FAKE') lbl.className = 'result-label ai';
            if (data.prediction === 'REAL') lbl.className = 'result-label real';

            document.getElementById('realBar').style.width = data.real_prob + '%';
            document.getElementById('fakeBar').style.width = data.fake_prob + '%';
            document.getElementById('realPct').textContent = data.real_prob + '%';
            document.getElementById('fakePct').textContent = data.fake_prob + '%';

            document.getElementById('metaInfo').textContent =
                `${data.model_used} · ${data.features} features · ${data.conf_level} confidence`;
        })
        .catch(() => {
            loading.style.display = 'none';
            alert('Server error - check terminal');
            resetAudioUI();
        });
}

function resetAudioUI() {
    if (!uploadBox) return;
    uploadBox.style.display = 'block';
    loading.style.display = 'none';
    resultBox.style.display = 'none';
    audioInput.value = '';
}
