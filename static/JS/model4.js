/* model4.js - Deepfake Audio Detection */

const uploadBox = document.getElementById('uploadBox');
const audioInput = document.getElementById('audioInput');
const loading = document.getElementById('loading');
const resultBox = document.getElementById('resultBox');
const startRecBtn = document.getElementById('startRecBtn');
const stopRecBtn = document.getElementById('stopRecBtn');
const recStatus = document.getElementById('recStatus');

let mediaStream = null;
let audioContext = null;
let sourceNode = null;
let processorNode = null;
let recordedChunks = [];
let isRecording = false;

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

if (startRecBtn && stopRecBtn) {
    startRecBtn.addEventListener('click', startRecording);
    stopRecBtn.addEventListener('click', stopRecording);
}

async function startRecording() {
    if (isRecording) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Microphone recording is not supported in this browser.');
        return;
    }

    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 22050 });
        sourceNode = audioContext.createMediaStreamSource(mediaStream);
        processorNode = audioContext.createScriptProcessor(4096, 1, 1);

        recordedChunks = [];
        processorNode.onaudioprocess = (event) => {
            if (!isRecording) return;
            const input = event.inputBuffer.getChannelData(0);
            recordedChunks.push(new Float32Array(input));
        };

        sourceNode.connect(processorNode);
        processorNode.connect(audioContext.destination);

        isRecording = true;
        startRecBtn.disabled = true;
        stopRecBtn.disabled = false;
        recStatus.textContent = 'Recording...';
        recStatus.classList.add('active');
    } catch (err) {
        alert('Unable to access microphone. Please allow microphone permission.');
    }
}

function stopRecording() {
    if (!isRecording) return;
    isRecording = false;

    startRecBtn.disabled = false;
    stopRecBtn.disabled = true;
    recStatus.textContent = 'Processing recording...';

    try {
        if (processorNode) processorNode.disconnect();
        if (sourceNode) sourceNode.disconnect();
        if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
        if (audioContext) audioContext.close();
    } catch (_) {
        // noop
    }

    if (!recordedChunks.length) {
        recStatus.textContent = 'No audio captured';
        recStatus.classList.remove('active');
        return;
    }

    const wavBlob = float32ChunksToWav(recordedChunks, 22050);
    const file = new File([wavBlob], 'recorded_audio.wav', { type: 'audio/wav' });
    recStatus.textContent = 'Recorded audio ready';
    recStatus.classList.remove('active');
    analyzeAudio(file);
}

function float32ChunksToWav(chunks, sampleRate) {
    const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
    const pcm = new Int16Array(totalLength);
    let offset = 0;

    for (const chunk of chunks) {
        for (let i = 0; i < chunk.length; i++) {
            const s = Math.max(-1, Math.min(1, chunk[i]));
            pcm[offset++] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
    }

    const buffer = new ArrayBuffer(44 + pcm.length * 2);
    const view = new DataView(buffer);

    writeAscii(view, 0, 'RIFF');
    view.setUint32(4, 36 + pcm.length * 2, true);
    writeAscii(view, 8, 'WAVE');
    writeAscii(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // fmt chunk length
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // byte rate
    view.setUint16(32, 2, true); // block align
    view.setUint16(34, 16, true); // bits per sample
    writeAscii(view, 36, 'data');
    view.setUint32(40, pcm.length * 2, true);

    let idx = 44;
    for (let i = 0; i < pcm.length; i++, idx += 2) {
        view.setInt16(idx, pcm[i], true);
    }

    return new Blob([view], { type: 'audio/wav' });
}

function writeAscii(view, offset, text) {
    for (let i = 0; i < text.length; i++) {
        view.setUint8(offset + i, text.charCodeAt(i));
    }
}

function resetAudioUI() {
    if (!uploadBox) return;
    uploadBox.style.display = 'block';
    loading.style.display = 'none';
    resultBox.style.display = 'none';
    audioInput.value = '';
}
